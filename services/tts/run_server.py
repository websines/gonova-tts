#!/usr/bin/env python3
"""
TTS Server Entry Point

This file MUST be the entry point to avoid multiprocessing issues with vLLM.
Keep this file minimal - vLLM uses spawn multiprocessing which re-executes
the main script in subprocesses.
"""

from chatterbox_vllm.tts import ChatterboxTTS

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    # Add parent to path for local imports
    sys.path.insert(0, str(Path(__file__).parent))

    # Now import everything else AFTER chatterbox_vllm has registered its tokenizer
    import asyncio
    import signal
    import time
    from uuid import uuid4

    import structlog
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
    from fastapi.responses import JSONResponse
    import uvicorn

    from core.synthesizer import StreamingSynthesizer
    from core.voice_manager import VoiceManager
    from core.queue_manager import TTSQueueManager

    # Setup logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ]
    )
    logger = structlog.get_logger()

    class RateLimiter:
        def __init__(self, max_requests: int = 100, window: int = 60):
            self.max_requests = max_requests
            self.window = window
            self.requests = {}

        def check(self, client_id: str) -> bool:
            now = time.time()
            if client_id in self.requests:
                self.requests[client_id] = [
                    t for t in self.requests[client_id] if now - t < self.window
                ]
            else:
                self.requests[client_id] = []
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            self.requests[client_id].append(now)
            return True

    class TTSService:
        def __init__(self, device_index=0, chunk_size=50, max_connections=50):
            self.device = "cuda"
            self.device_index = device_index
            self.chunk_size = chunk_size
            self.max_connections = max_connections

            self.synthesizer = StreamingSynthesizer(
                device="cuda",
                device_index=device_index,
                chunk_size=chunk_size,
            )
            self.voice_manager = VoiceManager(
                cache_dir="./voices",
                synthesizer=self.synthesizer,
            )
            self.queue_manager = TTSQueueManager()

            self.connections = {}
            self.active_connections = 0
            self.rate_limiter = RateLimiter()
            self.is_shutting_down = False

            logger.info("tts_service_initialized", device=f"cuda:{device_index}")

        def load_model_sync(self):
            logger.info("loading_tts_model_sync")
            self.synthesizer.load_sync()
            logger.info("model_loaded")

        async def _tts_worker(self):
            logger.info("tts_worker_started")
            while not self.is_shutting_down:
                try:
                    request = await self.queue_manager.get_next_request()
                    if request is None:
                        await asyncio.sleep(0.01)
                        continue

                    voice_embedding = None
                    if request.voice_id and request.voice_id != "default":
                        voice_embedding = await self.voice_manager.get_voice(request.voice_id)

                    chunk_id = 0
                    try:
                        async for audio_chunk in self.synthesizer.synthesize_streaming(
                            text=request.text,
                            voice_embedding=voice_embedding,
                            chunk_size=request.chunk_size,
                            exaggeration=request.exaggeration,
                        ):
                            await self.queue_manager.enqueue_audio_chunk(
                                request.connection_id,
                                audio_chunk.tobytes(),
                                chunk_id,
                                is_final=False
                            )
                            chunk_id += 1

                        await self.queue_manager.enqueue_audio_chunk(
                            request.connection_id, b'', chunk_id, is_final=True
                        )
                        logger.info("synthesis_completed",
                                   connection_id=request.connection_id,
                                   chunks=chunk_id)
                    except Exception as e:
                        logger.error("synthesis_failed", error=str(e))

                    await self.queue_manager.mark_request_done()
                except Exception as e:
                    logger.error("tts_worker_error", error=str(e))
                    await asyncio.sleep(1.0)

        async def handle_connection(self, websocket: WebSocket, conn_id: str):
            output_queue = self.queue_manager.register_connection(conn_id)
            self.connections[conn_id] = {'websocket': websocket, 'connected_at': time.time()}
            self.active_connections += 1
            logger.info("connection_established", connection_id=conn_id)

            try:
                async def receive_requests():
                    import json
                    async for message in websocket.iter_text():
                        try:
                            data = json.loads(message)
                            if data.get('type') == 'synthesize':
                                await self.queue_manager.enqueue_request(
                                    connection_id=conn_id,
                                    text=data.get('text', ''),
                                    voice_id=data.get('voice_id', 'default'),
                                    chunk_size=data.get('chunk_size', self.chunk_size),
                                    exaggeration=data.get('exaggeration', 0.5),
                                    streaming=data.get('streaming', True),
                                )
                            elif data.get('type') == 'register_voice':
                                voice_id = data.get('voice_id')
                                ref_audio = data.get('reference_audio')
                                if voice_id and ref_audio:
                                    try:
                                        await self.voice_manager.register_voice(
                                            voice_id=voice_id,
                                            reference_audio_b64=ref_audio,
                                            description=data.get('description', '')
                                        )
                                        await websocket.send_json({'type': 'voice_registered', 'voice_id': voice_id})
                                    except Exception as e:
                                        await websocket.send_json({'type': 'error', 'message': str(e)})
                            elif data.get('type') == 'list_voices':
                                voices = self.voice_manager.list_voices()
                                await websocket.send_json({'type': 'voice_list', 'voices': voices})
                        except Exception as e:
                            logger.error("request_error", error=str(e))

                async def send_audio():
                    while True:
                        chunk = await output_queue.get()
                        try:
                            if not chunk.is_final:
                                await websocket.send_bytes(chunk.audio_data)
                            else:
                                await websocket.send_json({'type': 'synthesis_complete', 'chunk_id': chunk.chunk_id})
                        except WebSocketDisconnect:
                            break
                        except Exception as e:
                            logger.error("send_error", error=str(e))
                            break

                await asyncio.gather(receive_requests(), send_audio())
            except WebSocketDisconnect:
                pass
            finally:
                self.queue_manager.unregister_connection(conn_id)
                self.connections.pop(conn_id, None)
                self.active_connections -= 1

        async def shutdown(self):
            logger.info("shutting_down")
            self.is_shutting_down = True
            await self.queue_manager.wait_until_empty(timeout=30.0)
            await self.queue_manager.stop()
            await self.synthesizer.cleanup()

    # Create FastAPI app
    app = FastAPI(title="TTS Service", version="0.1.0")
    service = None

    @app.on_event("startup")
    async def startup():
        global service
        if service is None:
            raise RuntimeError("Service not initialized")
        await service.queue_manager.start()
        asyncio.create_task(service._tts_worker())
        logger.info("tts_service_ready")

        def signal_handler(sig, frame):
            asyncio.create_task(service.shutdown())
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    @app.websocket("/v1/stream/tts")
    async def websocket_endpoint(websocket: WebSocket):
        client_ip = websocket.client.host
        if not service.rate_limiter.check(client_ip):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if service.active_connections >= service.max_connections:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        await service.handle_connection(websocket, str(uuid4()))

    @app.get("/health")
    async def health_check():
        if not service or not service.synthesizer.is_loaded:
            return JSONResponse(status_code=503, content={"status": "unhealthy"})
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_id": service.device_index,
                    "gpu_name": torch.cuda.get_device_name(service.device_index),
                    "memory_allocated_gb": torch.cuda.memory_allocated(service.device_index) / 1e9,
                }
        except:
            pass
        return {
            "status": "healthy",
            "active_connections": service.active_connections,
            "gpu": gpu_info,
        }

    @app.get("/metrics")
    async def metrics():
        return service.queue_manager.get_metrics()

    # Main execution
    port = int(os.getenv("TTS_PORT", "8002"))
    max_connections = int(os.getenv("TTS_MAX_CONNECTIONS", "50"))

    logger.info("starting_tts_server", port=port, max_connections=max_connections)

    service = TTSService(device_index=0, max_connections=max_connections)
    service.load_model_sync()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
