#!/usr/bin/env python3
"""
Streaming TTS Server Entry Point

Uses vLLM's AsyncLLM for token-level streaming with progressive S3Gen audio synthesis.
This achieves much lower TTFB than the batch-based approach.

IMPORTANT: This file MUST be the entry point. The chatterbox_vllm import at module
level is CRITICAL for vLLM's spawn multiprocessing to work correctly.
"""

# CRITICAL: Import chatterbox_vllm at TOP LEVEL before anything else
# This registers the EnTokenizer with vLLM's TokenizerRegistry
from chatterbox_vllm.tts import ChatterboxTTS as _ChatterboxTTS  # noqa: F401

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    # Add parent to path for local imports
    sys.path.insert(0, str(Path(__file__).parent))

    import asyncio
    import signal
    import time
    from uuid import uuid4

    import structlog
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
    from fastapi.responses import JSONResponse
    import uvicorn

    from core.streaming_synthesizer import StreamingSynthesizer
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

    class StreamingTTSService:
        """Streaming TTS service using vLLM AsyncLLM for low-latency generation."""

        def __init__(self, device_index=0, chunk_size=25, context_window=50, max_connections=50):
            self.device = "cuda"
            self.device_index = device_index
            self.chunk_size = chunk_size
            self.context_window = context_window
            self.max_connections = max_connections

            self.synthesizer = StreamingSynthesizer(
                device="cuda",
                device_index=device_index,
                chunk_size=chunk_size,
                context_window=context_window,
            )
            self.voice_manager = VoiceManager(
                cache_dir="./voices",
                synthesizer=self.synthesizer,
            )

            self.connections = {}
            self.active_connections = 0
            self.rate_limiter = RateLimiter()
            self.is_shutting_down = False

            logger.info("streaming_tts_service_initialized", device=f"cuda:{device_index}")

        def load_model_sync(self):
            logger.info("loading_streaming_tts_model")
            self.synthesizer.load_sync()
            logger.info("model_loaded")

        async def handle_connection(self, websocket: WebSocket, conn_id: str):
            """Handle WebSocket connection with streaming audio."""
            self.connections[conn_id] = {'websocket': websocket, 'connected_at': time.time()}
            self.active_connections += 1
            logger.info("connection_established", connection_id=conn_id)

            try:
                import json
                async for message in websocket.iter_text():
                    try:
                        data = json.loads(message)

                        if data.get('type') == 'synthesize':
                            text = data.get('text', '')
                            voice_id = data.get('voice_id', 'default')
                            exaggeration = data.get('exaggeration', 0.5)
                            chunk_size = data.get('chunk_size', self.chunk_size)

                            # Get voice embedding if not default
                            voice_embedding = None
                            if voice_id and voice_id != 'default':
                                voice_embedding = await self.voice_manager.get_voice(voice_id)

                            # Stream audio chunks directly to client
                            chunk_id = 0
                            try:
                                async for audio_chunk, metrics in self.synthesizer.generate_stream(
                                    text=text,
                                    voice_embedding=voice_embedding,
                                    exaggeration=exaggeration,
                                    chunk_size=chunk_size,
                                ):
                                    # Send binary audio chunk
                                    await websocket.send_bytes(audio_chunk.tobytes())
                                    chunk_id += 1

                                    # Log first chunk latency
                                    if chunk_id == 1:
                                        logger.info(
                                            "first_chunk_sent",
                                            connection_id=conn_id,
                                            ttfb_ms=metrics.latency_to_first_chunk * 1000
                                        )

                                # Send completion message
                                await websocket.send_json({
                                    'type': 'synthesis_complete',
                                    'chunk_id': chunk_id,
                                    'metrics': {
                                        'ttfb_ms': metrics.latency_to_first_chunk * 1000 if metrics.latency_to_first_chunk else None,
                                        'total_ms': metrics.total_generation_time * 1000 if metrics.total_generation_time else None,
                                        'rtf': metrics.rtf,
                                        'chunks': metrics.chunk_count,
                                    }
                                })

                                logger.info(
                                    "synthesis_completed",
                                    connection_id=conn_id,
                                    chunks=chunk_id,
                                    rtf=metrics.rtf
                                )

                            except Exception as e:
                                logger.error("synthesis_failed", error=str(e))
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': str(e)
                                })

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
                                    await websocket.send_json({
                                        'type': 'voice_registered',
                                        'voice_id': voice_id
                                    })
                                except Exception as e:
                                    await websocket.send_json({
                                        'type': 'error',
                                        'message': str(e)
                                    })

                        elif data.get('type') == 'list_voices':
                            voices = self.voice_manager.list_voices()
                            await websocket.send_json({
                                'type': 'voice_list',
                                'voices': voices
                            })

                    except Exception as e:
                        logger.error("request_error", error=str(e))

            except WebSocketDisconnect:
                pass
            finally:
                self.connections.pop(conn_id, None)
                self.active_connections -= 1
                logger.info("connection_closed", connection_id=conn_id)

        async def shutdown(self):
            logger.info("shutting_down")
            self.is_shutting_down = True
            await self.synthesizer.cleanup()
            logger.info("shutdown_complete")

    # Create FastAPI app
    app = FastAPI(title="Streaming TTS Service", version="0.2.0")
    service = None

    @app.on_event("startup")
    async def startup():
        global service
        if service is None:
            raise RuntimeError("Service not initialized")
        logger.info("streaming_tts_service_ready")

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
            "mode": "streaming",
            "active_connections": service.active_connections,
            "chunk_size": service.chunk_size,
            "context_window": service.context_window,
            "gpu": gpu_info,
            "stats": service.synthesizer.get_stats(),
        }

    @app.get("/metrics")
    async def metrics():
        return service.synthesizer.get_stats()

    # Main execution
    port = int(os.getenv("TTS_PORT", "8002"))
    max_connections = int(os.getenv("TTS_MAX_CONNECTIONS", "50"))
    chunk_size = int(os.getenv("TTS_CHUNK_SIZE", "25"))
    context_window = int(os.getenv("TTS_CONTEXT_WINDOW", "50"))

    logger.info(
        "starting_streaming_tts_server",
        port=port,
        max_connections=max_connections,
        chunk_size=chunk_size,
        context_window=context_window,
    )

    service = StreamingTTSService(
        device_index=0,
        chunk_size=chunk_size,
        context_window=context_window,
        max_connections=max_connections,
    )
    service.load_model_sync()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
