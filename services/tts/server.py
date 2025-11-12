"""
TTS Microservice Server

Real-time text-to-speech with:
- WebSocket streaming
- Voice cloning
- Low-latency chunked synthesis
- Queue-based processing (zero data loss)
- Graceful shutdown
- Rate limiting
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
import uvicorn
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.synthesizer import StreamingSynthesizer
from core.voice_manager import VoiceManager
from core.queue_manager import TTSQueueManager, SynthesisRequest, AudioChunk

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


class TTSService:
    """
    Main TTS service with all components integrated.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        device_index: int = 1,  # GPU 1 for TTS
        chunk_size: int = 50,
        max_connections: int = 50,
    ):
        self.model_path = model_path
        self.device = device
        self.device_index = device_index
        self.chunk_size = chunk_size
        self.max_connections = max_connections

        # Components
        self.synthesizer = StreamingSynthesizer(
            model_path=model_path,
            device=device,
            device_index=device_index,
            chunk_size=chunk_size,
        )

        self.voice_manager = VoiceManager(
            cache_dir="./voices",
            synthesizer=self.synthesizer,
        )

        self.queue_manager = TTSQueueManager()

        # Connection tracking
        self.connections = {}
        self.active_connections = 0
        self.rate_limiter = RateLimiter()

        # Shutdown handling
        self.is_shutting_down = False

        logger.info(
            "tts_service_initialized",
            device=f"{device}:{device_index}",
            chunk_size=chunk_size
        )

    async def start(self):
        """Start service (load models, start workers)"""
        logger.info("starting_tts_service")

        # Load synthesizer model (takes 3-5 seconds)
        await self.synthesizer.load()

        # Start queue manager workers
        await self.queue_manager.start()

        # Start TTS worker
        asyncio.create_task(self._tts_worker())

        logger.info("tts_service_ready")

    async def _tts_worker(self):
        """
        Background worker: Process TTS queue.

        Gets synthesis requests and generates audio.
        """
        logger.info("tts_worker_started")

        while not self.is_shutting_down:
            try:
                # Get next synthesis request
                request = await self.queue_manager.get_next_request()

                if request is None:
                    await asyncio.sleep(0.01)
                    continue

                # Get voice embedding
                voice_embedding = None
                if request.voice_id and request.voice_id != "default":
                    voice_embedding = await self.voice_manager.get_voice(request.voice_id)

                    if voice_embedding is None:
                        logger.warning(
                            "voice_not_found",
                            connection_id=request.connection_id,
                            voice_id=request.voice_id
                        )
                        # Use default voice
                        voice_embedding = None

                # Synthesize with streaming
                chunk_id = 0
                try:
                    async for audio_chunk in self.synthesizer.synthesize_streaming(
                        text=request.text,
                        voice_embedding=voice_embedding,
                        chunk_size=request.chunk_size,
                        exaggeration=request.exaggeration,
                    ):
                        # Enqueue audio chunk
                        await self.queue_manager.enqueue_audio_chunk(
                            request.connection_id,
                            audio_chunk.tobytes(),
                            chunk_id,
                            is_final=False
                        )
                        chunk_id += 1

                    # Send final marker
                    await self.queue_manager.enqueue_audio_chunk(
                        request.connection_id,
                        b'',
                        chunk_id,
                        is_final=True
                    )

                    logger.info(
                        "synthesis_completed",
                        connection_id=request.connection_id,
                        text_length=len(request.text),
                        chunks=chunk_id
                    )

                except Exception as e:
                    logger.error(
                        "synthesis_failed",
                        connection_id=request.connection_id,
                        error=str(e),
                        exc_info=True
                    )

                # Mark request as done
                await self.queue_manager.mark_request_done()

            except Exception as e:
                logger.error("tts_worker_error", error=str(e), exc_info=True)
                await asyncio.sleep(1.0)

    async def handle_connection(self, websocket: WebSocket, conn_id: str):
        """Handle WebSocket connection"""

        # Register connection
        output_queue = self.queue_manager.register_connection(conn_id)

        self.connections[conn_id] = {
            'websocket': websocket,
            'connected_at': time.time(),
        }
        self.active_connections += 1

        logger.info(
            "connection_established",
            connection_id=conn_id,
            active_connections=self.active_connections
        )

        try:
            # Receiver task: WebSocket → Queue
            async def receive_requests():
                async for message in websocket.iter_text():
                    try:
                        import json
                        data = json.loads(message)

                        if data.get('type') == 'synthesize':
                            # Enqueue synthesis request
                            await self.queue_manager.enqueue_request(
                                connection_id=conn_id,
                                text=data.get('text', ''),
                                voice_id=data.get('voice_id', 'default'),
                                chunk_size=data.get('chunk_size', self.chunk_size),
                                exaggeration=data.get('exaggeration', 0.5),
                                streaming=data.get('streaming', True),
                            )

                        elif data.get('type') == 'register_voice':
                            # Register new voice
                            voice_id = data.get('voice_id')
                            reference_audio = data.get('reference_audio')

                            if voice_id and reference_audio:
                                try:
                                    await self.voice_manager.register_voice(
                                        voice_id=voice_id,
                                        reference_audio_b64=reference_audio,
                                        description=data.get('description', '')
                                    )

                                    await websocket.send_json({
                                        'type': 'voice_registered',
                                        'voice_id': voice_id,
                                    })

                                except Exception as e:
                                    await websocket.send_json({
                                        'type': 'error',
                                        'message': f"Voice registration failed: {e}"
                                    })

                        elif data.get('type') == 'list_voices':
                            # List available voices
                            voices = self.voice_manager.list_voices()
                            await websocket.send_json({
                                'type': 'voice_list',
                                'voices': voices
                            })

                    except Exception as e:
                        logger.error(
                            "request_processing_error",
                            connection_id=conn_id,
                            error=str(e)
                        )

            # Sender task: Queue → WebSocket
            async def send_audio():
                while True:
                    chunk = await output_queue.get()

                    try:
                        # Send audio chunk
                        if not chunk.is_final:
                            await websocket.send_bytes(chunk.audio_data)
                        else:
                            # Send completion marker
                            await websocket.send_json({
                                'type': 'synthesis_complete',
                                'chunk_id': chunk.chunk_id
                            })

                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(
                            "send_error",
                            connection_id=conn_id,
                            error=str(e)
                        )
                        break

            # Run both tasks
            await asyncio.gather(receive_requests(), send_audio())

        except WebSocketDisconnect:
            logger.info("connection_closed", connection_id=conn_id)

        except Exception as e:
            logger.error(
                "connection_error",
                connection_id=conn_id,
                error=str(e),
                exc_info=True
            )

        finally:
            # Cleanup
            self.queue_manager.unregister_connection(conn_id)
            if conn_id in self.connections:
                del self.connections[conn_id]
            self.active_connections -= 1

            logger.info(
                "connection_cleaned_up",
                connection_id=conn_id,
                active_connections=self.active_connections
            )

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("shutting_down")
        self.is_shutting_down = True

        # Wait for queues to empty
        await self.queue_manager.wait_until_empty(timeout=30.0)

        # Stop workers
        await self.queue_manager.stop()

        # Cleanup synthesizer
        await self.synthesizer.cleanup()

        logger.info("shutdown_complete")


class RateLimiter:
    """Simple rate limiter by IP"""

    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}

    def check(self, client_id: str) -> bool:
        """Returns True if allowed, False if rate limited"""
        now = time.time()

        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window
            ]
        else:
            self.requests[client_id] = []

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True


# Create FastAPI app
app = FastAPI(title="TTS Service", version="0.1.0")

# Global service instance
service: Optional[TTSService] = None


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global service

    # Get GPU index from environment or default to 1
    import os
    device_index = int(os.getenv("CUDA_VISIBLE_DEVICES", "1"))

    service = TTSService(
        model_path=None,  # Set path to Chatterbox model if needed
        device="cuda",
        device_index=device_index,
        chunk_size=50,
        max_connections=50,
    )

    await service.start()

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logger.info("received_signal", signal=sig)
        asyncio.create_task(service.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


@app.websocket("/v1/stream/tts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS"""

    # Get client IP for rate limiting
    client_ip = websocket.client.host

    # Check rate limit
    if not service.rate_limiter.check(client_ip):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Rate limit exceeded")
        return

    # Check max connections
    if service.active_connections >= service.max_connections:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Max connections reached")
        return

    await websocket.accept()

    # Generate connection ID
    conn_id = str(uuid4())

    # Handle connection
    await service.handle_connection(websocket, conn_id)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not service or not service.synthesizer.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"}
        )

    # Get GPU info
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_id = service.device_index
        gpu_info = {
            "gpu_id": gpu_id,
            "gpu_name": torch.cuda.get_device_name(gpu_id),
            "memory_allocated_gb": torch.cuda.memory_allocated(gpu_id) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(gpu_id) / 1e9,
        }

    return {
        "status": "healthy",
        "device": f"{service.device}:{service.device_index}",
        "active_connections": service.active_connections,
        "queue_metrics": service.queue_manager.get_metrics(),
        "synthesizer_stats": service.synthesizer.get_stats(),
        "voice_stats": service.voice_manager.get_stats(),
        "gpu": gpu_info,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    return service.queue_manager.get_metrics()


if __name__ == "__main__":
    # Get port from environment (for load balancing multiple instances)
    import os
    port = int(os.getenv("TTS_PORT", "8002"))
    instance_id = os.getenv("TTS_INSTANCE_ID", "1")

    logger.info(
        "starting_tts_server",
        port=port,
        instance_id=instance_id
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
