"""
ASR Microservice Server

Real-time speech-to-text with:
- WebSocket streaming
- Hybrid VAD (Smart Turn v3 + fallback)
- faster-whisper transcription
- Multi-format audio support
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.audio_processor import AudioProcessor, AudioBuffer
from core.vad import HybridVAD
from core.transcriber import StreamingTranscriber
from core.queue_manager import ASRQueueManager, AudioChunk, TranscriptionResult

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


class ASRService:
    """
    Main ASR service with all components integrated.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        vad_model_path: Optional[str] = None,
        max_connections: int = 50,
    ):
        self.model_name = model_name
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.max_connections = max_connections

        # Components
        self.audio_processor = AudioProcessor()
        self.vad = HybridVAD(model_path=vad_model_path)
        self.transcriber = StreamingTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            device_index=device_index,
        )
        self.queue_manager = ASRQueueManager()

        # Connection tracking
        self.connections = {}
        self.active_connections = 0
        self.rate_limiter = RateLimiter()

        # Shutdown handling
        self.is_shutting_down = False

        logger.info(
            "asr_service_initialized",
            model=model_name,
            device=f"{device}:{device_index}",
            compute_type=compute_type
        )

    async def start(self):
        """Start service (load models, start workers)"""
        logger.info("starting_asr_service")

        # Load transcriber model (takes 3-5 seconds)
        await self.transcriber.load()

        # Start queue manager workers
        await self.queue_manager.start()

        # Start ASR worker
        asyncio.create_task(self._asr_worker())

        logger.info("asr_service_ready")

    async def _asr_worker(self):
        """
        Background worker: Process ASR queue.

        Gets batches of audio from queue and transcribes them.
        """
        logger.info("asr_worker_started")

        while not self.is_shutting_down:
            try:
                # Get batch of audio chunks
                batch = await self.queue_manager.get_batch_for_processing()

                if not batch:
                    await asyncio.sleep(0.01)
                    continue

                # Process batch
                for chunk in batch:
                    try:
                        # Transcribe
                        result = await self.transcriber.transcribe(
                            chunk.audio_data,
                            beam_size=5,
                            word_timestamps=True
                        )

                        # Send result back to connection
                        await self.queue_manager.enqueue_result(
                            chunk.connection_id,
                            TranscriptionResult(
                                connection_id=chunk.connection_id,
                                text=result['text'],
                                is_final=True,
                                confidence=result.get('language_probability', 0.0),
                                words=result.get('segments', [{}])[0].get('words', [])
                            )
                        )

                        logger.info(
                            "transcription_completed",
                            connection_id=chunk.connection_id,
                            text_length=len(result['text']),
                            language=result.get('language', 'unknown')
                        )

                    except Exception as e:
                        logger.error(
                            "transcription_failed",
                            connection_id=chunk.connection_id,
                            error=str(e),
                            exc_info=True
                        )

                # Mark batch as done
                await self.queue_manager.mark_batch_done(len(batch))

            except Exception as e:
                logger.error("asr_worker_error", error=str(e), exc_info=True)
                await asyncio.sleep(1.0)

    async def handle_connection(self, websocket: WebSocket, conn_id: str):
        """Handle WebSocket connection"""

        # Register connection
        output_queue = self.queue_manager.register_connection(conn_id)
        buffer = AudioBuffer()

        self.connections[conn_id] = {
            'websocket': websocket,
            'buffer': buffer,
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
            async def receive_audio():
                async for message in websocket.iter_bytes():
                    # Process audio
                    try:
                        # Normalize audio format (handles any input format)
                        audio = self.audio_processor.process(
                            message,
                            input_sample_rate=16000,  # TODO: Make configurable
                            input_format="float32",
                            input_channels=1,
                        )

                        # Add to buffer
                        buffer.add(audio)

                        # Run VAD
                        vad_result = await self.vad.detect_turn_end(audio)

                        if vad_result['turn_ended']:
                            # Get buffered audio
                            buffered_audio = buffer.get_and_clear()

                            # Validate audio quality
                            validation = self.audio_processor.validate_audio(
                                buffered_audio,
                                sample_rate=16000
                            )

                            if not validation['valid']:
                                logger.warning(
                                    "invalid_audio",
                                    connection_id=conn_id,
                                    reason=validation['reason']
                                )
                                continue

                            # Enqueue for transcription
                            await self.queue_manager.enqueue_audio(
                                conn_id,
                                buffered_audio,
                                sample_rate=16000
                            )

                            logger.debug(
                                "audio_enqueued",
                                connection_id=conn_id,
                                duration=len(buffered_audio) / 16000,
                                vad_method=vad_result['method']
                            )

                    except Exception as e:
                        logger.error(
                            "audio_processing_error",
                            connection_id=conn_id,
                            error=str(e)
                        )

            # Sender task: Queue → WebSocket
            async def send_results():
                while True:
                    result = await output_queue.get()

                    try:
                        # Send transcription result
                        await websocket.send_json({
                            'type': 'final' if result.is_final else 'partial',
                            'text': result.text,
                            'confidence': result.confidence,
                            'words': result.words,
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
            await asyncio.gather(receive_audio(), send_results())

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

        # Cleanup transcriber
        await self.transcriber.cleanup()

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

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window
            ]
        else:
            self.requests[client_id] = []

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add request
        self.requests[client_id].append(now)
        return True


# Create FastAPI app
app = FastAPI(title="ASR Service", version="0.1.0")

# Global service instance
service: Optional[ASRService] = None


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global service

    # When CUDA_VISIBLE_DEVICES is set, always use device 0
    # (the startup script controls which physical GPU is visible)
    import os
    device_index = 0

    service = ASRService(
        model_name="large-v3",
        device="cuda",
        device_index=device_index,
        compute_type="float16",
        vad_model_path="/mnt/d/voice-system/gonova-asr-tts/models/smart-turn-v3.0.onnx",
        max_connections=50,
    )

    await service.start()

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logger.info("received_signal", signal=sig)
        asyncio.create_task(service.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


@app.websocket("/v1/stream/asr")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming ASR"""

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
    if not service or not service.transcriber.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"}
        )

    # Get GPU info
    import torch
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
        "model": service.model_name,
        "device": f"{service.device}:{service.device_index}",
        "active_connections": service.active_connections,
        "queue_metrics": service.queue_manager.get_metrics(),
        "transcriber_stats": service.transcriber.get_stats(),
        "vad_stats": service.vad.get_stats(),
        "gpu": gpu_info,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    # TODO: Implement prometheus_client metrics
    return service.queue_manager.get_metrics()


if __name__ == "__main__":
    # Get port from environment (for load balancing multiple instances)
    import os
    port = int(os.getenv("ASR_PORT", "8001"))
    instance_id = os.getenv("ASR_INSTANCE_ID", "1")

    logger.info(
        "starting_asr_server",
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
