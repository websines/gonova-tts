"""
TTS WebSocket Server - Simple streaming implementation.

Endpoints:
- WebSocket /v1/stream/tts - Streaming TTS
- GET /health - Health check
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="TTS Service", version="1.0.0")

# Global synthesizer - set in main
synthesizer = None

# Thread pool for running sync generator in async context
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if synthesizer is None or not synthesizer.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"}
        )
    return {
        "status": "healthy",
        "sample_rate": synthesizer.sample_rate,
    }


@app.websocket("/v1/stream/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Send JSON: {"type": "synthesize", "text": "Hello world", "exaggeration": 0.5}
    Receive: Binary audio chunks (float32, 24kHz) + JSON completion message
    """
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            # Receive request
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("type") != "synthesize":
                continue

            text = data.get("text", "")
            if not text.strip():
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue

            exaggeration = data.get("exaggeration", 0.5)
            voice_path = data.get("voice_path")

            logger.info(f"Synthesizing: '{text[:50]}...'")

            # Run sync generator in thread pool and stream results
            loop = asyncio.get_event_loop()
            chunk_count = 0

            def generate():
                """Run generator in thread."""
                return list(synthesizer.generate_stream(
                    text=text,
                    voice_path=voice_path,
                    exaggeration=exaggeration,
                ))

            # Get all chunks from thread
            chunks = await loop.run_in_executor(executor, generate)

            # Send chunks to client
            for audio_bytes in chunks:
                await websocket.send_bytes(audio_bytes)
                chunk_count += 1
                logger.info(f"Sent chunk {chunk_count}: {len(audio_bytes)} bytes")

            # Send completion
            await websocket.send_json({
                "type": "synthesis_complete",
                "chunks": chunk_count
            })
            logger.info(f"Synthesis complete: {chunk_count} chunks")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


if __name__ == "__main__":
    import os

    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent))

    from core.synthesizer import StreamingSynthesizer

    # Config
    port = int(os.getenv("TTS_PORT", "8002"))

    logger.info(f"Starting TTS server on port {port}")

    # Create and load synthesizer
    synthesizer = StreamingSynthesizer(
        device="cuda",
        chunk_size=25,
    )

    logger.info("Loading model...")
    synthesizer.load()
    logger.info("Model ready!")

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
