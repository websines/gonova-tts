"""
TTS synthesizer using Chatterbox-streaming.

Handles:
- Streaming speech synthesis
- Voice cloning from reference audio
- Low-latency chunked generation
- GPU-accelerated inference
"""

import logging
import time
import torch
import torchaudio
import numpy as np
from typing import Optional, AsyncGenerator
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class StreamingSynthesizer:
    """
    Chatterbox-streaming TTS with voice cloning.

    Pre-loads model at startup to avoid cold start latency.
    Supports streaming audio generation chunk-by-chunk.

    NOTE: Chatterbox must be installed separately from GitHub:
    git clone https://github.com/davidbrowne17/chatterbox-streaming
    pip install -e ./chatterbox-streaming
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        device_index: int = 1,  # GPU 1 for TTS
        chunk_size: int = 50,
        sample_rate: int = 24000,
    ):
        """
        Args:
            model_path: Path to Chatterbox model
            device: "cuda" or "cpu"
            device_index: GPU index (1 for second 3090)
            chunk_size: Tokens per audio chunk (affects latency)
            sample_rate: Output sample rate
        """
        self.model_path = model_path
        self.device = f"{device}:{device_index}" if device == "cuda" else device
        self.device_index = device_index
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        self.model = None
        self.is_loaded = False

        # Stats
        self.stats = {
            'syntheses': 0,
            'total_latency': 0.0,
            'first_chunk_latency': 0.0,
            'errors': 0,
        }

    async def load(self):
        """
        Load model at startup (avoid cold start).

        This takes 3-5 seconds but only happens once.
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(
            f"Loading Chatterbox model on {self.device}"
        )

        start_time = time.time()

        try:
            # Import Chatterbox (must be installed separately)
            try:
                from chatterbox.tts import ChatterboxTTS
            except ImportError:
                raise ImportError(
                    "chatterbox-streaming not installed. "
                    "Install from: https://github.com/davidbrowne17/chatterbox-streaming"
                )

            # Load model using from_pretrained
            self.model = ChatterboxTTS.from_pretrained(device=self.device)

            # Warm up GPU with dummy synthesis
            logger.info("Warming up GPU...")
            dummy_text = "Hello, this is a test."
            loop = asyncio.get_event_loop()
            _ = await loop.run_in_executor(None, self._synthesize_sync, dummy_text)

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            self.is_loaded = True

        except ImportError as e:
            logger.error(f"Import error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def synthesize_streaming(
        self,
        text: str,
        voice_embedding: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        exaggeration: float = 0.5,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with streaming.

        Yields audio chunks as they're generated (low latency).

        Args:
            text: Text to synthesize
            voice_embedding: Voice embedding for cloning (None = default voice)
            chunk_size: Tokens per chunk (None = use default)
            exaggeration: Emotion intensity (0.0-1.0+)

        Yields:
            np.ndarray: Audio chunks (Float32, sample_rate)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first")

        if not text.strip():
            return

        chunk_size = chunk_size or self.chunk_size
        start_time = time.time()
        first_chunk_time = None

        try:
            # Run synthesis in executor (blocking call)
            loop = asyncio.get_event_loop()

            # Generate chunks
            chunk_id = 0
            async for audio_chunk in self._stream_generate(
                text,
                voice_embedding,
                chunk_size,
                exaggeration
            ):
                # Track first chunk latency
                if chunk_id == 0:
                    first_chunk_time = time.time() - start_time
                    self.stats['first_chunk_latency'] += first_chunk_time
                    logger.debug(f"First chunk in {first_chunk_time*1000:.0f}ms")

                yield audio_chunk
                chunk_id += 1

            total_time = time.time() - start_time
            self.stats['syntheses'] += 1
            self.stats['total_latency'] += total_time

            logger.debug(
                f"Synthesized '{text[:30]}...' in {total_time*1000:.0f}ms "
                f"({chunk_id} chunks)"
            )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Synthesis error: {e}")
            raise

    async def _stream_generate(
        self,
        text: str,
        voice_embedding: Optional[torch.Tensor],
        chunk_size: int,
        exaggeration: float
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream-generate audio chunks.

        Runs the synchronous Chatterbox generator in a thread pool
        and yields chunks via an async queue to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        error_holder = [None]  # Hold any exception from the thread

        def run_sync_generator():
            """Run blocking generator in thread, push chunks to async queue."""
            try:
                for audio_chunk, metrics in self.model.generate_stream(
                    text,
                    audio_prompt_path=voice_embedding if isinstance(voice_embedding, str) else None,
                    chunk_size=chunk_size,
                    exaggeration=exaggeration,
                    cfg_weight=3.0,
                ):
                    # Convert to numpy if needed
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_chunk = audio_chunk.cpu().numpy()

                    # Push chunk to async queue (thread-safe)
                    asyncio.run_coroutine_threadsafe(queue.put(audio_chunk), loop)

                # Signal completion
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            except Exception as e:
                error_holder[0] = e
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        try:
            # Start generator in thread pool
            loop.run_in_executor(None, run_sync_generator)

            # Yield chunks as they arrive from the queue
            while True:
                chunk = await queue.get()

                if chunk is None:
                    # Check if there was an error
                    if error_holder[0] is not None:
                        raise error_holder[0]
                    break

                yield chunk

        except Exception as e:
            # Fallback: Non-streaming synthesis
            logger.warning(f"Streaming failed ({e}), using full synthesis")
            audio = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                voice_embedding,
                exaggeration
            )

            # Chunk the output manually
            chunk_samples = int(self.sample_rate * 0.5)  # 0.5s chunks
            for i in range(0, len(audio), chunk_samples):
                yield audio[i:i + chunk_samples]

    def _synthesize_sync(
        self,
        text: str,
        voice_embedding: Optional[torch.Tensor] = None,
        exaggeration: float = 0.5
    ) -> np.ndarray:
        """
        Synchronous full synthesis (fallback).
        """
        # Use Chatterbox generate method
        audio = self.model.generate(
            text,
            audio_prompt_path=voice_embedding if isinstance(voice_embedding, str) else None,
            exaggeration=exaggeration,
        )

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        return audio

    async def extract_voice_embedding(
        self,
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Extract voice embedding from reference audio.

        Args:
            reference_audio: Audio as numpy array
            sample_rate: Sample rate of reference audio

        Returns:
            torch.Tensor: Voice embedding
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(reference_audio).float()

            # Resample if needed
            if sample_rate != self.sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )

            # Move to device
            audio_tensor = audio_tensor.to(self.device)

            # Extract embedding
            # NOTE: Adjust to actual Chatterbox API
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.model.extract_voice_embedding,
                audio_tensor
            )

            logger.info("Voice embedding extracted")
            return embedding

        except Exception as e:
            logger.error(f"Voice embedding extraction failed: {e}")
            raise

    def get_stats(self) -> dict:
        """Get synthesis statistics"""
        stats = self.stats.copy()
        if stats['syntheses'] > 0:
            stats['avg_latency'] = stats['total_latency'] / stats['syntheses']
            stats['avg_first_chunk'] = stats['first_chunk_latency'] / stats['syntheses']
        else:
            stats['avg_latency'] = 0.0
            stats['avg_first_chunk'] = 0.0
        return stats

    async def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded")
