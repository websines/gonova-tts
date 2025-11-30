"""
TTS synthesizer using chatterbox-streaming.

Simple, direct implementation matching the official example.
"""

import logging
import time
import torch
import numpy as np
from typing import Optional, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamingSynthesizer:
    """
    Chatterbox TTS with streaming support.

    Based on: https://github.com/davidbrowne17/chatterbox-streaming
    """

    def __init__(
        self,
        device: str = "cuda",
        chunk_size: int = 25,
    ):
        self.device = device
        self.chunk_size = chunk_size
        self.model = None
        self.is_loaded = False

    @property
    def sample_rate(self) -> int:
        """Output sample rate (24kHz for Chatterbox)"""
        return 24000

    def load(self):
        """Load Chatterbox model."""
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info("Loading Chatterbox model...")
        start_time = time.time()

        from chatterbox.tts import ChatterboxTTS

        # Detect device
        if self.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("CUDA not available, using CPU")

        # Load model exactly like the example
        self.model = ChatterboxTTS.from_pretrained(device=device)

        # Warmup
        logger.info("Warming up model...")
        _ = self.model.generate("Hello.")

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        self.is_loaded = True

    def generate_stream(
        self,
        text: str,
        voice_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        chunk_size: Optional[int] = None,
    ) -> Generator[bytes, None, None]:
        """
        Generate audio stream from text.

        Yields raw float32 audio bytes at 24kHz.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if not text or not text.strip():
            return

        _chunk_size = chunk_size or self.chunk_size

        logger.info(f"Generating: '{text[:50]}...' chunk_size={_chunk_size}")

        # Use generate_stream exactly like the example
        for audio_chunk, metrics in self.model.generate_stream(
            text=text,
            audio_prompt_path=voice_path,
            chunk_size=_chunk_size,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            print_metrics=True,  # Shows RTF in console
        ):
            # Convert torch tensor to float32 bytes
            audio_np = audio_chunk.squeeze().numpy().astype(np.float32)
            yield audio_np.tobytes()

        logger.info("Generation complete")

    def generate(
        self,
        text: str,
        voice_path: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> bytes:
        """
        Generate complete audio (non-streaming).

        Returns raw float32 audio bytes at 24kHz.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        wav = self.model.generate(
            text=text,
            audio_prompt_path=voice_path,
            exaggeration=exaggeration,
        )

        audio_np = wav.squeeze().numpy().astype(np.float32)
        return audio_np.tobytes()

    def cleanup(self):
        """Cleanup resources."""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Synthesizer cleaned up")
