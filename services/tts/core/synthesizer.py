"""
TTS synthesizer using Chatterbox.

Sentence-by-sentence generation for lower perceived latency.
Each sentence is generated separately and yielded immediately.
"""

import logging
import re
import time
import torch
import numpy as np
from typing import Optional, Generator, List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for running sync generation
_executor = ThreadPoolExecutor(max_workers=2)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on .!? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class StreamingSynthesizer:
    """
    Chatterbox TTS with sentence-by-sentence streaming.

    Instead of token-level streaming (which is slow), we split
    text into sentences and generate each one separately.
    This gives lower perceived latency for the first audio.
    """

    def __init__(
        self,
        device: str = "cuda",
        device_index: int = 0,
    ):
        self.device = device
        self.device_index = device_index
        self.model = None
        self.is_loaded = False

    @property
    def sample_rate(self) -> int:
        return 24000

    def load(self):
        """Load Chatterbox model."""
        if self.is_loaded:
            return

        logger.info("Loading Chatterbox model...")
        start = time.time()

        from chatterbox.tts import ChatterboxTTS

        # Device setup
        if self.device == "cuda" and torch.cuda.is_available():
            device_str = f"cuda:{self.device_index}"
            logger.info(f"Using: {torch.cuda.get_device_name(self.device_index)}")
        else:
            device_str = "cpu"
            logger.warning("Using CPU")

        # Enable optimizations
        if "cuda" in device_str:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load model
        self.model = ChatterboxTTS.from_pretrained(device=device_str)

        # Warmup
        logger.info("Warming up...")
        self._generate_sync("Hello.")
        self._generate_sync("Test warmup.")

        logger.info(f"Model loaded in {time.time()-start:.1f}s")
        self.is_loaded = True

    def _generate_sync(
        self,
        text: str,
        voice_path: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> np.ndarray:
        """Generate audio for text (synchronous)."""
        wav = self.model.generate(
            text,
            audio_prompt_path=voice_path,
            exaggeration=exaggeration,
            cfg_weight=0.5,
            temperature=0.8,
        )
        audio = wav.squeeze().cpu().numpy().astype(np.float32)
        return audio

    def generate_stream(
        self,
        text: str,
        voice_path: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> Generator[bytes, None, None]:
        """
        Generate audio sentence-by-sentence.

        Yields audio bytes (float32, 24kHz) for each sentence
        as soon as it's generated.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if not text or not text.strip():
            return

        sentences = split_into_sentences(text)
        logger.info(f"Generating {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            logger.info(f"[{i+1}/{len(sentences)}] '{sentence[:40]}...'")
            start = time.time()

            # Generate this sentence
            audio = self._generate_sync(sentence, voice_path, exaggeration)

            logger.info(f"[{i+1}/{len(sentences)}] Done in {time.time()-start:.2f}s, {len(audio)/24000:.2f}s audio")

            yield audio.tobytes()

    def generate(
        self,
        text: str,
        voice_path: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> bytes:
        """Generate complete audio (non-streaming)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        audio = self._generate_sync(text, voice_path, exaggeration)
        return audio.tobytes()

    def cleanup(self):
        """Cleanup resources."""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up")
