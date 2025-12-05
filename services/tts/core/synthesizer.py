"""
TTS synthesizer using Chatterbox-streaming.

Handles:
- Sentence-by-sentence streaming (low latency)
- Voice cloning from reference audio
- GPU-accelerated inference
"""

import logging
import re
import time
import torch
import torchaudio
import numpy as np
from typing import Optional, AsyncGenerator, List
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for streaming.

    Uses regex to split on sentence-ending punctuation while preserving
    the punctuation marks.
    """
    # Split on .!? followed by space or end of string
    # Keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


class StreamingSynthesizer:
    """
    Chatterbox-streaming TTS with voice cloning.

    Uses sentence-by-sentence generation for low latency streaming.
    Pre-loads model at startup to avoid cold start latency.

    NOTE: Chatterbox must be installed separately from GitHub:
    git clone https://github.com/davidbrowne17/chatterbox-streaming
    pip install -e ./chatterbox-streaming
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        device_index: int = 1,  # GPU 1 for TTS
        chunk_size: int = 15,  # Smaller chunks for lower latency
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

            # Enable CUDA optimizations
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("CUDA optimizations enabled")
            except AttributeError:
                pass

            # Load model using from_pretrained
            self.model = ChatterboxTTS.from_pretrained(device=self.device)

            # Warm up GPU with dummy synthesis (multiple times for better warmup)
            logger.info("Warming up GPU...")
            loop = asyncio.get_event_loop()
            for _ in range(2):
                _ = await loop.run_in_executor(None, self._synthesize_sync, "Hello warmup.")

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
        exaggeration: float = 0.25,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with sentence-by-sentence streaming.

        Splits text into sentences and generates each sentence separately,
        yielding audio as soon as each sentence is complete. This provides
        much lower perceived latency than waiting for the full text.

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
            # Split text into sentences
            sentences = split_into_sentences(text)
            logger.info(f"Split text into {len(sentences)} sentences")

            sentence_idx = 0
            for sentence in sentences:
                if not sentence.strip():
                    continue

                logger.debug(f"Generating sentence {sentence_idx + 1}/{len(sentences)}: '{sentence[:50]}...'")

                # Generate audio for this sentence
                async for audio_chunk in self._generate_sentence(
                    sentence,
                    voice_embedding,
                    chunk_size,
                    exaggeration
                ):
                    # Track first chunk latency
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        self.stats['first_chunk_latency'] += first_chunk_time
                        logger.info(f"First chunk in {first_chunk_time*1000:.0f}ms")

                    yield audio_chunk

                sentence_idx += 1

            total_time = time.time() - start_time
            self.stats['syntheses'] += 1
            self.stats['total_latency'] += total_time

            logger.info(
                f"Synthesized {len(sentences)} sentences in {total_time*1000:.0f}ms"
            )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Synthesis error: {e}")
            raise

    async def _generate_sentence(
        self,
        sentence: str,
        voice_embedding: Optional[torch.Tensor],
        chunk_size: int,
        exaggeration: float
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Generate audio for a single sentence using non-streaming for better RTF.

        Non-streaming has ~1.8x RTF vs ~3-5x RTF for streaming on 3090.
        """
        loop = asyncio.get_event_loop()

        try:
            # Use non-streaming generation (faster RTF)
            audio = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                sentence,
                voice_embedding,
                exaggeration
            )

            # Yield the entire sentence audio as one chunk
            # This gives better quality than chunking mid-sentence
            yield audio

        except Exception as e:
            logger.error(f"Sentence generation failed: {e}")
            raise

    def _synthesize_sync(
        self,
        text: str,
        voice_embedding: Optional[torch.Tensor] = None,
        exaggeration: float = 0.25
    ) -> np.ndarray:
        """
        Synchronous synthesis for a single sentence.

        Uses optimized settings from local-chatterbox-tts:
        - cfg_weight: 1.1 (balanced quality/speed)
        - temperature: 1.0
        - exaggeration: 0.25 (more natural)
        """
        # Use Chatterbox generate method with optimized settings
        audio = self.model.generate(
            text,
            audio_prompt_path=voice_embedding if isinstance(voice_embedding, str) else None,
            exaggeration=exaggeration,
            cfg_weight=1.1,
            temperature=1.0,
        )

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

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
