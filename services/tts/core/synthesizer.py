"""
TTS synthesizer using Chatterbox TTS.

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

# Lazy load spacy model
_nlp = None


def _get_nlp():
    """Lazy load spacy model for sentence segmentation."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Use small model for speed - only need sentencizer
            try:
                _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except OSError:
                # Model not installed, download it
                logger.info("Downloading spacy en_core_web_sm model...")
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            logger.info("Spacy model loaded for sentence segmentation")
        except ImportError:
            logger.warning("Spacy not installed, falling back to regex sentence splitting")
            _nlp = None
    return _nlp


def split_into_sentences(text: str, max_chars: int = 150) -> List[str]:
    """
    Split text into sentences for streaming.

    Uses spacy for accurate sentence boundary detection (handles abbreviations,
    decimals, etc.). Falls back to regex if spacy unavailable.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (splits on comma if exceeded)

    Returns:
        List of sentences
    """
    text = text.strip()
    if not text:
        return []

    nlp = _get_nlp()

    if nlp is not None:
        # Use spacy for accurate sentence segmentation
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Fallback: regex-based splitting (less accurate)
        # Avoid splitting on common abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    # Further split long sentences on commas/semicolons for lower latency
    result = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            result.append(sentence)
        else:
            # Split on comma or semicolon followed by space
            parts = re.split(r'[,;]\s+', sentence)
            current = ""
            for part in parts:
                if not current:
                    current = part
                elif len(current) + len(part) + 2 <= max_chars:
                    current += ", " + part
                else:
                    result.append(current)
                    current = part

            if current:
                result.append(current)

    return result


class StreamingSynthesizer:
    """
    Chatterbox TTS with sentence-by-sentence streaming.

    Uses sentence-by-sentence generation for low latency streaming.
    Pre-loads model at startup to avoid cold start.

    NOTE: Install chatterbox from:
    pip install chatterbox-tts
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        device_index: int = 1,  # GPU 1 for TTS
        chunk_size: int = 15,  # Not used with non-streaming, kept for API compat
        sample_rate: int = 24000,
    ):
        """
        Args:
            model_path: Path to Chatterbox model
            device: "cuda" or "cpu"
            device_index: GPU index (1 for second 3090)
            chunk_size: Tokens per audio chunk (kept for API compatibility)
            sample_rate: Output sample rate
        """
        self.model_path = model_path
        self.device = f"{device}:{device_index}" if device == "cuda" else device
        self.device_index = device_index
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        self.model = None
        self.is_loaded = False
        self._warmup_done = False

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

        Also warms up CUDA graphs which takes a few extra seconds but
        makes subsequent inference much faster.
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(
            f"Loading optimized Chatterbox model on {self.device}"
        )

        start_time = time.time()

        try:
            # Import Chatterbox
            try:
                from chatterbox.tts import ChatterboxTTS
            except ImportError:
                raise ImportError(
                    "Chatterbox not installed. "
                    "Install with: pip install chatterbox-tts"
                )

            # Enable CUDA optimizations
            if torch.cuda.is_available():
                try:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("CUDA optimizations enabled (cudnn.benchmark, tf32)")
                except AttributeError:
                    pass

            # Load model using from_pretrained
            self.model = ChatterboxTTS.from_pretrained(device=self.device)

            # Try to compile model with torch.compile for faster inference
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile (reduce-overhead mode)")
                except Exception as e:
                    logger.warning(f"torch.compile failed, using eager mode: {e}")

            # Warmup runs - first few are slow due to torch.compile tracing
            logger.info("Warming up model...")
            loop = asyncio.get_event_loop()

            warmup_texts = [
                "Hello.",
                "Hello, this is a warmup test.",
                "The quick brown fox jumps over the lazy dog, and this is a longer sentence to warm up the model properly.",
            ]

            for i, text in enumerate(warmup_texts):
                logger.info(f"Warmup {i+1}/{len(warmup_texts)}: '{text[:30]}...'")
                _ = await loop.run_in_executor(None, self._synthesize_sync, text, None, 0.5)

            self._warmup_done = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded and warmed up in {load_time:.2f}s")

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
        chunk_size: Optional[int] = None,  # noqa: ARG002 - kept for API compatibility
        exaggeration: float = 0.25,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with sentence-by-sentence streaming.

        Splits text into sentences and generates each sentence separately,
        yielding audio as soon as each sentence is complete. This provides
        low perceived latency while using fast non-streaming generation.

        Args:
            text: Text to synthesize
            voice_embedding: Path to reference audio for voice cloning (None = default voice)
            chunk_size: Unused, kept for API compatibility
            exaggeration: Emotion intensity (0.0-1.0+)

        Yields:
            np.ndarray: Audio chunks (Float32, sample_rate)
        """
        _ = chunk_size  # Unused, kept for API compatibility
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first")

        if not text.strip():
            return

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
        voice_embedding: Optional[str],
        exaggeration: float
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Generate audio for a single sentence.

        Uses optimized non-streaming generation with CUDA graphs for
        ~0.8x RTF (faster than realtime).
        """
        loop = asyncio.get_event_loop()

        try:
            # Use non-streaming generation with CUDA graphs
            audio = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                sentence,
                voice_embedding,
                exaggeration
            )

            # Yield the entire sentence audio as one chunk
            yield audio

        except Exception as e:
            logger.error(f"Sentence generation failed: {e}")
            raise

    def _synthesize_sync(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.25
    ) -> np.ndarray:
        """
        Synchronous synthesis for a single sentence.

        Uses optimized settings:
        - cfg_weight: 0.5 (fast, good quality)
        - temperature: 0.8
        - exaggeration: 0.25 (natural)

        With CUDA graphs, achieves ~0.8x RTF on 3090.
        """
        # Use Chatterbox generate method
        audio = self.model.generate(
            text,
            audio_prompt_path=voice_embedding if isinstance(voice_embedding, str) else None,
            exaggeration=exaggeration,
            cfg_weight=0.5,
            temperature=0.8,
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
    ) -> str:
        """
        Save reference audio for voice cloning.

        The optimized chatterbox uses audio file paths for voice cloning,
        so we save the audio to a temp file and return the path.

        Args:
            reference_audio: Audio as numpy array
            sample_rate: Sample rate of reference audio

        Returns:
            str: Path to saved reference audio file
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            import tempfile

            # Convert to tensor
            audio_tensor = torch.from_numpy(reference_audio).float()

            # Resample if needed
            if sample_rate != self.sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )

            # Ensure 2D for torchaudio.save
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_file.name, audio_tensor, self.sample_rate)

            logger.info(f"Voice reference saved to {temp_file.name}")
            return temp_file.name

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
            self._warmup_done = False
            logger.info("Model unloaded")
