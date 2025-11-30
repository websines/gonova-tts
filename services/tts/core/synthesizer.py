"""
TTS synthesizer using Marvis TTS for fast, high-quality inference.

Handles:
- Streaming sentence-by-sentence generation
- Voice cloning from reference audio (10 seconds needed)
- Transformers-based inference with CUDA support

Model: Marvis-AI/marvis-tts-250m-v0.2-transformers
- 250M parameter multimodal backbone + 60M audio decoder
- Based on Sesame CSM-1B architecture
- 24kHz output sample rate
- Supports English, French, German

Performance: Much faster than Chatterbox with similar quality.
"""

import logging
import re
import time
import numpy as np
from typing import Optional, AsyncGenerator, List, TYPE_CHECKING
from pathlib import Path
import asyncio

if TYPE_CHECKING:
    import torch
    import soundfile as sf

logger = logging.getLogger(__name__)


def split_into_sentences(text: str, max_chars: int = 150) -> List[str]:
    """
    Split text into sentences for streaming.

    Uses regex to split on sentence-ending punctuation while preserving
    the punctuation marks. Also splits long sentences on commas.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (splits on comma if exceeded)
    """
    # First split on .!? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    # Further split long sentences on commas
    result = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            result.append(sentence)
        else:
            # Split on comma followed by space
            parts = re.split(r',\s+', sentence)
            current = ""
            for part in parts:
                if not current:
                    current = part
                elif len(current) + len(part) + 2 <= max_chars:
                    current += ", " + part
                else:
                    # Add comma back if not ending with punctuation
                    if current and not current[-1] in '.!?':
                        current += ","
                    result.append(current)
                    current = part

            if current:
                # Ensure last part has proper ending
                if not current[-1] in '.!?,':
                    current += "."
                result.append(current)

    return result


class StreamingSynthesizer:
    """
    High-performance Marvis TTS using Transformers for inference.

    Uses Marvis-AI/marvis-tts-250m-v0.2-transformers model.
    Supports voice cloning with 10 seconds of reference audio.

    NOTE: Install dependencies:
    pip install transformers torch soundfile huggingface_hub
    """

    # Default model ID - use v0.1 which has better documented PyTorch support
    MODEL_ID = "Marvis-AI/marvis-tts-250m-v0.1-transformers"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        device_index: int = 0,
        chunk_size: int = 15,  # Kept for API compat
        sample_rate: int = 24000,
    ):
        """
        Args:
            model_path: HuggingFace model ID (defaults to Marvis v0.2)
            device: "cuda" or "cpu"
            device_index: GPU index
            chunk_size: Kept for API compatibility
            sample_rate: Output sample rate (24000 for Marvis)
        """
        self.model_path = model_path or self.MODEL_ID
        self.device = device
        self.device_index = device_index
        self.chunk_size = chunk_size

        self.model = None
        self.processor = None
        self.is_loaded = False
        self._warmup_done = False

        # Marvis outputs at 24kHz
        self._model_sample_rate = 24000

        # Stats
        self.stats = {
            'syntheses': 0,
            'total_latency': 0.0,
            'first_chunk_latency': 0.0,
            'errors': 0,
        }

    @property
    def sample_rate(self) -> int:
        """Output sample rate from model (24kHz for Marvis)"""
        return self._model_sample_rate

    def load_sync(self):
        """
        Load Marvis TTS model synchronously.

        Downloads model weights from HuggingFace and initializes
        the Transformers inference pipeline.
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(f"Loading Marvis TTS model: {self.model_path}")

        start_time = time.time()

        try:
            import torch
            from transformers import AutoProcessor, CsmForConditionalGeneration

            # Determine device
            if self.device == "cuda" and torch.cuda.is_available():
                device_str = f"cuda:{self.device_index}"
            else:
                device_str = "cpu"
                logger.warning("CUDA not available, using CPU")

            logger.info(f"Loading model to device: {device_str}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=device_str,
                torch_dtype=torch.float16 if "cuda" in device_str else torch.float32,
            )

            # Enable CUDA optimizations
            if "cuda" in device_str:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("CUDA optimizations enabled")

            # Skip warmup for faster startup - first request will be slower
            # Warmup was causing hangs with some model versions

            self._warmup_done = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded and warmed up in {load_time:.2f}s")

            self.is_loaded = True

        except ImportError as e:
            logger.error(f"Import error: {e}")
            raise ImportError(
                "Missing dependencies. Install: pip install transformers torch soundfile"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def load(self):
        """Async wrapper - just calls load_sync."""
        self.load_sync()

    async def synthesize_streaming(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
        chunk_size: Optional[int] = None,
        exaggeration: float = 0.5,  # Kept for API compat, not used by Marvis
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with sentence-by-sentence streaming.

        Generates ONE sentence at a time for minimal TTFB (time to first byte).
        Each sentence is yielded immediately after generation.

        Args:
            text: Text to synthesize
            voice_embedding: Path to reference audio for voice cloning (None = default voice)
            chunk_size: Not used (kept for API compatibility)
            exaggeration: Not used by Marvis (kept for API compatibility)

        Yields:
            np.ndarray: Audio chunks (Float32, 24kHz)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first")

        if not text.strip():
            return

        start_time = time.time()
        first_chunk_time = None

        try:
            # Split text into sentences for streaming
            sentences = split_into_sentences(text)
            print(f"[TTS] Split into {len(sentences)} sentences: {sentences}")
            logger.info(f"Split text into {len(sentences)} sentences for streaming")

            # Generate ONE sentence at a time for low TTFB
            for i, sentence in enumerate(sentences):
                logger.debug(f"Generating sentence {i+1}/{len(sentences)}: {sentence[:50]}...")

                # Generate single sentence
                audio = self._generate_single(sentence, voice_embedding)

                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    self.stats['first_chunk_latency'] += first_chunk_time
                    logger.info(f"First chunk in {first_chunk_time*1000:.0f}ms")

                yield audio

            total_time = time.time() - start_time
            self.stats['syntheses'] += 1
            self.stats['total_latency'] += total_time

            logger.info(
                f"Streamed {len(sentences)} sentences in {total_time*1000:.0f}ms"
            )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Synthesis error: {e}")
            raise

    def synthesize_batch(
        self,
        texts: List[str],
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.5,  # Kept for API compat
    ) -> List[np.ndarray]:
        """
        Synthesize multiple texts sequentially.

        Note: Marvis doesn't support native batching like vLLM,
        so this processes texts one at a time.

        Args:
            texts: List of texts to synthesize
            voice_embedding: Path to reference audio for voice cloning
            exaggeration: Not used by Marvis (kept for API compatibility)

        Returns:
            List[np.ndarray]: List of audio arrays (Float32, 24kHz)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first")

        if not texts:
            return []

        start_time = time.time()

        try:
            audios = []
            for text in texts:
                audio = self._generate_single(text, voice_embedding)
                audios.append(audio)

            total_time = time.time() - start_time
            self.stats['syntheses'] += len(texts)
            self.stats['total_latency'] += total_time

            logger.info(
                f"Batch synthesized {len(texts)} texts in {total_time*1000:.0f}ms "
                f"({total_time/len(texts)*1000:.0f}ms per text)"
            )

            return audios

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Batch synthesis error: {e}")
            raise

    def _generate_single(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate audio for a single text using Marvis TTS.

        Args:
            text: Text to synthesize
            voice_embedding: Path to reference audio for voice cloning (not yet supported)

        Returns:
            np.ndarray: Audio array (Float32, 24kHz)
        """
        import torch

        try:
            # Format text with speaker ID [0]
            formatted_text = f"[0]{text}"
            print(f"[TTS] Generating: {formatted_text[:60]}...")

            # Get input_ids directly
            input_ids = self.processor(
                formatted_text,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.model.device).pop("input_ids")
            print(f"[TTS] Input IDs shape: {input_ids.shape}, device: {input_ids.device}")

            # Generate audio
            print("[TTS] Starting model.generate()...")
            with torch.no_grad():
                audio = self.model.generate(
                    input_ids=input_ids,
                    output_audio=True,
                    max_new_tokens=1024,  # Limit generation length
                )
            print(f"[TTS] Generation complete, audio type: {type(audio)}")

            # Convert to numpy array
            audio_np = audio[0].cpu().numpy()
            print(f"[TTS] Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")

            # Ensure float32
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)

            return audio_np

        except Exception as e:
            print(f"[TTS] ERROR: {e}")
            logger.error(f"Generation failed: {e}")
            raise

    async def extract_voice_embedding(
        self,
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> str:
        """
        Save reference audio for voice cloning.

        Marvis uses audio file paths for voice cloning,
        so we save the audio to a file and return the path.

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
            import soundfile as sf

            # Resample to model sample rate if needed
            target_sr = self.sample_rate
            if sample_rate != target_sr:
                import torch
                import torchaudio

                audio_tensor = torch.from_numpy(reference_audio).float()
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=sample_rate,
                    new_freq=target_sr
                )
                reference_audio = audio_tensor.numpy()

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, reference_audio, target_sr)

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

        stats['model'] = self.model_path
        stats['sample_rate'] = self.sample_rate

        return stats

    async def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None

        if self.processor:
            del self.processor
            self.processor = None

        self.is_loaded = False
        self._warmup_done = False

        # Clear CUDA cache
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded and CUDA cache cleared")
