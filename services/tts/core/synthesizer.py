"""
TTS synthesizer using Chatterbox-vLLM for high-performance inference.

Handles:
- Batched generation (4-10x faster than standard Chatterbox)
- Voice cloning from reference audio
- vLLM continuous batching for high throughput

Performance (RTX 3090):
- 40 minutes of audio in ~87 seconds
- T3 token generation: ~13 seconds
- S3Gen waveform synthesis: ~60 seconds
"""

import logging
import re
import time
import torch
import torchaudio
import numpy as np
from typing import Optional, AsyncGenerator, List, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for running sync vLLM operations
_executor = ThreadPoolExecutor(max_workers=2)


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
    High-performance Chatterbox TTS using vLLM for inference.

    Uses vLLM's continuous batching for 4-10x speedup over standard Chatterbox.
    Supports batched generation for maximum throughput.

    NOTE: Install chatterbox-vllm:
    pip install chatterbox-vllm
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        device_index: int = 0,  # GPU index
        chunk_size: int = 15,  # Not used, kept for API compat
        sample_rate: int = 24000,
    ):
        """
        Args:
            model_path: Path to Chatterbox model (uses HF if None)
            device: "cuda" or "cpu"
            device_index: GPU index
            chunk_size: Kept for API compatibility
            sample_rate: Output sample rate (24000 for Chatterbox)
        """
        self.model_path = model_path
        self.device = device
        self.device_index = device_index
        self.chunk_size = chunk_size

        self.model = None
        self.is_loaded = False
        self._warmup_done = False

        # Sample rate is set by the model (S3GEN_SR = 24000)
        self._model_sample_rate = None

        # Batch size for processing (chatterbox-vllm default is 10)
        self.max_batch_size = 10

        # Stats
        self.stats = {
            'syntheses': 0,
            'total_latency': 0.0,
            'first_chunk_latency': 0.0,
            't3_time': 0.0,
            's3gen_time': 0.0,
            'errors': 0,
        }

    @property
    def sample_rate(self) -> int:
        """Output sample rate from model"""
        if self._model_sample_rate:
            return self._model_sample_rate
        return 24000  # Default S3GEN_SR

    async def load(self):
        """
        Load vLLM-optimized Chatterbox model at startup.

        This downloads model weights from HuggingFace and initializes
        the vLLM inference engine for high-throughput generation.
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info("Loading Chatterbox-vLLM model...")

        start_time = time.time()

        try:
            # Import chatterbox-vllm
            try:
                from chatterbox_vllm.tts import ChatterboxTTS
            except ImportError:
                raise ImportError(
                    "chatterbox-vllm not installed. "
                    "Install from: pip install chatterbox-vllm"
                )

            # Enable CUDA optimizations
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("CUDA optimizations enabled")
            except AttributeError:
                pass

            # Load model using from_pretrained (downloads from HuggingFace)
            # This also initializes the vLLM engine
            # max_model_len=1000 prevents CUDA assertion errors during profiling
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                _executor,
                lambda: ChatterboxTTS.from_pretrained(
                    max_model_len=1000,
                )
            )

            # Get sample rate from model
            self._model_sample_rate = self.model.sr

            # Warmup with a simple generation
            logger.info("Warming up vLLM engine...")
            warmup_texts = ["Hello, this is a warmup test."]
            _ = await loop.run_in_executor(
                _executor,
                lambda: self.model.generate(
                    warmup_texts,
                    exaggeration=0.5,
                    temperature=0.8,
                    top_p=0.8,
                    repetition_penalty=2.0,
                )
            )

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
        voice_embedding: Optional[str] = None,
        chunk_size: Optional[int] = None,
        exaggeration: float = 0.5,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with sentence-by-sentence streaming.

        Splits text into sentences and generates each sentence separately,
        yielding audio as soon as each sentence is complete.

        For maximum throughput, use synthesize_batch() instead.

        Args:
            text: Text to synthesize
            voice_embedding: Path to reference audio for voice cloning (None = default voice)
            chunk_size: Not used (kept for API compatibility)
            exaggeration: Emotion intensity (0.0-1.0+, default 0.5)

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
            # Split text into sentences
            sentences = split_into_sentences(text)
            logger.info(f"Split text into {len(sentences)} sentences")

            # Generate sentences in batches for efficiency
            batch_size = min(self.max_batch_size, len(sentences))

            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                logger.debug(f"Generating batch {i//batch_size + 1}: {len(batch)} sentences")

                # Generate batch
                audios = await self._generate_batch(
                    batch,
                    voice_embedding,
                    exaggeration
                )

                # Yield each audio in order
                for audio in audios:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        self.stats['first_chunk_latency'] += first_chunk_time
                        logger.info(f"First chunk in {first_chunk_time*1000:.0f}ms")

                    yield audio

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

    async def synthesize_batch(
        self,
        texts: List[str],
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Synthesize multiple texts in a single batch for maximum throughput.

        This is the most efficient method when you have multiple texts to
        synthesize - vLLM will batch them together for ~4-10x speedup.

        Args:
            texts: List of texts to synthesize
            voice_embedding: Path to reference audio for voice cloning
            exaggeration: Emotion intensity (0.0-1.0+)

        Returns:
            List[np.ndarray]: List of audio arrays (Float32, 24kHz)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first")

        if not texts:
            return []

        start_time = time.time()

        try:
            audios = await self._generate_batch(texts, voice_embedding, exaggeration)

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

    async def _generate_batch(
        self,
        texts: List[str],
        voice_embedding: Optional[str],
        exaggeration: float
    ) -> List[np.ndarray]:
        """
        Generate audio for a batch of texts using vLLM.

        Args:
            texts: List of texts to synthesize
            voice_embedding: Path to reference audio
            exaggeration: Emotion intensity

        Returns:
            List of audio arrays
        """
        loop = asyncio.get_event_loop()

        try:
            # Run vLLM generation in thread pool (it's synchronous)
            audios = await loop.run_in_executor(
                _executor,
                self._synthesize_batch_sync,
                texts,
                voice_embedding,
                exaggeration
            )

            return audios

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise

    def _synthesize_batch_sync(
        self,
        texts: List[str],
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.5
    ) -> List[np.ndarray]:
        """
        Synchronous batch synthesis using vLLM.

        Args:
            texts: List of texts to synthesize
            voice_embedding: Path to reference audio
            exaggeration: Emotion intensity

        Returns:
            List of numpy audio arrays
        """
        # Use chatterbox-vllm generate method (supports batching natively)
        # Parameters match the original Chatterbox defaults for quality
        audio_tensors = self.model.generate(
            texts,
            audio_prompt_path=voice_embedding,
            exaggeration=exaggeration,
            temperature=0.8,
            top_p=0.8,
            repetition_penalty=2.0,
        )

        # Convert tensors to numpy arrays
        # generate() returns list[torch.Tensor], one per input text
        result = []
        for audio in audio_tensors:
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().cpu().numpy()

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            result.append(audio)

        return result

    async def extract_voice_embedding(
        self,
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> str:
        """
        Save reference audio for voice cloning.

        Chatterbox-vLLM uses audio file paths for voice cloning,
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

            # Convert to tensor
            audio_tensor = torch.from_numpy(reference_audio).float()

            # Resample to model sample rate if needed
            target_sr = self.sample_rate
            if sample_rate != target_sr:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=sample_rate,
                    new_freq=target_sr
                )

            # Ensure 2D for torchaudio.save
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_file.name, audio_tensor, target_sr)

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

        stats['max_batch_size'] = self.max_batch_size
        stats['sample_rate'] = self.sample_rate

        return stats

    async def cleanup(self):
        """Clean up resources"""
        if self.model:
            try:
                self.model.shutdown()
            except Exception as e:
                logger.warning(f"Error during model shutdown: {e}")

            del self.model
            self.model = None
            self.is_loaded = False
            self._warmup_done = False

            # Clear CUDA cache
            torch.cuda.empty_cache()

            logger.info("Model unloaded and CUDA cache cleared")
