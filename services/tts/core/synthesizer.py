"""
TTS synthesizer using chatterbox-streaming for real-time inference.

Handles:
- True streaming generation (yields audio chunks as generated)
- Voice cloning from reference audio (5-10s needed)
- Optimized with torch.compile for faster inference

Model: ResembleAI/chatterbox (MIT License)
- 0.5B Llama backbone
- 24kHz output sample rate
- Emotion exaggeration control

Performance targets:
- RTF < 1.0 (faster than realtime)
- First chunk latency < 500ms
"""

import logging
import time
import numpy as np
from typing import Optional, AsyncGenerator, List
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamingSynthesizer:
    """
    High-performance Chatterbox TTS with true streaming.

    Uses chatterbox-streaming for token-level streaming with
    progressive S3Gen audio synthesis.

    NOTE: Install:
    pip install chatterbox-streaming
    """

    def __init__(
        self,
        model_path: Optional[str] = None,  # Not used, kept for API compat
        device: str = "cuda",
        device_index: int = 0,
        chunk_size: int = 25,  # Tokens per chunk (lower = lower latency)
        sample_rate: int = 24000,
    ):
        """
        Args:
            model_path: Not used (model downloaded from HF)
            device: "cuda" or "cpu"
            device_index: GPU index
            chunk_size: Speech tokens per chunk (default 25, lower = faster first chunk)
            sample_rate: Output sample rate (24000 for Chatterbox)
        """
        self.device = device
        self.device_index = device_index
        self.chunk_size = chunk_size

        self.model = None
        self.is_loaded = False
        self._warmup_done = False

        # Chatterbox outputs at 24kHz
        self._model_sample_rate = 24000

        # Optimization settings
        self.context_window = 25  # Reduced from 50 for speed
        self.use_compile = False  # Disabled - causes CUDA graph issues with streaming

        # Stats
        self.stats = {
            'syntheses': 0,
            'total_latency': 0.0,
            'first_chunk_latency': 0.0,
            'avg_rtf': 0.0,
            'errors': 0,
        }

    @property
    def sample_rate(self) -> int:
        """Output sample rate from model (24kHz for Chatterbox)"""
        return self._model_sample_rate

    def load_sync(self):
        """
        Load Chatterbox model synchronously.

        Downloads model weights from HuggingFace and initializes
        with optional torch.compile optimization.
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info("Loading Chatterbox-streaming model...")

        start_time = time.time()

        try:
            import torch
            from chatterbox.tts import ChatterboxTTS

            # Determine device
            if self.device == "cuda" and torch.cuda.is_available():
                device_str = f"cuda:{self.device_index}"
            else:
                device_str = "cpu"
                logger.warning("CUDA not available, using CPU")

            logger.info(f"Loading model to device: {device_str}")

            # Load model
            self.model = ChatterboxTTS.from_pretrained(device=device_str)

            # Enable CUDA optimizations (keep FP32 - FP16 causes dtype mismatch with conds)
            if "cuda" in device_str:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Apply torch.compile for speedup
                if self.use_compile:
                    try:
                        logger.info("Applying torch.compile optimization...")
                        self.model.t3.tfmr = torch.compile(
                            self.model.t3.tfmr,
                            mode="reduce-overhead",
                            fullgraph=False,
                        )
                        logger.info("torch.compile applied to T3 transformer")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}")

                logger.info("CUDA optimizations enabled")

            # Warmup with a simple generation
            logger.info("Warming up model...")
            warmup_text = "Hello."
            for chunk, metrics in self.model.generate_stream(
                warmup_text,
                chunk_size=self.chunk_size,
                context_window=self.context_window,
                print_metrics=False,
            ):
                pass  # Just run through to warm up

            self._warmup_done = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded and warmed up in {load_time:.2f}s")

            self.is_loaded = True

        except ImportError as e:
            logger.error(f"Import error: {e}")
            raise ImportError(
                "chatterbox-streaming not installed. "
                "Install: pip install chatterbox-streaming"
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
        exaggeration: float = 0.5,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with true token-level streaming.

        Yields audio chunks as they are generated - no waiting for
        full sentence completion.

        Args:
            text: Text to synthesize
            voice_embedding: Path to reference audio for voice cloning
            chunk_size: Override default chunk size (tokens per chunk)
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
        total_audio_duration = 0.0

        # Use provided chunk_size or default
        _chunk_size = chunk_size or self.chunk_size

        try:
            logger.info(f"Streaming synthesis: {len(text)} chars, chunk_size={_chunk_size}")

            # Generate with streaming
            for audio_chunk, metrics in self.model.generate_stream(
                text=text,
                audio_prompt_path=voice_embedding,
                exaggeration=exaggeration,
                cfg_weight=0.5,
                temperature=0.8,
                chunk_size=_chunk_size,
                context_window=self.context_window,
                fade_duration=0.02,
                print_metrics=False,
            ):
                # Track first chunk latency
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    self.stats['first_chunk_latency'] += first_chunk_time
                    logger.info(f"First chunk in {first_chunk_time*1000:.0f}ms")

                # Convert to numpy float32
                audio_np = audio_chunk.squeeze().numpy()
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)

                # Track audio duration
                chunk_duration = len(audio_np) / self._model_sample_rate
                total_audio_duration += chunk_duration

                yield audio_np

            # Final stats
            total_time = time.time() - start_time
            self.stats['syntheses'] += 1
            self.stats['total_latency'] += total_time

            if total_audio_duration > 0:
                rtf = total_time / total_audio_duration
                self.stats['avg_rtf'] = (
                    (self.stats['avg_rtf'] * (self.stats['syntheses'] - 1) + rtf)
                    / self.stats['syntheses']
                )
                logger.info(
                    f"Synthesis complete: {total_time:.2f}s, "
                    f"audio={total_audio_duration:.2f}s, RTF={rtf:.3f}"
                )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Synthesis error: {e}")
            raise

    def synthesize_batch(
        self,
        texts: List[str],
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Synthesize multiple texts (non-streaming).

        For streaming, use synthesize_streaming() instead.

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
        results = []

        try:
            for text in texts:
                # Use non-streaming generate for batch
                wav = self.model.generate(
                    text=text,
                    audio_prompt_path=voice_embedding,
                    exaggeration=exaggeration,
                    cfg_weight=0.5,
                    temperature=0.8,
                )

                audio_np = wav.squeeze().numpy()
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)

                results.append(audio_np)

            total_time = time.time() - start_time
            self.stats['syntheses'] += len(texts)
            self.stats['total_latency'] += total_time

            logger.info(
                f"Batch synthesized {len(texts)} texts in {total_time*1000:.0f}ms"
            )

            return results

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Batch synthesis error: {e}")
            raise

    async def extract_voice_embedding(
        self,
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> str:
        """
        Save reference audio for voice cloning.

        Chatterbox uses audio file paths for voice cloning,
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
                import librosa
                reference_audio = librosa.resample(
                    reference_audio,
                    orig_sr=sample_rate,
                    target_sr=target_sr
                )

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

        stats['chunk_size'] = self.chunk_size
        stats['context_window'] = self.context_window
        stats['sample_rate'] = self.sample_rate
        stats['use_compile'] = self.use_compile

        return stats

    async def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None

        self.is_loaded = False
        self._warmup_done = False

        # Clear CUDA cache
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded and CUDA cache cleared")
