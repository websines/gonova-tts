"""
ASR transcriber using faster-whisper.

Handles:
- GPU-accelerated transcription
- Batched inference for efficiency
- Streaming results
- Error recovery
"""

import logging
import time
import numpy as np
from typing import List, Optional, AsyncGenerator
import asyncio

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """
    faster-whisper transcriber with streaming support.

    Pre-loads model at startup to avoid cold start latency.
    Supports batched inference for better GPU utilization.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        device_index: int = 0,
    ):
        """
        Args:
            model_name: Model size (tiny, base, small, medium, large-v3)
            device: "cuda" or "cpu"
            compute_type: "float16", "int8", "int8_float16"
            device_index: GPU index (0 or 1 for your dual-GPU setup)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.device_index = device_index

        self.model = None
        self.is_loaded = False

        # Stats
        self.stats = {
            'transcriptions': 0,
            'total_latency': 0.0,
            'errors': 0,
        }

    async def load(self):
        """
        Load model at startup (avoid cold start).

        This takes 3-5 seconds but only happens once at startup.
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(
            f"Loading faster-whisper model: {self.model_name} "
            f"on {self.device}:{self.device_index} ({self.compute_type})"
        )

        start_time = time.time()

        try:
            # Import here to catch errors early
            from faster_whisper import WhisperModel

            # Load model
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                num_workers=1,  # Single worker per GPU
            )

            # Warm up GPU with dummy inference
            logger.info("Warming up GPU...")
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second
            _ = list(self.model.transcribe(dummy_audio, beam_size=1))

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            self.is_loaded = True

        except ImportError:
            raise ImportError(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> dict:
        """
        Transcribe audio chunk.

        Args:
            audio: Audio as float32 numpy array (16kHz)
            language: Language code (None for auto-detect)
            beam_size: Beam search size (1-10, lower is faster)
            word_timestamps: Include word-level timestamps

        Returns:
            dict: {
                'text': str,
                'segments': list,
                'language': str,
                'language_probability': float,
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first")

        start_time = time.time()

        try:
            # Run transcription (blocking, so run in executor)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio,
                language,
                beam_size,
                word_timestamps
            )

            latency = time.time() - start_time
            self.stats['transcriptions'] += 1
            self.stats['total_latency'] += latency

            logger.debug(
                f"Transcribed in {latency*1000:.0f}ms: "
                f"{result['text'][:50]}..."
            )

            return result

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Transcription error: {e}")
            raise

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        language: Optional[str],
        beam_size: int,
        word_timestamps: bool
    ) -> dict:
        """Synchronous transcription (runs in executor)"""

        # Run faster-whisper
        segments_iter, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=False,  # We do VAD ourselves
        )

        # Convert generator to list
        segments = []
        for segment in segments_iter:
            seg_dict = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'confidence': segment.avg_logprob,
            }

            # Add word-level timestamps if available
            if word_timestamps and hasattr(segment, 'words'):
                seg_dict['words'] = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'confidence': word.probability,
                    }
                    for word in segment.words
                ]

            segments.append(seg_dict)

        # Combine all segment texts
        full_text = ' '.join(seg['text'] for seg in segments).strip()

        return {
            'text': full_text,
            'segments': segments,
            'language': info.language,
            'language_probability': info.language_probability,
        }

    async def transcribe_batch(
        self,
        audio_batch: List[np.ndarray],
        **kwargs
    ) -> List[dict]:
        """
        Transcribe multiple audio chunks.

        NOTE: faster-whisper doesn't natively support batching,
        but we can process them concurrently.
        """
        tasks = [
            self.transcribe(audio, **kwargs)
            for audio in audio_batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                # Return empty result
                valid_results.append({
                    'text': '',
                    'segments': [],
                    'language': 'en',
                    'language_probability': 0.0,
                    'error': str(result)
                })
            else:
                valid_results.append(result)

        return valid_results

    def get_stats(self) -> dict:
        """Get transcription statistics"""
        stats = self.stats.copy()
        if stats['transcriptions'] > 0:
            stats['avg_latency'] = stats['total_latency'] / stats['transcriptions']
        else:
            stats['avg_latency'] = 0.0
        return stats

    async def cleanup(self):
        """Clean up resources"""
        if self.model:
            # faster-whisper cleanup
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded")
