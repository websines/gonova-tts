"""
Parallel TTS synthesizer using vLLM batch processing + concurrent S3Gen.

Strategy:
1. Split text into sentences upfront
2. Batch ALL sentences to vLLM for parallel T3 token generation (4-10x faster)
3. Run S3Gen on multiple chunks concurrently using thread pool
4. Stream audio chunks to client as they complete (in order)

This combines vLLM's batch efficiency with streaming output.
"""

# CRITICAL: This import MUST be at module level for vLLM spawn multiprocessing
from chatterbox_vllm.tts import ChatterboxTTS as _ChatterboxTTS  # noqa: F401

import asyncio
import logging
import re
import time
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple, List, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Import after chatterbox_vllm to ensure tokenizer is registered
from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
from chatterbox_vllm.models.s3gen import S3GEN_SR
from chatterbox_vllm.models.s3tokenizer import S3_SR, drop_invalid_tokens
from chatterbox_vllm.text_utils import punc_norm


def split_into_sentences(text: str, max_chars: int = 150) -> List[str]:
    """Split text into sentences for batch processing."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    result = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            result.append(sentence)
        else:
            parts = re.split(r',\s+', sentence)
            current = ""
            for part in parts:
                if not current:
                    current = part
                elif len(current) + len(part) + 2 <= max_chars:
                    current += ", " + part
                else:
                    if current and current[-1] not in '.!?':
                        current += ","
                    result.append(current)
                    current = part
            if current:
                if current[-1] not in '.!?,':
                    current += "."
                result.append(current)

    return result


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class ParallelSynthesizer:
    """
    High-performance TTS using vLLM batch processing + parallel S3Gen.

    Uses vLLM's batch generation for parallel T3 token generation,
    then processes S3Gen chunks concurrently for maximum throughput.
    """

    S3_SR = 16000
    S3GEN_SR = 24000
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        device: str = "cuda",
        device_index: int = 0,
        max_batch_size: int = 10,
        max_model_len: int = 1000,
        sample_rate: int = 24000,
        s3gen_workers: int = 2,  # Parallel S3Gen workers
        diffusion_steps: int = 3,  # Fast diffusion
    ):
        self.device = device
        self.device_index = device_index
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self._sample_rate = sample_rate
        self.s3gen_workers = s3gen_workers
        self.diffusion_steps = diffusion_steps

        # Components - initialized in load()
        self.tts = None  # ChatterboxTTS instance
        self.executor = None  # Thread pool for S3Gen

        self.is_loaded = False

        # Stats
        self.stats = {
            'syntheses': 0,
            'total_latency': 0.0,
            'first_chunk_latency': 0.0,
            'errors': 0,
        }

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def sr(self) -> int:
        return self._sample_rate

    def load_sync(self):
        """Load ChatterboxTTS model."""
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info("Loading parallel TTS models...")
        start_time = time.time()

        try:
            from chatterbox_vllm.tts import ChatterboxTTS

            # Load the full ChatterboxTTS - it handles vLLM setup
            self.tts = ChatterboxTTS.from_pretrained(
                max_batch_size=self.max_batch_size,
                max_model_len=self.max_model_len,
            )

            # Create thread pool for parallel S3Gen
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.s3gen_workers
            )

            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Parallel TTS loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load parallel TTS: {e}")
            raise

    def _process_s3gen(
        self,
        speech_tokens: torch.Tensor,
        s3gen_ref: dict,
        chunk_idx: int,
    ) -> Tuple[int, np.ndarray, float]:
        """Process a single chunk through S3Gen (runs in thread pool)."""
        try:
            with torch.inference_mode():
                wav, _ = self.tts.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=s3gen_ref,
                    n_timesteps=self.diffusion_steps,
                )
            audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
            duration = len(audio) / self._sample_rate
            return (chunk_idx, audio, duration)
        except Exception as e:
            logger.error(f"S3Gen error for chunk {chunk_idx}: {e}")
            return (chunk_idx, None, 0.0)

    async def generate_stream(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
    ) -> AsyncGenerator[Tuple[np.ndarray, StreamingMetrics], None]:
        """
        Stream audio generation with parallel processing.

        1. Splits text into sentences
        2. Batches all sentences to vLLM for parallel T3 generation
        3. Processes S3Gen chunks in parallel
        4. Yields audio chunks in order as they complete
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_sync() first")

        if not text.strip():
            return

        start_time = time.time()
        metrics = StreamingMetrics()

        try:
            # Split text into sentences
            sentences = split_into_sentences(text)
            num_sentences = len(sentences)
            print(f"[PARALLEL] Split into {num_sentences} sentences")

            if num_sentences == 0:
                return

            # Get audio conditionals
            s3gen_ref, cond_emb = self.tts.get_audio_conditionals(voice_embedding)
            cond_emb = self.tts.update_exaggeration(cond_emb, exaggeration)

            # Batch generate ALL T3 tokens in parallel via vLLM
            print(f"[PARALLEL] Starting batch T3 generation...")
            t3_start = time.time()

            from vllm import SamplingParams

            prompts = ["[START]" + punc_norm(s) + "[STOP]" for s in sentences]

            batch_results = self.tts.t3.generate(
                [
                    {
                        "prompt": p,
                        "multi_modal_data": {"conditionals": [cond_emb]},
                    }
                    for p in prompts
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    stop_token_ids=[self.tts.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(1000, self.max_model_len),
                    top_p=0.8,
                    repetition_penalty=2.0,
                )
            )

            t3_time = time.time() - t3_start
            print(f"[PARALLEL] T3 batch done in {t3_time:.2f}s for {num_sentences} sentences")

            # Extract speech tokens for each sentence
            all_speech_tokens = []
            for batch_result in batch_results:
                for output in batch_result.outputs:
                    speech_tokens = torch.tensor(
                        [token - SPEECH_TOKEN_OFFSET for token in output.token_ids],
                        device="cuda"
                    )
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    all_speech_tokens.append(speech_tokens)

            # Process S3Gen in parallel and yield results in order
            print(f"[PARALLEL] Starting parallel S3Gen with {self.s3gen_workers} workers...")

            # Submit all S3Gen jobs
            loop = asyncio.get_event_loop()
            futures = []
            for idx, tokens in enumerate(all_speech_tokens):
                future = loop.run_in_executor(
                    self.executor,
                    self._process_s3gen,
                    tokens,
                    s3gen_ref,
                    idx,
                )
                futures.append(future)

            # Collect results and yield in order
            results = {}
            total_audio_duration = 0.0
            next_to_yield = 0

            for future in asyncio.as_completed(futures):
                chunk_idx, audio, duration = await future

                if audio is not None:
                    results[chunk_idx] = (audio, duration)

                    # Yield chunks in order
                    while next_to_yield in results:
                        audio_chunk, audio_dur = results.pop(next_to_yield)

                        if metrics.chunk_count == 0:
                            metrics.latency_to_first_chunk = time.time() - start_time
                            print(f"[PARALLEL] TTFB: {metrics.latency_to_first_chunk*1000:.0f}ms")

                        metrics.chunk_count += 1
                        total_audio_duration += audio_dur

                        yield audio_chunk, metrics
                        next_to_yield += 1

            # Final metrics
            metrics.total_generation_time = time.time() - start_time
            metrics.total_audio_duration = total_audio_duration
            if total_audio_duration > 0:
                metrics.rtf = metrics.total_generation_time / total_audio_duration

            self.stats['syntheses'] += 1
            self.stats['total_latency'] += metrics.total_generation_time
            if metrics.latency_to_first_chunk:
                self.stats['first_chunk_latency'] += metrics.latency_to_first_chunk

            print(f"\n[PARALLEL] ========== SUMMARY ==========")
            print(f"[PARALLEL] Sentences: {num_sentences}")
            print(f"[PARALLEL] T3 batch time: {t3_time:.2f}s")
            print(f"[PARALLEL] Total time: {metrics.total_generation_time:.2f}s")
            print(f"[PARALLEL] Audio duration: {total_audio_duration:.2f}s")
            print(f"[PARALLEL] TTFB: {metrics.latency_to_first_chunk*1000:.0f}ms")
            print(f"[PARALLEL] RTF: {metrics.rtf:.2f}")
            print(f"[PARALLEL] ================================\n")

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Parallel synthesis error: {e}")
            raise

    async def synthesize_streaming(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
        chunk_size: Optional[int] = None,
        exaggeration: float = 0.5,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Compatibility wrapper."""
        async for audio_chunk, _ in self.generate_stream(
            text=text,
            voice_embedding=voice_embedding,
            exaggeration=exaggeration,
        ):
            yield audio_chunk

    def get_stats(self) -> dict:
        """Get synthesis statistics."""
        stats = self.stats.copy()
        if stats['syntheses'] > 0:
            stats['avg_latency'] = stats['total_latency'] / stats['syntheses']
            stats['avg_first_chunk'] = stats['first_chunk_latency'] / stats['syntheses']
        else:
            stats['avg_latency'] = 0.0
            stats['avg_first_chunk'] = 0.0

        stats['s3gen_workers'] = self.s3gen_workers
        stats['diffusion_steps'] = self.diffusion_steps
        stats['sample_rate'] = self._sample_rate

        return stats

    async def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=False)

        if self.tts:
            try:
                self.tts.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down TTS: {e}")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.is_loaded = False
        logger.info("Parallel synthesizer cleaned up")
