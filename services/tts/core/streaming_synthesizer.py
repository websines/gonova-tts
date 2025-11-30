"""
Streaming TTS synthesizer using vLLM's AsyncLLM for token-level streaming.

Combines:
- vLLM's AsyncLLM with DELTA mode for fast token streaming (~40 tok/s)
- Progressive S3Gen audio generation with context window (from chatterbox-streaming)

This achieves low TTFB by yielding audio chunks as tokens are generated,
rather than waiting for full generation to complete.

Architecture:
1. T3 (vLLM AsyncLLM) generates speech tokens in streaming mode
2. Tokens are buffered into chunks (default 25 tokens)
3. S3Gen processes each chunk with a context window for coherence
4. Audio is cropped to remove context overlap and yielded immediately

Expected performance (RTX 3090):
- TTFB: ~600ms (25 tokens / 40 tok/s)
- RTF: ~0.3-0.5x (faster than realtime)

IMPORTANT: Import chatterbox_vllm at module level to register tokenizer!
This is required for vLLM's spawn multiprocessing to work correctly.
"""

# CRITICAL: This import MUST be at module level for vLLM spawn multiprocessing
# It registers the EnTokenizer with vLLM's TokenizerRegistry
from chatterbox_vllm.tts import ChatterboxTTS as _ChatterboxTTS  # noqa: F401

import asyncio
import logging
import time
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


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class StreamingSynthesizer:
    """
    High-performance streaming TTS using vLLM's AsyncLLM.

    Uses vLLM V1's async streaming to get tokens as they're generated,
    then progressively converts them to audio using S3Gen with a
    sliding context window.
    """

    # Constants from chatterbox
    S3_SR = 16000  # S3 tokenizer sample rate
    S3GEN_SR = 24000  # S3Gen output sample rate
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        device: str = "cuda",
        device_index: int = 0,
        max_batch_size: int = 10,
        max_model_len: int = 1000,
        chunk_size: int = 25,  # Tokens per audio chunk
        context_window: int = 50,  # Context tokens for S3Gen coherence
        sample_rate: int = 24000,
    ):
        self.device = device
        self.device_index = device_index
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.chunk_size = chunk_size
        self.context_window = context_window
        self._sample_rate = sample_rate

        # Components - initialized in load()
        self.engine = None  # AsyncLLM
        self.s3gen = None
        self.ve = None  # Voice encoder
        self.t3_cond_enc = None
        self.t3_speech_emb = None
        self.t3_speech_pos_emb = None
        self.t3_config = None
        self.default_conds = None

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
        """
        Load models synchronously.

        Initializes:
        - vLLM AsyncLLM engine for T3 token generation
        - S3Gen for waveform synthesis
        - Voice encoder and conditioning modules
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info("Loading streaming TTS models...")
        start_time = time.time()

        try:
            # Import dependencies
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download

            # Import chatterbox components
            from chatterbox_vllm.models.s3gen import S3Gen, S3GEN_SR
            from chatterbox_vllm.models.s3tokenizer import drop_invalid_tokens
            from chatterbox_vllm.models.voice_encoder import VoiceEncoder
            from chatterbox_vllm.models.t3.modules.t3_config import T3Config
            from chatterbox_vllm.models.t3.modules.cond_enc import T3Cond, T3CondEnc
            from chatterbox_vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
            from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET

            # Store imports for later use
            self._drop_invalid_tokens = drop_invalid_tokens
            self._T3Cond = T3Cond
            self._SPEECH_TOKEN_OFFSET = SPEECH_TOKEN_OFFSET
            self._S3GEN_SR = S3GEN_SR

            # Download model files
            REPO_ID = "ResembleAI/chatterbox"
            REVISION = "1b475dffa71fb191cb6d5901215eb6f55635a9b6"

            for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
                local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath, revision=REVISION)

            ckpt_dir = Path(local_path).parent

            # Setup T3 model symlink for vLLM
            t3_cfg_path = ckpt_dir / "t3_cfg.safetensors"
            model_dir = Path.cwd() / "t3-model"
            model_dir.mkdir(exist_ok=True)
            model_safetensors_path = model_dir / "model.safetensors"
            model_safetensors_path.unlink(missing_ok=True)
            model_safetensors_path.symlink_to(t3_cfg_path)

            # Load T3 config and conditioning modules
            self.t3_config = T3Config()
            t3_weights = load_file(ckpt_dir / "t3_cfg.safetensors")

            self.t3_cond_enc = T3CondEnc(self.t3_config)
            self.t3_cond_enc.load_state_dict({
                k.replace('cond_enc.', ''): v
                for k, v in t3_weights.items()
                if k.startswith('cond_enc.')
            })
            self.t3_cond_enc = self.t3_cond_enc.to(device=self.device).eval()

            self.t3_speech_emb = torch.nn.Embedding(
                self.t3_config.speech_tokens_dict_size,
                self.t3_config.n_channels
            )
            self.t3_speech_emb.load_state_dict({
                k.replace('speech_emb.', ''): v
                for k, v in t3_weights.items()
                if k.startswith('speech_emb.')
            })
            self.t3_speech_emb = self.t3_speech_emb.to(device=self.device).eval()

            self.t3_speech_pos_emb = LearnedPositionEmbeddings(
                self.t3_config.max_speech_tokens + 2 + 2,
                self.t3_config.n_channels
            )
            self.t3_speech_pos_emb.load_state_dict({
                k.replace('speech_pos_emb.', ''): v
                for k, v in t3_weights.items()
                if k.startswith('speech_pos_emb.')
            })
            self.t3_speech_pos_emb = self.t3_speech_pos_emb.to(device=self.device).eval()

            # Calculate vLLM memory allocation
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
            vllm_memory_needed = (1.55 * 1024 * 1024 * 1024) + (self.max_batch_size * self.max_model_len * 1024 * 128)
            vllm_memory_percent = min(0.9, vllm_memory_needed / unused_gpu_memory)

            logger.info(f"Allocating {vllm_memory_percent * 100:.1f}% GPU memory to vLLM")

            # Initialize vLLM AsyncLLM engine
            engine_args = AsyncEngineArgs(
                model="./t3-model",
                task="generate",
                tokenizer="EnTokenizer",
                tokenizer_mode="custom",
                gpu_memory_utilization=vllm_memory_percent,
                enforce_eager=True,  # More stable for streaming
                max_model_len=self.max_model_len,
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)

            # Load voice encoder
            self.ve = VoiceEncoder()
            self.ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
            self.ve = self.ve.to(device=self.device).eval()

            # Load S3Gen
            self.s3gen = S3Gen(use_fp16=False)
            self.s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
            self.s3gen = self.s3gen.to(device=self.device).eval()

            # Load default conditionals
            conds_data = torch.load(ckpt_dir / "conds.pt", weights_only=True)

            # Create conditionals object manually
            @dataclass
            class Conditionals:
                t3: Any
                gen: dict

                def to(self, device):
                    self.t3 = self.t3.to(device=device)
                    for k, v in self.gen.items():
                        if torch.is_tensor(v):
                            self.gen[k] = v.to(device=device)
                    return self

            self.default_conds = Conditionals(
                T3Cond(**conds_data['t3']),
                conds_data['gen']
            )
            self.default_conds.to(device=self.device)

            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Streaming TTS loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load streaming TTS: {e}")
            raise

    def _get_conditionals(
        self,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5
    ) -> Tuple[dict, torch.Tensor]:
        """Get S3Gen reference dict and T3 conditioning embedding."""
        import librosa
        from chatterbox_vllm.models.s3tokenizer import S3_SR

        if audio_prompt_path is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            # Load reference audio
            s3gen_ref_wav, _ = librosa.load(audio_prompt_path, sr=self._S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=self._S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, self._S3GEN_SR)

            # Get speech conditioning tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:self.ENC_COND_LEN]],
                max_len=self.t3_config.speech_cond_prompt_len
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Get speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        # Build conditioning embedding
        cond_prompt_speech_emb = (
            self.t3_speech_emb(t3_cond_prompt_tokens)[0] +
            self.t3_speech_pos_emb(t3_cond_prompt_tokens)
        )

        cond_emb = self.t3_cond_enc(self._T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=exaggeration * torch.ones(1, 1)
        ).to(device=self.device)).to(device="cpu")

        return s3gen_ref_dict, cond_emb

    def _process_token_chunk(
        self,
        new_tokens: torch.Tensor,
        all_tokens_so_far: torch.Tensor,
        s3gen_ref: dict,
        fade_duration: float = 0.02,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Process a chunk of tokens through S3Gen with context window.

        Uses a sliding context window from previous tokens for coherence,
        then crops the output audio to remove the context portion.

        Args:
            new_tokens: New speech tokens to process
            all_tokens_so_far: All previously processed tokens (for context)
            s3gen_ref: S3Gen reference dictionary for voice
            fade_duration: Fade-in duration in seconds for smoothing

        Returns:
            Tuple of (audio_chunk, audio_duration) or (None, 0) if failed
        """
        # Build tokens with context window
        if len(all_tokens_so_far) > 0:
            context_tokens = (
                all_tokens_so_far[-self.context_window:]
                if len(all_tokens_so_far) > self.context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        # Clean tokens
        clean_tokens = self._drop_invalid_tokens(tokens_to_process).to(self.device)
        clean_tokens = clean_tokens[clean_tokens < 6561]  # Valid token range

        if len(clean_tokens) == 0:
            return None, 0.0

        # Run S3Gen inference
        with torch.inference_mode():
            wav, _ = self.s3gen.inference(
                speech_tokens=clean_tokens,
                ref_dict=s3gen_ref,
                n_timesteps=5,  # Reduced from 10 for faster streaming
            )

        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop out context portion
        if context_length > 0:
            samples_per_token = len(wav) / len(clean_tokens)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0

        # Apply fade-in for smooth transitions
        fade_samples = int(fade_duration * self._sample_rate)
        if fade_samples > 0 and fade_samples < len(audio_chunk):
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / self._sample_rate

        return audio_chunk.astype(np.float32), audio_duration

    async def generate_stream(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        chunk_size: Optional[int] = None,
    ) -> AsyncGenerator[Tuple[np.ndarray, StreamingMetrics], None]:
        """
        Stream audio generation token-by-token.

        Uses vLLM's AsyncLLM with DELTA mode to get tokens as they're generated,
        then progressively converts chunks to audio using S3Gen.

        Args:
            text: Text to synthesize
            voice_embedding: Path to reference audio for voice cloning
            exaggeration: Emotion intensity (0.0-1.0+)
            temperature: Sampling temperature
            chunk_size: Tokens per audio chunk (default: self.chunk_size)

        Yields:
            Tuple of (audio_chunk, metrics)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_sync() first")

        if not text.strip():
            return

        chunk_size = chunk_size or self.chunk_size
        start_time = time.time()
        metrics = StreamingMetrics()

        try:
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
            from chatterbox_vllm.text_utils import punc_norm

            # Get conditionals
            s3gen_ref, cond_emb = self._get_conditionals(voice_embedding, exaggeration)

            # Prepare text
            text = "[START]" + punc_norm(text) + "[STOP]"

            # Configure streaming sampling
            sampling_params = SamplingParams(
                max_tokens=min(1000, self.max_model_len),
                temperature=temperature,
                top_p=0.8,
                repetition_penalty=2.0,
                stop_token_ids=[self.t3_config.stop_speech_token + self._SPEECH_TOKEN_OFFSET],
                output_kind=RequestOutputKind.DELTA,  # Stream tokens!
            )

            # Token buffers
            token_buffer = []
            all_tokens = torch.tensor([], dtype=torch.long, device=self.device)
            total_audio_duration = 0.0
            request_id = f"stream-{time.time()}"

            # Timing for profiling
            first_token_time = None
            token_times = []
            s3gen_times = []

            # Stream tokens from vLLM
            async for output in self.engine.generate(
                request_id=request_id,
                prompt={
                    "prompt": text,
                    "multi_modal_data": {
                        "conditionals": [cond_emb],
                    },
                },
                sampling_params=sampling_params,
            ):
                # Get new tokens from DELTA output
                for completion in output.outputs:
                    if completion.token_ids:
                        for token_id in completion.token_ids:
                            # Convert from vLLM token space to speech token space
                            speech_token = token_id - self._SPEECH_TOKEN_OFFSET
                            token_buffer.append(speech_token)

                            # Track timing
                            now = time.time()
                            if first_token_time is None:
                                first_token_time = now - start_time
                                logger.info(f"First token in {first_token_time*1000:.0f}ms")
                            token_times.append(now)

                # Process chunk when buffer is full
                if len(token_buffer) >= chunk_size:
                    new_tokens = torch.tensor(token_buffer, dtype=torch.long, device=self.device)

                    # Time S3Gen
                    s3gen_start = time.time()
                    audio_chunk, audio_duration = self._process_token_chunk(
                        new_tokens, all_tokens, s3gen_ref
                    )
                    s3gen_time = time.time() - s3gen_start
                    s3gen_times.append(s3gen_time)
                    logger.info(f"S3Gen chunk took {s3gen_time*1000:.0f}ms")

                    if audio_chunk is not None:
                        # Update metrics
                        if metrics.chunk_count == 0:
                            metrics.latency_to_first_chunk = time.time() - start_time
                            logger.info(f"TTFB: {metrics.latency_to_first_chunk*1000:.0f}ms")

                        metrics.chunk_count += 1
                        total_audio_duration += audio_duration

                        yield audio_chunk, metrics

                    # Update token history
                    all_tokens = torch.cat([all_tokens, new_tokens])
                    token_buffer = []

                if output.finished:
                    break

            # Process remaining tokens
            if token_buffer:
                new_tokens = torch.tensor(token_buffer, dtype=torch.long, device=self.device)

                audio_chunk, audio_duration = self._process_token_chunk(
                    new_tokens, all_tokens, s3gen_ref
                )

                if audio_chunk is not None:
                    if metrics.chunk_count == 0:
                        metrics.latency_to_first_chunk = time.time() - start_time

                    metrics.chunk_count += 1
                    total_audio_duration += audio_duration

                    yield audio_chunk, metrics

            # Final metrics
            metrics.total_generation_time = time.time() - start_time
            metrics.total_audio_duration = total_audio_duration
            if total_audio_duration > 0:
                metrics.rtf = metrics.total_generation_time / total_audio_duration

            self.stats['syntheses'] += 1
            self.stats['total_latency'] += metrics.total_generation_time
            if metrics.latency_to_first_chunk:
                self.stats['first_chunk_latency'] += metrics.latency_to_first_chunk

            # Performance summary
            total_tokens = len(token_times)
            if total_tokens > 1 and token_times[-1] > token_times[0]:
                token_gen_time = token_times[-1] - token_times[0]
                tok_per_sec = (total_tokens - 1) / token_gen_time
                avg_s3gen = sum(s3gen_times) / len(s3gen_times) if s3gen_times else 0
                logger.info(
                    f"T3 speed: {tok_per_sec:.1f} tok/s, "
                    f"Avg S3Gen: {avg_s3gen*1000:.0f}ms/chunk, "
                    f"First token: {first_token_time*1000:.0f}ms"
                )

            logger.info(
                f"Streamed {metrics.chunk_count} chunks in {metrics.total_generation_time:.2f}s, "
                f"TTFB={metrics.latency_to_first_chunk*1000:.0f}ms, RTF={metrics.rtf:.2f}"
            )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Streaming synthesis error: {e}")
            raise

    async def synthesize_streaming(
        self,
        text: str,
        voice_embedding: Optional[str] = None,
        chunk_size: Optional[int] = None,
        exaggeration: float = 0.5,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Compatibility wrapper for existing server code.

        Yields only audio chunks (no metrics).
        """
        async for audio_chunk, _ in self.generate_stream(
            text=text,
            voice_embedding=voice_embedding,
            exaggeration=exaggeration,
            chunk_size=chunk_size,
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

        stats['chunk_size'] = self.chunk_size
        stats['context_window'] = self.context_window
        stats['sample_rate'] = self._sample_rate

        return stats

    async def cleanup(self):
        """Clean up resources."""
        if self.engine:
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down engine: {e}")

        # Clear GPU memory
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.is_loaded = False
        logger.info("Streaming synthesizer cleaned up")
