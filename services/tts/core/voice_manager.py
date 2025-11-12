"""
Voice manager for TTS service.

Handles:
- Voice cloning from reference audio
- Voice embedding caching (disk + memory)
- Voice quality validation
"""

import logging
import base64
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import asyncio

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Manage voice embeddings for TTS.

    Caches voice embeddings to avoid re-extracting.
    Validates reference audio quality.
    """

    def __init__(
        self,
        cache_dir: str = "./voices",
        max_cached: int = 100,
        synthesizer=None,
    ):
        """
        Args:
            cache_dir: Directory to cache voice embeddings
            max_cached: Max voices to keep in memory
            synthesizer: StreamingSynthesizer instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cached = max_cached
        self.synthesizer = synthesizer

        # In-memory cache
        self.voice_cache: Dict[str, torch.Tensor] = {}
        self.voice_metadata: Dict[str, dict] = {}

        # Stats
        self.stats = {
            'registrations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        logger.info(f"Voice manager initialized, cache: {cache_dir}")

    async def register_voice(
        self,
        voice_id: str,
        reference_audio_b64: str,
        description: str = "",
    ) -> torch.Tensor:
        """
        Register a new voice from reference audio.

        Args:
            voice_id: Unique identifier for this voice
            reference_audio_b64: Base64-encoded WAV file (3-10 seconds)
            description: Optional description

        Returns:
            torch.Tensor: Voice embedding

        Raises:
            ValueError: If audio quality is insufficient
        """
        if self.synthesizer is None:
            raise RuntimeError("Synthesizer not set")

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(reference_audio_b64)

            # Save temporarily
            temp_path = f"/tmp/{voice_id}_ref.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)

            # Load audio
            audio, sr = sf.read(temp_path)

            # Validate audio quality
            validation = self._validate_reference_audio(audio, sr)
            if not validation['valid']:
                raise ValueError(f"Invalid reference audio: {validation['reason']}")

            # Convert to float32 mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            # Extract voice embedding
            logger.info(f"Extracting voice embedding for {voice_id}")
            embedding = await self.synthesizer.extract_voice_embedding(audio, sr)

            # Cache to disk
            cache_path = self.cache_dir / f"{voice_id}.pt"
            torch.save(embedding, cache_path)

            # Cache in memory
            self.voice_cache[voice_id] = embedding
            self.voice_metadata[voice_id] = {
                'description': description,
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'created_at': torch.tensor(asyncio.get_event_loop().time()),
            }

            # Cleanup LRU if cache too large
            if len(self.voice_cache) > self.max_cached:
                self._cleanup_cache()

            self.stats['registrations'] += 1

            logger.info(f"Voice {voice_id} registered successfully")
            return embedding

        except Exception as e:
            logger.error(f"Voice registration failed: {e}")
            raise

    async def get_voice(self, voice_id: str) -> Optional[torch.Tensor]:
        """
        Get voice embedding.

        Checks memory cache → disk cache → returns None if not found.

        Args:
            voice_id: Voice identifier

        Returns:
            torch.Tensor or None
        """
        # Check memory cache
        if voice_id in self.voice_cache:
            self.stats['cache_hits'] += 1
            return self.voice_cache[voice_id]

        # Check disk cache
        cache_path = self.cache_dir / f"{voice_id}.pt"
        if cache_path.exists():
            try:
                embedding = torch.load(cache_path)
                self.voice_cache[voice_id] = embedding
                self.stats['cache_hits'] += 1
                logger.debug(f"Loaded voice {voice_id} from disk")
                return embedding
            except Exception as e:
                logger.error(f"Failed to load voice {voice_id}: {e}")

        # Not found
        self.stats['cache_misses'] += 1
        logger.warning(f"Voice {voice_id} not found")
        return None

    def list_voices(self) -> list:
        """
        List all registered voices.

        Returns:
            list: List of voice info dicts
        """
        voices = []

        # Check disk cache
        for voice_file in self.cache_dir.glob("*.pt"):
            voice_id = voice_file.stem

            voice_info = {
                'voice_id': voice_id,
                'description': self.voice_metadata.get(voice_id, {}).get('description', ''),
                'is_cached': voice_id in self.voice_cache,
            }

            voices.append(voice_info)

        return voices

    def _validate_reference_audio(self, audio: np.ndarray, sr: int) -> dict:
        """
        Validate reference audio quality.

        Returns:
            dict: {"valid": bool, "reason": str}
        """
        # Check duration
        duration = len(audio) / sr

        if duration < 3.0:
            return {"valid": False, "reason": "Too short (minimum 3 seconds)"}

        if duration > 10.0:
            return {"valid": False, "reason": "Too long (maximum 10 seconds)"}

        # Check SNR (signal-to-noise ratio)
        energy = np.mean(audio ** 2)
        if energy < 0.01:
            return {"valid": False, "reason": "Audio too quiet"}

        # Check for clipping
        if np.max(np.abs(audio)) > 0.99:
            return {"valid": False, "reason": "Audio clipped (reduce volume)"}

        # Estimate SNR
        noise_floor = np.percentile(np.abs(audio), 10)
        signal_level = np.percentile(np.abs(audio), 90)

        if signal_level / (noise_floor + 1e-6) < 5.0:
            return {"valid": False, "reason": "Too noisy (poor SNR)"}

        return {"valid": True, "reason": "OK"}

    def _cleanup_cache(self):
        """Remove least recently used voices from memory cache"""
        if len(self.voice_cache) <= self.max_cached:
            return

        # Simple LRU: remove oldest half
        to_remove = len(self.voice_cache) - (self.max_cached // 2)

        # Sort by creation time
        sorted_voices = sorted(
            self.voice_metadata.items(),
            key=lambda x: x[1].get('created_at', 0)
        )

        # Remove oldest
        for voice_id, _ in sorted_voices[:to_remove]:
            if voice_id in self.voice_cache:
                del self.voice_cache[voice_id]
                logger.debug(f"Evicted voice {voice_id} from memory cache")

    def get_stats(self) -> dict:
        """Get voice manager statistics"""
        stats = self.stats.copy()
        stats['total_voices'] = len(list(self.cache_dir.glob("*.pt")))
        stats['cached_in_memory'] = len(self.voice_cache)
        return stats
