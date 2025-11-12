"""
Audio format processor for ASR service.

Handles conversion from ANY audio format to ASR-ready format:
- Target: 16kHz, mono, float32
- Supports: 8kHz-48kHz input, stereo/mono, int16/float32/μ-law
"""

import audioop
import logging
import numpy as np
import librosa
from typing import Tuple

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Convert any audio format to ASR-compatible format.

    Handles:
    - Sample rate conversion (8kHz-48kHz → 16kHz)
    - Channel conversion (stereo → mono)
    - Format conversion (int16/μ-law → float32)
    - Normalization
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(self):
        self.stats = {
            'conversions': 0,
            'resamples': 0,
            'stereo_to_mono': 0,
            'format_changes': 0,
        }

    def process(
        self,
        audio_bytes: bytes,
        input_sample_rate: int = 16000,
        input_format: str = "float32",
        input_channels: int = 1,
    ) -> np.ndarray:
        """
        Convert audio to ASR format.

        Args:
            audio_bytes: Raw audio bytes
            input_sample_rate: Source sample rate (8000-48000)
            input_format: "float32", "int16", or "mulaw"
            input_channels: 1 (mono) or 2 (stereo)

        Returns:
            np.ndarray: Audio as float32, 16kHz, mono
        """
        self.stats['conversions'] += 1

        # Step 1: Decode bytes to numpy array
        audio = self._decode_audio(audio_bytes, input_format)

        # Step 2: Convert stereo to mono if needed
        if input_channels == 2:
            audio = self._stereo_to_mono(audio)

        # Step 3: Resample to 16kHz if needed
        if input_sample_rate != self.TARGET_SAMPLE_RATE:
            audio = self._resample(audio, input_sample_rate, self.TARGET_SAMPLE_RATE)

        # Step 4: Normalize to [-1, 1] range
        audio = self._normalize(audio)

        return audio

    def _decode_audio(self, audio_bytes: bytes, format: str) -> np.ndarray:
        """Decode bytes to float32 numpy array"""

        if format == "float32":
            audio = np.frombuffer(audio_bytes, dtype=np.float32)

        elif format == "int16":
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            # Convert to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            self.stats['format_changes'] += 1

        elif format == "mulaw":
            # μ-law (used by Twilio, phone systems)
            # Decode to linear PCM int16 first
            linear_bytes = audioop.ulaw2lin(audio_bytes, 2)
            audio = np.frombuffer(linear_bytes, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            self.stats['format_changes'] += 1

        else:
            raise ValueError(f"Unsupported audio format: {format}")

        return audio

    def _stereo_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo to mono by averaging channels"""
        if len(audio.shape) > 1 and audio.shape[1] == 2:
            audio = audio.mean(axis=1)
            self.stats['stereo_to_mono'] += 1

        return audio

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio using high-quality librosa resampler"""
        if orig_sr != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type='kaiser_best'  # High quality
            )
            self.stats['resamples'] += 1

        return audio

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()

        if max_val > 0:
            # Only normalize if needed (avoid amplifying quiet audio)
            if max_val > 1.0:
                audio = audio / max_val

        return audio

    def compute_snr(self, audio: np.ndarray) -> float:
        """
        Compute signal-to-noise ratio.

        Returns:
            float: SNR estimate (0.0-1.0, higher is better)
        """
        energy = np.mean(audio ** 2)
        noise_floor = 0.001  # Estimated noise floor

        if energy < noise_floor:
            return 0.0

        snr = min(energy / noise_floor, 1.0)
        return snr

    def validate_audio(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Validate audio quality.

        Returns:
            dict: {"valid": bool, "reason": str, "snr": float}
        """
        # Check if empty
        if len(audio) == 0:
            return {"valid": False, "reason": "Empty audio", "snr": 0.0}

        # Check duration (min 0.1s, max 60s)
        duration = len(audio) / sample_rate
        if duration < 0.1:
            return {"valid": False, "reason": "Too short (< 0.1s)", "snr": 0.0}
        if duration > 60.0:
            return {"valid": False, "reason": "Too long (> 60s)", "snr": 0.0}

        # Check SNR
        snr = self.compute_snr(audio)
        if snr < 0.1:
            return {"valid": False, "reason": "Too noisy (SNR < 0.1)", "snr": snr}

        # Check clipping
        if np.max(np.abs(audio)) > 0.99:
            return {"valid": False, "reason": "Audio clipped", "snr": snr}

        return {"valid": True, "reason": "OK", "snr": snr}

    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()


class AudioBuffer:
    """
    Circular buffer for audio chunks.
    Prevents memory leaks from unbounded buffering.
    """

    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        self.max_samples = int(sample_rate * max_duration_seconds)
        self.sample_rate = sample_rate
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.position = 0
        self.total_samples = 0

    def add(self, audio: np.ndarray):
        """Add audio to buffer (circular)"""
        samples_to_add = len(audio)

        # Check if we'd overflow
        if self.position + samples_to_add > self.max_samples:
            # Wrap around (circular buffer)
            logger.warning(f"Audio buffer overflow, wrapping around")
            self.position = 0

        # Add to buffer
        self.buffer[self.position:self.position + samples_to_add] = audio
        self.position += samples_to_add
        self.total_samples += samples_to_add

    def get_and_clear(self) -> np.ndarray:
        """Get all buffered audio and clear"""
        audio = self.buffer[:self.position].copy()
        self.position = 0
        return audio

    def clear(self):
        """Clear buffer without returning data"""
        self.position = 0

    def get_duration(self) -> float:
        """Get current buffer duration in seconds"""
        return self.position / self.sample_rate

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self.position == 0
