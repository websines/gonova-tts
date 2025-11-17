"""
Improved VAD with Silero VAD + Smart Turn v3

Architecture (following Pipecat design):
1. Silero VAD detects pauses (lightweight, runs on every chunk)
2. When pause detected -> Smart Turn analyzes last 8 seconds
3. Smart Turn returns turn complete/incomplete
4. Fallback: if incomplete but silence > stop_secs, force end
"""

import logging
import time
import numpy as np
from typing import Optional
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class ImprovedVAD:
    """
    Two-stage VAD following Pipecat architecture:
    - Stage 1: Silero VAD (fast pause detection)
    - Stage 2: Smart Turn v3 (semantic turn completion)
    """

    def __init__(
        self,
        smart_turn_model_path: Optional[str] = None,
        silero_model_path: Optional[str] = None,
        pause_threshold: float = 0.2,      # Silero detects pause after 0.2s silence
        stop_secs: float = 2.0,            # Force stop after 2s silence (with Smart Turn)
        max_duration_secs: float = 8.0,    # Smart Turn context window
        smart_turn_confidence: float = 0.5,
    ):
        """
        Args:
            smart_turn_model_path: Path to Smart Turn v3 ONNX model
            silero_model_path: Path to Silero VAD model (optional, will download if None)
            pause_threshold: Seconds of silence for Silero to detect pause
            stop_secs: Seconds of silence to force turn end (Smart Turn fallback)
            max_duration_secs: Audio context for Smart Turn (8 seconds)
            smart_turn_confidence: Minimum confidence for Smart Turn
        """
        self.pause_threshold = pause_threshold
        self.stop_secs = stop_secs
        self.max_duration_secs = max_duration_secs
        self.smart_turn_confidence = smart_turn_confidence

        # Sample rate (assumed 16kHz)
        self.sample_rate = 16000
        self.max_buffer_samples = int(max_duration_secs * self.sample_rate)

        # Audio buffer for Smart Turn context (last 8 seconds)
        self.audio_buffer = deque(maxlen=self.max_buffer_samples)

        # Load Silero VAD
        self.silero_vad = None
        try:
            import torch
            torch.set_num_threads(1)  # Optimize for low latency

            # Download and load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False  # Use PyTorch version
            )
            self.silero_vad = model
            self.get_speech_timestamps = utils[0]

            logger.info("Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

        # Load Smart Turn v3
        self.smart_turn = None
        if smart_turn_model_path:
            try:
                self.smart_turn = SmartTurnV3(smart_turn_model_path)
                logger.info(f"Smart Turn v3 loaded from {smart_turn_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Smart Turn v3: {e}")

        # State tracking
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.pause_detected_time: Optional[float] = None
        self.has_speech = False
        self.in_pause = False

        # Stats
        self.stats = {
            'silero_triggers': 0,
            'smart_turn_complete': 0,
            'smart_turn_incomplete': 0,
            'fallback_triggers': 0,
            'total_detections': 0,
        }

    async def detect_turn_end(self, audio_chunk: np.ndarray) -> dict:
        """
        Two-stage detection:
        1. Add chunk to buffer
        2. Run Silero VAD to detect pause
        3. If pause detected -> run Smart Turn on buffered audio
        4. Return turn_ended decision

        Returns:
            dict: {
                'turn_ended': bool,
                'has_speech': bool,
                'confidence': float,
                'method': str
            }
        """
        now = time.time()

        # Add to buffer (maintains last 8 seconds)
        self.audio_buffer.extend(audio_chunk)

        # Stage 1: Silero VAD - detect speech/pause
        silero_result = await self._silero_detect(audio_chunk, now)

        if not silero_result['has_speech']:
            # Currently in pause
            if self.has_speech and not self.in_pause:
                # Just entered pause
                self.pause_detected_time = now
                self.in_pause = True
                logger.debug(f"Pause detected by Silero")

            # Check pause duration
            if self.pause_detected_time:
                pause_duration = now - self.pause_detected_time

                # Stage 2: After short pause, run Smart Turn
                if pause_duration >= self.pause_threshold and self.smart_turn:
                    smart_turn_result = await self._smart_turn_detect(now)
                    if smart_turn_result['turn_ended']:
                        return smart_turn_result

                # Fallback: Silence timeout
                if pause_duration >= self.stop_secs:
                    self.stats['fallback_triggers'] += 1
                    self.stats['total_detections'] += 1
                    self._reset_state()

                    return {
                        'turn_ended': True,
                        'has_speech': False,
                        'confidence': 0.8,
                        'method': 'silence_timeout'
                    }

        else:
            # Speech detected
            if not self.has_speech:
                self.speech_start_time = now
                self.has_speech = True

            self.last_speech_time = now
            self.in_pause = False
            self.pause_detected_time = None

        return {
            'turn_ended': False,
            'has_speech': silero_result['has_speech'],
            'confidence': silero_result['confidence'],
            'method': 'silero'
        }

    async def _silero_detect(self, audio_chunk: np.ndarray, now: float) -> dict:
        """
        Run Silero VAD on audio chunk.

        Returns:
            dict: {'has_speech': bool, 'confidence': float}
        """
        try:
            import torch

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()

            # Run Silero VAD
            speech_prob = self.silero_vad(audio_tensor, self.sample_rate).item()

            has_speech = speech_prob > 0.5  # Silero threshold

            return {
                'has_speech': has_speech,
                'confidence': speech_prob,
            }

        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            # Fallback to energy-based detection
            energy = np.mean(audio_chunk ** 2)
            return {
                'has_speech': energy > 0.01,
                'confidence': 0.5,
            }

    async def _smart_turn_detect(self, now: float) -> dict:
        """
        Run Smart Turn v3 on buffered audio (last 8 seconds).

        Returns:
            dict: {'turn_ended': bool, 'confidence': float, 'method': str}
        """
        if not self.smart_turn:
            return {'turn_ended': False, 'confidence': 0.0, 'method': 'no_smart_turn'}

        try:
            # Get buffered audio
            buffered_audio = np.array(self.audio_buffer, dtype=np.float32)

            if len(buffered_audio) < self.sample_rate:  # Less than 1 second
                return {'turn_ended': False, 'confidence': 0.0, 'method': 'insufficient_audio'}

            # Check minimum speech duration
            if self.speech_start_time:
                speech_duration = self.last_speech_time - self.speech_start_time
                if speech_duration < 0.3:  # Too short
                    return {'turn_ended': False, 'confidence': 0.0, 'method': 'speech_too_short'}

            # Run Smart Turn
            result = await self.smart_turn.detect(buffered_audio)

            logger.debug(f"Smart Turn result: turn_end={result['turn_end']}, confidence={result['confidence']:.3f}")

            if result['turn_end'] and result['confidence'] >= self.smart_turn_confidence:
                # Turn complete!
                self.stats['smart_turn_complete'] += 1
                self.stats['total_detections'] += 1
                self._reset_state()

                return {
                    'turn_ended': True,
                    'has_speech': False,
                    'confidence': result['confidence'],
                    'method': 'smart_turn'
                }
            else:
                # Turn incomplete
                self.stats['smart_turn_incomplete'] += 1
                return {
                    'turn_ended': False,
                    'has_speech': True,
                    'confidence': result['confidence'],
                    'method': 'smart_turn_incomplete'
                }

        except Exception as e:
            logger.error(f"Smart Turn error: {e}")
            return {'turn_ended': False, 'confidence': 0.0, 'method': 'smart_turn_error'}

    def _reset_state(self):
        """Reset state after turn end"""
        self.speech_start_time = None
        self.last_speech_time = None
        self.pause_detected_time = None
        self.has_speech = False
        self.in_pause = False
        self.audio_buffer.clear()

    def get_stats(self) -> dict:
        """Get VAD statistics"""
        stats = self.stats.copy()
        if stats['total_detections'] > 0:
            stats['smart_turn_accuracy'] = (
                stats['smart_turn_complete'] / stats['total_detections']
            )
        return stats


class SmartTurnV3:
    """
    Smart Turn v3 ONNX inference.

    Analyzes audio to detect semantic turn completion.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

        try:
            import onnxruntime as ort

            # Use CPU for Smart Turn (12ms inference time)
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )

            logger.info(f"Smart Turn v3 ONNX model loaded")

        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"Failed to load Smart Turn model: {e}")

    async def detect(self, audio: np.ndarray) -> dict:
        """
        Run Smart Turn v3 inference.

        Args:
            audio: Audio buffer (last 8 seconds max), float32, 16kHz

        Returns:
            dict: {
                'has_speech': bool,
                'turn_end': bool,
                'confidence': float
            }
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare input
            audio_input = self._preprocess(audio)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_inference,
                audio_input
            )

            return result

        except Exception as e:
            logger.error(f"Smart Turn inference error: {e}")
            raise

    def _run_inference(self, audio_input: np.ndarray) -> dict:
        """Synchronous ONNX inference"""
        outputs = self.session.run(None, {'audio': audio_input})

        # Parse output (adjust based on actual model output format)
        turn_end_prob = float(outputs[0][0])

        return {
            'has_speech': turn_end_prob < 0.5,
            'turn_end': turn_end_prob > 0.5,
            'confidence': abs(turn_end_prob - 0.5) * 2  # Convert to 0-1 range
        }

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for Smart Turn v3.

        NOTE: Adjust based on actual model requirements.
        Smart Turn v3 likely expects specific input shape/format.
        """
        # Ensure correct shape and type
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)

        return audio.astype(np.float32)
