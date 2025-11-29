#!/usr/bin/env python3
"""
Clone a voice from a WAV file.

Usage:
    python scripts/clone_voice.py path/to/audio.wav my_voice_name
    python scripts/clone_voice.py --list  # List all voices
"""

import asyncio
import argparse
import base64
import io
import json
import ssl
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)

try:
    import numpy as np
    import soundfile as sf
    AUDIO_PROCESSING = True
except ImportError:
    AUDIO_PROCESSING = False
    print("Warning: soundfile/numpy not installed. Audio won't be normalized.")
    print("Install with: pip install soundfile numpy")

try:
    import noisereduce as nr
    NOISE_REDUCE = True
except ImportError:
    NOISE_REDUCE = False

TTS_URL = "wss://lmstudio.subh-dev.xyz/tts/v1/stream/tts"
TARGET_SAMPLE_RATE = 24000  # Chatterbox expects 24kHz


def get_ssl_context():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


def process_audio(audio_path: Path) -> bytes:
    """
    Load, clean, and normalize audio for voice cloning.

    - Converts to mono
    - Resamples to 24kHz
    - Removes silence at start/end
    - Reduces background noise (if noisereduce installed)
    - Normalizes volume
    - Trims to max 15 seconds
    """
    if not AUDIO_PROCESSING:
        # Fallback: just read raw bytes
        with open(audio_path, 'rb') as f:
            return f.read()

    print("Processing audio...")

    # Load audio
    audio, sr = sf.read(audio_path)
    print(f"  Original: {len(audio)/sr:.1f}s, {sr}Hz, {'stereo' if len(audio.shape) > 1 else 'mono'}")

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        print("  Converted to mono")

    # Resample to target sample rate if needed
    if sr != TARGET_SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
            print(f"  Resampled to {sr}Hz")
        except ImportError:
            print("  Warning: librosa not installed, skipping resample")

    # Trim silence from start and end
    def trim_silence(audio, threshold=0.01):
        # Find first and last samples above threshold
        above_threshold = np.abs(audio) > threshold * np.max(np.abs(audio))
        if not np.any(above_threshold):
            return audio
        first = np.argmax(above_threshold)
        last = len(audio) - np.argmax(above_threshold[::-1])
        # Add small padding
        padding = int(0.1 * sr)  # 100ms padding
        first = max(0, first - padding)
        last = min(len(audio), last + padding)
        return audio[first:last]

    original_len = len(audio)
    audio = trim_silence(audio)
    if len(audio) < original_len:
        print(f"  Trimmed silence: {original_len/sr:.1f}s -> {len(audio)/sr:.1f}s")

    # Noise reduction (if available)
    if NOISE_REDUCE:
        try:
            audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.7)
            print("  Applied noise reduction")
        except Exception as e:
            print(f"  Warning: Noise reduction failed: {e}")

    # Normalize volume (peak normalization to -1dB)
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 0.9  # -1dB headroom
        audio = audio * (target_peak / peak)
        print(f"  Normalized volume (peak: {peak:.3f} -> {target_peak:.3f})")

    # Trim to max 15 seconds (optimal for voice cloning)
    max_samples = 15 * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"  Trimmed to 15 seconds")

    print(f"  Final: {len(audio)/sr:.1f}s, {sr}Hz")

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()


async def clone_voice(audio_path: str, voice_id: str, description: str = ""):
    """Clone a voice from an audio file."""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return False

    # Process and clean audio
    print(f"Loading audio file: {audio_path}")
    audio_bytes = process_audio(audio_path)

    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    print(f"Audio size: {len(audio_bytes)} bytes ({len(audio_b64)} base64 chars)")

    print(f"Connecting to TTS server...")
    async with websockets.connect(TTS_URL, ssl=get_ssl_context()) as ws:
        # Register the voice
        request = {
            "type": "register_voice",
            "voice_id": voice_id,
            "reference_audio": audio_b64,
            "description": description or f"Cloned voice: {voice_id}"
        }

        print(f"Registering voice '{voice_id}'...")
        await ws.send(json.dumps(request))

        response = await ws.recv()
        data = json.loads(response)

        if data.get("type") == "voice_registered":
            print(f"\n✅ Voice '{voice_id}' registered successfully!")
            print(f"\nYou can now use it with:")
            print(f'  python scripts/test_tts_realtime.py "Hello" --voice {voice_id}')
            return True
        elif data.get("type") == "error":
            print(f"\n❌ Error: {data.get('message')}")
            return False
        else:
            print(f"\nUnexpected response: {data}")
            return False


async def list_voices():
    """List all available voices."""

    print("Connecting to TTS server...")
    async with websockets.connect(TTS_URL, ssl=get_ssl_context()) as ws:
        await ws.send(json.dumps({"type": "list_voices"}))

        response = await ws.recv()
        data = json.loads(response)

        if data.get("type") == "voice_list":
            voices = data.get("voices", [])
            if voices:
                print(f"\n📢 Available voices ({len(voices)}):")
                print("-" * 40)
                for voice in voices:
                    if isinstance(voice, dict):
                        print(f"  • {voice.get('id', voice)} - {voice.get('description', '')}")
                    else:
                        print(f"  • {voice}")
            else:
                print("\nNo custom voices registered yet.")
                print("Use: python scripts/clone_voice.py <audio.wav> <voice_name>")
        else:
            print(f"Unexpected response: {data}")


async def test_voice(voice_id: str, text: str = "Hello, this is a test of the cloned voice."):
    """Test a cloned voice."""

    print(f"Testing voice '{voice_id}'...")
    async with websockets.connect(TTS_URL, ssl=get_ssl_context()) as ws:
        request = {
            "type": "synthesize",
            "text": text,
            "voice_id": voice_id,
            "exaggeration": 0.3,
            "streaming": True
        }

        await ws.send(json.dumps(request))

        audio_chunks = []
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                audio_chunks.append(message)
                print(f"  Received chunk: {len(message)} bytes")
            else:
                data = json.loads(message)
                if data.get("type") == "synthesis_complete":
                    break
                elif data.get("type") == "error":
                    print(f"❌ Error: {data.get('message')}")
                    return

        if audio_chunks:
            print(f"\n✅ Voice '{voice_id}' works! Received {len(audio_chunks)} chunks.")
        else:
            print(f"\n⚠️ No audio received for voice '{voice_id}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone a voice from a WAV file")
    parser.add_argument("audio_file", nargs="?", help="Path to WAV audio file (5-15 seconds recommended)")
    parser.add_argument("voice_id", nargs="?", help="Name for the cloned voice")
    parser.add_argument("-d", "--description", default="", help="Description of the voice")
    parser.add_argument("-l", "--list", action="store_true", help="List all available voices")
    parser.add_argument("-t", "--test", metavar="VOICE_ID", help="Test a cloned voice")

    args = parser.parse_args()

    if args.list:
        asyncio.run(list_voices())
    elif args.test:
        asyncio.run(test_voice(args.test))
    elif args.audio_file and args.voice_id:
        asyncio.run(clone_voice(args.audio_file, args.voice_id, args.description))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Clone a voice:  python scripts/clone_voice.py my_audio.wav my_voice")
        print("  List voices:    python scripts/clone_voice.py --list")
        print("  Test a voice:   python scripts/clone_voice.py --test my_voice")
