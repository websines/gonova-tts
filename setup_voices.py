#!/usr/bin/env python3
"""
Setup script to register default voices for TTS service
"""

import asyncio
import shutil
from pathlib import Path

async def setup():
    print("=" * 50)
    print("TTS Service Voice Setup")
    print("=" * 50)

    # Copy default voice to voices cache directory
    print("\n1. Setting up default voice (urek)...")

    source_voice = Path("services/tts/voices/urek.wav")
    voices_dir = Path("./voices")
    voices_dir.mkdir(exist_ok=True)

    if source_voice.exists():
        dest_voice = voices_dir / "urek.wav"
        shutil.copy(source_voice, dest_voice)
        print(f"   ✓ Copied {source_voice} -> {dest_voice}")
        print(f"   Size: {dest_voice.stat().st_size / 1024:.2f} KB")
    else:
        print(f"   ✗ Source voice not found: {source_voice}")
        print(f"   Please add reference audio (3-10 seconds) to: {source_voice}")

    # List all available voices
    print("\n2. Available voices:")

    voice_count = 0
    for voice_file in voices_dir.glob("*.wav"):
        print(f"   - {voice_file.stem} ({voice_file.stat().st_size / 1024:.2f} KB)")
        voice_count += 1

    if voice_count == 0:
        print("   No voices found. Add .wav files to ./voices/")

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Start TTS service:")
    print("   ./scripts/start_all.sh")
    print("\n2. Use voice 'urek' in your application:")
    print('   {"type": "synthesize", "text": "Hello", "voice_id": "urek"}')
    print("\n3. Register more voices via WebSocket API:")
    print('   {"type": "register_voice", "voice_id": "myvoice", "reference_audio": "<base64>"}')
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(setup())
