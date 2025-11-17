#!/usr/bin/env python3
"""
Setup script to register urek voice and verify Smart Turn v3
"""

import asyncio
import shutil
from pathlib import Path

async def setup():
    print("=" * 50)
    print("Voice Agent Setup")
    print("=" * 50)

    # 1. Copy urek.wav to voices cache directory
    print("\n1. Setting up urek voice...")

    source_voice = Path("services/tts/voices/urek.wav")
    voices_dir = Path("./voices")
    voices_dir.mkdir(exist_ok=True)

    if source_voice.exists():
        dest_voice = voices_dir / "urek.wav"
        shutil.copy(source_voice, dest_voice)
        print(f"   ✓ Copied {source_voice} -> {dest_voice}")
    else:
        print(f"   ✗ Source voice not found: {source_voice}")
        print(f"   Please create/upload reference audio to: {source_voice}")

    # 2. Verify Smart Turn model
    print("\n2. Verifying Smart Turn v3 model...")

    model_path = Path("/mnt/d/voice-system/gonova-asr-tts/models/smart-turn-v3.0.onnx")
    if model_path.exists():
        print(f"   ✓ Smart Turn model found: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"   ✗ Smart Turn model not found: {model_path}")
        print(f"   Download from: https://huggingface.co/pipecat-ai/smart-turn-v3")

    # 3. List all available voices
    print("\n3. Available voices:")

    for voice_file in voices_dir.glob("*.wav"):
        print(f"   - {voice_file.stem} ({voice_file.stat().st_size / 1024:.2f} KB)")

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Update services/asr/server.py line 369:")
    print(f'   vad_model_path="{model_path}",')
    print("\n2. Restart services:")
    print("   ./scripts/stop_all.sh")
    print("   ./scripts/start_all.sh")
    print("\n3. Use voice 'urek' in HTML interface or API")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(setup())
