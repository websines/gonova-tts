#!/usr/bin/env python3
"""
Minimal test script - mirrors the working example-tts.py
"""

from chatterbox_vllm.tts import ChatterboxTTS

if __name__ == "__main__":
    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    print("Generating...")
    prompts = ["Hello, this is a test of the TTS system."]
    audios = model.generate(prompts, exaggeration=0.5)

    print(f"Generated {len(audios)} audio(s)")
    print("Success!")

    model.shutdown()
