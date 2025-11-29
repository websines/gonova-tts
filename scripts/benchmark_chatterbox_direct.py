#!/usr/bin/env python3
"""
Direct Chatterbox benchmark - bypasses our server to test raw performance
"""
import torch
import time

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

from chatterbox.tts import ChatterboxTTS

# Load model
print("\nLoading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
print(f"Model loaded on {device}")

# Warmup
print("\nWarming up...")
_ = model.generate("Hello warmup.")
print("Warmup done")

# Test text
text = "In a world where technology continues to evolve at an unprecedented pace, artificial intelligence stands at the forefront of innovation."

print(f"\n{'='*60}")
print(f"Text: {text}")
print(f"Text length: {len(text)} chars")
print(f"{'='*60}\n")

# Test 1: Non-streaming generation
print("Test 1: Non-streaming generation")
start = time.perf_counter()
wav = model.generate(text, exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
elapsed = time.perf_counter() - start
duration = wav.shape[-1] / model.sr
rtf = elapsed / duration
print(f"  Time: {elapsed:.2f}s")
print(f"  Audio: {duration:.2f}s")
print(f"  RTF: {rtf:.3f}x")

# Test 2: Streaming generation
print("\nTest 2: Streaming generation (chunk_size=25)")
start = time.perf_counter()
first_chunk_time = None
chunks = []
chunk_count = 0

for audio_chunk, metrics in model.generate_stream(
    text=text,
    chunk_size=25,
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5,
    print_metrics=False
):
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter() - start
        print(f"  TTFB: {first_chunk_time*1000:.0f}ms")

    chunks.append(audio_chunk)
    chunk_count += 1
    chunk_dur = audio_chunk.shape[-1] / model.sr
    print(f"  Chunk {chunk_count}: {chunk_dur:.2f}s audio")

elapsed = time.perf_counter() - start
total_audio = torch.cat(chunks, dim=-1)
duration = total_audio.shape[-1] / model.sr
rtf = elapsed / duration

print(f"\n  Total time: {elapsed:.2f}s")
print(f"  Total audio: {duration:.2f}s")
print(f"  RTF: {rtf:.3f}x")
print(f"  Chunks: {chunk_count}")

# Test 3: Streaming with smaller chunks
print("\nTest 3: Streaming generation (chunk_size=10)")
start = time.perf_counter()
first_chunk_time = None
chunks = []
chunk_count = 0

for audio_chunk, metrics in model.generate_stream(
    text=text,
    chunk_size=10,
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5,
    print_metrics=False
):
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter() - start
        print(f"  TTFB: {first_chunk_time*1000:.0f}ms")

    chunks.append(audio_chunk)
    chunk_count += 1

elapsed = time.perf_counter() - start
total_audio = torch.cat(chunks, dim=-1)
duration = total_audio.shape[-1] / model.sr
rtf = elapsed / duration

print(f"\n  Total time: {elapsed:.2f}s")
print(f"  Total audio: {duration:.2f}s")
print(f"  RTF: {rtf:.3f}x")
print(f"  Chunks: {chunk_count}")

print(f"\n{'='*60}")
print("BENCHMARK COMPLETE")
print(f"{'='*60}")
