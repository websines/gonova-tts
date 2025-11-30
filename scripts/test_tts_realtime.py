#!/usr/bin/env python3
"""
TTS Realtime Benchmark - Measures latency and plays audio in real-time

Works with Marvis TTS server (WebSocket API).
"""

import asyncio
import json
import ssl
import time
import sys
import argparse
import numpy as np

try:
    import websockets
    import sounddevice as sd
except ImportError:
    print("Install dependencies: pip install websockets sounddevice numpy")
    sys.exit(1)

# Configuration
TTS_URL = "wss://lmstudio.subh-dev.xyz/tts/v1/stream/tts"
SAMPLE_RATE = 24000


async def test_tts_realtime(text: str, chunk_size: int = 50, play_realtime: bool = False):
    """Test TTS with real-time audio playback and latency measurement"""

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    print(f"\n{'='*60}")
    print(f"Text: \"{text}\"")
    print(f"Text length: {len(text)} chars")
    print(f"Chunk size: {chunk_size} tokens")
    print(f"Realtime playback: {play_realtime}")
    print(f"{'='*60}\n")

    # Timing
    start_time = time.perf_counter()
    first_chunk_time = None

    # For realtime playback, use a stream
    stream = None
    if play_realtime:
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        stream.start()

    async with websockets.connect(TTS_URL, ssl=ssl_context) as ws:
        connect_time = time.perf_counter()
        print(f"[{(connect_time - start_time)*1000:.0f}ms] Connected to server")

        # Send request
        request = {
            "type": "synthesize",
            "text": text,
            "voice_id": "default",
            "streaming": True
        }

        await ws.send(json.dumps(request))
        send_time = time.perf_counter()
        print(f"[{(send_time - start_time)*1000:.0f}ms] Request sent, waiting for audio...\n")

        audio_chunks = []
        chunk_count = 0

        while True:
            message = await ws.recv()
            recv_time = time.perf_counter()

            if isinstance(message, bytes):
                chunk_count += 1

                if first_chunk_time is None:
                    first_chunk_time = recv_time
                    ttfb = (first_chunk_time - start_time) * 1000
                    print(f"[{ttfb:.0f}ms] ⚡ FIRST CHUNK RECEIVED (TTFB)")

                # Convert float32 to int16 for playback
                audio_float = np.frombuffer(message, dtype=np.float32)
                audio_int16 = (audio_float * 32767).astype(np.int16)
                audio_chunks.append(audio_int16)

                chunk_duration = len(audio_float) / SAMPLE_RATE * 1000
                print(f"[{(recv_time - start_time)*1000:.0f}ms] Chunk {chunk_count}: {len(message)} bytes ({chunk_duration:.0f}ms audio)")

                # Play chunk immediately if realtime mode
                if play_realtime and stream:
                    stream.write(audio_int16)

            else:
                data = json.loads(message)
                if data.get("type") == "synthesis_complete":
                    end_time = time.perf_counter()
                    print(f"\n[{(end_time - start_time)*1000:.0f}ms] Synthesis complete")
                    break

    # Stop stream if realtime
    if stream:
        stream.stop()
        stream.close()

    # Calculate stats
    total_time = (end_time - start_time) * 1000
    ttfb = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0

    # Combine audio
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        audio_duration = len(full_audio) / SAMPLE_RATE * 1000
    else:
        full_audio = np.array([], dtype=np.int16)
        audio_duration = 0

    # Real-time factor (RTF) - lower is better, <1 means faster than real-time
    rtf = total_time / audio_duration if audio_duration > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Time to First Byte (TTFB): {ttfb:.0f}ms")
    print(f"Total synthesis time:      {total_time:.0f}ms")
    print(f"Audio duration:            {audio_duration:.0f}ms ({audio_duration/1000:.2f}s)")
    print(f"Real-time factor (RTF):    {rtf:.2f}x")
    print(f"Chunks received:           {chunk_count}")
    print(f"{'='*60}\n")

    # Play audio at end if not realtime
    if not play_realtime and len(full_audio) > 0:
        print("Playing audio...")
        sd.play(full_audio, SAMPLE_RATE)
        sd.wait()
        print("Done!")

    return {
        "ttfb_ms": ttfb,
        "total_ms": total_time,
        "audio_ms": audio_duration,
        "rtf": rtf,
        "chunks": chunk_count
    }


async def benchmark(chunk_size: int = 50):
    """Run benchmark with multiple texts"""

    texts = [
        "Hello.",
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology continues to evolve at an unprecedented pace, artificial intelligence stands at the forefront of innovation.",
    ]

    print("\n" + "="*60)
    print(f"TTS REALTIME BENCHMARK (chunk_size={chunk_size})")
    print("="*60)

    results = []
    for text in texts:
        result = await test_tts_realtime(text, chunk_size=chunk_size)
        results.append(result)
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Text Length':<15} {'TTFB':<10} {'Total':<10} {'Audio':<10} {'RTF':<10}")
    print("-"*60)

    for text, result in zip(texts, results):
        print(f"{len(text):<15} {result['ttfb_ms']:<10.0f} {result['total_ms']:<10.0f} {result['audio_ms']:<10.0f} {result['rtf']:<10.2f}")

    avg_ttfb = sum(r['ttfb_ms'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)
    print("-"*60)
    print(f"{'Average':<15} {avg_ttfb:<10.0f} {'':<10} {'':<10} {avg_rtf:<10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Realtime Benchmark")
    parser.add_argument("text", nargs="*", help="Text to synthesize")
    parser.add_argument("-c", "--chunk-size", type=int, default=50, help="Chunk size in tokens (default: 50)")
    parser.add_argument("-r", "--realtime", action="store_true", help="Play audio in realtime as chunks arrive")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmark with multiple texts")

    args = parser.parse_args()

    if args.benchmark:
        asyncio.run(benchmark(chunk_size=args.chunk_size))
    elif args.text:
        text = " ".join(args.text)
        asyncio.run(test_tts_realtime(text, chunk_size=args.chunk_size, play_realtime=args.realtime))
    else:
        # Default test
        asyncio.run(test_tts_realtime("Hello, this is a real-time text to speech test.", chunk_size=args.chunk_size, play_realtime=args.realtime))
