#!/usr/bin/env python3
"""
TTS Realtime Test - Test the WebSocket TTS server.
"""

import asyncio
import json
import time
import sys
import argparse
import numpy as np

try:
    import websockets
    import sounddevice as sd
except ImportError:
    print("Install: pip install websockets sounddevice numpy")
    sys.exit(1)

# Default server URL
DEFAULT_URL = "ws://localhost:8002/v1/stream/tts"
SAMPLE_RATE = 24000


async def test_tts(
    text: str,
    url: str = DEFAULT_URL,
    exaggeration: float = 0.5,
    play_audio: bool = True,
):
    """Test TTS with timing measurements."""

    print(f"\n{'='*60}")
    print(f"URL: {url}")
    print(f"Text: \"{text}\"")
    print(f"Exaggeration: {exaggeration}")
    print(f"{'='*60}\n")

    start_time = time.perf_counter()
    first_chunk_time = None
    audio_chunks = []

    async with websockets.connect(url) as ws:
        connect_time = time.perf_counter()
        print(f"[{(connect_time - start_time)*1000:.0f}ms] Connected")

        # Send request
        request = {
            "type": "synthesize",
            "text": text,
            "exaggeration": exaggeration,
        }
        await ws.send(json.dumps(request))
        print(f"[{(time.perf_counter() - start_time)*1000:.0f}ms] Request sent\n")

        # Receive chunks
        chunk_count = 0
        while True:
            message = await ws.recv()

            if isinstance(message, bytes):
                chunk_count += 1
                recv_time = time.perf_counter()

                if first_chunk_time is None:
                    first_chunk_time = recv_time
                    ttfb = (first_chunk_time - start_time) * 1000
                    print(f"[{ttfb:.0f}ms] ⚡ FIRST CHUNK (TTFB)")

                # Convert float32 bytes to numpy
                audio = np.frombuffer(message, dtype=np.float32)
                audio_chunks.append(audio)

                duration_ms = len(audio) / SAMPLE_RATE * 1000
                print(f"[{(recv_time - start_time)*1000:.0f}ms] Chunk {chunk_count}: {len(message)} bytes ({duration_ms:.0f}ms audio)")

            else:
                data = json.loads(message)
                if data.get("type") == "synthesis_complete":
                    end_time = time.perf_counter()
                    print(f"\n[{(end_time - start_time)*1000:.0f}ms] Complete")
                    break
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    return

    # Stats
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        audio_duration = len(full_audio) / SAMPLE_RATE
        total_time = end_time - start_time
        ttfb = (first_chunk_time - start_time) if first_chunk_time else 0
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"TTFB:           {ttfb*1000:.0f}ms")
        print(f"Total time:     {total_time*1000:.0f}ms")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"RTF:            {rtf:.3f}x {'✓' if rtf < 1 else '✗'}")
        print(f"Chunks:         {chunk_count}")
        print(f"{'='*60}\n")

        # Play audio
        if play_audio:
            print("Playing audio...")
            # Convert to int16 for playback
            audio_int16 = (full_audio * 32767).astype(np.int16)
            sd.play(audio_int16, SAMPLE_RATE)
            sd.wait()
            print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Test")
    parser.add_argument("text", nargs="*", default=["Hello, this is a test of the text to speech system."])
    parser.add_argument("-u", "--url", default=DEFAULT_URL)
    parser.add_argument("-e", "--exaggeration", type=float, default=0.5)
    parser.add_argument("--no-play", action="store_true", help="Don't play audio")

    args = parser.parse_args()
    text = " ".join(args.text)

    asyncio.run(test_tts(
        text=text,
        url=args.url,
        exaggeration=args.exaggeration,
        play_audio=not args.no_play,
    ))
