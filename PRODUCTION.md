# Production Considerations

Critical optimizations and gotchas for real-world deployment with AI agents.

## Table of Contents

1. [Latency Optimization](#latency-optimization)
2. [VAD Reliability](#vad-reliability)
3. [Streaming Everything](#streaming-everything)
4. [Memory Integration](#memory-integration)
5. [Error Handling](#error-handling)
6. [Performance Tuning](#performance-tuning)

---

## Latency Optimization

### The Problem: Non-Streaming is Too Slow

```
❌ BAD: Non-streaming pipeline
User speaks → ASR → Wait for full LLM response → TTS → Audio
                    ↑
                    User waits 2-3 seconds (feels broken)

Total latency: ~2200ms
User experience: Unacceptable
```

### The Solution: Stream Everything

```
✅ GOOD: Fully streaming pipeline
User speaks → ASR → LLM streams → TTS streams → Audio plays
                    ↑             ↑
                    Starts fast   Parallel processing

Total latency: ~730ms
User experience: Natural conversation
```

### Implementation: Streaming LLM + TTS

```python
# examples/streaming_voice_agent.py

import asyncio
import websockets
from openai import AsyncOpenAI

openai = AsyncOpenAI()

async def streaming_voice_agent():
    """
    Optimized voice agent with <1s latency using streaming.
    """
    asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
    tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

    async def process_conversations():
        async for asr_message in asr_ws:
            data = json.loads(asr_message)

            if data['type'] == 'final':
                user_text = data['text']
                print(f"User: {user_text}")

                # === OPTIMIZATION: Stream LLM response ===
                llm_stream = await openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": user_text}],
                    stream=True  # ← Enable streaming!
                )

                # === OPTIMIZATION: Start TTS immediately ===
                full_response = ""
                sentence_buffer = ""

                async for chunk in llm_stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        sentence_buffer += token

                        # Send complete sentences to TTS (don't wait for full response)
                        if token in '.!?\n':
                            if sentence_buffer.strip():
                                # Send sentence to TTS immediately
                                await tts_ws.send(json.dumps({
                                    'type': 'synthesize',
                                    'text': sentence_buffer.strip(),
                                    'streaming': True
                                }))
                                sentence_buffer = ""

                # Send any remaining text
                if sentence_buffer.strip():
                    await tts_ws.send(json.dumps({
                        'type': 'synthesize',
                        'text': sentence_buffer.strip(),
                        'streaming': True
                    }))

                print(f"Assistant: {full_response}")

    # Play TTS audio
    async def play_audio():
        output_stream = setup_audio_output(sample_rate=24000)

        async for tts_message in tts_ws:
            if isinstance(tts_message, bytes):
                output_stream.write(tts_message)

    await asyncio.gather(
        capture_and_send_audio(asr_ws),
        process_conversations(),
        play_audio()
    )
```

### Latency Comparison

| Approach | Time to First Audio | User Experience |
|----------|-------------------|-----------------|
| **Non-streaming** | 2200ms | ❌ Feels broken |
| **Streaming LLM only** | 1200ms | 🟡 Acceptable |
| **Streaming LLM + sentence-based TTS** | **730ms** | ✅ Natural |
| **Streaming LLM + word-based TTS** | **500ms** | ✅✅ Excellent |

### Advanced: Word-Level Streaming

```python
# Even faster: Stream TTS word-by-word (experimental)

word_buffer = ""
for token in llm_stream:
    word_buffer += token

    # Send every 3-5 words to TTS
    if ' ' in word_buffer and len(word_buffer.split()) >= 3:
        await tts_ws.send(json.dumps({
            'type': 'synthesize',
            'text': word_buffer.strip(),
            'streaming': True
        }))
        word_buffer = ""

# Result: Audio starts playing in ~300ms
# Tradeoff: More TTS calls, slightly higher GPU usage
```

---

## VAD Reliability

### Problem: Smart Turn v3 is Semantic but NOT Perfect

**Real-world issues you WILL encounter:**

### Issue 1: False Positives (Interrupting User)

```python
# User: "I want to... [thinks] ...order pizza"
#       ↑                ↑
#       Starts           VAD thinks done → Interrupts!

# Solution: Add minimum speech duration
class ImprovedVAD:
    def __init__(self):
        self.smart_turn = SmartTurnV3()
        self.min_speech_duration = 1.0  # seconds
        self.speech_start_time = None

    async def detect_turn_end(self, audio):
        result = await self.smart_turn.detect(audio)

        if result['has_speech']:
            if self.speech_start_time is None:
                self.speech_start_time = time.time()
            return False  # Still speaking

        if result['turn_end']:
            # Check minimum duration
            speech_duration = time.time() - (self.speech_start_time or 0)

            if speech_duration < self.min_speech_duration:
                # Too short, probably mid-sentence pause
                return False

            # Turn actually ended
            self.speech_start_time = None
            return True

        return False
```

### Issue 2: False Negatives (Not Detecting End)

```python
# User: "What's the weather?" [done, waits]
#                            ↑
#                            VAD doesn't fire

# Solution: Add silence timeout fallback
class HybridVAD:
    """Combines Smart Turn v3 with silence detection"""

    def __init__(self):
        self.smart_turn = SmartTurnV3()
        self.last_speech_time = None
        self.silence_timeout = 1.5  # seconds

    async def detect_turn_end(self, audio):
        smart_result = await self.smart_turn.detect(audio)

        # Update last speech time
        if smart_result['has_speech']:
            self.last_speech_time = time.time()

        # Primary: Smart Turn detection
        if smart_result['turn_end']:
            return True

        # Fallback: Silence timeout
        if self.last_speech_time:
            silence_duration = time.time() - self.last_speech_time

            if silence_duration > self.silence_timeout:
                self.last_speech_time = None
                return True  # Assume turn ended

        return False
```

### Issue 3: Background Noise

```python
# User: [speaks] ... [dog barks] ... [continues]
#                     ↑
#                     Noise causes issues

# Solution: Add audio quality gating
class RobustVAD:
    def __init__(self):
        self.vad = HybridVAD()
        self.noise_threshold = 0.3

    async def detect_turn_end(self, audio):
        # Check signal-to-noise ratio
        snr = compute_snr(audio)

        if snr < self.noise_threshold:
            # Too noisy, ignore this chunk
            return False

        return await self.vad.detect_turn_end(audio)

def compute_snr(audio: np.ndarray) -> float:
    """Compute signal-to-noise ratio"""
    # Simple energy-based SNR
    energy = np.mean(audio ** 2)
    noise_floor = 0.01  # Calibrate based on your environment

    if energy < noise_floor:
        return 0.0

    return min(energy / noise_floor, 1.0)
```

### Recommended VAD Configuration

```yaml
# config.yaml - Production VAD settings

vad:
  type: "hybrid"  # Smart Turn + silence fallback

  smart_turn:
    model_path: "./models/smart-turn-v3.onnx"
    threshold: 0.5
    confidence_threshold: 0.7  # Only trigger on high confidence

  fallback:
    enabled: true
    silence_timeout: 1.5  # seconds
    min_speech_duration: 0.8  # seconds

  quality_gating:
    enabled: true
    min_snr: 0.3
    max_noise_level: 0.5

  # Per-user tuning (advanced)
  adaptive:
    enabled: false  # Learn user's speech patterns over time
    adjustment_rate: 0.1
```

---

## Streaming Everything

### Architecture: Fully Streaming Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              OPTIMIZED STREAMING ARCHITECTURE                │
└─────────────────────────────────────────────────────────────┘

User Mic
   ↓
   ↓ (continuous stream)
   ↓
ASR Service (streaming)
   ↓
   ↓ (partial + final transcriptions)
   ↓
LLM API (streaming)
   ↓
   ↓ (tokens as generated)
   ↓
Sentence Boundary Detection
   ↓
   ↓ (complete sentences)
   ↓
TTS Service (streaming)
   ↓
   ↓ (audio chunks)
   ↓
User Speaker

TOTAL LATENCY: ~500-700ms (feels natural!)
```

### Implementation: Orchestrator with Streaming

```python
# examples/streaming_orchestrator.py

class StreamingVoiceOrchestrator:
    """
    Orchestrates ASR → LLM → TTS with full streaming.
    Optimized for <1s latency.
    """

    def __init__(self):
        self.asr_ws = None
        self.tts_ws = None
        self.llm_client = AsyncOpenAI()

    async def start(self):
        # Connect to services
        self.asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
        self.tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

        # Start pipeline
        await asyncio.gather(
            self._asr_to_llm_pipeline(),
            self._tts_playback()
        )

    async def _asr_to_llm_pipeline(self):
        """ASR → LLM streaming → TTS"""
        async for asr_msg in self.asr_ws:
            data = json.loads(asr_msg)

            if data['type'] == 'final':
                # User finished speaking
                user_text = data['text']

                # Stream LLM response
                await self._stream_llm_to_tts(user_text)

    async def _stream_llm_to_tts(self, user_text: str):
        """
        Stream LLM response to TTS in real-time.
        Sends complete sentences as they're generated.
        """
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_text}],
            stream=True
        )

        sentence_buffer = ""

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                sentence_buffer += token

                # Detect sentence boundaries
                if self._is_sentence_end(token):
                    if sentence_buffer.strip():
                        # Send complete sentence to TTS
                        await self._synthesize_sentence(sentence_buffer.strip())
                        sentence_buffer = ""

        # Send remaining text
        if sentence_buffer.strip():
            await self._synthesize_sentence(sentence_buffer.strip())

    def _is_sentence_end(self, token: str) -> bool:
        """Detect sentence boundaries"""
        return token.rstrip() in ('.', '!', '?', '\n')

    async def _synthesize_sentence(self, text: str):
        """Send sentence to TTS"""
        await self.tts_ws.send(json.dumps({
            'type': 'synthesize',
            'text': text,
            'streaming': True
        }))

    async def _tts_playback(self):
        """Play TTS audio chunks"""
        output_stream = setup_audio_output()

        async for tts_msg in self.tts_ws:
            if isinstance(tts_msg, bytes):
                output_stream.write(tts_msg)
```

---

## Memory Integration

### Architecture: AI Agent with Memory

```python
# examples/voice_agent_with_memory.py

import redis.asyncio as redis
from openai import AsyncOpenAI

class VoiceAgentWithMemory:
    """
    Voice agent with conversation memory.
    Uses Redis for persistence across sessions.
    """

    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)
        self.llm = AsyncOpenAI()
        self.asr_ws = None
        self.tts_ws = None

    async def start(self, user_id: str):
        """Start voice agent for specific user"""
        # Connect to services
        self.asr_ws = await websockets.connect('ws://localhost:8001/v1/stream/asr')
        self.tts_ws = await websockets.connect('ws://localhost:8002/v1/stream/tts')

        # Load conversation history
        history = await self._load_history(user_id)

        # Start conversation loop
        await self._conversation_loop(user_id, history)

    async def _load_history(self, user_id: str) -> list:
        """Load conversation history from Redis"""
        messages = await self.redis.lrange(f"conv:{user_id}", 0, -1)
        return [json.loads(msg) for msg in messages]

    async def _save_message(self, user_id: str, role: str, content: str):
        """Save message to Redis"""
        message = json.dumps({"role": role, "content": content})
        await self.redis.rpush(f"conv:{user_id}", message)

        # Keep only last 50 messages
        await self.redis.ltrim(f"conv:{user_id}", -50, -1)

        # Set expiry (optional)
        await self.redis.expire(f"conv:{user_id}", 86400)  # 24 hours

    async def _conversation_loop(self, user_id: str, history: list):
        """Main conversation loop with memory"""
        async for asr_msg in self.asr_ws:
            data = json.loads(asr_msg)

            if data['type'] == 'final':
                user_text = data['text']

                # Save user message
                await self._save_message(user_id, "user", user_text)

                # Add to history
                history.append({"role": "user", "content": user_text})

                # Get LLM response with full history
                response = await self._get_llm_response(history)

                # Save assistant response
                await self._save_message(user_id, "assistant", response)

                # Add to history
                history.append({"role": "assistant", "content": response})

                # Synthesize response
                await self._stream_tts(response)

    async def _get_llm_response(self, history: list) -> str:
        """
        Get LLM response with streaming.
        Returns full response text.
        """
        stream = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant."},
                *history[-20:]  # Last 20 messages for context
            ],
            stream=True
        )

        full_response = ""
        sentence_buffer = ""

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                sentence_buffer += token

                # Stream sentences to TTS
                if token.rstrip() in ('.', '!', '?'):
                    await self._stream_tts(sentence_buffer.strip())
                    sentence_buffer = ""

        # Send remaining
        if sentence_buffer.strip():
            await self._stream_tts(sentence_buffer.strip())

        return full_response

    async def _stream_tts(self, text: str):
        """Send text to TTS"""
        await self.tts_ws.send(json.dumps({
            'type': 'synthesize',
            'text': text,
            'streaming': True
        }))
```

### Memory Performance Considerations

```yaml
# Redis memory optimization

memory_settings:
  # Keep only recent conversation
  max_messages_per_user: 50  # ~25 turns
  message_ttl: 86400  # 24 hours

  # Summarize old conversations
  summarization:
    enabled: true
    trigger_at: 40 messages
    keep_summary: true
    keep_recent: 10 messages

  # LLM context window
  context_window:
    max_tokens: 4000  # GPT-4 can handle 128k but costs more
    keep_recent_messages: 20  # Last 10 turns
```

---

## Error Handling

### Critical Error Scenarios

#### 1. ASR Failure

```python
class ResilientASR:
    """ASR with fallback and retry"""

    async def transcribe_with_fallback(self, audio):
        try:
            # Primary: faster-whisper
            return await self.faster_whisper.transcribe(audio)
        except GPUOutOfMemory:
            logger.error("GPU OOM, falling back to CPU")
            return await self.faster_whisper_cpu.transcribe(audio)
        except Exception as e:
            logger.error(f"ASR failed: {e}")
            # Ultimate fallback: Ask user to repeat
            return {
                'text': '[Could not transcribe. Please repeat.]',
                'confidence': 0.0
            }
```

#### 2. LLM Timeout

```python
async def get_llm_response_with_timeout(text, timeout=10.0):
    """LLM with timeout and fallback"""
    try:
        response = await asyncio.wait_for(
            llm.chat.completions.create(...),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.error("LLM timeout")
        return "I'm having trouble processing that. Could you rephrase?"
```

#### 3. TTS Failure

```python
class ResilientTTS:
    """TTS with fallback"""

    async def synthesize_with_fallback(self, text):
        try:
            # Primary: Chatterbox
            return await self.chatterbox.synthesize(text)
        except Exception as e:
            logger.error(f"TTS failed: {e}, using fallback")
            # Fallback: Pre-recorded audio or simpler TTS
            return await self.fallback_tts.synthesize(text)
```

---

## Performance Tuning

### Latency Budget

```
Target: <1000ms total latency

Budget allocation:
- VAD detection: 12ms      (1.2%)
- ASR transcription: 150ms  (15%)
- LLM first tokens: 200ms   (20%)
- TTS first chunk: 470ms    (47%)
- Network overhead: 100ms   (10%)
- Buffer/Queue: 68ms        (6.8%)
────────────────────────────────
TOTAL: 1000ms (1 second)

Critical path: TTS latency (47%)
Optimization priority: Reduce TTS first chunk time
```

### GPU Optimization

```yaml
# Optimize for latency vs throughput

asr:
  model: "medium"  # Faster than large-v3, still accurate
  compute_type: "int8_float16"  # 2x faster
  beam_size: 3  # Lower = faster (5 is default)

tts:
  chunk_size: 25  # Smaller = faster first chunk (50 is default)
  batch_size: 1  # No batching for lowest latency
```

### Monitoring

```python
# Track latency at each stage

class LatencyMonitor:
    async def track_conversation(self):
        start = time.time()

        # VAD
        vad_start = time.time()
        turn_ended = await vad.detect()
        metrics.record('vad_latency', time.time() - vad_start)

        # ASR
        asr_start = time.time()
        text = await asr.transcribe()
        metrics.record('asr_latency', time.time() - asr_start)

        # LLM
        llm_start = time.time()
        response = await llm.generate()
        metrics.record('llm_latency', time.time() - llm_start)
        metrics.record('llm_ttft', llm.time_to_first_token)

        # TTS
        tts_start = time.time()
        audio = await tts.synthesize()
        metrics.record('tts_latency', time.time() - tts_start)

        # Total
        metrics.record('total_latency', time.time() - start)
```

---

## Recommendations

### For Production Deployment:

1. **✅ USE:**
   - Hybrid VAD (Smart Turn + silence fallback)
   - Streaming LLM responses
   - Sentence-based TTS streaming
   - Memory with Redis (optional)
   - Error handling with fallbacks

2. **⚠️ MONITOR:**
   - Queue sizes (alert if >80% full)
   - Latency per stage (track p50, p95, p99)
   - VAD false positive/negative rates
   - GPU utilization

3. **🔧 TUNE:**
   - VAD thresholds per use case
   - ASR model size vs accuracy
   - LLM streaming chunk size
   - TTS chunk size vs quality

### Expected Performance:

```
Optimized Setup:
├─ Total latency: 500-800ms
├─ VAD accuracy: 85-95% (with hybrid approach)
├─ ASR accuracy: 95%+ (faster-whisper large-v3)
├─ User experience: Natural conversation
└─ Concurrent users: 20-30 on single GPU
```

**Bottom line: YES, it will work well IF you implement streaming + hybrid VAD!**
