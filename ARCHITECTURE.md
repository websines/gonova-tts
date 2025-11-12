# System Architecture

## Overview

The Modular Voice Agent system is designed as **two independent microservices** that communicate via WebSocket/HTTP APIs. This separation allows for:

- Independent scaling of ASR and TTS
- Use of either service standalone
- Easy integration with external systems
- Flexible deployment options

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CLIENT APPLICATIONS                            │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  Website │  │  Mobile  │  │  Twilio  │  │  Custom  │           │
│  │  Browser │  │   App    │  │  Phone   │  │  System  │           │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
└────────┼────────────┼─────────────┼─────────────┼──────────────────┘
         │            │             │             │
         └────────────┴─────────────┴─────────────┘
                      │
         ┌────────────▼────────────┐
         │   WebSocket / HTTP      │
         └────────────┬────────────┘
                      │
         ┌────────────▼─────────────────────────┐
         │     Optional Orchestrator            │
         │   (Routes between services + LLM)    │
         └────────────┬─────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼─────────┐      ┌───────▼──────────┐
│  ASR SERVICE     │      │  TTS SERVICE     │
│  Port: 8001      │      │  Port: 8002      │
│                  │      │                  │
│  ┌────────────┐  │      │  ┌────────────┐ │
│  │    VAD     │  │      │  │ Chatterbox │ │
│  │ Smart Turn │  │      │  │ Streaming  │ │
│  └─────┬──────┘  │      │  └─────┬──────┘ │
│        │         │      │        │        │
│  ┌─────▼──────┐  │      │  ┌─────▼──────┐ │
│  │   faster   │  │      │  │   Voice    │ │
│  │  -whisper  │  │      │  │  Cloning   │ │
│  └────────────┘  │      │  └────────────┘ │
│                  │      │                  │
│  Audio → Text    │      │  Text → Audio   │
└──────────────────┘      └──────────────────┘
         ↓                         ↓
    ┌────────────────────────────────┐
    │         GPU (CUDA)             │
    │  Shared Model Inference        │
    └────────────────────────────────┘
```

## Component Details

### 1. ASR Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ASR SERVICE (Port 8001)                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              WebSocket Handler                         │ │
│  │  • Connection management (30 concurrent)               │ │
│  │  • Audio buffer management                             │ │
│  │  • Async message routing                               │ │
│  └──────────┬─────────────────────────────────────────────┘ │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │   Audio Processor    │                                   │
│  │  • Format detection  │                                   │
│  │  • PCM conversion    │                                   │
│  │  • Resampling 16kHz  │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │  VAD (Smart Turn v3) │                                   │
│  │  • ONNX model        │                                   │
│  │  • CPU inference     │                                   │
│  │  • 12ms latency      │                                   │
│  │  • Semantic turn     │                                   │
│  │    detection         │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │   Speech Buffer      │                                   │
│  │  • Accumulate chunks │                                   │
│  │  • Detect boundaries │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │ Transcription Queue  │                                   │
│  │  • Async queue       │                                   │
│  │  • Batch processing  │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │  faster-whisper ASR  │                                   │
│  │  • GPU inference     │                                   │
│  │  • Shared model      │                                   │
│  │  • Streaming mode    │                                   │
│  │  • Word timestamps   │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │   Result Formatter   │                                   │
│  │  • JSON encoding     │                                   │
│  │  • Confidence scores │                                   │
│  │  • Timestamps        │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │  WebSocket Response  │                                   │
│  └──────────────────────┘                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Client sends audio chunks via WebSocket
2. Audio processor converts to 16kHz mono PCM
3. VAD detects speech segments
4. Speech buffered until turn end detected
5. Queued for GPU transcription
6. faster-whisper transcribes with timestamps
7. Results formatted as JSON
8. Streamed back to client

### 2. TTS Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TTS SERVICE (Port 8002)                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              WebSocket Handler                         │ │
│  │  • Connection management (30 concurrent)               │ │
│  │  • Message parsing                                     │ │
│  │  • Streaming control                                   │ │
│  └──────────┬─────────────────────────────────────────────┘ │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │   Voice Manager      │                                   │
│  │  • Voice cache       │                                   │
│  │  • Embedding extract │                                   │
│  │  • Reference storage │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │   Text Preprocessor  │                                   │
│  │  • Text cleaning     │                                   │
│  │  • Chunking          │                                   │
│  │  • Token counting    │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │  Synthesis Queue     │                                   │
│  │  • Async queue       │                                   │
│  │  • Priority handling │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │ Chatterbox Streaming │                                   │
│  │  • GPU inference     │                                   │
│  │  • Shared model      │                                   │
│  │  • Chunk generation  │                                   │
│  │  • Voice cloning     │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │  Audio Encoder       │                                   │
│  │  • PCM encoding      │                                   │
│  │  • Optional MP3/Opus │                                   │
│  │  • Sample rate conv  │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │ Streaming Controller │                                   │
│  │  • Chunk buffering   │                                   │
│  │  • Flow control      │                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│             ↓                                                │
│  ┌──────────────────────┐                                   │
│  │  WebSocket Response  │                                   │
│  └──────────────────────┘                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Client sends text + voice config via WebSocket
2. Voice manager loads/caches voice embedding
3. Text preprocessed and chunked
4. Queued for GPU synthesis
5. Chatterbox generates audio chunks
6. Audio encoded to PCM/MP3
7. Chunks streamed back to client
8. Client plays audio as received

## Concurrency Model

### Connection Management

```python
# Both services use similar concurrency patterns

class ServiceConnectionManager:
    def __init__(self, max_connections=30):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = {}
        self.gpu_queue = asyncio.Queue()

    async def handle_connection(self, websocket):
        """Handle a single WebSocket connection"""
        async with self.semaphore:  # Limit concurrent connections
            conn_id = str(uuid4())
            self.active_connections[conn_id] = {
                'websocket': websocket,
                'buffer': AudioBuffer(),
                'state': {}
            }

            try:
                await self.process_connection(conn_id, websocket)
            finally:
                del self.active_connections[conn_id]
```

### GPU Resource Sharing

```
┌─────────────────────────────────────────────────────────┐
│                  GPU Memory Layout                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │         Model Weights (Loaded Once)                │ │
│  │  • faster-whisper: ~3GB (large-v3)                 │ │
│  │  • Chatterbox: ~2GB                                │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │         Inference Memory (Dynamic)                 │ │
│  │  • Connection 1: Audio batch                       │ │
│  │  • Connection 2: Audio batch                       │ │
│  │  • ...                                             │ │
│  │  • Connection N: Audio batch                       │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘

Strategy:
1. Load model once (shared across connections)
2. Use async queue for GPU access
3. Batch small requests when possible
4. INT8 quantization for less VRAM
```

## Communication Protocols

### WebSocket Message Format

#### ASR Service Messages

**Client → Server (Audio)**
```
Binary Frame: Raw PCM audio (Float32Array or Int16Array)
Sample Rate: 16000 Hz
Channels: 1 (mono)
```

**Client → Server (Config)**
```json
{
  "type": "config",
  "sample_rate": 16000,
  "language": "en",
  "model": "large-v3"
}
```

**Server → Client (Transcription)**
```json
{
  "type": "partial",
  "text": "Hello world",
  "confidence": 0.95,
  "timestamp": 1.234
}

{
  "type": "final",
  "text": "Hello world, how are you?",
  "start": 0.0,
  "end": 2.5,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.98},
    {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.97}
  ]
}

{
  "type": "speech_start",
  "timestamp": 0.5
}

{
  "type": "speech_end",
  "timestamp": 2.5
}
```

#### TTS Service Messages

**Client → Server (Synthesis Request)**
```json
{
  "type": "synthesize",
  "text": "Hello, how can I help you?",
  "voice_id": "default",
  "streaming": true,
  "chunk_size": 50,
  "emotion": 0.5
}
```

**Client → Server (Voice Registration)**
```json
{
  "type": "register_voice",
  "voice_id": "user_123",
  "reference_audio": "base64_encoded_wav"
}
```

**Server → Client (Audio Chunks)**
```
Binary Frame: Raw PCM audio (Float32Array)
Sample Rate: 24000 Hz
Channels: 1 (mono)
```

**Server → Client (Metadata)**
```json
{
  "type": "audio_chunk",
  "chunk_id": 0,
  "duration_ms": 500
}

{
  "type": "synthesis_complete",
  "total_duration_ms": 2500,
  "chunks_sent": 5
}
```

## Scaling Strategy

### Vertical Scaling (Single Machine)

```
Current Setup (20-30 connections):
- GPU: 1x NVIDIA GPU (12GB+ VRAM)
- CPU: 8+ cores
- RAM: 32GB+

For 50-100 connections:
- GPU: 1x High-end GPU (24GB+ VRAM)
- CPU: 16+ cores
- RAM: 64GB+
- Optimization: Batch inference, INT8 quantization
```

### Horizontal Scaling (Multiple Machines)

```
┌─────────────┐
│ Load        │
│ Balancer    │
│ (nginx)     │
└──────┬──────┘
       │
   ┌───┴────┬────────┬────────┐
   ↓        ↓        ↓        ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ ASR  │ │ ASR  │ │ TTS  │ │ TTS  │
│ Node │ │ Node │ │ Node │ │ Node │
│  1   │ │  2   │ │  1   │ │  2   │
└──────┘ └──────┘ └──────┘ └──────┘

Strategy:
- Deploy multiple instances of each service
- Use nginx for WebSocket load balancing
- Sticky sessions for stateful connections
- Shared Redis for voice cache (TTS)
```

## Deployment Patterns

### Pattern 1: Standalone Services
```
Use ASR or TTS independently

Examples:
- Transcription service (ASR only)
- Text-to-speech API (TTS only)
```

### Pattern 2: Direct Pipeline
```
Client → ASR → TTS → Client

No LLM, direct voice-to-voice
```

### Pattern 3: With External LLM
```
Client → ASR → Client → LLM → Client → TTS → Client

Client orchestrates the pipeline
```

### Pattern 4: Server-side Orchestration
```
Client → Orchestrator → ASR → LLM → TTS → Orchestrator → Client

Server handles full pipeline
```

## Error Handling

### Connection Errors
- Automatic reconnection with exponential backoff
- Graceful degradation (e.g., skip VAD if overloaded)
- Circuit breaker pattern for GPU failures

### Processing Errors
- Fallback to smaller model if OOM
- Timeout handling for stuck inference
- Partial results on failures

### Monitoring
```
Metrics to track:
- Active connections
- GPU utilization
- Memory usage
- Latency (p50, p95, p99)
- Error rates
- Throughput (requests/sec)
```

## Security Considerations

### Network Security
- TLS/SSL for production WebSocket (wss://)
- API key authentication
- Rate limiting per client
- CORS configuration

### Resource Protection
- Connection limits (30 concurrent)
- Request size limits
- Timeout enforcement
- Memory limits per connection

### Data Privacy
- No audio/text logging by default
- Optional encryption at rest
- GDPR compliance options

## Performance Benchmarks

### ASR Service
- **VAD Latency**: 12ms (Smart Turn v3, CPU)
- **ASR Latency**: 80-150ms per chunk (GPU, large-v3)
- **Total Latency**: <200ms from audio to text
- **Throughput**: 30 concurrent streams @ 16kHz
- **GPU Memory**: ~3GB (large-v3 model)

### TTS Service
- **First Chunk**: 472ms (Chatterbox)
- **Streaming Chunks**: ~100ms per chunk
- **Real-time Factor**: 0.499 (faster than playback)
- **Throughput**: 30 concurrent syntheses
- **GPU Memory**: ~2GB (Chatterbox model)

## Technology Decisions

### Why WebSocket over HTTP?
- **Bidirectional**: Real-time streaming both ways
- **Low latency**: No HTTP overhead per chunk
- **Stateful**: Maintain connection state
- **Efficient**: Binary frames for audio

### Why Separate Services?
- **Scalability**: Scale ASR and TTS independently
- **Flexibility**: Use only what you need
- **Maintenance**: Update services separately
- **Fault isolation**: Failure in one doesn't affect other

### Why Smart Turn v3 over Silero?
- **Semantic**: Understands conversation flow
- **Fast**: 12ms CPU inference
- **Accurate**: Better than silence-based VAD
- **Multi-language**: 23 languages supported

### Why faster-whisper?
- **Speed**: 4x faster than Whisper
- **Quality**: Same accuracy as Whisper
- **Efficient**: Lower VRAM usage
- **Mature**: Production-ready

### Why Chatterbox-streaming?
- **Low latency**: 472ms first chunk
- **Voice cloning**: Reference audio support
- **Streaming**: True chunk-by-chunk generation
- **Quality**: Natural-sounding speech

## Future Enhancements

### Planned Features
- [ ] WebRTC transport option
- [ ] Multiple ASR backends (Groq, Deepgram)
- [ ] Multiple TTS backends (ElevenLabs, Coqui)
- [ ] Voice embedding cache (Redis)
- [ ] Prometheus metrics
- [ ] Health check endpoints
- [ ] Auto-scaling based on load
- [ ] Multi-GPU support

### Under Consideration
- [ ] Batch file processing API
- [ ] Speaker diarization (ASR)
- [ ] Emotion detection (ASR)
- [ ] Multilingual voice cloning (TTS)
- [ ] Custom model fine-tuning
