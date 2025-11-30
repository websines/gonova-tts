# TTS API Specification

## WebSocket Endpoint

### `ws://host:port/v1/stream/tts`

Streaming text-to-speech synthesis.

**Request (JSON):**
```json
{
  "type": "synthesize",
  "text": "Hello world. This is a test.",
  "exaggeration": 0.5,
  "voice_path": null
}
```

**Response:**
- Binary audio chunks (float32, 24kHz)
- Final JSON message: `{"type": "synthesis_complete", "chunks": N}`

---

## HTTP Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "healthy",
  "sample_rate": 24000
}
```

---

## Requirements

- Low latency streaming (first audio chunk < 500ms ideal)
- 24kHz sample rate, float32 audio
- Voice cloning support (optional voice_path parameter)
- Exaggeration control (0.0-1.0+)
