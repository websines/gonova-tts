# Modular Voice Agent

A production-ready, high-performance voice agent system with separate ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) microservices. Built for bare metal deployment with support for 20-30 concurrent connections per instance, **scalable to 60-80+ users with load balancing**.

## Overview

This project provides two independent microservices that can be used separately or together:

- **ASR Service**: Converts audio streams to text in real-time
- **TTS Service**: Converts text to natural-sounding speech with voice cloning

## Key Features

- **Modular Architecture**: Use ASR and TTS independently or together
- **High Concurrency**: Handle 20-30 connections per instance, **60-80+ with load balancing**
- **Horizontal Scaling**: Run multiple instances with nginx load balancing
- **Real-time Streaming**: WebSocket-based streaming for low latency
- **GPU Optimized**: Efficient GPU usage with async processing
- **Zero Data Loss**: In-memory queue system prevents dropped audio
- **Connection Isolation**: Each user's audio stream is completely isolated
- **Voice Cloning**: TTS supports custom voice creation from reference audio
- **Semantic VAD**: Smart Turn v3 for intelligent turn detection
- **Production Ready**: Bare metal deployment with systemd support

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                       │
│  [Website] [Phone] [Mobile App] [Internal Systems]         │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        WebSocket/HTTP
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
┌───────────────────┐                     ┌──────────────────┐
│   ASR SERVICE     │                     │   TTS SERVICE    │
│   Port: 8001      │                     │   Port: 8002     │
│                   │                     │                  │
│ Audio → Text      │                     │ Text → Audio     │
│                   │                     │                  │
│ • Smart Turn VAD  │                     │ • Chatterbox TTS │
│ • faster-whisper  │                     │ • Voice Cloning  │
│ • Streaming       │                     │ • Streaming      │
└───────────────────┘                     └──────────────────┘
```

## Tech Stack

### ASR Service
- **VAD**: Smart Turn v3 (12ms CPU inference, semantic turn detection)
- **ASR**: faster-whisper (4x faster than Whisper, GPU accelerated)
- **Streaming**: whisper-streaming for real-time chunks
- **Framework**: FastAPI + WebSockets

### TTS Service
- **TTS**: Chatterbox-streaming (0.472s first chunk latency)
- **Voice Cloning**: Reference audio support
- **Real-time Factor**: 0.499 (faster than real-time)
- **Framework**: FastAPI + WebSockets

### Infrastructure
- **Package Manager**: UV (fast Python dependency management)
- **Server**: Uvicorn with async workers
- **Deployment**: Bare metal with systemd services
- **GPU**: CUDA-enabled for both services

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support
- UV package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd voice-agent

# Install dependencies with UV
uv sync

# Download models
python scripts/download_models.py
```

### Running Services

**Option 1: Single Instance (Simple)**
```bash
# Start ASR service
./scripts/start_asr.sh

# Start TTS service (in another terminal)
./scripts/start_tts.sh
```

**Option 2: Load Balanced Cluster (Production)**
```bash
# Start 2 ASR + 2 TTS instances with load balancing
./scripts/start_all.sh

# Requires nginx - see LOAD_BALANCING.md for setup
```

### Test with WebSocket

```javascript
// ASR Example
const asrWs = new WebSocket('ws://localhost:8001/v1/stream/asr');

// Send audio chunks
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    // ... capture audio and send to asrWs
  });

// Receive transcriptions
asrWs.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Transcription:', result.text);
};
```

## Project Structure

```
voice-agent/
├── README.md                  # This file
├── ARCHITECTURE.md            # Detailed system architecture
├── INTEGRATION.md             # Integration examples
├── pyproject.toml             # UV dependencies
│
├── services/
│   ├── asr/                   # ASR Microservice
│   │   ├── README.md         # ASR documentation
│   │   ├── server.py
│   │   ├── vad.py
│   │   ├── transcriber.py
│   │   └── config.yaml
│   │
│   └── tts/                   # TTS Microservice
│       ├── README.md         # TTS documentation
│       ├── server.py
│       ├── synthesizer.py
│       └── config.yaml
│
├── examples/
│   ├── simple_pipeline.py    # ASR → TTS direct
│   ├── with_llm.py           # ASR → LLM → TTS
│   ├── website_client.html   # Browser integration
│   └── twilio_phone.py       # Phone system
│
└── scripts/
    ├── download_models.py    # Download AI models
    └── systemd/              # Service files
```

## Use Cases

### 1. Simple Voice Bot (No LLM)
Direct audio-to-audio conversion:
```
Audio Input → ASR Service → TTS Service → Audio Output
```

### 2. LLM Voice Agent
Add intelligence with any LLM:
```
Audio → ASR → Your LLM (OpenAI/Claude/Local) → TTS → Audio
```

### 3. Phone System (Twilio)
Handle phone calls:
```
Twilio → ASR Service → Your Business Logic → TTS → Twilio
```

### 4. Website Voice Chat
Browser-based voice interface:
```
Browser Mic → WebSocket → ASR → Your Backend → TTS → Browser Speaker
```

### 5. Transcription Service
Use ASR alone:
```
Audio Files/Streams → ASR Service → Text Output
```

## Performance

### ASR Service
- **Latency**: <100ms per chunk (VAD + transcription)
- **Throughput**: 20-30 concurrent audio streams
- **VAD Speed**: 12ms per chunk (Smart Turn v3)
- **GPU Usage**: Shared model across connections

### TTS Service
- **First Chunk**: 0.472s latency
- **Real-time Factor**: 0.499 (faster than playback)
- **Throughput**: 20-30 concurrent syntheses
- **Voice Cloning**: <1s to extract embedding

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[Load Balancing](LOAD_BALANCING.md)** - Scale to 60-80+ concurrent users (2 instances each)
- [Architecture Details](ARCHITECTURE.md) - System design and data flow
- [Hardware Optimization](HARDWARE_OPTIMIZATION.md) - Dual-GPU setup guide
- [Production Guide](PRODUCTION.md) - Production deployment best practices
- [Integration Guide](INTEGRATION.md) - Connect with LLMs, websites, phones
- [ASR Service](services/asr/README.md) - Complete ASR documentation
- [TTS Service](services/tts/README.md) - Complete TTS documentation

## Deployment

### Systemd Services

```bash
# Install systemd services
sudo cp scripts/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl start voice-agent-asr
sudo systemctl start voice-agent-tts

# Enable on boot
sudo systemctl enable voice-agent-asr
sudo systemctl enable voice-agent-tts
```

### Monitoring

```bash
# Check service status
sudo systemctl status voice-agent-asr
sudo systemctl status voice-agent-tts

# View logs
sudo journalctl -u voice-agent-asr -f
sudo journalctl -u voice-agent-tts -f
```

## API Endpoints

### ASR Service (Port 8001)
- `ws://localhost:8001/v1/stream/asr` - WebSocket streaming
- `POST http://localhost:8001/v1/transcribe` - File upload

### TTS Service (Port 8002)
- `ws://localhost:8002/v1/stream/tts` - WebSocket streaming
- `POST http://localhost:8002/v1/synthesize` - Simple TTS

## Configuration

### ASR Config (`services/asr/config.yaml`)
```yaml
model:
  name: "large-v3"
  device: "cuda"
  compute_type: "float16"

vad:
  type: "smart_turn_v3"
  threshold: 0.5

server:
  host: "0.0.0.0"
  port: 8001
  max_connections: 30
```

### TTS Config (`services/tts/config.yaml`)
```yaml
model:
  device: "cuda"
  chunk_size: 50

server:
  host: "0.0.0.0"
  port: 8002
  max_connections: 30
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check .
```

## Contributing

Contributions welcome! Please read the contribution guidelines first.

## License

MIT License - see LICENSE file for details

## Credits

Built with:
- [Pipecat](https://github.com/pipecat-ai/pipecat) - Voice AI framework
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Fast ASR
- [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) - Semantic VAD
- [Chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) - Streaming TTS
# gonova-asr-tts
