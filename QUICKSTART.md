# Quick Start Guide

Get the voice agent running in 5 minutes.

## Prerequisites

Your system (already set up):
- ✅ 2x RTX 3090 GPUs (24GB each)
- ✅ 96GB RAM
- ✅ UV package manager

## Installation

### 1. Install Dependencies

```bash
cd voice-agent

# Install with UV
uv sync

# Install additional packages
uv pip install fastapi uvicorn websockets
uv pip install faster-whisper torch torchaudio
uv pip install librosa soundfile onnxruntime
uv pip install structlog
```

### 2. Install Chatterbox-streaming (TTS)

```bash
# Clone from GitHub
git clone https://github.com/davidbrowne17/chatterbox-streaming.git
cd chatterbox-streaming
uv pip install -e .
cd ..
```

### 3. Download Models (Optional)

**Smart Turn v3 VAD:**
```bash
mkdir -p models
wget https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/model.onnx \
  -O models/smart-turn-v3.onnx
```

**faster-whisper** (auto-downloads on first run)

**Chatterbox** (auto-downloads on first run)

## Running Services

### Option 1: Quick Start (Separate Terminals)

**Terminal 1 - ASR Service (GPU 0):**
```bash
./scripts/start_asr.sh
```

**Terminal 2 - TTS Service (GPU 1):**
```bash
./scripts/start_tts.sh
```

**Terminal 3 - Monitor GPUs:**
```bash
watch -n 1 nvidia-smi
```

### Option 2: Systemd (Production)

```bash
# Copy systemd files
sudo cp scripts/systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Start services
sudo systemctl start voice-agent-asr
sudo systemctl start voice-agent-tts

# Enable on boot
sudo systemctl enable voice-agent-asr
sudo systemctl enable voice-agent-tts

# Check status
sudo systemctl status voice-agent-asr
sudo systemctl status voice-agent-tts

# View logs
sudo journalctl -u voice-agent-asr -f
sudo journalctl -u voice-agent-tts -f
```

## Testing

### 1. Check Health

```bash
# ASR service
curl http://localhost:8001/health

# TTS service
curl http://localhost:8002/health
```

### 2. Test WebSocket (Python)

```python
import asyncio
import websockets
import numpy as np

async def test_asr():
    # Connect to ASR
    async with websockets.connect('ws://localhost:8001/v1/stream/asr') as ws:
        # Send dummy audio
        audio = np.random.rand(16000).astype(np.float32)
        await ws.send(audio.tobytes())

        # Receive transcription
        response = await ws.recv()
        print(f"ASR: {response}")

async def test_tts():
    # Connect to TTS
    async with websockets.connect('ws://localhost:8002/v1/stream/tts') as ws:
        # Request synthesis
        import json
        await ws.send(json.dumps({
            'type': 'synthesize',
            'text': 'Hello, this is a test!',
            'streaming': True
        }))

        # Receive audio chunks
        chunk_count = 0
        async for message in ws:
            if isinstance(message, bytes):
                chunk_count += 1
                print(f"Received audio chunk {chunk_count}")
            else:
                data = json.loads(message)
                if data.get('type') == 'synthesis_complete':
                    print(f"Done! Received {chunk_count} chunks")
                    break

asyncio.run(test_asr())
asyncio.run(test_tts())
```

## Expected Output

### ASR Service Startup:
```
Loading faster-whisper model: large-v3 on cuda:0 (float16)
Warming up GPU...
Model loaded in 4.23s
Starting ASR queue manager workers
ASR service ready
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### TTS Service Startup:
```
Loading Chatterbox model on cuda:1
Warming up GPU...
Model loaded in 3.87s
Starting TTS queue manager workers
TTS service ready
INFO:     Uvicorn running on http://0.0.0.0:8002
```

### GPU Usage (nvidia-smi):
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.05    Driver Version: 535.86.05    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P2    65W / 350W |   4096MiB / 24576MiB |     25%      Default |
|-------------------------------+----------------------+----------------------|
|   1  NVIDIA GeForce ...  Off  | 00000000:02:00.0 Off |                  N/A |
| 32%   48C    P2    70W / 350W |   3584MiB / 24576MiB |     35%      Default |
+-----------------------------------------------------------------------------+

GPU 0: ASR service (4GB VRAM, 25% utilization)
GPU 1: TTS service (3.5GB VRAM, 35% utilization)
```

## Troubleshooting

### GPU Out of Memory
```bash
# Use INT8 quantization (ASR)
# Edit services/asr/config.yaml:
compute_type: "int8_float16"  # Instead of float16
```

### Models Not Loading
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

### WebSocket Connection Fails
```bash
# Check if services are running
curl http://localhost:8001/health
curl http://localhost:8002/health

# Check firewall
sudo ufw allow 8001
sudo ufw allow 8002
```

### Slow Performance
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check queue sizes
curl http://localhost:8001/health | jq '.queue_metrics'
curl http://localhost:8002/health | jq '.queue_metrics'
```

## Next Steps

1. **Integrate with LLM**: See [INTEGRATION.md](INTEGRATION.md)
2. **Production setup**: See [PRODUCTION.md](PRODUCTION.md)
3. **Hardware optimization**: See [HARDWARE_OPTIMIZATION.md](HARDWARE_OPTIMIZATION.md)
4. **Examples**: Check `examples/` directory

## Performance Targets

With your 2x RTX 3090 setup:

- **ASR Latency**: 80-150ms per transcription
- **TTS First Chunk**: 450-500ms
- **Total Pipeline**: <1 second (with streaming LLM)
- **Concurrent Users**: 50+ easily (up to 100+ possible)
- **GPU Utilization**: 20-30% at 30 connections
- **VRAM Usage**: ~8GB total (plenty of headroom)

## Quick Commands

```bash
# Start everything
./scripts/start_asr.sh &
./scripts/start_tts.sh &

# Monitor
watch -n 1 nvidia-smi

# Health check
curl http://localhost:8001/health
curl http://localhost:8002/health

# Stop
pkill -f "python server.py"
```

**You're ready to go!** 🚀
