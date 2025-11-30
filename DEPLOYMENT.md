# TTS Service Deployment Guide

High-performance TTS service using Chatterbox-vLLM.

## Requirements

- Linux / WSL2
- NVIDIA GPU (RTX 3090 recommended, 24GB VRAM)
- CUDA 12.x
- Python 3.10+

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Fix missing tokenizer.json (PyPI package bug)
curl -sL "https://raw.githubusercontent.com/randombk/chatterbox-vllm/master/src/chatterbox_vllm/models/t3/tokenizer.json" \
  -o .venv/lib/python3.12/site-packages/chatterbox_vllm/models/t3/tokenizer.json

# 3. Create required directories
mkdir -p t3-model t3-model-multilingual voices logs

# 4. Start the service
./scripts/start_tts.sh
```

First startup downloads ~2GB of model weights and takes 2-5 minutes.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `1` | GPU index to use |
| `TTS_PORT` | `8002` | Server port |
| `TTS_MAX_CONNECTIONS` | `50` | Max WebSocket connections |

Example:

```bash
CUDA_VISIBLE_DEVICES=0 TTS_PORT=8003 ./scripts/start_tts.sh
```

## API Endpoints

### WebSocket Streaming

```
ws://localhost:8002/v1/stream/tts
```

**Synthesize text:**
```json
{
  "type": "synthesize",
  "text": "Hello, how are you?",
  "voice_id": "default",
  "streaming": true,
  "exaggeration": 0.5
}
```

**Register custom voice:**
```json
{
  "type": "register_voice",
  "voice_id": "my_voice",
  "reference_audio": "<base64 WAV data>"
}
```

**Response:** Binary audio chunks (PCM Float32, 24kHz mono)

### REST API

```
POST /v1/synthesize
GET /health
```

### Health Check

```bash
curl http://localhost:8002/health
```

## Performance

RTX 3090 benchmarks:

| Metric | Value |
|--------|-------|
| 40 min audio generation | ~87 seconds |
| Batch throughput | 4-10x vs standard Chatterbox |
| Sample rate | 24kHz |

## Systemd Service

Create `/etc/systemd/system/tts.service`:

```ini
[Unit]
Description=TTS Service (Chatterbox-vLLM)
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/voice-agent
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="TTS_PORT=8002"
ExecStart=/path/to/voice-agent/scripts/start_tts.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tts
sudo systemctl start tts
sudo systemctl status tts
```

## Logs

```bash
# View logs
tail -f logs/tts.log

# Check GPU usage
nvidia-smi
```

## Troubleshooting

### CUDA assertion errors / "srcIndex < srcSelectDimSize"

This is caused by a vLLM version mismatch. chatterbox-vllm requires exactly vLLM 0.9.2.

**Fix:** Recreate your venv and reinstall:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or just reinstall vLLM:

```bash
pip uninstall vllm -y
pip install vllm==0.9.2
```

### "t3-model/config.json not found"

The start script creates this automatically. If missing:

```bash
./scripts/start_tts.sh
```

### "No such file or directory" / tokenizer.json missing

The PyPI package is missing tokenizer.json. Download it manually:

```bash
curl -sL "https://raw.githubusercontent.com/randombk/chatterbox-vllm/master/src/chatterbox_vllm/models/t3/tokenizer.json" \
  -o .venv/lib/python3.12/site-packages/chatterbox_vllm/models/t3/tokenizer.json
```

Note: Re-run this after reinstalling chatterbox-vllm.

### Out of Memory

RTX 3090 (24GB) should handle default settings. If OOM occurs, the model context length (131072) may be too large for profiling. This is usually a one-time issue during first load.

### Slow first request

First request after startup is slower due to vLLM warmup. Subsequent requests are fast.

### Port already in use

```bash
TTS_PORT=8003 ./scripts/start_tts.sh
```

## Directory Structure

```
voice-agent/
├── services/tts/
│   ├── server.py           # FastAPI + WebSocket server
│   ├── config.yaml         # Service config
│   └── core/
│       ├── synthesizer.py  # Chatterbox-vLLM wrapper
│       ├── voice_manager.py
│       └── queue_manager.py
├── scripts/
│   ├── start_tts.sh        # Start script
│   └── install_vllm.sh     # Installation script
├── t3-model/               # vLLM model config (auto-created)
├── voices/                 # Cached voice embeddings
└── logs/                   # Service logs
```
