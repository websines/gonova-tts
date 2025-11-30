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

## Server Modes

### Batch Mode (Default)

High throughput, higher latency. Best for offline processing.

```bash
./scripts/start_tts.sh
```

- **TTFB**: 5-10+ seconds (processes all sentences before streaming)
- **Throughput**: 4-10x faster than realtime
- **Use case**: Batch audio generation, pre-rendering

### Streaming Mode (Low Latency)

Token-level streaming for real-time applications.

```bash
./scripts/start_streaming_tts.sh
```

- **TTFB**: ~1-2 seconds (streams after 25 tokens)
- **Throughput**: Slightly lower than batch mode
- **Use case**: Voice agents, real-time TTS

Streaming mode environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_CHUNK_SIZE` | `25` | Tokens per audio chunk |
| `TTS_CONTEXT_WINDOW` | `50` | Context tokens for S3Gen coherence |

## Performance

RTX 3090 benchmarks:

| Metric | Batch Mode | Streaming Mode |
|--------|------------|----------------|
| TTFB | 5-10+ sec | ~1-2 sec |
| 40 min audio | ~87 sec | ~100 sec |
| RTF | 0.04x | 0.3-0.5x |
| Sample rate | 24kHz | 24kHz |

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

### "Tokenizer EnTokenizer not found" / vLLM spawn multiprocessing error

**This is the most common issue.** vLLM uses `spawn` multiprocessing which re-executes the main script in subprocesses. The `EnTokenizer` must be registered before vLLM initializes.

**The fix:** Your entry point script MUST import `chatterbox_vllm` at the **top level** (outside `if __name__ == "__main__":`):

```python
# CORRECT - import at top level, BEFORE the main guard
from chatterbox_vllm.tts import ChatterboxTTS

if __name__ == "__main__":
    # ... rest of your code
```

```python
# WRONG - import inside the main guard won't work with spawn
if __name__ == "__main__":
    from chatterbox_vllm.tts import ChatterboxTTS  # TOO LATE!
```

When vLLM spawns a subprocess, it re-imports your script. The top-level import triggers `chatterbox_vllm/models/t3/__init__.py` which registers the tokenizer with vLLM's `TokenizerRegistry`. Without this, the subprocess doesn't know about `EnTokenizer`.

Use `services/tts/run_server.py` as the entry point - it has this pattern correct.

### CUDA assertion errors / "srcIndex < srcSelectDimSize"

This can be caused by:
1. **Large max_model_len** - The model config uses 131072 tokens which causes OOM during profiling. Our code sets `max_model_len=1000` to fix this.
2. **vLLM version mismatch** - chatterbox-vllm (git master) requires vLLM 0.10.0.

**Fix:** Reinstall from git:

```bash
pip uninstall chatterbox-vllm vllm -y
pip install "chatterbox-vllm @ git+https://github.com/randombk/chatterbox-vllm.git"
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
│   ├── run_server.py              # Batch mode entry point
│   ├── run_streaming_server.py    # Streaming mode entry point
│   ├── server.py                  # FastAPI + WebSocket server
│   └── core/
│       ├── synthesizer.py         # Batch synthesizer (sentences)
│       ├── streaming_synthesizer.py  # Streaming synthesizer (tokens)
│       ├── voice_manager.py
│       └── queue_manager.py
├── scripts/
│   ├── start_tts.sh               # Start batch mode
│   ├── start_streaming_tts.sh     # Start streaming mode
│   └── test_tts_realtime.py       # Benchmark client
├── t3-model/                      # vLLM model config (auto-created)
├── voices/                        # Cached voice embeddings
└── logs/                          # Service logs
```
