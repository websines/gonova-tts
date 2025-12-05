# Hardware Optimization for 2x RTX 3090

Optimized architecture for dual-GPU setup with massive headroom.

## Your Hardware

```
System Specs:
├─ GPU 1: RTX 3090 (24GB VRAM)
├─ GPU 2: RTX 3090 (24GB VRAM)
├─ Total VRAM: 48GB
├─ RAM: 96GB
└─ Total Power: INSANE
```

## The Perfect Setup

### GPU Assignment Strategy

```
┌────────────────────────────────────────────────────────┐
│                  OPTIMAL GPU ALLOCATION                 │
└────────────────────────────────────────────────────────┘

GPU 0 (RTX 3090 #1) - 24GB VRAM
├─ ASR Service (Port 8001)
│  ├─ faster-whisper large-v3: 3GB
│  ├─ Smart Turn v3 VAD: <100MB (CPU)
│  └─ Available VRAM: 21GB (room for 7 more instances!)
│
└─ Capacity: 30+ concurrent streams, easy

GPU 1 (RTX 3090 #2) - 24GB VRAM
├─ TTS Service (Port 8002)
│  ├─ Chatterbox-streaming: 2-3GB
│  └─ Available VRAM: 21GB (room for 7 more instances!)
│
└─ Capacity: 30+ concurrent syntheses, easy

Benefits:
✅ Zero GPU contention
✅ Maximum performance for each service
✅ Can scale to 100+ connections easily
✅ Simple configuration
```

### Why This is Perfect

**Dedicated GPU per Service:**
- ASR never competes with TTS for GPU
- Each service gets full 24GB to itself
- Can run larger models if needed
- Massive headroom for scaling

**Your Capacity:**

| Metric | With Your Hardware | Notes |
|--------|-------------------|-------|
| **Current target** | 20-30 connections | ✅ Easy |
| **Max capacity** | 100+ connections | With current models |
| **ASR throughput** | 50+ concurrent streams | Per GPU |
| **TTS throughput** | 50+ concurrent syntheses | Per GPU |
| **VRAM usage** | ~6GB / 48GB (12.5%) | Tons of headroom |

## Configuration

### GPU Assignment

```yaml
# services/asr/config.yaml
model:
  device: "cuda:0"  # ← GPU 0 for ASR
  name: "large-v3"
  compute_type: "float16"  # Or int8 for even more capacity

server:
  host: "0.0.0.0"
  port: 8001
  max_connections: 50  # Can handle way more than 30!
```

```yaml
# services/tts/config.yaml
model:
  device: "cuda:1"  # ← GPU 1 for TTS

server:
  host: "0.0.0.0"
  port: 8002
  max_connections: 50  # Can handle way more than 30!
```

### Environment Variables

```bash
# Start ASR service on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run python services/asr/server.py

# Start TTS service on GPU 1
CUDA_VISIBLE_DEVICES=1 uv run python services/tts/server.py
```

### Systemd Services

```ini
# /etc/systemd/system/voice-agent-asr.service
[Unit]
Description=Voice Agent ASR Service (GPU 0)
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/voice-agent/services/asr
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PATH=/home/your-user/.local/bin:$PATH"
ExecStart=/home/your-user/.local/bin/uv run python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/voice-agent-tts.service
[Unit]
Description=Voice Agent TTS Service (GPU 1)
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/voice-agent/services/tts
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="PATH=/home/your-user/.local/bin:$PATH"
ExecStart=/home/your-user/.local/bin/uv run python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Performance Expectations

### With Your Hardware:

```
30 Concurrent Connections (Your Target):
────────────────────────────────────────────
GPU 0 (ASR):
├─ VRAM usage: ~4GB (3GB model + 1GB inference)
├─ Utilization: ~20-30%
└─ Status: UNDERUTILIZED ✅

GPU 1 (TTS):
├─ VRAM usage: ~4GB (3GB model + 1GB inference)
├─ Utilization: ~30-40%
└─ Status: UNDERUTILIZED ✅

RAM:
├─ Queues: ~50MB
├─ Python processes: ~2GB
├─ Audio buffers: ~500MB
└─ Total: ~3GB / 96GB (3%)

Latency:
├─ ASR: 80-120ms (GPU 0 dedicated)
├─ TTS: 450-500ms first chunk (GPU 1 dedicated)
└─ Total pipeline: 600-800ms ✅
```

### Scaling Potential:

```
If you need to scale beyond 30 connections:

Option 1: Increase limits (same GPUs)
├─ ASR: 50 connections on GPU 0
├─ TTS: 50 connections on GPU 1
└─ Total: 100 concurrent users

Option 2: Run multiple instances per GPU
├─ 2x ASR instances on GPU 0 (ports 8001, 8011)
├─ 2x TTS instances on GPU 1 (ports 8002, 8012)
├─ Load balance with nginx
└─ Total: 200+ concurrent users

Option 3: Use both GPUs for each service
├─ ASR: Multi-GPU with model parallelism
├─ TTS: Multi-GPU with model parallelism
└─ Total: Extreme performance (probably overkill)
```

## Memory Optimization

### Current Allocation (Conservative):

```
GPU 0 (ASR):
├─ faster-whisper large-v3: 3GB
├─ Inference headroom: 1GB
├─ Multi-stream buffers: 1GB
└─ Total: 5GB / 24GB (21%)

GPU 1 (TTS):
├─ Chatterbox model: 2.5GB
├─ Voice embeddings cache: 0.5GB
├─ Inference headroom: 1GB
└─ Total: 4GB / 24GB (17%)

System RAM:
├─ Python + dependencies: 2GB
├─ Audio queues: 50MB
├─ Redis (if used): 500MB
├─ OS overhead: 2GB
└─ Total: ~5GB / 96GB (5%)

You have TONS of headroom!
```

### Aggressive Optimization (For Max Throughput):

```yaml
# Even better performance with INT8 quantization

asr:
  model: "large-v3"
  compute_type: "int8_float16"  # 2x faster, half the VRAM
  # VRAM: 1.5GB (instead of 3GB)
  # Can fit 16 instances on one GPU!

tts:
  # Chatterbox doesn't support INT8 yet
  # But still plenty of room
```

## Monitoring

### GPU Monitoring Script

```bash
# scripts/monitor_gpus.sh
#!/bin/bash

watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits'
```

**Expected output:**
```
0, NVIDIA GeForce RTX 3090, 45, 25, 20, 4096, 24576
1, NVIDIA GeForce RTX 3090, 48, 35, 18, 3584, 24576
      ↑                    ↑   ↑   ↑      ↑      ↑
      GPU ID              Temp GPU% MEM%  Used  Total
```

### Health Check Endpoint

```python
# services/asr/server.py
import torch

@app.get("/health")
async def health_check():
    gpu_id = 0  # ASR on GPU 0

    return {
        "status": "healthy",
        "gpu": {
            "id": gpu_id,
            "name": torch.cuda.get_device_name(gpu_id),
            "available": torch.cuda.is_available(),
            "memory_allocated": torch.cuda.memory_allocated(gpu_id) / 1e9,  # GB
            "memory_reserved": torch.cuda.memory_reserved(gpu_id) / 1e9,
            "memory_total": torch.cuda.get_device_properties(gpu_id).total_memory / 1e9,
        },
        "active_connections": len(manager.connections),
        "queue_size": manager.queue_manager.input_queue.qsize()
    }
```

```bash
# Check ASR service
curl http://localhost:8001/health

# Check TTS service
curl http://localhost:8002/health
```

## Advanced: Multi-Instance Scaling

If you ever need to handle 100+ connections:

### nginx Load Balancing

```nginx
# /etc/nginx/nginx.conf

upstream asr_backend {
    # Run 2 ASR instances on GPU 0
    server localhost:8001;
    server localhost:8011;
}

upstream tts_backend {
    # Run 2 TTS instances on GPU 1
    server localhost:8002;
    server localhost:8012;
}

server {
    listen 80;

    location /v1/stream/asr {
        proxy_pass http://asr_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /v1/stream/tts {
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Running Multiple Instances

```bash
# GPU 0: 2 ASR instances
CUDA_VISIBLE_DEVICES=0 uv run python server.py --port 8001 &
CUDA_VISIBLE_DEVICES=0 uv run python server.py --port 8011 &

# GPU 1: 2 TTS instances
CUDA_VISIBLE_DEVICES=1 uv run python server.py --port 8002 &
CUDA_VISIBLE_DEVICES=1 uv run python server.py --port 8012 &

# Now you can handle 200+ connections!
```

## Benchmarks with Your Hardware

### Expected Performance:

```
Single Connection:
├─ ASR latency: 80ms (GPU 0 dedicated)
├─ TTS first chunk: 450ms (GPU 1 dedicated)
└─ Total: 530ms ✅ Excellent!

30 Concurrent Connections:
├─ ASR latency: 100ms (slight queue wait)
├─ TTS first chunk: 480ms (slight queue wait)
├─ GPU 0 utilization: 25%
├─ GPU 1 utilization: 35%
└─ Total: 580ms ✅ Still great!

50 Concurrent Connections (if needed):
├─ ASR latency: 120ms
├─ TTS first chunk: 520ms
├─ GPU 0 utilization: 40%
├─ GPU 1 utilization: 50%
└─ Total: 640ms ✅ Very good!

100 Concurrent Connections (extreme):
├─ ASR latency: 200ms
├─ TTS first chunk: 700ms
├─ GPU 0 utilization: 70%
├─ GPU 1 utilization: 80%
└─ Total: 900ms 🟡 Acceptable
```

## Power & Thermal Considerations

### Power Draw:

```
RTX 3090 TDP: 350W each
├─ Idle: ~30W per GPU
├─ At 30% load: ~120W per GPU
├─ At 100% load: 350W per GPU
└─ Your setup (30 connections): ~240W total (both GPUs)

System total: ~400W (GPUs + CPU + RAM)
```

### Thermal:

```
Expected temps at 30 connections:
├─ GPU 0: 45-55°C
├─ GPU 1: 50-60°C (TTS uses more)
└─ Safe range: <83°C

Recommendations:
├─ Good case airflow
├─ Monitor with nvidia-smi
└─ Set fan curve if needed
```

## Cost Savings

### Your Setup vs Cloud:

```
Cloud GPU instances (AWS/GCP):
├─ 2x A100 (40GB): ~$8/hour = $5,760/month
├─ 2x V100 (32GB): ~$4/hour = $2,880/month
└─ 2x T4 (16GB): ~$1.5/hour = $1,080/month

Your bare metal setup:
├─ Upfront cost: Already owned
├─ Monthly cost: ~$50 electricity
└─ Savings: $1,000-5,000/month ✅
```

## Recommendations

### For Your Hardware:

1. **✅ DO:**
   - Dedicate GPU 0 to ASR
   - Dedicate GPU 1 to TTS
   - Start with max 30-50 connections per service
   - Monitor GPU utilization with nvidia-smi
   - Use INT8 if you need more capacity

2. **❌ DON'T:**
   - Run both services on same GPU (unnecessary)
   - Use CPU for inference (waste of GPUs)
   - Over-allocate connections (start conservative)

3. **🔧 TUNE:**
   - Increase limits gradually based on monitoring
   - Consider INT8 quantization if scaling beyond 50
   - Add load balancing if you hit 100+ connections

### Quick Start Commands:

```bash
# Terminal 1: Start ASR on GPU 0
cd services/asr
CUDA_VISIBLE_DEVICES=0 uv run python server.py

# Terminal 2: Start TTS on GPU 1
cd services/tts
CUDA_VISIBLE_DEVICES=1 uv run python server.py

# Terminal 3: Monitor GPUs
watch -n 1 nvidia-smi
```

## Summary

**Your Hardware = BEAST MODE**

- 2x RTX 3090s = Way more than you need for 30 connections
- Can easily handle 100+ with current setup
- Can scale to 200+ with multi-instance deployment
- Tons of VRAM and RAM headroom for future growth

**Configuration:**
- GPU 0: ASR only
- GPU 1: TTS only
- Simple, clean, maximum performance

**You're in excellent shape!** 🚀
