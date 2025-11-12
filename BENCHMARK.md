# Voice Agent Benchmarking Guide

Performance benchmarking tools for ASR and TTS services.

## Quick Start

### Benchmark Everything

```bash
# Test local services
./scripts/benchmark_all.sh

# Test remote/production services
ASR_URL="https://asr.gonova.one" TTS_URL="https://tts.gonova.one" ./scripts/benchmark_all.sh
```

### Benchmark Individual Services

```bash
# ASR only
./scripts/benchmark_asr.sh

# TTS only
./scripts/benchmark_tts.sh
```

## Configuration

### Environment Variables

```bash
# ASR Configuration
export ASR_URL="http://localhost"           # Service URL
export ASR_HOST="localhost"                 # Host header
export ITERATIONS=5                         # Number of test iterations

# TTS Configuration
export TTS_URL="http://localhost"
export TTS_HOST="localhost"
export ITERATIONS=5
```

### Example: Test Production

```bash
# Test through Cloudflare
ASR_URL="https://asr.gonova.one" \
TTS_URL="https://tts.gonova.one" \
ASR_HOST="asr.gonova.one" \
TTS_HOST="tts.gonova.one" \
ITERATIONS=10 \
./scripts/benchmark_all.sh
```

### Example: Test via VPS

```bash
# Test VPS reverse proxy
ASR_URL="http://your-vps-ip" \
TTS_URL="http://your-vps-ip" \
ASR_HOST="asr.gonova.one" \
TTS_HOST="tts.gonova.one" \
./scripts/benchmark_all.sh
```

## What Gets Measured

### ASR Benchmarks

1. **Health Check Latency** - Time to respond to `/health` endpoint
2. **Transcription Latency** - Average time to transcribe 3-second audio
3. **WebSocket Connectivity** - Connection time and stability
4. **Load Performance** - Concurrent request handling
5. **Success Rate** - Percentage of successful transcriptions

### TTS Benchmarks

1. **Health Check Latency** - Time to respond to `/health` endpoint
2. **Synthesis Latency** - Time to synthesize different text lengths:
   - Short (2 words): ~10 characters
   - Medium (9 words): ~40 characters
   - Long (19 words): ~100 characters
3. **Streaming Performance**:
   - Time to first audio chunk (TTFB)
   - Total synthesis time
   - Number of chunks
   - Streaming efficiency
4. **Load Performance** - Concurrent synthesis handling
5. **Success Rate** - Percentage of successful syntheses

## Performance Targets

### ASR Service

| Metric | Excellent | Good | Moderate | Poor |
|--------|-----------|------|----------|------|
| Health check | < 50ms | < 100ms | < 200ms | > 200ms |
| Transcription | < 200ms | < 500ms | < 1000ms | > 1000ms |
| Success rate | 100% | > 95% | > 90% | < 90% |

### TTS Service

| Metric | Excellent | Good | Moderate | Poor |
|--------|-----------|------|----------|------|
| Health check | < 50ms | < 100ms | < 200ms | > 200ms |
| Synthesis | < 300ms | < 600ms | < 1500ms | > 1500ms |
| First chunk | < 300ms | < 500ms | < 1000ms | > 1000ms |
| Success rate | 100% | > 95% | > 90% | < 90% |

### Voice Agent (Combined)

For real-time voice agents, the **total round-trip latency** should be:

- **Excellent**: < 800ms (ASR + TTS + network)
- **Good**: < 1500ms
- **Acceptable**: < 2500ms
- **Poor**: > 2500ms

## Interpreting Results

### Example Output

```
ASR Service Benchmark
=========================================
Health Check Latency:  45ms
Average Transcription: 320ms
Success Rate:          5/5 (100%)

Performance Analysis:
  ✓ Good - Acceptable for real-time use
```

### What to Check if Performance is Poor

**High Health Check Latency (> 200ms)**
- Network issues between benchmark and service
- Server overloaded
- Check: `ping <service-url>`

**High Transcription Latency (> 1000ms)**
- GPU not being used (check `nvidia-smi`)
- Model not loaded properly
- Audio processing bottleneck
- Check: `tail -f logs/asr/*.log`

**High Synthesis Latency (> 1500ms)**
- GPU memory full
- Model loading issues
- Text too complex
- Check: `nvidia-smi` and `tail -f logs/tts/*.log`

**Low Success Rate (< 90%)**
- Service crashes under load
- Audio format incompatibility
- Network timeouts
- Check service logs for errors

**Slow Time to First Chunk (> 1000ms)**
- Not streaming properly (buffering)
- Model initialization delay
- Network buffering
- Check WebSocket connection

## Comparing Configurations

### Before/After Comparison

```bash
# Benchmark local (baseline)
./scripts/benchmark_all.sh > baseline.txt

# Deploy to VPS/Cloudflare

# Benchmark production
ASR_URL="https://asr.gonova.one" \
TTS_URL="https://tts.gonova.one" \
./scripts/benchmark_all.sh > production.txt

# Compare
diff baseline.txt production.txt
```

### Network Overhead Calculation

```bash
# Local latency
LOCAL_ASR=320ms  # From local benchmark
LOCAL_TTS=450ms

# Production latency
PROD_ASR=365ms   # From production benchmark
PROD_TTS=495ms

# Network overhead
ASR_OVERHEAD = 365 - 320 = 45ms
TTS_OVERHEAD = 495 - 450 = 45ms
```

**Good**: < 50ms overhead
**Acceptable**: < 100ms overhead
**Poor**: > 150ms overhead

## Continuous Benchmarking

### Automated Monitoring

```bash
# Run benchmark every hour and log results
*/60 * * * * cd /path/to/voice-agent && ./scripts/benchmark_all.sh >> logs/benchmarks.log 2>&1
```

### Performance Regression Detection

```bash
# Create a performance baseline
./scripts/benchmark_all.sh > benchmarks/baseline.txt

# After changes, compare
./scripts/benchmark_all.sh > benchmarks/current.txt

# Alert if performance degrades by > 20%
```

## Dependencies

### Required

- `curl` - HTTP requests
- `bash` - Shell scripting

### Optional (for full features)

```bash
# For WebSocket testing
cargo install websocat
# or
sudo apt install websocat

# For audio generation (ASR tests)
sudo apt install sox

# For TTS WebSocket testing
pip install websockets
```

## Troubleshooting

### "Service unhealthy" Error

```bash
# Check if service is running
curl http://localhost/health -H "Host: asr.gonova.one"

# Check service logs
tail -f logs/asr/*.log
tail -f logs/tts/*.log
```

### "websocat not found" Warning

This is optional. WebSocket tests will be skipped.

```bash
# To enable WebSocket tests
cargo install websocat
```

### "sox not found" Warning

ASR benchmark will use alternative method to generate test audio.

```bash
# For better test audio
sudo apt install sox
```

### Timeouts

If tests timeout, increase the timeout in the scripts or check:
- Service is actually running
- Network connectivity
- Firewall not blocking
- GPU has available memory

## Best Practices

1. **Run benchmarks after deployment** to verify performance
2. **Benchmark during different load conditions**
3. **Compare local vs production** to measure network overhead
4. **Set up monitoring** for continuous performance tracking
5. **Benchmark before/after changes** to detect regressions
6. **Test from different geographic locations** if serving global users

## Example Workflow

```bash
# 1. Start services
./scripts/start_with_logs.sh

# 2. Wait for services to be fully loaded (2-3 minutes)

# 3. Benchmark local performance
./scripts/benchmark_all.sh

# 4. Deploy to VPS or setup Cloudflare Tunnel

# 5. Benchmark production
ASR_URL="https://asr.gonova.one" \
TTS_URL="https://tts.gonova.one" \
./scripts/benchmark_all.sh

# 6. Compare results and optimize if needed
```

---

**Questions or issues?** Check service logs and GPU usage first.
