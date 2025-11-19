# Voice Agent - Current Status

**Last Updated:** 2025-11-17

---

## ✅ Working Components

### 1. **TTS Service (Text-to-Speech)**
- ✅ Chatterbox-streaming integration
- ✅ Streaming audio synthesis
- ✅ WebSocket endpoint: `ws://localhost:8003/v1/stream/tts`
- ✅ Voice cloning ready (urek.wav voice available)
- ✅ Multiple instances running (ports 8002, 8012)
- ✅ GPU acceleration (CUDA)
- ✅ Audio output is clear (no noise)

### 2. **NGINX Load Balancing**
- ✅ Running on port 80, 8000, 8003, 8080
- ✅ Routes to ASR backends (8001, 8011)
- ✅ Routes to TTS backends (8002, 8012)
- ✅ Health checks working
- ✅ WebSocket support enabled

### 3. **HTML Test Interface**
- ✅ ASR file upload UI
- ✅ TTS text-to-speech UI
- ✅ WebSocket connections
- ✅ Audio playback and download
- ✅ Real-time logging

### 4. **Infrastructure**
- ✅ Whisper large-v3 model loaded
- ✅ GPU available (2x RTX 3090)
- ✅ Services running on WSL (Windows Server 2022)
- ✅ Python environment with uv

---

## ⚠️ Issues to Fix

### **CRITICAL: ASR Service - VAD Not Working**

**Problem:**
- Audio is uploaded successfully
- Audio is resampled and sent to ASR service
- **VAD never detects turn end** - transcription never happens
- System gets stuck waiting for VAD to detect speech completion

**Recent Changes:**
- Implemented Silero VAD + Smart Turn v3 architecture (following Pipecat design)
- New file: `services/asr/core/vad_improved.py`
- Updated ASR server to use `ImprovedVAD`

**What Needs Debugging:**
1. Check if Silero VAD auto-downloaded successfully
2. Check if Smart Turn v3 model is loading from: `/mnt/d/voice-system/gonova-asr-tts/models/smart-turn-v3.0.onnx`
3. Verify VAD is receiving audio chunks
4. Check server logs for VAD detection events
5. Test if pause detection is working

**Files Involved:**
- `services/asr/core/vad_improved.py` (new VAD implementation)
- `services/asr/server.py` (updated to use ImprovedVAD)
- `models/smart-turn-v3.0.onnx` (Smart Turn model)

**Debug Steps:**
```bash
# Restart ASR with full logging
pkill -f "services/asr/server.py"
CUDA_VISIBLE_DEVICES=0 ASR_PORT=8001 ASR_INSTANCE_ID=1 uv run services/asr/server.py 2>&1 | tee asr_debug.log

# Test with HTML interface
# Check asr_debug.log for:
# - "Silero VAD loaded successfully"
# - "Smart Turn v3 loaded"
# - "Pause detected by Silero"
# - Any error messages
```

**Possible Solutions:**
- [ ] Verify Silero VAD downloads correctly
- [ ] Check Smart Turn model path is correct
- [ ] Add debug logging to VAD detection
- [ ] Test with shorter audio samples
- [ ] Consider temporary fallback to simple silence detection

---

## 🚀 Next Steps (External Access)

### **1. Set Up Reverse Proxy with frp**

**Goal:** Expose services to the internet via AWS EC2

**Architecture:**
```
Internet → AWS EC2 (frp server) → Windows Server (frp client) → NGINX → Services
```

**Tasks:**
- [ ] Set up frp server on AWS EC2
- [ ] Configure AWS Security Groups (ports 80, 443, 7000)
- [ ] Set up frp client on Windows Server (WSL)
- [ ] Configure NGINX for HTTPS (SSL certificates)
- [ ] Set up DNS (asr.gonova.one, tts.gonova.one → EC2 IP)
- [ ] Test complete setup end-to-end

**Reference:**
- frp guide already prepared (see conversation history)
- AWS EC2 instance available
- Domain: gonova.one (needs DNS configuration)

---

## 📋 Configuration Summary

### **Ports:**
- **8001, 8011** - ASR service instances
- **8002, 8012** - TTS service instances
- **8000** - NGINX → ASR load balancer
- **8003** - NGINX → TTS load balancer
- **8080** - NGINX health/status

### **Key Files:**
- `services/asr/server.py` - ASR WebSocket server
- `services/tts/server.py` - TTS WebSocket server
- `services/asr/core/vad_improved.py` - VAD implementation (NEW)
- `config/nginx.conf` - NGINX configuration (basic)
- `config/nginx-cloudflare.conf` - NGINX for Cloudflare
- `test-interface-fixed.html` - HTML test UI

### **Models:**
- `/mnt/d/voice-system/gonova-asr-tts/models/smart-turn-v3.0.onnx` - Smart Turn v3
- Whisper large-v3 (auto-downloaded by faster-whisper)
- Silero VAD (auto-downloaded via torch.hub)
- Chatterbox TTS (installed separately)

### **Custom Voice:**
- `services/tts/voices/urek.wav` - Custom voice reference
- Voice ID: `"urek"` (use in TTS requests)

---

## 🔧 Quick Commands

### **Start All Services:**
```bash
# ASR instances
CUDA_VISIBLE_DEVICES=0 ASR_PORT=8001 ASR_INSTANCE_ID=1 uv run services/asr/server.py &
CUDA_VISIBLE_DEVICES=0 ASR_PORT=8011 ASR_INSTANCE_ID=2 uv run services/asr/server.py &

# TTS instances
CUDA_VISIBLE_DEVICES=0 TTS_PORT=8002 TTS_INSTANCE_ID=1 uv run services/tts/server.py &
CUDA_VISIBLE_DEVICES=0 TTS_PORT=8012 TTS_INSTANCE_ID=2 uv run services/tts/server.py &

# NGINX
sudo systemctl restart nginx
```

### **Stop All Services:**
```bash
pkill -f "services/asr/server.py"
pkill -f "services/tts/server.py"
sudo systemctl stop nginx
```

### **Check Status:**
```bash
# Health checks
curl http://localhost:8000/health  # ASR
curl http://localhost:8003/health  # TTS

# Active processes
ps aux | grep -E "(asr|tts)" | grep python3

# GPU usage
nvidia-smi
```

---

## 📝 Notes

- **WSL Environment:** Services run in WSL on Windows Server 2022
- **GPU Setup:** 2x NVIDIA RTX 3090 (CUDA 12.7, Driver 566.03)
- **Network:** Server IP: 69.54.59.154
- **Domain:** gonova.one (needs DNS configuration for external access)

---

## 🐛 Known Issues

1. **ASR VAD stuck** - Critical, prevents transcription
2. **External access not set up** - Services only work on localhost
3. **SSL certificates needed** - For HTTPS/WSS external access

---

## ✨ Future Improvements

- [ ] Add rate limiting per user/IP
- [ ] Implement prometheus metrics
- [ ] Add authentication/API keys
- [ ] Create systemd services for auto-start
- [ ] Add monitoring dashboard
- [ ] Implement voice registration API
- [ ] Add audio format validation
- [ ] Create production deployment scripts
