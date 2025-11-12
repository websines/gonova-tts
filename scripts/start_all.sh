#!/bin/bash
# Start entire voice agent cluster with load balancing
# - 2 ASR instances on GPU 0 (ports 8001, 8011)
# - 2 TTS instances on GPU 1 (ports 8002, 8012)
# - nginx load balancer (ASR: port 8000, TTS: port 8003)

set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "Starting Voice Agent Cluster"
echo "========================================="
echo ""

# Start ASR cluster
echo "1. Starting ASR cluster (GPU 0)..."
bash scripts/start_asr_cluster.sh
echo ""

# Wait a bit for ASR instances to initialize
sleep 5

# Start TTS cluster
echo "2. Starting TTS cluster (GPU 1)..."
bash scripts/start_tts_cluster.sh
echo ""

# Wait for TTS instances to initialize
sleep 5

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "⚠️  WARNING: nginx not found! Install it first:"
    echo "   macOS: brew install nginx"
    echo "   Ubuntu: sudo apt install nginx"
    echo ""
    echo "Skipping nginx startup..."
    echo ""
else
    # Start nginx
    echo "3. Starting nginx load balancer..."
    echo ""

    # Copy nginx config if needed
    if [ -f "config/nginx.conf" ]; then
        echo "Using nginx config: config/nginx.conf"
        echo ""
        echo "To start nginx manually:"
        echo "  sudo nginx -c $(pwd)/config/nginx.conf"
        echo ""
    fi
fi

echo "========================================="
echo "Cluster Status"
echo "========================================="
echo ""

# Check ASR instances
echo "ASR instances:"
for PORT in 8001 8011; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  ✓ Port $PORT: Running"
    else
        echo "  ✗ Port $PORT: Not responding (may still be starting)"
    fi
done
echo ""

# Check TTS instances
echo "TTS instances:"
for PORT in 8002 8012; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  ✓ Port $PORT: Running"
    else
        echo "  ✗ Port $PORT: Not responding (may still be starting)"
    fi
done
echo ""

echo "========================================="
echo "Usage"
echo "========================================="
echo ""
echo "Connect to services via nginx load balancer:"
echo "  ASR:  ws://localhost:8000/v1/stream/asr"
echo "  TTS:  ws://localhost:8003/v1/stream/tts"
echo ""
echo "Or connect directly to instances:"
echo "  ASR:  ports 8001, 8011"
echo "  TTS:  ports 8002, 8012"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/asr/instance_*.log"
echo "  tail -f logs/tts/instance_*.log"
echo ""
echo "Monitor GPUs:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop cluster:"
echo "  bash scripts/stop_all.sh"
echo ""
