#!/bin/bash
# Start a single TTS service instance
# Usage: ./start_tts_single.sh [PORT] [GPU_ID]

set -e

# Configuration with defaults
PORT=${1:-8002}
GPU_ID=${2:-1}

echo "Starting TTS instance on port $PORT (GPU $GPU_ID)..."

# Create logs directory
mkdir -p logs/tts

# Get CUDA libs path
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")

# Start the instance
CUDA_VISIBLE_DEVICES=$GPU_ID \
TTS_PORT=$PORT \
TTS_INSTANCE_ID=1 \
LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH" \
nohup uv run python services/tts/server.py \
    > logs/tts/instance_single_${PORT}.log 2>&1 &

PID=$!
echo $PID > logs/tts/instance_single.pid

echo "Started TTS instance (PID: $PID, Port: $PORT)"
echo ""
echo "Check logs:"
echo "  tail -f logs/tts/instance_single_${PORT}.log"
echo ""
echo "Check health:"
echo "  curl http://localhost:$PORT/health"
echo ""
echo "Stop:"
echo "  kill $PID"
