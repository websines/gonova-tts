#!/bin/bash
# Start multiple TTS service instances on GPU 0
# Each instance runs on a different port for load balancing

set -e

# Configuration
GPU_ID=0
INSTANCES=2
PORTS=(8002 8012)

echo "Starting $INSTANCES TTS instances on GPU $GPU_ID..."

# Create logs directory
mkdir -p logs/tts

# Start each instance in background
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    INSTANCE_ID=$((i + 1))

    echo "Starting TTS instance $INSTANCE_ID on port $PORT..."

    # Set environment variables and start
    # Try to get CUDA libs path, fall back if fails
    CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    TTS_PORT=$PORT \
    TTS_INSTANCE_ID=$INSTANCE_ID \
    LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH" \
    nohup uv run python services/tts/server.py \
        > logs/tts/instance_${INSTANCE_ID}_${PORT}.log 2>&1 &

    PID=$!
    echo $PID > logs/tts/instance_${INSTANCE_ID}.pid

    echo "  → Started TTS instance $INSTANCE_ID (PID: $PID, Port: $PORT)"

    # Small delay between starts to avoid GPU initialization conflicts
    sleep 2
done

echo ""
echo "All TTS instances started!"
echo ""
echo "Check status:"
echo "  ps aux | grep 'services/tts/server.py'"
echo ""
echo "Check logs:"
echo "  tail -f logs/tts/instance_*.log"
echo ""
echo "Check health:"
for PORT in "${PORTS[@]}"; do
    echo "  curl http://localhost:$PORT/health"
done
