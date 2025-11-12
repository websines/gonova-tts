#!/bin/bash
# Start multiple ASR service instances on GPU 0
# Each instance runs on a different port for load balancing

set -e

# Configuration
GPU_ID=0
INSTANCES=2
BASE_PORT=8001
PORTS=(8001 8011)

echo "Starting $INSTANCES ASR instances on GPU $GPU_ID..."

# Create logs directory
mkdir -p logs/asr

# Start each instance in background
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    INSTANCE_ID=$((i + 1))

    echo "Starting ASR instance $INSTANCE_ID on port $PORT..."

    # Set environment variables and start
    # Add CUDA libraries to LD_LIBRARY_PATH
    CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    ASR_PORT=$PORT \
    ASR_INSTANCE_ID=$INSTANCE_ID \
    LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH" \
    nohup uv run python services/asr/server.py \
        > logs/asr/instance_${INSTANCE_ID}_${PORT}.log 2>&1 &

    PID=$!
    echo $PID > logs/asr/instance_${INSTANCE_ID}.pid

    echo "  → Started ASR instance $INSTANCE_ID (PID: $PID, Port: $PORT)"

    # Small delay between starts to avoid GPU initialization conflicts
    sleep 2
done

echo ""
echo "All ASR instances started!"
echo ""
echo "Check status:"
echo "  ps aux | grep 'services/asr/server.py'"
echo ""
echo "Check logs:"
echo "  tail -f logs/asr/instance_*.log"
echo ""
echo "Check health:"
for PORT in "${PORTS[@]}"; do
    echo "  curl http://localhost:$PORT/health"
done
