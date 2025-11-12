#!/bin/bash
# Interactive startup script with live log output
# Starts services and shows logs in real-time

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GPU_ASR=0
GPU_TTS=1
ASR_PORTS=(8001 8011)
TTS_PORTS=(8002 8012)

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Voice Agent Cluster - Interactive Start${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Create logs directory
mkdir -p logs/asr logs/tts

# Function to start a service and tail its log
start_service() {
    local service=$1
    local gpu_id=$2
    local port=$3
    local instance_id=$4

    echo -e "${YELLOW}Starting $service instance $instance_id on port $port...${NC}"

    # Get CUDA libs path
    CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")

    # Start service in background
    if [ "$service" = "ASR" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id \
        ASR_PORT=$port \
        ASR_INSTANCE_ID=$instance_id \
        LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH" \
        uv run python services/asr/server.py \
            > logs/asr/instance_${instance_id}_${port}.log 2>&1 &

        PID=$!
        echo $PID > logs/asr/instance_${instance_id}.pid
        LOG_FILE="logs/asr/instance_${instance_id}_${port}.log"
    else
        CUDA_VISIBLE_DEVICES=$gpu_id \
        TTS_PORT=$port \
        TTS_INSTANCE_ID=$instance_id \
        LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH" \
        uv run python services/tts/server.py \
            > logs/tts/instance_${instance_id}_${port}.log 2>&1 &

        PID=$!
        echo $PID > logs/tts/instance_${instance_id}.pid
        LOG_FILE="logs/tts/instance_${instance_id}_${port}.log"
    fi

    echo -e "${GREEN}  → Started PID: $PID${NC}"
    echo -e "${BLUE}  → Watching logs (wait for model to load)...${NC}"
    echo ""

    # Wait a moment for log file to be created
    sleep 1

    # Tail log file until we see success or error
    tail -f "$LOG_FILE" 2>/dev/null &
    TAIL_PID=$!

    # Wait for startup indicators (max 180 seconds for TTS model loading)
    local waited=0
    local max_wait=180

    while [ $waited -lt $max_wait ]; do
        if [ -f "$LOG_FILE" ]; then
            # Check for success indicators
            if grep -q "Uvicorn running\|Application startup complete\|Model loaded successfully" "$LOG_FILE" 2>/dev/null; then
                sleep 2  # Give it a moment to stabilize
                kill $TAIL_PID 2>/dev/null || true
                echo ""
                echo -e "${GREEN}✓ $service instance $instance_id started successfully!${NC}"
                echo ""
                return 0
            fi

            # Check for errors
            if grep -qi "error\|exception\|failed\|traceback" "$LOG_FILE" 2>/dev/null; then
                sleep 2  # Show a bit more of the error
                kill $TAIL_PID 2>/dev/null || true
                echo ""
                echo -e "${RED}✗ $service instance $instance_id failed to start!${NC}"
                echo -e "${YELLOW}Check full log: $LOG_FILE${NC}"
                echo ""
                return 1
            fi
        fi

        sleep 5
        waited=$((waited + 5))

        # Show progress every 15 seconds
        if [ $((waited % 15)) -eq 0 ]; then
            echo -e "${YELLOW}  ... still loading ($waited/${max_wait}s)${NC}"
        fi
    done

    # Timeout
    kill $TAIL_PID 2>/dev/null || true
    echo ""
    echo -e "${YELLOW}⚠ Timeout waiting for $service instance $instance_id${NC}"
    echo -e "${YELLOW}Service may still be loading. Check: $LOG_FILE${NC}"
    echo ""
    return 0
}

# Start ASR instances
echo -e "${BLUE}1. Starting ASR instances on GPU $GPU_ASR${NC}"
echo ""

for i in "${!ASR_PORTS[@]}"; do
    PORT=${ASR_PORTS[$i]}
    INSTANCE_ID=$((i + 1))
    start_service "ASR" "$GPU_ASR" "$PORT" "$INSTANCE_ID"
    sleep 2
done

# Start TTS instances
echo -e "${BLUE}2. Starting TTS instances on GPU $GPU_TTS${NC}"
echo ""

for i in "${!TTS_PORTS[@]}"; do
    PORT=${TTS_PORTS[$i]}
    INSTANCE_ID=$((i + 1))
    start_service "TTS" "$GPU_TTS" "$PORT" "$INSTANCE_ID"
    sleep 2
done

# Final status check
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Startup Complete - Status Check${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

echo "ASR Health Checks:"
for PORT in "${ASR_PORTS[@]}"; do
    if curl -s -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Port $PORT: Healthy"
    else
        echo -e "  ${YELLOW}⚠${NC} Port $PORT: Not ready yet"
    fi
done

echo ""
echo "TTS Health Checks:"
for PORT in "${TTS_PORTS[@]}"; do
    if curl -s -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Port $PORT: Healthy"
    else
        echo -e "  ${YELLOW}⚠${NC} Port $PORT: Not ready yet"
    fi
done

echo ""
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Monitor all logs:"
echo "  tail -f logs/asr/*.log logs/tts/*.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Stop cluster:"
echo "  bash scripts/stop_all.sh"
echo ""
