#!/bin/bash
# Debug TTS startup - run in foreground to see errors

set -e

echo "========================================"
echo "TTS Debug Mode - Running in Foreground"
echo "========================================"
echo ""

# Configuration
GPU_ID=1
PORT=8002

echo "GPU: $GPU_ID"
echo "Port: $PORT"
echo ""

# Get CUDA libs path
echo "Getting CUDA library path..."
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
echo "CUDA libs: $CUDA_LIBS"
echo ""

echo "Checking tokenizers version..."
python -c "import tokenizers; print('Tokenizers:', tokenizers.__version__)"
echo ""

echo "Starting TTS server..."
echo "========================================"
echo ""

# Run TTS server in foreground
CUDA_VISIBLE_DEVICES=$GPU_ID \
TTS_PORT=$PORT \
TTS_INSTANCE_ID=1 \
LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH" \
uv run python services/tts/server.py
