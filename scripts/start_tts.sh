#!/bin/bash
# Start TTS service with Chatterbox-vLLM (high-performance)

set -e

echo "Starting TTS service (Chatterbox-vLLM)..."

# Go to project root (chatterbox-vllm needs t3-model/ in cwd)
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Set GPU (use GPU 1 by default, or set CUDA_VISIBLE_DEVICES externally)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Configuration (chatterbox-vllm uses its own defaults for batch_size=10, model_len=1000)
export TTS_MAX_CONNECTIONS=${TTS_MAX_CONNECTIONS:-50}

echo "Configuration:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Max Connections: $TTS_MAX_CONNECTIONS"
echo "  Working Dir: $PROJECT_ROOT"

# Add CUDA libraries to LD_LIBRARY_PATH
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
if [ -n "$CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"
fi

# Create required directories
mkdir -p logs
mkdir -p voices
mkdir -p t3-model
mkdir -p t3-model-multilingual

# Note: t3-model/ contents are created by chatterbox-vllm's from_pretrained()
# It downloads weights and creates symlinks automatically

# Start server from project root (so t3-model/ is accessible)
echo ""
echo "Starting server..."
uv run python services/tts/server.py 2>&1 | tee logs/tts.log
