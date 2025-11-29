#!/bin/bash
# Start TTS service with Chatterbox-vLLM (high-performance)

set -e

echo "Starting TTS service (Chatterbox-vLLM)..."

cd "$(dirname "$0")/../services/tts"

# Set GPU (use GPU 0 for single GPU, or set CUDA_VISIBLE_DEVICES externally)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# vLLM configuration
export TTS_MAX_BATCH_SIZE=${TTS_MAX_BATCH_SIZE:-8}
export TTS_MAX_MODEL_LEN=${TTS_MAX_MODEL_LEN:-1000}
export TTS_MAX_CONNECTIONS=${TTS_MAX_CONNECTIONS:-50}

echo "Configuration:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Max Batch Size: $TTS_MAX_BATCH_SIZE"
echo "  Max Model Len: $TTS_MAX_MODEL_LEN"
echo "  Max Connections: $TTS_MAX_CONNECTIONS"

# Add CUDA libraries to LD_LIBRARY_PATH
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"

# Create logs and model dirs
mkdir -p ../../logs
mkdir -p ../../t3-model
mkdir -p ../../t3-model-multilingual

# Start server
uv run python server.py 2>&1 | tee ../../logs/tts.log
