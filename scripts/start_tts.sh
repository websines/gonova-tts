#!/bin/bash
# Start TTS service on GPU 1

set -e

echo "Starting TTS service on GPU 1..."

cd "$(dirname "$0")/../services/tts"

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Add CUDA libraries to LD_LIBRARY_PATH
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"

# Create logs dir
mkdir -p ../../logs

# Start server
uv run python server.py 2>&1 | tee ../../logs/tts.log
