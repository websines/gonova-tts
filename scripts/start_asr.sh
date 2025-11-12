#!/bin/bash
# Start ASR service on GPU 0

set -e

echo "Starting ASR service on GPU 0..."

cd "$(dirname "$0")/../services/asr"

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Add CUDA libraries to LD_LIBRARY_PATH
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"

# Create logs dir
mkdir -p ../../logs

# Start server
uv run python server.py 2>&1 | tee ../../logs/asr.log
