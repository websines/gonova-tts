#!/bin/bash
# Start ASR service on GPU 0

set -e

echo "Starting ASR service on GPU 0..."

cd "$(dirname "$0")/../services/asr"

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Create logs dir
mkdir -p ../../logs

# Start server
uv run python server.py 2>&1 | tee ../../logs/asr.log
