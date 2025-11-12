#!/bin/bash
# Start TTS service on GPU 1

set -e

echo "Starting TTS service on GPU 1..."

cd "$(dirname "$0")/../services/tts"

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Create logs dir
mkdir -p ../../logs

# Start server
uv run python server.py 2>&1 | tee ../../logs/tts.log
