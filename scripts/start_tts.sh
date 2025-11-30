#!/bin/bash
# Start TTS Server with Marvis TTS
#
# Uses Marvis-AI/marvis-tts-250m-v0.2-transformers for fast inference.
# Supports sentence-level streaming and voice cloning.
#
# Environment variables:
#   TTS_PORT - Server port (default: 8002)
#   TTS_MAX_CONNECTIONS - Max WebSocket connections (default: 50)
#   CUDA_VISIBLE_DEVICES - GPU index (default: 0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default to GPU 0 if not set
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Create required directories
mkdir -p voices logs

echo "Starting Marvis TTS Server..."
echo "  Model: Marvis-AI/marvis-tts-250m-v0.2-transformers"
echo "  Port: ${TTS_PORT:-8002}"
echo "  Max connections: ${TTS_MAX_CONNECTIONS:-50}"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run the server
exec python services/tts/server.py 2>&1 | tee -a logs/tts.log
