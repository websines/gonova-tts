#!/bin/bash
# Start Streaming TTS Server with vLLM AsyncLLM
#
# This uses token-level streaming for low TTFB (~1-2s target)
# instead of batch processing.
#
# Environment variables:
#   TTS_PORT - Server port (default: 8002)
#   TTS_CHUNK_SIZE - Tokens per audio chunk (default: 25)
#   TTS_CONTEXT_WINDOW - Context tokens for S3Gen (default: 50)
#   TTS_MAX_CONNECTIONS - Max WebSocket connections (default: 50)
#   CUDA_VISIBLE_DEVICES - GPU index (default: 0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default to GPU 0 if not set
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Create required directories
mkdir -p t3-model voices logs

# Ensure t3-model has config.json
if [ ! -f "t3-model/config.json" ]; then
    echo "Creating t3-model/config.json..."
    cat > t3-model/config.json << 'EOF'
{
    "architectures": ["ChatterboxT3"],
    "model_type": "ChatterboxT3"
}
EOF
fi

echo "Starting Streaming TTS Server..."
echo "  Port: ${TTS_PORT:-8002}"
echo "  Chunk size: ${TTS_CHUNK_SIZE:-25} tokens"
echo "  Context window: ${TTS_CONTEXT_WINDOW:-50} tokens"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Expected TTFB: ~1-2 seconds (25 tokens @ 40 tok/s)"
echo ""

# Run the streaming server
exec python services/tts/run_streaming_server.py 2>&1 | tee -a logs/streaming_tts.log
