#!/bin/bash
# Start Parallel TTS Server with vLLM Batch Processing
#
# This uses batch processing for T3 + concurrent S3Gen for maximum throughput.
# Sentences are processed in parallel, not sequentially.
#
# Environment variables:
#   TTS_PORT - Server port (default: 8002)
#   TTS_S3GEN_WORKERS - Parallel S3Gen workers (default: 2)
#   TTS_DIFFUSION_STEPS - S3Gen diffusion steps (default: 3)
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

echo "Starting Parallel TTS Server..."
echo "  Port: ${TTS_PORT:-8002}"
echo "  S3Gen workers: ${TTS_S3GEN_WORKERS:-2}"
echo "  Diffusion steps: ${TTS_DIFFUSION_STEPS:-3}"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Strategy: Batch T3 + Parallel S3Gen"
echo ""

# Run the parallel server
exec python services/tts/run_parallel_server.py 2>&1 | tee -a logs/parallel_tts.log
