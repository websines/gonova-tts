#!/bin/bash
# Run TTS benchmark

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}TTS Service Benchmark${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

# Parse arguments
DIRECT_MODE=false
if [ "$1" = "--direct" ]; then
    DIRECT_MODE=true
    shift
fi

# Configure URLs
if [ "$DIRECT_MODE" = true ]; then
    # Test backend service directly (bypass nginx)
    TTS_URL="${TTS_URL:-http://localhost:8002}"
    TTS_HOST="${TTS_HOST:-localhost}"
    echo -e "${YELLOW}Direct mode: Testing backend service${NC}"
else
    # Test through nginx
    TTS_URL="${TTS_URL:-http://localhost}"
    TTS_HOST="${TTS_HOST:-localhost}"
fi

ITERATIONS="${ITERATIONS:-5}"

# Check if remote URL provided as argument
if [ "$1" ]; then
    TTS_URL="$1"
    TTS_HOST="${2:-tts.gonova.one}"
fi

echo "Configuration:"
echo "  TTS URL: $TTS_URL"
echo "  TTS Host: $TTS_HOST"
echo "  Iterations: $ITERATIONS"
echo ""

# Run TTS benchmark
TTS_URL="$TTS_URL" TTS_HOST="$TTS_HOST" ITERATIONS="$ITERATIONS" \
    "$SCRIPT_DIR/benchmark_tts.sh"
TTS_RESULT=$?

echo ""
echo ""

# Final summary
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}TTS Service Performance${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

if [ $TTS_RESULT -eq 0 ]; then
    echo -e "TTS Service: ${GREEN}✓ PASSED${NC}"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Service is performing optimally! 🚀${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "TTS Service: ${YELLOW}⚠ NEEDS ATTENTION${NC}"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Service needs attention ⚠${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check GPU usage: nvidia-smi"
    echo "  - Check service logs: tail -f logs/tts/*.log"
    echo "  - Restart service: ./scripts/stop_all.sh && ./scripts/start_all.sh"
    echo "  - Check network: ping <your-server>"
    exit 1
fi
