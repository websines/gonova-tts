#!/bin/bash
# Run all benchmarks and provide comprehensive report

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Voice Agent System Benchmark${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""
echo "Benchmarking both ASR and TTS services..."
echo ""

# Parse arguments for URL/host
ASR_URL="${ASR_URL:-http://localhost}"
TTS_URL="${TTS_URL:-http://localhost}"
ASR_HOST="${ASR_HOST:-localhost}"
TTS_HOST="${TTS_HOST:-localhost}"
ITERATIONS="${ITERATIONS:-5}"

# Check if remote URL provided
if [ "$1" ]; then
    ASR_URL="$1"
    TTS_URL="$1"
    ASR_HOST="${2:-asr.gonova.one}"
    TTS_HOST="${2:-tts.gonova.one}"
fi

echo "Configuration:"
echo "  ASR URL: $ASR_URL"
echo "  ASR Host: $ASR_HOST"
echo "  TTS URL: $TTS_URL"
echo "  TTS Host: $TTS_HOST"
echo "  Iterations: $ITERATIONS"
echo ""

# Run ASR benchmark
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}ASR Service Benchmark${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

ASR_URL="$ASR_URL" ASR_HOST="$ASR_HOST" ITERATIONS="$ITERATIONS" \
    "$SCRIPT_DIR/benchmark_asr.sh"
ASR_RESULT=$?

echo ""
echo ""

# Run TTS benchmark
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}TTS Service Benchmark${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

TTS_URL="$TTS_URL" TTS_HOST="$TTS_HOST" ITERATIONS="$ITERATIONS" \
    "$SCRIPT_DIR/benchmark_tts.sh"
TTS_RESULT=$?

echo ""
echo ""

# Final summary
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}Overall System Performance${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""

if [ $ASR_RESULT -eq 0 ]; then
    echo -e "ASR Service: ${GREEN}✓ PASSED${NC}"
else
    echo -e "ASR Service: ${YELLOW}⚠ NEEDS ATTENTION${NC}"
fi

if [ $TTS_RESULT -eq 0 ]; then
    echo -e "TTS Service: ${GREEN}✓ PASSED${NC}"
else
    echo -e "TTS Service: ${YELLOW}⚠ NEEDS ATTENTION${NC}"
fi

echo ""

if [ $ASR_RESULT -eq 0 ] && [ $TTS_RESULT -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}System is performing optimally! 🚀${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Some services need attention ⚠${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check GPU usage: nvidia-smi"
    echo "  - Check service logs: tail -f logs/{asr,tts}/*.log"
    echo "  - Restart services: ./scripts/stop_all.sh && ./scripts/start_with_logs.sh"
    echo "  - Check network: ping <your-server>"
    exit 1
fi
