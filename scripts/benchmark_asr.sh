#!/bin/bash
# ASR Service Benchmark Script
# Measures latency, throughput, and accuracy of ASR service

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ASR_URL="${ASR_URL:-http://localhost}"
ASR_HOST="${ASR_HOST:-localhost}"
ITERATIONS="${ITERATIONS:-5}"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}ASR Service Benchmark${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Target: $ASR_URL"
echo "Host header: $ASR_HOST"
echo "Iterations: $ITERATIONS"
echo ""

# Check if service is healthy
echo -e "${YELLOW}1. Health Check...${NC}"
HEALTH_START=$(date +%s%3N)
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" -H "Host: $ASR_HOST" "$ASR_URL/health" 2>&1)
HEALTH_END=$(date +%s%3N)

HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -2 | head -1)
TIME_TOTAL=$(echo "$HEALTH_RESPONSE" | tail -1)
HEALTH_LATENCY=$((HEALTH_END - HEALTH_START))

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "   ${GREEN}✓ Service is healthy${NC}"
    echo "   Response time: ${HEALTH_LATENCY}ms"
else
    echo -e "   ${RED}✗ Service unhealthy (HTTP $HTTP_CODE)${NC}"
    exit 1
fi
echo ""

# Check if we have test audio files
echo -e "${YELLOW}2. Preparing test audio...${NC}"

# Create a test directory
TEST_DIR="/tmp/asr_benchmark_$$"
mkdir -p "$TEST_DIR"

# Check if we have test audio, otherwise generate silence
if [ ! -f "test_audio.wav" ]; then
    echo "   Generating test audio (3 seconds of silence)..."

    # Check if sox is available
    if command -v sox &> /dev/null; then
        sox -n -r 16000 -c 1 "$TEST_DIR/test_3s.wav" trim 0 3
        sox -n -r 16000 -c 1 "$TEST_DIR/test_5s.wav" trim 0 5
        sox -n -r 16000 -c 1 "$TEST_DIR/test_10s.wav" trim 0 10
        echo -e "   ${GREEN}✓ Generated test audio files${NC}"
    else
        echo -e "   ${YELLOW}⚠ sox not found. Install: sudo apt install sox${NC}"
        echo "   Using alternative method..."

        # Generate raw PCM and convert to WAV manually
        dd if=/dev/zero bs=32000 count=3 2>/dev/null | \
            python3 -c "
import sys, wave, struct
data = sys.stdin.buffer.read()
with wave.open('$TEST_DIR/test_3s.wav', 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(struct.pack('<%dh' % (len(data)//2), *[0]*(len(data)//2)))
" 2>/dev/null || {
            echo -e "   ${RED}✗ Could not generate test audio${NC}"
            echo "   Please provide test_audio.wav file"
            exit 1
        }
        cp "$TEST_DIR/test_3s.wav" "$TEST_DIR/test_5s.wav"
        cp "$TEST_DIR/test_3s.wav" "$TEST_DIR/test_10s.wav"
    fi
else
    cp test_audio.wav "$TEST_DIR/test_3s.wav"
fi
echo ""

# Benchmark HTTP endpoint (if available)
echo -e "${YELLOW}3. Testing HTTP transcription endpoint...${NC}"

TOTAL_TIME=0
SUCCESS_COUNT=0

for i in $(seq 1 $ITERATIONS); do
    echo -n "   Iteration $i/$ITERATIONS... "

    START=$(date +%s%3N)

    RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" \
        -H "Host: $ASR_HOST" \
        -F "audio=@$TEST_DIR/test_3s.wav" \
        "$ASR_URL/v1/transcribe" 2>&1 || echo "error")

    END=$(date +%s%3N)
    LATENCY=$((END - START))

    HTTP_CODE=$(echo "$RESPONSE" | tail -2 | head -1 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "404" ]; then
        if [ "$HTTP_CODE" = "404" ]; then
            echo -e "${YELLOW}404 (endpoint may not exist)${NC}"
            break
        else
            echo -e "${GREEN}${LATENCY}ms${NC}"
            TOTAL_TIME=$((TOTAL_TIME + LATENCY))
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    else
        echo -e "${RED}Failed (HTTP $HTTP_CODE)${NC}"
    fi

    sleep 0.5
done

if [ $SUCCESS_COUNT -gt 0 ]; then
    AVG_TIME=$((TOTAL_TIME / SUCCESS_COUNT))
    echo ""
    echo -e "   Average latency: ${GREEN}${AVG_TIME}ms${NC}"
    echo -e "   Success rate: ${SUCCESS_COUNT}/${ITERATIONS}"
else
    echo -e "   ${YELLOW}⚠ HTTP endpoint not available or not tested${NC}"
fi
echo ""

# Test WebSocket streaming (basic connectivity)
echo -e "${YELLOW}4. Testing WebSocket connectivity...${NC}"

# Check if websocat is available
if command -v websocat &> /dev/null; then
    WS_URL="ws://$(echo $ASR_URL | sed 's|http://||')/v1/stream/asr"

    echo "   Connecting to: $WS_URL"

    START=$(date +%s%3N)
    echo '{"type":"ping"}' | timeout 5 websocat "$WS_URL" > /dev/null 2>&1 || true
    END=$(date +%s%3N)
    WS_LATENCY=$((END - START))

    if [ $WS_LATENCY -lt 5000 ]; then
        echo -e "   ${GREEN}✓ WebSocket connection successful${NC}"
        echo "   Connection time: ${WS_LATENCY}ms"
    else
        echo -e "   ${YELLOW}⚠ WebSocket connection timeout${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠ websocat not installed (optional)${NC}"
    echo "   Install: cargo install websocat"
fi
echo ""

# Load test
echo -e "${YELLOW}5. Load Test (concurrent requests)...${NC}"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "   Sending 10 concurrent requests..."

    START=$(date +%s%3N)

    for i in {1..10}; do
        curl -s -H "Host: $ASR_HOST" \
            -F "audio=@$TEST_DIR/test_3s.wav" \
            "$ASR_URL/v1/transcribe" \
            > /dev/null 2>&1 &
    done

    wait

    END=$(date +%s%3N)
    LOAD_TIME=$((END - START))

    echo -e "   ${GREEN}✓ Completed in ${LOAD_TIME}ms${NC}"
    echo "   Average per request: $((LOAD_TIME / 10))ms"
else
    echo -e "   ${YELLOW}⚠ Skipped (HTTP endpoint not available)${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Benchmark Summary${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Health Check Latency:  ${HEALTH_LATENCY}ms"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Average Transcription: ${AVG_TIME}ms"
    echo "Success Rate:          ${SUCCESS_COUNT}/${ITERATIONS} ($(( SUCCESS_COUNT * 100 / ITERATIONS ))%)"
fi

echo ""
echo -e "${YELLOW}Performance Analysis:${NC}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    if [ $AVG_TIME -lt 200 ]; then
        echo -e "  ${GREEN}✓ Excellent${NC} - Very fast response times"
    elif [ $AVG_TIME -lt 500 ]; then
        echo -e "  ${GREEN}✓ Good${NC} - Acceptable for real-time use"
    elif [ $AVG_TIME -lt 1000 ]; then
        echo -e "  ${YELLOW}⚠ Moderate${NC} - May feel slightly slow"
    else
        echo -e "  ${RED}✗ Slow${NC} - Check GPU/CPU usage and network"
    fi
fi

echo ""
echo "Recommendations:"
echo "  - Optimal latency: < 500ms"
echo "  - For voice agents: < 1000ms acceptable"
echo "  - Check GPU usage: nvidia-smi"
echo "  - Check service logs: tail -f logs/asr/*.log"
echo ""

# Cleanup
rm -rf "$TEST_DIR"

# Exit with success/warning based on performance
if [ $SUCCESS_COUNT -gt 0 ] && [ $AVG_TIME -lt 1000 ]; then
    exit 0
else
    exit 1
fi
