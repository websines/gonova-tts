#!/bin/bash
# TTS Service Benchmark Script
# Measures latency, throughput, and quality of TTS service

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TTS_URL="${TTS_URL:-http://localhost}"
TTS_HOST="${TTS_HOST:-localhost}"
ITERATIONS="${ITERATIONS:-5}"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}TTS Service Benchmark${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Target: $TTS_URL"
echo "Host header: $TTS_HOST"
echo "Iterations: $ITERATIONS"
echo ""

# Check if service is healthy
echo -e "${YELLOW}1. Health Check...${NC}"
HEALTH_START=$(date +%s%3N)
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" -H "Host: $TTS_HOST" "$TTS_URL/health" 2>&1)
HEALTH_END=$(date +%s%3N)

HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -2 | head -1)
TIME_TOTAL=$(echo "$HEALTH_RESPONSE" | tail -1)
HEALTH_LATENCY=$((HEALTH_END - HEALTH_START))

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "   ${GREEN}✓ Service is healthy${NC}"
    echo "   Response time: ${HEALTH_LATENCY}ms"

    # Parse GPU memory info if available
    BODY=$(echo "$HEALTH_RESPONSE" | head -n -2)
    if echo "$BODY" | grep -q "memory_allocated_gb"; then
        GPU_MEM=$(echo "$BODY" | grep -o '"memory_allocated_gb":[0-9.]*' | cut -d: -f2)
        echo "   GPU memory: ${GPU_MEM}GB"
    fi
else
    echo -e "   ${RED}✗ Service unhealthy (HTTP $HTTP_CODE)${NC}"
    exit 1
fi
echo ""

# Test different text lengths
echo -e "${YELLOW}2. Testing synthesis latency...${NC}"
echo ""

# Test cases: short, medium, long text
TEST_TEXTS=(
    "Hello world"
    "The quick brown fox jumps over the lazy dog"
    "This is a longer test sentence to measure the text-to-speech synthesis performance with more realistic input length"
)

TEST_NAMES=("Short (2 words)" "Medium (9 words)" "Long (19 words)")

TOTAL_TIME=0
TOTAL_TESTS=0
SUCCESS_COUNT=0

for idx in "${!TEST_TEXTS[@]}"; do
    TEXT="${TEST_TEXTS[$idx]}"
    NAME="${TEST_NAMES[$idx]}"

    echo "   Testing: $NAME"

    SUB_TOTAL=0
    SUB_SUCCESS=0

    for i in $(seq 1 $ITERATIONS); do
        echo -n "     Iteration $i/$ITERATIONS... "

        START=$(date +%s%3N)

        RESPONSE=$(curl -s -w "\n%{http_code}" \
            -H "Host: $TTS_HOST" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$TEXT\", \"voice_id\": \"default\"}" \
            "$TTS_URL/v1/synthesize" 2>&1 || echo "error\n000")

        END=$(date +%s%3N)
        LATENCY=$((END - START))

        HTTP_CODE=$(echo "$RESPONSE" | tail -1)

        if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "404" ]; then
            if [ "$HTTP_CODE" = "404" ]; then
                echo -e "${YELLOW}404 (endpoint may not exist)${NC}"
                break 2
            else
                echo -e "${GREEN}${LATENCY}ms${NC}"
                SUB_TOTAL=$((SUB_TOTAL + LATENCY))
                SUB_SUCCESS=$((SUB_SUCCESS + 1))
                TOTAL_TIME=$((TOTAL_TIME + LATENCY))
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
        else
            echo -e "${RED}Failed (HTTP $HTTP_CODE)${NC}"
        fi

        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        sleep 0.3
    done

    if [ $SUB_SUCCESS -gt 0 ]; then
        AVG=$((SUB_TOTAL / SUB_SUCCESS))
        echo "     Average: ${AVG}ms"
    fi

    echo ""
done

if [ $SUCCESS_COUNT -gt 0 ]; then
    OVERALL_AVG=$((TOTAL_TIME / SUCCESS_COUNT))
    echo -e "   Overall average: ${GREEN}${OVERALL_AVG}ms${NC}"
    echo -e "   Success rate: ${SUCCESS_COUNT}/${TOTAL_TESTS}"
else
    echo -e "   ${YELLOW}⚠ HTTP endpoint not available or not tested${NC}"
fi
echo ""

# Test WebSocket streaming
echo -e "${YELLOW}3. Testing WebSocket streaming...${NC}"

if command -v websocat &> /dev/null && command -v python3 &> /dev/null; then
    WS_URL="ws://$(echo $TTS_URL | sed 's|http://||')/v1/stream/tts"

    echo "   Testing time to first audio chunk..."

    # Create a Python script to test WebSocket and measure time to first chunk
    cat > /tmp/test_tts_ws.py << 'EOF'
import asyncio
import websockets
import json
import time
import sys

async def test_tts():
    uri = sys.argv[1]
    text = "Hello, this is a test of streaming text to speech synthesis."

    try:
        start = time.time()
        async with websockets.connect(uri, ping_interval=None) as ws:
            connect_time = (time.time() - start) * 1000

            # Send synthesis request
            await ws.send(json.dumps({
                "type": "synthesize",
                "text": text,
                "voice_id": "default",
                "streaming": True
            }))

            send_time = (time.time() - start) * 1000

            # Wait for first audio chunk
            first_chunk = True
            chunk_count = 0

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)

                    if isinstance(msg, bytes):
                        chunk_count += 1
                        if first_chunk:
                            first_audio_time = (time.time() - start) * 1000
                            print(f"FIRST_CHUNK:{first_audio_time:.0f}")
                            first_chunk = False
                    elif isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get('type') == 'synthesis_complete':
                            total_time = (time.time() - start) * 1000
                            print(f"COMPLETE:{total_time:.0f}")
                            print(f"CHUNKS:{chunk_count}")
                            break
                except asyncio.TimeoutError:
                    print("TIMEOUT")
                    break

            print(f"CONNECT:{connect_time:.0f}")
            print(f"SEND:{send_time:.0f}")

    except Exception as e:
        print(f"ERROR:{str(e)}")

asyncio.run(test_tts())
EOF

    RESULT=$(python3 /tmp/test_tts_ws.py "$WS_URL" 2>&1)

    if echo "$RESULT" | grep -q "FIRST_CHUNK"; then
        FIRST_CHUNK=$(echo "$RESULT" | grep "FIRST_CHUNK" | cut -d: -f2)
        COMPLETE=$(echo "$RESULT" | grep "COMPLETE" | cut -d: -f2)
        CHUNKS=$(echo "$RESULT" | grep "CHUNKS" | cut -d: -f2)

        echo -e "   ${GREEN}✓ WebSocket streaming working${NC}"
        echo "   Time to first chunk: ${FIRST_CHUNK}ms"
        echo "   Total synthesis time: ${COMPLETE}ms"
        echo "   Audio chunks received: $CHUNKS"

        # Calculate streaming efficiency
        if [ -n "$FIRST_CHUNK" ] && [ -n "$COMPLETE" ]; then
            EFFICIENCY=$((FIRST_CHUNK * 100 / COMPLETE))
            echo "   Streaming efficiency: ${EFFICIENCY}% of time before first chunk"
        fi
    else
        echo -e "   ${YELLOW}⚠ WebSocket test failed${NC}"
        echo "   Result: $RESULT"
    fi

    rm -f /tmp/test_tts_ws.py
else
    echo -e "   ${YELLOW}⚠ websocat or python3 not installed${NC}"
    echo "   Install: sudo apt install python3-websockets"
fi
echo ""

# Load test
echo -e "${YELLOW}4. Load Test (concurrent requests)...${NC}"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "   Sending 5 concurrent synthesis requests..."

    START=$(date +%s%3N)

    for i in {1..5}; do
        curl -s -H "Host: $TTS_HOST" \
            -H "Content-Type: application/json" \
            -d '{"text": "This is a concurrent load test", "voice_id": "default"}' \
            "$TTS_URL/v1/synthesize" \
            > /dev/null 2>&1 &
    done

    wait

    END=$(date +%s%3N)
    LOAD_TIME=$((END - START))

    echo -e "   ${GREEN}✓ Completed in ${LOAD_TIME}ms${NC}"
    echo "   Average per request: $((LOAD_TIME / 5))ms"

    # Check if processing is parallelized (should be faster than 5x serial)
    if [ -n "$OVERALL_AVG" ]; then
        EXPECTED_SERIAL=$((OVERALL_AVG * 5))
        if [ $LOAD_TIME -lt $EXPECTED_SERIAL ]; then
            SPEEDUP=$((EXPECTED_SERIAL * 100 / LOAD_TIME))
            echo -e "   ${GREEN}✓ Good parallelization${NC} (${SPEEDUP}% vs serial)"
        else
            echo -e "   ${YELLOW}⚠ Limited parallelization${NC}"
        fi
    fi
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
    echo "Average Synthesis:     ${OVERALL_AVG}ms"
    echo "Success Rate:          ${SUCCESS_COUNT}/${TOTAL_TESTS} ($(( SUCCESS_COUNT * 100 / TOTAL_TESTS ))%)"
fi

if [ -n "$FIRST_CHUNK" ]; then
    echo "Time to First Chunk:   ${FIRST_CHUNK}ms"
    echo "Streaming Total:       ${COMPLETE}ms"
fi

echo ""
echo -e "${YELLOW}Performance Analysis:${NC}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    if [ $OVERALL_AVG -lt 300 ]; then
        echo -e "  ${GREEN}✓ Excellent${NC} - Very fast synthesis"
    elif [ $OVERALL_AVG -lt 600 ]; then
        echo -e "  ${GREEN}✓ Good${NC} - Acceptable for real-time use"
    elif [ $OVERALL_AVG -lt 1500 ]; then
        echo -e "  ${YELLOW}⚠ Moderate${NC} - May feel slightly slow"
    else
        echo -e "  ${RED}✗ Slow${NC} - Check GPU/CPU usage and model"
    fi
fi

if [ -n "$FIRST_CHUNK" ]; then
    echo ""
    if [ $FIRST_CHUNK -lt 500 ]; then
        echo -e "  ${GREEN}✓ Excellent streaming${NC} - Quick first response"
    elif [ $FIRST_CHUNK -lt 1000 ]; then
        echo -e "  ${GREEN}✓ Good streaming${NC} - Acceptable latency"
    else
        echo -e "  ${YELLOW}⚠ Slow first chunk${NC} - Users may notice delay"
    fi
fi

echo ""
echo "Recommendations:"
echo "  - Optimal synthesis: < 600ms"
echo "  - Optimal first chunk: < 500ms"
echo "  - For voice agents: < 1500ms total acceptable"
echo "  - Check GPU usage: nvidia-smi"
echo "  - Check service logs: tail -f logs/tts/*.log"
echo ""

# Exit with success/warning based on performance
if [ $SUCCESS_COUNT -gt 0 ] && [ $OVERALL_AVG -lt 1500 ]; then
    exit 0
else
    exit 1
fi
