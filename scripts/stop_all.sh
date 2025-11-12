#!/bin/bash
# Stop entire voice agent cluster

set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "Stopping Voice Agent Cluster"
echo "========================================="
echo ""

# Stop nginx if running
if command -v nginx &> /dev/null; then
    echo "1. Stopping nginx..."
    sudo nginx -s stop 2>/dev/null || echo "  (nginx not running)"
    echo ""
fi

# Stop TTS cluster
echo "2. Stopping TTS cluster..."
bash scripts/stop_tts_cluster.sh
echo ""

# Stop ASR cluster
echo "3. Stopping ASR cluster..."
bash scripts/stop_asr_cluster.sh
echo ""

echo "========================================="
echo "Cluster Stopped"
echo "========================================="
