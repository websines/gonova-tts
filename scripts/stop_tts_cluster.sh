#!/bin/bash
# Stop all TTS service instances

set -e

echo "Stopping TTS instances..."

# Kill by PID files
if [ -d "logs/tts" ]; then
    for pidfile in logs/tts/instance_*.pid; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile")
            INSTANCE=$(basename "$pidfile" .pid)

            if kill -0 "$PID" 2>/dev/null; then
                echo "Stopping $INSTANCE (PID: $PID)..."
                kill -TERM "$PID"

                # Wait for graceful shutdown (max 10 seconds)
                for i in {1..10}; do
                    if ! kill -0 "$PID" 2>/dev/null; then
                        echo "  → $INSTANCE stopped gracefully"
                        break
                    fi
                    sleep 1
                done

                # Force kill if still running
                if kill -0 "$PID" 2>/dev/null; then
                    echo "  → Force killing $INSTANCE..."
                    kill -9 "$PID"
                fi
            else
                echo "  → $INSTANCE already stopped"
            fi

            rm -f "$pidfile"
        fi
    done
fi

# Fallback: kill any remaining TTS processes
pkill -f "services/tts/server.py" 2>/dev/null || true

echo ""
echo "All TTS instances stopped!"
