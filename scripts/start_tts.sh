#!/bin/bash
# Start TTS service with Chatterbox-vLLM (high-performance)

set -e

echo "Starting TTS service (Chatterbox-vLLM)..."

# Go to project root (chatterbox-vllm needs t3-model/ in cwd)
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Set GPU (use GPU 0 for single GPU, or set CUDA_VISIBLE_DEVICES externally)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# vLLM configuration
export TTS_MAX_BATCH_SIZE=${TTS_MAX_BATCH_SIZE:-8}
export TTS_MAX_MODEL_LEN=${TTS_MAX_MODEL_LEN:-1000}
export TTS_MAX_CONNECTIONS=${TTS_MAX_CONNECTIONS:-50}

echo "Configuration:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Max Batch Size: $TTS_MAX_BATCH_SIZE"
echo "  Max Model Len: $TTS_MAX_MODEL_LEN"
echo "  Max Connections: $TTS_MAX_CONNECTIONS"
echo "  Working Dir: $PROJECT_ROOT"

# Add CUDA libraries to LD_LIBRARY_PATH
CUDA_LIBS=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
if [ -n "$CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"
fi

# Create required directories
mkdir -p logs
mkdir -p t3-model
mkdir -p t3-model-multilingual
mkdir -p voices

# Ensure t3-model has config.json (required by vLLM)
if [ ! -f "t3-model/config.json" ]; then
    echo "Creating t3-model/config.json..."
    cat > t3-model/config.json << 'VLLM_CONFIG'
{
    "architectures": ["ChatterboxT3"],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_implementation": "sdpa",
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 30,
    "num_key_value_heads": 16,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "use_cache": true,
    "vocab_size": 8
}
VLLM_CONFIG
    cp t3-model/config.json t3-model-multilingual/config.json
fi

# Start server from project root (so t3-model/ is accessible)
echo ""
echo "Starting server..."
uv run python services/tts/server.py 2>&1 | tee logs/tts.log
