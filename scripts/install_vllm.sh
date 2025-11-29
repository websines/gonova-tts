#!/bin/bash
# Install chatterbox-vllm for high-performance TTS
# Requirements: Linux/WSL2 with NVIDIA GPU, CUDA 12.x

set -e

echo "=== Chatterbox-vLLM Installation ==="
echo ""

# Check if we're on Linux/WSL2
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: chatterbox-vllm requires Linux or WSL2"
    echo "Current OS: $OSTYPE"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' || echo "not found")
echo "CUDA Version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "not found" ]]; then
    echo "Warning: nvcc not found. Make sure CUDA toolkit is installed."
fi

echo ""
echo "=== Installing dependencies ==="

cd "$(dirname "$0")/.."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.10
fi

# Install PyTorch with CUDA support first
echo ""
echo "=== Installing PyTorch with CUDA support ==="
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (requires specific version)
echo ""
echo "=== Installing vLLM ==="
uv pip install vllm==0.10.0

# Install chatterbox-vllm
echo ""
echo "=== Installing chatterbox-vllm ==="
uv pip install chatterbox-vllm

# Install remaining project dependencies
echo ""
echo "=== Installing project dependencies ==="
uv pip install -e .

# Create t3-model directories for vLLM model symlinks
# These directories must exist - chatterbox-vllm will create symlinks inside them
echo ""
echo "=== Setting up model directories ==="
mkdir -p t3-model
mkdir -p t3-model-multilingual

# Copy the vLLM model config files
# These are required by vLLM to load the model correctly
# Config from: https://github.com/randombk/chatterbox-vllm/blob/main/t3-model/config.json
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

echo "Created t3-model/ and t3-model-multilingual/ directories with vLLM configs"

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except Exception as e:
    print(f'vLLM import error: {e}')

try:
    from chatterbox_vllm.tts import ChatterboxTTS
    print('chatterbox-vllm: OK')
except Exception as e:
    print(f'chatterbox-vllm import error: {e}')
"

echo ""
echo "=== Installation complete ==="
echo ""
echo "To start the TTS service:"
echo "  ./scripts/start_tts.sh"
echo ""
echo "Environment variables:"
echo "  TTS_MAX_BATCH_SIZE=8      # Max batch size for vLLM"
echo "  TTS_MAX_MODEL_LEN=1000    # Max tokens per generation"
echo "  TTS_MAX_CONNECTIONS=50    # Max WebSocket connections"
echo ""
