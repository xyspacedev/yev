#!/usr/bin/env bash
# Training box for yev2 (different bases on the Stage 1c recipe): Ubuntu 22.04 Deep Learning Base AMI, NVIDIA RTX PRO
# 6000 Blackwell (sm_120). Same packages as setup.sh; Python 3.12 from uv (the AMI ships 3.10) and the AMI's CUDA 13.0.
# gemma-4-E4B ships without a chat template: it gets gemma-4-E4B-it's, which uses the same vocabulary.
set -euo pipefail

CUDA_VERSION=13.0
TORCH_INDEX=https://download.pytorch.org/whl/cu130
CUDA_ARCH="${CUDA_ARCH:-12.0}"
GEMMA_IT_REV=4ecc2ba63bcac23ba4a767fc87025b1450f84c9c

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/venv
export VIRTUAL_ENV=~/venv PATH="$HOME/venv/bin:$PATH"
uv pip install -U pip wheel setuptools ninja
uv pip install torch --index-url "$TORCH_INDEX"
uv pip install 'transformers>=5.0' 'peft>=0.17' 'accelerate>=1.0' 'tensorboard>=2.17' \
  datasets huggingface_hub safetensors nvidia-ml-py flash-linear-attention

export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION" PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=6
uv pip install --no-build-isolation --no-cache causal-conv1d

python - <<'EOF'
import torch
from causal_conv1d import causal_conv1d_fn
import fla  # noqa: F401
x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
assert causal_conv1d_fn(x, w).shape == x.shape
print("fast kernels ok on", torch.cuda.get_device_name())
EOF

uv pip install -e ~/yev
sudo chown -R "$USER" ~/models  # bases downloaded through Docker are owned by root
hf download google/gemma-4-E4B-it chat_template.jinja --revision "$GEMMA_IT_REV" --local-dir ~/models/gemma-4-E4B
python - <<'EOF'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("/home/ubuntu/models/gemma-4-E4B")
assert t.chat_template, "gemma-4-E4B still has no chat template"
print("gemma chat template ok")
EOF
echo SETUP_OK
