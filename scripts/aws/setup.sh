#!/usr/bin/env bash
# Rebuild the training box environment. Copy to the box and run it there.
# The venv is isolated (no system site packages): Ubuntu's cryptography and
# pyOpenSSL break transformers' imports.
#
# Both fast kernels are required. Without them Qwen3.5's Gated DeltaNet layers fall back to PyTorch
# reference code: 13.3 s/step instead of ~2,140 tokens/s on an L40S in Stage 1.
#   - flash-linear-attention installs from pip.
#   - causal-conv1d has no wheel for this torch, so it is built against NVIDIA's apt CUDA toolkit whose
#     version matches torch's CUDA (13.0). The pip nvcc/cccl wheels are newer than 13.0 and fail with
#     "CUDA compiler and CUDA toolkit headers are incompatible".
set -euo pipefail

CUDA_VERSION=13.0                    # must match torch's +cuXYZ build below
TORCH_INDEX=https://download.pytorch.org/whl/cu130
CUDA_ARCH="${CUDA_ARCH:-8.9}"        # L40S; set e.g. 9.0 for an H100

/opt/pytorch/bin/python -m venv ~/venv
PIP=~/venv/bin/pip
PY=~/venv/bin/python

$PIP install -U pip wheel setuptools ninja
$PIP install torch --index-url "$TORCH_INDEX"
$PIP install 'transformers>=5.0' 'peft>=0.17' 'accelerate>=1.0' 'tensorboard>=2.17' \
  datasets huggingface_hub safetensors nvidia-ml-py
$PIP install flash-linear-attention

# CUDA toolkit for building causal-conv1d.
if [ ! -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-${CUDA_VERSION/./-}"
  cd -
fi
export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION" PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4
$PIP install --no-build-isolation --no-cache-dir causal-conv1d

$PY - <<'EOF'
import torch
from causal_conv1d import causal_conv1d_fn
import fla  # noqa: F401
x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
assert causal_conv1d_fn(x, w).shape == x.shape
print("fast kernels ok")
EOF

mkdir -p ~/models ~/runs
~/venv/bin/hf download Qwen/Qwen3.5-4B-Base --local-dir ~/models/Qwen3.5-4B-Base
