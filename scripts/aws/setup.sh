#!/usr/bin/env bash
# Rebuild the training box environment. Copy to the box and run it there.
# The venv is isolated (no system site packages): Ubuntu's cryptography and
# pyOpenSSL break transformers' imports.
set -euo pipefail

/opt/pytorch/bin/python -m venv ~/venv
PIP=~/venv/bin/pip
PY=~/venv/bin/python

$PIP install -U pip wheel setuptools
$PIP install torch --index-url https://download.pytorch.org/whl/cu130
$PIP install 'transformers>=5.0' 'peft>=0.17' 'accelerate>=1.0' 'tensorboard>=2.17' \
  datasets huggingface_hub safetensors nvidia-ml-py
$PIP install flash-linear-attention

CUDA_HOME="$($PY -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
export CUDA_HOME
$PIP install causal-conv1d --no-build-isolation

mkdir -p ~/models ~/runs
~/venv/bin/hf download Qwen/Qwen3.5-4B-Base --local-dir ~/models/Qwen3.5-4B-Base
