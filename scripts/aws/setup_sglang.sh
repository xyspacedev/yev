#!/usr/bin/env bash
# Build ~/venv-sglang with the newest stable sglang, separate from the training venv; record what was installed.
set -euo pipefail
NAME=sglang
PKG="${PKG:-sglang[all]}"
VENV="$HOME/venv-$NAME"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -U "$PKG"
uv pip freeze --python "$VENV/bin/python" > "$HOME/venv-$NAME.freeze"
grep -iE "^($NAME|torch|transformers|flashinfer|triton)[=<>]" "$HOME/venv-$NAME.freeze" || true
