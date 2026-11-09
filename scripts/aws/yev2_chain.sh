#!/usr/bin/env bash
# yev2 on one box, one base after another: gemma-4-E4B, then Qwen3.6-35B-A3B. Each base runs the Stage 1 chain
# (train, calibrate, dev evals, position bias) with the Stage 1c recipe and data; Qwen first runs a 200-step smoke
# and stops there if it fails (out of memory, slow kernels), so a recipe change is decided before a long run.
# Run from ~/yev after setup_yev2.sh. Every step skips if its output exists, so the script can be re-run to resume.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/venv/bin:$PATH"
RUNS="$HOME/runs"
LOG="$RUNS/yev2_chain.log"
exec >> "$LOG" 2>&1
ts() { date +%H:%M:%S; }

chain() {  # $1 = run name, $2 = base model dir
  echo "=== $1 chain start $(ts)"
  STAGE1_SKIP_SMOKE=1 STAGE1_CONFIG="configs/yev2/$1.json" STAGE1_NAME="$1" STAGE1_BASE="$2" STAGE1_MIX=stage1c \
    bash scripts/aws/stage1_chain.sh
  grep -q STAGE1_DONE "$RUNS/$1/chain.log" || { echo "!!! $1 chain did not finish"; exit 1; }
  echo "=== $1 chain done $(ts)"
}

smoke() {  # $1 = run name
  [ -e "$RUNS/$1-smoke/train_summary.json" ] && { echo "=== $1 smoke skipped (done)"; return 0; }
  echo "=== $1 smoke start $(ts)"
  yev train --config "configs/yev2/$1-smoke.json" > "$RUNS/$1-smoke.log" 2>&1 \
    || { echo "!!! $1 smoke failed; tail:"; tail -20 "$RUNS/$1-smoke.log"; exit 1; }
  echo "=== $1 smoke done $(ts): $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in d if k in ("tokens_per_s", "steps", "peak_mem_gb")})' "$RUNS/$1-smoke/train_summary.json")"
}

until [ -e "$RUNS/yev2-gemma4-e4b-smoke/train_summary.json" ]; do sleep 30; done
chain yev2-gemma4-e4b "$HOME/models/gemma-4-E4B"
smoke yev2-qwen36-35b-a3b
chain yev2-qwen36-35b-a3b "$HOME/models/Qwen3.6-35B-A3B"
echo YEV2_DONE
