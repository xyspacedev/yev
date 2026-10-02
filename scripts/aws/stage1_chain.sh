#!/usr/bin/env bash
# Stage 1 chain, run ON the GPU box from ~/yev (L40S or H100):
#   scripts/aws/run.sh --bg --log stage1 bash scripts/aws/stage1_chain.sh
# smoke (200 steps, abort under 500 tokens/s) -> full training -> calibrate -> Stage 0 dev eval -> Stage 1 dev eval
# -> position-bias check -> STAGE1_DONE. Log: ~/runs/stage1/chain.log (=== <step> start|done HH:MM:SS).
# Resumable: a step whose output already exists is skipped; a failure prints a `!!!` line and stops the chain.
# --force and --skip-overlong are never passed: an over-long row makes eval/calibrate refuse, and that stops the chain.
# It never runs DecideBench test, WorkflowEvals or the Decision Index (separate, explicit steps).
# Overrides (tests and non-default setups): STAGE1_RUNS (default ~/runs), STAGE1_BASE, STAGE1_MAX_LEN, STAGE1_MIN_TPS.
set -uo pipefail

export PATH="$HOME/venv/bin:$PATH"
RUNS="${STAGE1_RUNS:-$HOME/runs}"
BASE="${STAGE1_BASE:-$HOME/models/Qwen3.5-4B-Base}"
MAX_LEN="${STAGE1_MAX_LEN:-12288}"   # = max_len_plain/token_budget of configs/stage1/main.json; the eval default (4096) would refuse rows
MIN_TPS="${STAGE1_MIN_TPS:-500}"
OUT="$RUNS/stage1"
SMOKE="$RUNS/stage1-smoke"
MIX=data/mix
mkdir -p "$OUT" "$SMOKE"
exec >> "$OUT/chain.log" 2>&1

ts() { date +%H:%M:%S; }
fail() { echo "!!! $1 failed (rc=$2); chain stopped"; exit 1; }

# step <name> <output-file> <cmd...>: skipped when the output exists.
step() {
  local name="$1" outfile="$2"; shift 2
  if [ -e "$outfile" ]; then echo "=== $name skipped ($outfile exists)"; return 0; fi
  echo "=== $name start $(ts)"
  "$@"; local rc=$?
  [ $rc -eq 0 ] || fail "$name" $rc
  echo "=== $name done $(ts)"
}

# --- smoke ---
step smoke "$SMOKE/train_summary.json" yev train --config configs/stage1/smoke.json
tps=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens_per_s"])' "$SMOKE/train_summary.json" 2>/dev/null) \
  || { echo "!!! smoke: no tokens_per_s in $SMOKE/train_summary.json; chain stopped"; exit 1; }
if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$tps" "$MIN_TPS"; then
  echo "!!! smoke: $tps tokens/s is under $MIN_TPS; chain stopped"; exit 1
fi
echo "=== smoke throughput $tps tokens/s"

# --- full training (summary is written only when a run ends; the trainer's finished-run guard also protects it) ---
step train "$OUT/train_summary.json" yev train --config configs/stage1/main.json

COMMON=(--model "$OUT/final" --base "$BASE" --max-len "$MAX_LEN")
step calibrate "$OUT/calibration.json" yev calibrate "${COMMON[@]}" \
  --data "$MIX/stage1/calibration.chat.jsonl" --out "$OUT/calibration.json"
step eval_stage0_dev "$OUT/dev_report_stage0.json" yev eval "${COMMON[@]}" --calibration "$OUT/calibration.json" \
  --data "$MIX/stage0/dev.chat.jsonl" --out "$OUT/dev_report_stage0.json"
step eval_stage1_dev "$OUT/dev_report.json" yev eval "${COMMON[@]}" --calibration "$OUT/calibration.json" \
  --data "$MIX/stage1/dev.chat.jsonl" --out "$OUT/dev_report.json"
step position_bias "$OUT/position_bias.json" yev eval "${COMMON[@]}" --calibration "$OUT/calibration.json" \
  --data "$MIX/stage1/dev.chat.jsonl" --shuffle-options 3 --min-options 7 --out "$OUT/position_bias.json"

echo STAGE1_DONE
