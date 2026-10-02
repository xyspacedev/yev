#!/usr/bin/env bash
# Run the four TypeSafe WorkflowEvals workflows against `jeff serve` on THIS box (the GPU box, not the laptop).
# Start the server first (e.g. scripts/aws/run.sh --bg --log serve jeff serve --model ... --port 8000), then:
#   scripts/aws/run.sh --bg --log workflowevals bash scripts/aws/workflowevals.sh
# Everything lands in ~/workflowevals/yev-4b/: <workflow>.log, <workflow>/{scores,results}.json, run_meta.json.
# WorkflowEvals rejects every TypeSafe model name except jev-1.13.0, so that name is passed; our server ignores it.
# Overrides (tests and non-default setups): WE_DIR (existing checkout, skips clone/sync), WE_RUNNER (command run in
# WE_DIR as `$WE_RUNNER <workflow> ...`), WE_SERVER, WE_OUT, WE_COMMIT, WE_REPO.
set -uo pipefail

WE_REPO="${WE_REPO:-https://github.com/typesafe-ai/WorkflowEvals}"
WE_COMMIT="${WE_COMMIT:-0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb}"  # origin HEAD on 2026-10-01
WE_DIR="${WE_DIR:-$HOME/workflowevals/repo}"
WE_OUT="${WE_OUT:-$HOME/workflowevals/yev-4b}"
WE_SERVER="${WE_SERVER:-http://127.0.0.1:8000}"
WE_RUNNER="${WE_RUNNER:-uv run python run.py}"
RUN_NAME=yev-4b
MODEL=typesafe:jev-1.13.0
WORKFLOWS=(invoice_processing customer_service agent_trace_observability security_incidents)

curl -fsS --max-time 10 "$WE_SERVER/health" > /dev/null \
  || { echo "jeff serve is not answering at $WE_SERVER/health; start it first" >&2; exit 3; }

if [ ! -f "$WE_DIR/run.py" ] && [ "$WE_RUNNER" = "uv run python run.py" ]; then
  mkdir -p "$(dirname "$WE_DIR")"
  git clone -q "$WE_REPO" "$WE_DIR" || exit 4
  git -C "$WE_DIR" checkout -q "$WE_COMMIT" || exit 4
  (cd "$WE_DIR" && uv sync --locked) || exit 4
fi
[ -d "$WE_DIR" ] || { echo "WE_DIR $WE_DIR does not exist" >&2; exit 4; }

mkdir -p "$WE_OUT"
# The SDK refuses an unset key; our server ignores it. Printable ASCII, no spaces.
export TYPESAFE_API_KEY="${TYPESAFE_API_KEY:-local}"

WE_SHA="$(git -C "$WE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
JEFF_SHA="$(cat "$HOME/jeff/GIT_SHA" 2>/dev/null || echo unknown)"
TIMES="$WE_OUT/.times.tsv"; : > "$TIMES"
FAILED=0

for wf in "${WORKFLOWS[@]}"; do
  echo "== $wf" >&2
  start=$(date +%s)
  # $WE_RUNNER is deliberately word-split (it is a command line).
  (cd "$WE_DIR" && $WE_RUNNER "$wf" --model "$MODEL" --base-url "$WE_SERVER" --name "$RUN_NAME") \
    > "$WE_OUT/$wf.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))
  printf '%s\t%s\t%s\n' "$wf" "$secs" "$rc" >> "$TIMES"
  [ "$rc" -eq 0 ] || { echo "$wf exited $rc (see $WE_OUT/$wf.log)" >&2; FAILED=1; }
  mkdir -p "$WE_OUT/$wf"
  for f in scores.json results.json; do
    src="$WE_DIR/runs/$wf/$RUN_NAME/$f"
    [ -f "$src" ] && cp "$src" "$WE_OUT/$wf/$f"
  done
done

python3 - "$TIMES" "$WE_OUT/run_meta.json" "$WE_SHA" "$JEFF_SHA" "$WE_SERVER" "$MODEL" <<'PY'
import json, sys
times, out, we_sha, jeff_sha, server, model = sys.argv[1:]
wf = {}
for line in open(times):
    name, secs, rc = line.rstrip("\n").split("\t")
    wf[name] = {"wall_clock_s": int(secs), "exit_code": int(rc)}
json.dump({"workflowevals_commit": we_sha, "jeff_git_sha": jeff_sha, "server": server,
           "model_arg": model, "workflows": wf}, open(out, "w"), indent=2)
PY
rm -f "$TIMES"
exit "$FAILED"
