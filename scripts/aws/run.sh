#!/usr/bin/env bash
# Run a command on the training box: run.sh [--bg] <cmd...>
# With --bg the command runs under nohup and logs to ~/runs/<first arg basename>.log.
# Requires JEFF_TRAIN_HOST (user@host) and JEFF_TRAIN_KEY (path to ssh key).
set -euo pipefail
[ -n "${JEFF_TRAIN_HOST:-}" ] || { echo "JEFF_TRAIN_HOST is not set" >&2; exit 2; }
[ -n "${JEFF_TRAIN_KEY:-}" ] || { echo "JEFF_TRAIN_KEY is not set" >&2; exit 2; }

BG=0
if [ "${1:-}" = "--bg" ]; then BG=1; shift; fi
[ "$#" -gt 0 ] || { echo "usage: run.sh [--bg] <cmd...>" >&2; exit 64; }

LOG="$(basename "$1")"
CMD="$(printf '%q ' "$@")"
PRE="cd ~/jeff && ~/venv/bin/pip install -q -e '.[train]' --no-deps"

if [ "$BG" = 1 ]; then
  ssh -i "$JEFF_TRAIN_KEY" "$JEFF_TRAIN_HOST" "mkdir -p ~/runs && nohup bash -c \"$PRE && $CMD\" > ~/runs/$LOG.log 2>&1 &"
else
  ssh -i "$JEFF_TRAIN_KEY" "$JEFF_TRAIN_HOST" "$PRE && $CMD"
fi
