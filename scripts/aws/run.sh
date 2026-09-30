#!/usr/bin/env bash
# Run a command on the training box: run.sh [--bg [--log NAME]] <cmd...>
# With --bg the command runs detached (nohup setsid) and ssh returns at once; it logs to ~/runs/<NAME>.log, where NAME is
# --log, else the value after --config (e.g. lc25), else basename of the command plus a UTC timestamp.
# Requires JEFF_TRAIN_HOST (user@host) and JEFF_TRAIN_KEY (path to ssh key).
set -euo pipefail
[ -n "${JEFF_TRAIN_HOST:-}" ] || { echo "JEFF_TRAIN_HOST is not set" >&2; exit 2; }
[ -n "${JEFF_TRAIN_KEY:-}" ] || { echo "JEFF_TRAIN_KEY is not set" >&2; exit 2; }

BG=0
LOG=""
if [ "${1:-}" = "--bg" ]; then
  BG=1; shift
  if [ "${1:-}" = "--log" ]; then
    [ "$#" -ge 2 ] || { echo "--log needs a name" >&2; exit 64; }
    LOG="$2"; shift 2
  fi
fi
[ "$#" -gt 0 ] || { echo "usage: run.sh [--bg [--log NAME]] <cmd...>" >&2; exit 64; }

if [ -z "$LOG" ]; then
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--config" ]; then LOG="$(basename "$a" .json)"; break; fi
    prev="$a"
  done
fi
[ -n "$LOG" ] || LOG="$(basename "$1")-$(date -u +%Y%m%dT%H%M%SZ)"

SSH=(ssh -i "$JEFF_TRAIN_KEY")
CMD="$(printf '%q ' "$@")"
# \$HOME and \$PATH expand on the remote side.
INNER="export PATH=\"\$HOME/venv/bin:\$PATH\" && cd ~/jeff && pip install -q -e '.[train]' --no-deps && $CMD"

if [ "$BG" = 1 ]; then
  # Only the job itself is backgrounded (not an "a && b &" list, whose subshell would keep ssh's
  # stdout open), and setsid puts it in its own session so it survives the ssh session closing.
  REMOTE="mkdir -p ~/runs || exit 1; nohup setsid bash -c $(printf %q "$INNER") > ~/runs/$(printf %q "$LOG").log 2>&1 < /dev/null &"
else
  REMOTE="bash -c $(printf %q "$INNER")"
fi
"${SSH[@]}" "$JEFF_TRAIN_HOST" "$REMOTE"
