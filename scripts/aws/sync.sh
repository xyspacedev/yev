#!/usr/bin/env bash
# Push the repo (git archive HEAD), the Stage 0 mix and the benchmark rows to the training box.
# Requires JEFF_TRAIN_HOST (user@host) and JEFF_TRAIN_KEY (path to ssh key).
set -euo pipefail
[ -n "${JEFF_TRAIN_HOST:-}" ] || { echo "JEFF_TRAIN_HOST is not set" >&2; exit 2; }
[ -n "${JEFF_TRAIN_KEY:-}" ] || { echo "JEFF_TRAIN_KEY is not set" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"
SSH=(ssh -i "$JEFF_TRAIN_KEY")

git archive HEAD | "${SSH[@]}" "$JEFF_TRAIN_HOST" 'mkdir -p ~/jeff && tar -x -C ~/jeff'

# Only the Stage 0 mix, the dev examples and data/bench go over; nothing else under data/.
"${SSH[@]}" "$JEFF_TRAIN_HOST" 'mkdir -p ~/jeff/data/mix/stage0 ~/jeff/data/dev'
rsync -az -e "ssh -i \"$JEFF_TRAIN_KEY\"" data/mix/stage0/ "$JEFF_TRAIN_HOST:jeff/data/mix/stage0/"
rsync -az -e "ssh -i \"$JEFF_TRAIN_KEY\"" data/dev/decidebench_examples.jsonl "$JEFF_TRAIN_HOST:jeff/data/dev/decidebench_examples.jsonl"
if [ -d data/bench ]; then
  "${SSH[@]}" "$JEFF_TRAIN_HOST" 'mkdir -p ~/jeff/data/bench'
  rsync -az -e "ssh -i \"$JEFF_TRAIN_KEY\"" data/bench/ "$JEFF_TRAIN_HOST:jeff/data/bench/"
fi
