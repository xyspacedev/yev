#!/usr/bin/env bash
# Push the repo (git archive HEAD), the Stage 0 mix and the benchmark rows to the training box.
# Requires YEV_TRAIN_HOST (user@host) and YEV_TRAIN_KEY (path to ssh key).
set -euo pipefail
[ -n "${YEV_TRAIN_HOST:-}" ] || { echo "YEV_TRAIN_HOST is not set" >&2; exit 2; }
[ -n "${YEV_TRAIN_KEY:-}" ] || { echo "YEV_TRAIN_KEY is not set" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"
SSH=(ssh -i "$YEV_TRAIN_KEY")

git archive HEAD | "${SSH[@]}" "$YEV_TRAIN_HOST" 'mkdir -p ~/yev && tar -x -C ~/yev'

# The archive has no .git: record the revision so metrics.json / train_summary.json can carry it.
GIT_SHA="$(git rev-parse HEAD)"
[ -z "$(git status --porcelain)" ] || GIT_SHA="$GIT_SHA-dirty"
"${SSH[@]}" "$YEV_TRAIN_HOST" "printf '%s\\n' '$GIT_SHA' > ~/yev/GIT_SHA"

# Only the Stage 0 mix, the dev examples and data/bench go over; nothing else under data/.
"${SSH[@]}" "$YEV_TRAIN_HOST" 'mkdir -p ~/yev/data/mix/stage0 ~/yev/data/dev'
rsync -az -e "ssh -i \"$YEV_TRAIN_KEY\"" data/mix/stage0/ "$YEV_TRAIN_HOST:yev/data/mix/stage0/"
rsync -az -e "ssh -i \"$YEV_TRAIN_KEY\"" data/dev/decidebench_examples.jsonl "$YEV_TRAIN_HOST:yev/data/dev/decidebench_examples.jsonl"
if [ -d data/bench ]; then
  "${SSH[@]}" "$YEV_TRAIN_HOST" 'mkdir -p ~/yev/data/bench'
  rsync -az -e "ssh -i \"$YEV_TRAIN_KEY\"" data/bench/ "$YEV_TRAIN_HOST:yev/data/bench/"
fi
