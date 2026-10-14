#!/usr/bin/env bash
# Start a sidecar engine on 127.0.0.1:8100 for yev serve --backend vllm|sglang. Usage: serve_engine.sh vllm|sglang MODEL_DIR
set -euo pipefail
NAME="$1"; MODEL="$2"; PORT="${PORT:-8100}"; MAXLEN="${MAXLEN:-16384}"
case "$NAME" in
  vllm)
    FLAGS=(serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name yev --dtype bfloat16
           --max-model-len "$MAXLEN" --enable-prefix-caching --logprobs-mode processed_logprobs
           --max-logprobs 26 --gpu-memory-utilization 0.85 --trust-remote-code)
    CMD=("$HOME/venv-vllm/bin/vllm" "${FLAGS[@]}") ;;
  sglang)
    FLAGS=(--model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16
           --context-length "$MAXLEN" --mem-fraction-static 0.85 --trust-remote-code)
    CMD=("$HOME/venv-sglang/bin/python" -m sglang.launch_server "${FLAGS[@]}") ;;
  *) echo "usage: $0 vllm|sglang MODEL_DIR" >&2; exit 2 ;;
esac
echo "${CMD[*]}" > "$HOME/engine-$NAME.flags"
setsid nohup "${CMD[@]}" > "$HOME/engine-$NAME.log" 2>&1 < /dev/null &
echo $! > "$HOME/engine-$NAME.pid"
for _ in $(seq 180); do
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "$NAME up on $PORT"; exit 0; }
  sleep 5
done
echo "!!! $NAME did not come up; tail of ~/engine-$NAME.log:" >&2; tail -40 "$HOME/engine-$NAME.log" >&2; exit 1
