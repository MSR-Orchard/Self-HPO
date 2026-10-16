#!/bin/bash
# serve_model.sh — serve a policy with SGLang, one replica per GPU (ports PORT_BASE..PORT_BASE+N-1).
#
# Usage:
#   bash scripts/eval/serve_model.sh --model <HF checkpoint dir> [--ngpu N] [--port-base 8000] [--context 131072]
#   --model      local HF checkpoint dir or Hugging Face model id (required),
#                e.g. self-hpo/Self-HPO-Qwen3.5-35B-A3B
#   --ngpu       replicas = GPUs to use        (default: all visible GPUs)
#   --port-base  first port                    (default: 8000)
#   --context    --context-length              (default: 131072)
# The same settings can also be given as env vars MODEL / NGPU / PORT_BASE / CONTEXT.
# Other env vars:
#   ATTN_BACKEND  SGLang attention backend   (default: trtllm_mha, B200 only)
#   FREE       1 = first kill every GPU process on this machine (default 0)
#   CHECK      1 = send one test request after the server is ready (default 1)
#   HOLD       1 = keep the script alive at the end (default 1)
#   WAIT_MIN   minutes to wait for all replicas to come up (default 120)
#   SERVE_HOST bind address (default 127.0.0.1)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
SGL=$SELFHPO_SERVE_ENV/bin/python
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --ngpu) NGPU="$2"; shift 2;;
  --port-base) PORT_BASE="$2"; shift 2;;
  --context) CONTEXT="$2"; shift 2;;
  *) echo "ERR: unknown arg $1"; exit 1;;
esac; done
MODEL="${MODEL:?pass --model <HF checkpoint dir or model id>}"
LOGDIR=$SELFHPO_WORK/logs; mkdir -p "$LOGDIR"; H=$(hostname)
PORT_BASE=${PORT_BASE:-8000}; CONTEXT=${CONTEXT:-131072}
if [ -z "${NGPU:-}" ]; then NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); fi
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || { echo "ERR: no GPUs detected (NGPU=$NGPU)"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/../resolve_model.sh"
MODEL=$(PY="$SELFHPO_EVAL_ENV/bin/python" resolve_model "$MODEL") || exit 1

echo "SERVE model=$MODEL"
echo "      ngpu=$NGPU ports=${PORT_BASE}..$((PORT_BASE+NGPU-1)) host=$H context=$CONTEXT"

# ---- free GPUs ----
if [ "${FREE:-0}" = 1 ]; then
  echo "freeing GPUs ..."
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  sleep 4
fi

# ---- launch replicas ----
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"; else GPU_IDS=($(seq 0 $((NGPU-1)))); fi
PIDS=()
for i in $(seq 0 $((NGPU-1))); do
  PORT=$((PORT_BASE+i))
  CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} setsid nohup $SGL -m sglang.launch_server \
    --host ${SERVE_HOST:-127.0.0.1} --port $PORT \
    --trust-remote-code --allow-auto-truncate \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --mem-fraction-static 0.85 --max-prefill-tokens 8192 \
    --context-length $CONTEXT --tp 1 \
    --attention-backend ${ATTN_BACKEND:-trtllm_mha} --mamba-scheduler-strategy extra_buffer \
    --model-path "$MODEL" \
    < /dev/null > "$LOGDIR/sglang_${H}_${PORT}.log" 2>&1 &
  PIDS+=($!)
  echo "  launched GPU$i port=$PORT pid=$!  log=$LOGDIR/sglang_${H}_${PORT}.log"
done

stop_replicas(){
  trap - INT TERM EXIT
  echo "stopping replicas ..."
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
  for t in $(seq 1 20); do
    alive=0; for p in "${PIDS[@]}"; do kill -0 -- "-$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break; sleep 1
  done
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  echo "replicas stopped"
}
trap 'stop_replicas; exit 130' INT
trap 'stop_replicas; exit 143' TERM

# ---- wait for replicas ----
WAIT_MIN=${WAIT_MIN:-120}
CHECK_HOST=${SERVE_HOST:-127.0.0.1}; [ "$CHECK_HOST" = 0.0.0.0 ] && CHECK_HOST=127.0.0.1
echo "waiting for /v1/models on all ports (up to ${WAIT_MIN} min)..."
NOTREADY=0; DEADLINE=$(( $(date +%s) + WAIT_MIN*60 ))
for i in $(seq 0 $((NGPU-1))); do
  PORT=$((PORT_BASE+i)); ok=0
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    curl -sf http://$CHECK_HOST:$PORT/v1/models >/dev/null 2>&1 && { echo "  port $PORT READY"; ok=1; break; }
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "  port $PORT: server process exited"; break; }
    sleep 15
  done
  [ "$ok" = 1 ] || { echo "  port $PORT NOT READY after ${WAIT_MIN} min — check $LOGDIR/sglang_${H}_${PORT}.log"; NOTREADY=1; }
done

# ---- generation check ----
if [ "${CHECK:-1}" = 1 ]; then
  echo "generation check on every port ..."
  printf '%s' '{"model":"m","messages":[{"role":"user","content":"reply with the single word ok"}],"max_tokens":8,"temperature":0}' > "$LOGDIR/.check_${H}.json"
  for i in $(seq 0 $((NGPU-1))); do
    PORT=$((PORT_BASE+i)); ok=0
    for t in $(seq 1 24); do
      R=$(curl -s http://$CHECK_HOST:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @"$LOGDIR/.check_${H}.json" 2>/dev/null)
      echo "$R" | grep -q '"choices"' && { echo "  port $PORT CHECK OK"; ok=1; break; }
      sleep 15
    done
    [ "$ok" = 1 ] || { echo "  port $PORT CHECK FAILED"; NOTREADY=1; }
  done
fi

[ "$NOTREADY" = 0 ] || { echo "SERVE_FAILED: some replicas are not serving — see the logs above"; stop_replicas; exit 1; }
echo "SERVE_READY $(date -u +%H:%M:%S)  model=$MODEL  ports=${PORT_BASE}..$((PORT_BASE+NGPU-1))"
echo "  -> gen with:  bash scripts/eval/gen_traj.sh ... --ngpu $NGPU --port-base $PORT_BASE"
if [ "${HOLD:-1}" = 1 ]; then
  echo "Serving. Press Ctrl-C to stop all replicas."
  trap stop_replicas EXIT
  DEAD=()
  while true; do
    sleep 60 & wait $!
    for i in $(seq 0 $((NGPU-1))); do
      [ -z "${DEAD[$i]:-}" ] && ! kill -0 "${PIDS[$i]}" 2>/dev/null && { DEAD[$i]=1;
        echo "WARNING: $(date -u +%T) replica on port $((PORT_BASE+i)) exited — see $LOGDIR/sglang_${H}_$((PORT_BASE+i)).log"; }
    done
  done
fi
