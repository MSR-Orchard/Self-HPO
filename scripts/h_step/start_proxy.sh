#!/bin/bash
# start_proxy.sh — expose the served policy (ports 8000..8000+N-1) to Claude Code as the model `policy`:
#   toolcall_shim.py on 8100..8100+N-1, litellm proxy on :4001
source "$(dirname "$0")/../env.sh"
cd "$SELFHPO_REPO/scripts/h_step"
PY="$SELFHPO_WORK/envs/proposer/bin"
N=0; while curl -sf "http://127.0.0.1:$((8000 + N))/v1/models" >/dev/null 2>&1; do N=$((N + 1)); done
[ "$N" -ge 1 ] || { echo "ERR: no policy served on 127.0.0.1:8000 (start scripts/eval/serve_model.sh first)"; exit 1; }
for port in 4001 $(seq 8100 $((8099 + N))); do
  curl -s -o /dev/null -m 5 "http://127.0.0.1:$port/" && { echo "ERR: port $port is already in use (stop the old proxy/shim first)"; exit 1; }
done
CFG="$SELFHPO_WORK/litellm_proxy.yaml"
"$PY/python" - "$N" "$CFG" <<'PY'
import sys, yaml
n, out = int(sys.argv[1]), sys.argv[2]
c = yaml.safe_load(open("litellm_proxy.yaml"))
c["model_list"] = [dict(model_name="policy", litellm_params=dict(model="hosted_vllm/policy",
                   api_base=f"http://127.0.0.1:{8100 + i}/v1", api_key="dummy")) for i in range(n)]
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
cp strip_thinking_hook.py "$SELFHPO_WORK/"
SHIM_N=$N setsid nohup "$PY/python" toolcall_shim.py > "$SELFHPO_WORK/logs/toolcall_shim_server.log" 2>&1 < /dev/null &
echo "tool-call shim pid=$!  ports 8100..$((8099 + N))  log=$SELFHPO_WORK/logs/toolcall_shim_server.log"
PYTHONPATH="$SELFHPO_REPO/scripts/h_step" setsid nohup "$PY/litellm" \
  --config "$CFG" --host 127.0.0.1 --port 4001 > "$SELFHPO_WORK/logs/litellm_proxy.log" 2>&1 < /dev/null &
echo "litellm proxy pid=$!  ($N replicas)  log=$SELFHPO_WORK/logs/litellm_proxy.log"
for port in $(seq 8100 $((8099 + N))); do
  for _ in $(seq 30); do curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 2; done
  curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "ERR: shim port $port did not come up (see $SELFHPO_WORK/logs/toolcall_shim_server.log)"; exit 1; }
done
for _ in $(seq 60); do curl -sf http://127.0.0.1:4001/health/liveliness >/dev/null && break; sleep 5; done
curl -sf -m 300 http://127.0.0.1:4001/v1/chat/completions -H "Authorization: Bearer sk-local" -H "Content-Type: application/json" \
  -d '{"model":"policy","messages":[{"role":"user","content":"Say OK."}],"max_tokens":512}' | grep -q '"choices"' \
  || { echo "ERR: a test request through the proxy on :4001 failed (see $SELFHPO_WORK/logs/litellm_proxy.log)"; exit 1; }
echo "proxy ready on :4001 ($N replicas, test request OK)"
