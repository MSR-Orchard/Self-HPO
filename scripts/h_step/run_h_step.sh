#!/bin/bash
# run_h_step.sh <work_dir> <out harness dir> [N_PROPOSALS=6] [SEED_RUNS=1]
# Harness search in a work dir created by init.sh; the best harness is copied to <out harness dir>.
# Needs the policy served on 127.0.0.1:8000-...; for PROPOSER=self (default) it starts the proxy if it is not running.
# Output: experience/frontier.json (all candidates and scores) and agents/<candidate>/.
# Rerunning the same command continues after the last recorded proposal.
source "$(dirname "$0")/../env.sh"
W=${1:?usage: run_h_step.sh <work_dir> <out harness dir> [N_PROPOSALS] [SEED_RUNS]}
HOUT=${2:?out harness dir}; mkdir -p "$(dirname "$HOUT")"; HOUT=$(cd "$(dirname "$HOUT")" && pwd)/$(basename "$HOUT"); MAX=${3:-6}; SEEDN=${4:-1}
source "$SELFHPO_EVAL_ENV/bin/activate"
[ "$(id -u)" = 0 ] && export IS_SANDBOX=1
if [ "${PROPOSER:-self}" = self ] && ! curl -sf -m 5 http://127.0.0.1:4001/health/liveliness >/dev/null; then
  bash "$SELFHPO_REPO/scripts/h_step/start_proxy.sh" || exit 1
fi
command -v claude >/dev/null || { echo 'ERR: claude not on PATH (export PATH=$HOME/.local/bin:$PATH)'; exit 1; }
set -o pipefail
infra_ok(){
  curl -sf -m 10 http://127.0.0.1:8000/v1/models >/dev/null || return 1
  [ "${PROPOSER:-self}" = self ] || return 0
  curl -sf -m 120 http://127.0.0.1:4001/v1/chat/completions -H "Authorization: Bearer sk-local" \
    -H "Content-Type: application/json" -d '{"model":"policy","messages":[{"role":"user","content":"Say OK."}],"max_tokens":512}' \
    | grep -q '"choices"'
}
wait_infra(){
  for _ in $(seq 60); do infra_ok && return 0; echo "waiting for the serve/proxy ($(date -u +%T))" | tee -a "$LOG"; sleep 30; done
  echo "ERR: serve or proxy down for 30 min; stopping (restart serve_model.sh / start_proxy.sh, then rerun to resume)" | tee -a "$LOG"; exit 1
}
cd "$W" || exit 1
LOG=runs/h_step.log
infra_ok || { echo "ERR: the policy serve (127.0.0.1:8000) or the proxy (start_proxy.sh, :4001) does not answer"; exit 1; }
if [ ! -f experience/frontier.json ]; then bash seed_bar.sh "$SEEDN" | tee -a "$LOG" || exit 1; fi
[ -f experience/frontier.json ] || { echo "ERR: seed bar did not finish"; exit 1; }
if [ ! -f experience/stats.json ]; then python build_experience.py | tee -a "$LOG" || exit 1; fi
python render_prompt.py | tee -a "$LOG" || exit 1
propose(){
  env -u WANDB_API_KEY -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u GITHUB_TOKEN -u GH_TOKEN \
    "$SELFHPO_REPO/scripts/h_step/proposer.sh" "$@" \
    --allowedTools "Read Write Edit Bash Glob Grep" --permission-mode bypassPermissions >> "$LOG" 2>&1
}
INFRA="search_eval.py fire_gate.py parent_check.py record_result.py inspect_trajs.py build_experience.py render_prompt.py propose_prompt.tmpl seed_bar.sh"
restore_infra(){ for f in $INFRA; do cp -f "$SELFHPO_REPO/scripts/h_step/$f" "$f"; done; python render_prompt.py > /dev/null; }
gate(){
  restore_infra
  local R="runs/gate_$1.txt"
  if cmp -s "agents/$1/agent.py" "agents/$2/agent.py"; then echo "agents/$1/agent.py is identical to the parent's: add your mechanism to it." > "$R"; return 1; fi
  local bad
  bad=$(python - "agents/$1/agent.py" <<'PY'
import ast, sys
try:
    t = ast.parse(open(sys.argv[1]).read())
except SyntaxError:
    sys.exit(0)
for c in ast.walk(t):
    if isinstance(c, ast.ClassDef) and c.name == "AgentHarness":
        names = {f.name for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
        m = next((m for m in ("run", "query") if m in names), "")
        if m:
            print(f"override {m}")
            break
        used = set()
        for n in ast.walk(c):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                if n.attr in ("progress_manager", "instance_id", "logger", "extra_template_vars"):
                    used.add(f"self.{n.attr}")
            if (isinstance(n, ast.Attribute) and n.attr == "query" and isinstance(n.value, ast.Attribute)
                    and n.value.attr == "model" and isinstance(n.value.value, ast.Name) and n.value.value.id == "self"):
                used.add("self.model.query")
        if used:
            print("uses " + ", ".join(sorted(used)))
        break
PY
)
  case "$bad" in
    override*) echo "agents/$1/agent.py overrides ${bad#override }(): RL training only calls step() and execute_actions(), so a mechanism in ${bad#override }() would never act during the P-step. Move it into step() or execute_actions() and remove the ${bad#override }() override." > "$R"; return 1;;
    uses*) echo "agents/$1/agent.py uses ${bad#uses }: these are not available during RL training, so every P-step rollout would fail. Remove them; call super().step() for the normal step and self.env.execute for repo access." > "$R"; return 1;;
  esac
  local imp
  if ! imp=$(PYTHONPATH="$SELFHPO_REPO/mini-swe-agent/src${PYTHONPATH:+:$PYTHONPATH}" python - "agents/$1/agent.py" 2>&1 <<'PY'
import importlib.util, sys, traceback
try:
    spec = importlib.util.spec_from_file_location("_gate_harness", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.AgentHarness
except BaseException:
    print(traceback.format_exc()[-3000:])
    sys.exit(1)
PY
); then
    { echo "agents/$1/agent.py fails to import, so it cannot run at all. Fix the error below:"; echo "$imp"; } > "$R"; return 1
  fi
  if [ ! -s "agents/$1/MARKER.txt" ]; then echo "agents/$1/MARKER.txt is missing or empty: write your marker token there." > "$R"; return 1; fi
  if [ ! -s "agents/$1/ACT_MARKER.txt" ] || cmp -s "agents/$1/ACT_MARKER.txt" "agents/$1/MARKER.txt"; then
    echo "agents/$1/ACT_MARKER.txt is missing or equals MARKER.txt: write a second token there and self.mark() it only when your mechanism acts." > "$R"; return 1; fi
  if ! python parent_check.py "$1" > "$R" 2>&1; then return 1; fi
  local rc=0
  python fire_gate.py "$1" > "$R.fg" 2>&1 || rc=$?
  if [ "$rc" -eq 3 ]; then
    cat "$R.fg" >> "$LOG"; wait_infra; rc=0; python fire_gate.py "$1" > "$R.fg" 2>&1 || rc=$?
    [ "$rc" -eq 3 ] && { cat "$R.fg" >> "$LOG"; echo "ERR: fire_gate failed twice on infrastructure; stopping" | tee -a "$LOG"; exit 1; }
  fi
  if [ "$rc" -ne 0 ] || ! grep -q "FIRED=yes" "$R.fg"; then
    { echo "fire_gate (the candidate run on 3 real tasks):"; tail -c 4000 "$R.fg"; } > "$R"; return 1
  fi
  cat "$R.fg" >> "$LOG"
}
# ---- proposals ----
START=$(python -c "import json,re;print(1+max([int(m.group(1)) for x in json.load(open('experience/frontier.json')) for m in [re.match(r'c(\d+)_a\d+\$',x['name'])] if m]+[0]))")
for i in $(seq "$START" "$MAX"); do
  PARENT=$(tr -d '[:space:]' < experience/current_parent.txt)
  echo "=== PROPOSAL $i START $(date -u +%FT%TZ) (parent=$PARENT) ===" | tee -a "$LOG"
  NAME=""
  for attempt in 1 2 3 4 5; do
    CN="c${i}_a${attempt}"
    rm -rf "agents/$CN"
    cp -r "agents/$PARENT" "agents/$CN"
    rm -f "agents/$CN/MARKER.txt" "agents/$CN/ACT_MARKER.txt" "agents/$CN/.hook_fired" "agents/$CN/purpose"
    echo "$PARENT" > "agents/$CN/parent"
    SID=$(python -c "import uuid; print(uuid.uuid4())")
    echo "=== $CN: proposer session $SID ===" | tee -a "$LOG"
    restore_infra
    wait_infra
    propose -p "$(sed "s/\bCAND\b/$CN/g" propose_prompt.txt)" --session-id "$SID"
    for fix in 0 1 2 3 4; do
      if gate "$CN" "$PARENT"; then echo "=== $CN accepted (FIRED=yes, fix $fix) ===" | tee -a "$LOG"; NAME=$CN; break; fi
      echo "=== $CN rejected (fix $fix): $(head -n 1 "runs/gate_$CN.txt" | cut -c1-200) ===" | tee -a "$LOG"
      cat "runs/gate_$CN.txt" >> "$LOG"
      [ "$fix" -lt 4 ] || break
      wait_infra
      propose -p "The search's automatic check REJECTED your candidate agents/$CN. Reason:
$(cat "runs/gate_$CN.txt")

Fix agents/$CN (keep every parent method and your one added mechanism; the marker must be recorded on the first line of your handler), then STOP. The check will run again automatically." --resume "$SID"
    done
    [ -n "$NAME" ] && break
  done
  [ -n "$NAME" ] || { echo "=== PROPOSAL $i: no accepted candidate after 5 attempts ===" | tee -a "$LOG"; continue; }
  restore_infra
  wait_infra
  rc=0; python search_eval.py "$NAME" r1 >> "$LOG" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    wait_infra; rc=0; python search_eval.py "$NAME" r1 >> "$LOG" 2>&1 || rc=$?
    [ "$rc" -ne 0 ] && { echo "ERR: search_eval of $NAME failed twice (rc=$rc); stopping" | tee -a "$LOG"; exit 1; }
  fi
  python record_result.py "$NAME" | tee -a "$LOG" || exit 1
done
echo "H_STEP_DONE  frontier:" | tee -a "$LOG"
python -c "import json;[print(f\"  {x['name']:10s} {x['avg']:6.2f}  kept={x['kept']}\") for x in json.load(open('experience/frontier.json'))]" | tee -a "$LOG"
BEST=$(cat experience/current_parent.txt)
rm -rf "$HOUT"; cp -r "agents/$BEST" "$HOUT"
echo "best harness: $BEST -> $HOUT" | tee -a "$LOG"
