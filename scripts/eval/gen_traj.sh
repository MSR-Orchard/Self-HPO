#!/bin/bash
# gen_traj.sh — generate agent trajectories for one evaluation run.
# Benchmarks: sbv (SWE-bench Verified) | multi (SWE-bench Multilingual) | sbp (SWE-bench Pro).
# Start the policy server first (serve_model.sh).
#
# Output: $SELFHPO_WORK/eval/<Method>/<Step>/<Benchmark>/<tag>_<harness>_temp_<T>_top_p_<P>_r<N>/
# The r<N> suffix auto-increments, so re-running the same command adds one more run.
#
# Usage:
#   bash scripts/eval/gen_traj.sh --method <name> [--step <name>] --benchmark <sbv|multi|sbp> [--harness <seed|self-hpo|hpo|DIR>] [options]
#
# Required:
#   --method            label for the output tree (e.g. --method Self-HPO); --step adds an optional sub-label
#   --benchmark         sbv | multi | sbp
# Options:
#   --harness           seed | self-hpo | hpo (directories under <repo>/harnesses), or a harness dir (default seed)
#   --temp 0.95 --top_p 0.95
#   --timeout           per-task agent budget in seconds (sbv/multi 900, sbp 1800)
#   --steplim           agent step limit (default: from the config, 250)
#   --passes 3          generation passes
#   --workers N         concurrent tasks (default 48)
#   --ngpu N            number of server replicas started by serve_model.sh (default: #GPUs, else #replicas answering)
#   --port-base 8000    first server port
#   --node-ip 127.0.0.1 host of the servers
#   --pool URL          sandbox service (default: $SANDBOX_BASE_URL)
#   --run NAME          resume / name a specific run dir
#   --filter REGEX      run only tasks whose instance_id matches REGEX
#   --output_dir DIR    root of the output tree (default: $SELFHPO_WORK/eval)
#   --sandbox-cpu N     sandbox CPUs (default: from the environment config)
#   --sandbox-mem M     sandbox memory, e.g. 16Gi (default: from the environment config)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EH=$SELFHPO_REPO/mini-swe-agent; EVAL_ROOT=$SELFHPO_WORK/eval

# ---- defaults / args ----
METHOD=""; STEP=""; BENCH=""; HARNESS="seed"; POOL=""; TEMP="0.95"; TOPP="0.95"
PASSES=3; TIMEOUT=""; STEPLIM=""; WORKERS=""; NGPU=""; PORT_BASE=8000; NODE_IP="127.0.0.1"
FILTER=""; RUNNAME=""; OUTROOT=""; SANDBOX_CPU=""; SANDBOX_MEM=""
while [ $# -gt 0 ]; do case "$1" in
  --method) METHOD="$2"; shift 2;;
  --step) STEP="$2"; shift 2;;
  --benchmark) BENCH="$2"; shift 2;;
  --harness) HARNESS="$2"; shift 2;;
  --pool) POOL="$2"; shift 2;;
  --temp) TEMP="$2"; shift 2;;
  --top_p) TOPP="$2"; shift 2;;
  --passes) PASSES="$2"; shift 2;;
  --timeout) TIMEOUT="$2"; shift 2;;
  --steplim) STEPLIM="$2"; shift 2;;
  --workers) WORKERS="$2"; shift 2;;
  --ngpu) NGPU="$2"; shift 2;;
  --port-base) PORT_BASE="$2"; shift 2;;
  --node-ip) NODE_IP="$2"; shift 2;;
  --filter) FILTER="$2"; shift 2;;
  --run) RUNNAME="$2"; shift 2;;
  --output_dir) OUTROOT="$2"; shift 2;;
  --sandbox-cpu) SANDBOX_CPU="$2"; shift 2;;
  --sandbox-mem) SANDBOX_MEM="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
die(){ echo "ERR: $*"; exit 1; }
[ -n "$METHOD" ] || die "--method required (a label, e.g. Self-HPO)"
[ -n "$BENCH" ] || die "--benchmark required (sbv multi sbp)"

# ---- benchmark registry ----
case "$BENCH" in
  sbv)   BENCHDIR=SWE_Bench_Verified;     BTAG=sbv; SUBCMD=swebench;     PROMPT=swebench;     ENVCFG=swebench_orchard;       SUBSET="--subset verified --split test";     DEF_W=48; DEF_TO=900;  DENOM=500;;
  multi) BENCHDIR=SWE_Bench_Multilingual; BTAG=ml;  SUBCMD=swebench;     PROMPT=swebench;     ENVCFG=swebench_multi_orchard; SUBSET="--subset multilingual --split test"; DEF_W=48; DEF_TO=900; DENOM=300;;
  sbp)   BENCHDIR=SWE_Bench_Pro;          BTAG=sbp; SUBCMD=swebench-pro; PROMPT=swebench_pro; ENVCFG=swebench_pro_orchard;   SUBSET="--subset swebench_pro --split test"; DEF_W=48; DEF_TO=1800; DENOM=642;;
  *) die "benchmark: sbv | multi | sbp";;
esac
[ -n "$TIMEOUT" ] || TIMEOUT=$DEF_TO
[ -n "$WORKERS" ] || WORKERS=$DEF_W
[ -n "$POOL" ] || POOL=${SANDBOX_BASE_URL:-}
[ -n "$POOL" ] || die "set --pool or SANDBOX_BASE_URL"
case "$POOL" in http://*|https://*) ;; *) POOL=http://$POOL;; esac

# ---- harness ----
if [ -d "$HARNESS" ]; then HD=$(cd "$HARNESS" && pwd); else HD=$SELFHPO_REPO/harnesses/$HARNESS; fi
[ -f "$HD/config.yaml" ] || die "harness not found: $HD (use seed, self-hpo, hpo, or a harness dir)"
HARNESS=$(basename "$HD")
HAF=""
[ "$HARNESS" != seed ] && HAF=$HD/agent.py

# ---- harness config ----
if [ "$BENCH" = sbp ]; then echo "NOTE: SWE-bench Pro uses its own prompt config ($PROMPT)"; else PROMPT=$HD/config.yaml; fi

# ---- server check ----
SERVED=$(curl -s -m 8 "http://$NODE_IP:$PORT_BASE/v1/models" 2>/dev/null | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
[ -n "$SERVED" ] || die "serve not reachable at http://$NODE_IP:$PORT_BASE/v1/models — start it with serve_model.sh (or fix --node-ip/--port-base)"
echo "served model: $SERVED"
UP=0; while curl -sf -m 8 "http://$NODE_IP:$((PORT_BASE + UP))/v1/models" >/dev/null 2>&1; do UP=$((UP + 1)); done

# ---- auto NGPU ----
[ -n "$NGPU" ] || NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=$UP
[ "$UP" -ge "$NGPU" ] || die "only $UP replica(s) answer from port $PORT_BASE but NGPU=$NGPU: tasks would go to dead ports (pass --ngpu $UP or restart the serve)"

# ---- output dir ----
[ -n "$OUTROOT" ] || OUTROOT=$EVAL_ROOT
LEAF=$OUTROOT/$METHOD${STEP:+/$STEP}/$BENCHDIR
mkdir -p "$LEAF"
PREFIX="${BTAG}_${HARNESS}_temp_${TEMP}_top_p_${TOPP}"
if [ -n "$RUNNAME" ]; then
  RUN="$RUNNAME"
else
  N=1
  for d in "$LEAF/${PREFIX}_r"*; do [ -d "$d" ] || continue; k=$(basename "$d"); k=${k##*_r}; k=${k%%_*}; case "$k" in ''|*[!0-9]*) ;; *) [ "$k" -ge "$N" ] && N=$((k+1));; esac; done
  RUN="${PREFIX}_r${N}"
fi
OUT="$LEAF/$RUN"

if [ -n "$STEPLIM" ]; then EFF_STEPLIM="$STEPLIM (--steplim)"; else
  PCFG=$PROMPT; [ -f "$PCFG" ] || PCFG=$EH/src/minisweagent/config/benchmarks/$PROMPT.yaml
  EFF_STEPLIM="$(grep -E '^ *step_limit:' "$PCFG" | head -1 | awk '{print $2}') (from $(basename "$PCFG"))"
fi
echo "=================================================================="
echo "GEN  method=$METHOD step=$STEP benchmark=$BENCH harness=$HARNESS"
echo "     temp=$TEMP top_p=$TOPP passes=$PASSES timeout=$TIMEOUT steplim=$EFF_STEPLIM workers=$WORKERS ngpu=$NGPU pool=$POOL"
echo "     harness_hook=${HAF:-<none: plain seed>}  prompt_cfg=$PROMPT"
echo "     sandbox_cpu=${SANDBOX_CPU:-<env config default>} sandbox_mem=${SANDBOX_MEM:-<env config default>}  subcmd=$SUBCMD env_cfg=$ENVCFG"
echo "     OUT=$OUT"
[ -n "$FILTER" ] && echo "     filter=$FILTER"
echo "=================================================================="
mkdir -p "$OUT"

# ---- env ----
source "$SELFHPO_EVAL_ENV/bin/activate"
cd "$EH"
export SANDBOX_API_KEY=${SANDBOX_API_KEY:?set SANDBOX_API_KEY}
export SANDBOX_BASE_URL=$POOL
export OPENAI_API_KEY=EMPTY
export SGLANG_PORT_BASE=$PORT_BASE; export SGLANG_NGPU=$NGPU
if [ "$BENCH" = sbp ]; then export SWE_CLOCK_AFTER_READY=${SWE_CLOCK_AFTER_READY:-1}; export SWE_PULL_ALLOWANCE=${SWE_PULL_ALLOWANCE:-5400}; fi
if [ -n "$HAF" ]; then export HARNESS_AGENT_FILE=$HAF; else unset HARNESS_AGENT_FILE; fi

FILTER_ARG=(); [ -n "$FILTER" ] && FILTER_ARG=(--filter "$FILTER")
STEPLIM_ARG=""; [ -n "$STEPLIM" ] && STEPLIM_ARG="-c agent.step_limit=$STEPLIM"
SANDBOX_ARG=""; [ -n "$SANDBOX_CPU" ] && SANDBOX_ARG="$SANDBOX_ARG -c environment.cpu=\"$SANDBOX_CPU\""; [ -n "$SANDBOX_MEM" ] && SANDBOX_ARG="$SANDBOX_ARG -c environment.memory=\"$SANDBOX_MEM\""
APIBASE="http://$NODE_IP:{port}/v1"

do_gen(){
  mini-extra "$SUBCMD" \
    -c "$PROMPT" -c "$ENVCFG" $SANDBOX_ARG \
    -c model.model_kwargs.temperature=$TEMP \
    -c model.model_kwargs.top_p=$TOPP \
    $STEPLIM_ARG \
    -c model.model_kwargs.api_key="$OPENAI_API_KEY" \
    -c "model.model_kwargs.api_base=$APIBASE" \
    $SUBSET ${FILTER_ARG[@]+"${FILTER_ARG[@]}"} \
    -o "$OUT" --n-samples 1 --workers "$WORKERS" --instance-timeout "$TIMEOUT" --no-preds
}

for p in $(seq 1 "$PASSES"); do
  echo "=== PASS $p/$PASSES gen start $(date -u +%H:%M:%S) ==="
  do_gen
  echo "=== PASS $p/$PASSES gen done $(date -u +%H:%M:%S) ==="
done

# ---- validate ----
echo "=== VALIDATE $(date -u +%H:%M:%S) ==="
"$SELFHPO_EVAL_ENV/bin/python" - "$OUT" "$DENOM" <<'PY'
import sys,glob,json,collections
out,denom=sys.argv[1],int(sys.argv[2]); c=collections.Counter(); nt=0
for f in glob.glob(out+"/*/*.traj.json"):
    nt+=1
    try: c[json.load(open(f)).get("info",{}).get("exit_status","?")]+=1
    except: c["_unreadable"]+=1
ise=c.get("InternalServerError",0); sub=c.get("Submitted",0)
print("trajs=%d/%d  exit_status=%s"%(nt,denom,dict(c)))
if nt<denom: print("WARNING: %d tasks have no trajectory: re-run the same command with --run %s to resume, then score with score_traj.sh --run <dir> --fresh"%(denom-nt,out.rstrip("/").split("/")[-1]))
if ise: print("WARNING: %d InternalServerError trajs: check the server, then re-run the same command with --run %s and score with score_traj.sh --run <dir> --fresh"%(ise,out.rstrip("/").split("/")[-1]))
print("Submitted=%d"%sub)
PY
echo "GEN_DONE $(date -u +%H:%M:%S)  OUT=$OUT"
echo "  -> score with:  bash scripts/eval/score_traj.sh --run $OUT --pool $POOL"
