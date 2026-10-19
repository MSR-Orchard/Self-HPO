#!/bin/bash
# score_traj.sh — grade one gen_traj.sh run.
# Benchmark is detected from the run path. Writes <run>/verify_results.json.
#
# Usage:
#   bash scripts/eval/score_traj.sh --run <run dir> [--pool URL] [--workers 48] [--maxpass 6]
#                                   [--sandbox-cpu N] [--sandbox-mem SIZE] [--fresh]
#   --pool         sandbox service URL (default: $SANDBOX_BASE_URL)
#   --workers      concurrent grading tasks (default 48)
#   --maxpass      max re-verify passes for grading errors (default 6)
#   --sandbox-cpu  sandbox CPUs (default: sbv 2, multi/sbp 4)
#   --sandbox-mem  sandbox memory (default: sbv 8Gi, multi/sbp 16Gi)
#   --fresh        delete the existing verify json first (default: incremental)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EH=$SELFHPO_REPO/mini-swe-agent
PY=$SELFHPO_EVAL_ENV/bin/python; MINI=$SELFHPO_EVAL_ENV/bin/mini-extra

RUN=""; POOL=""; MAXPASS=6; WORKERS=""; FRESH=0; SANDBOX_CPU=""; SANDBOX_MEM=""
while [ $# -gt 0 ]; do case "$1" in
  --run) RUN="$2"; shift 2;;
  --pool) POOL="$2"; shift 2;;
  --maxpass) MAXPASS="$2"; shift 2;;
  --workers) WORKERS="$2"; shift 2;;
  --sandbox-cpu) SANDBOX_CPU="$2"; shift 2;;
  --sandbox-mem) SANDBOX_MEM="$2"; shift 2;;
  --fresh) FRESH=1; shift;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
die(){ echo "ERR: $*"; exit 1; }
[ -n "$RUN" ] || die "--run <run dir> required"
[ -d "$RUN" ] || die "run dir not found: $RUN"
RUN=$(cd "$RUN" && pwd)

# ---- detect benchmark ----
case "$RUN" in
  *SWE_Bench_Verified*)      BENCH=sbv;   DENOM=500; DEF_SBOX_CPU=2; DEF_SBOX_MEM=8Gi;;
  *SWE_Bench_Multilingual*)  BENCH=multi; DENOM=300; DEF_SBOX_CPU=4; DEF_SBOX_MEM=16Gi;;
  *SWE_Bench_Pro*)           BENCH=sbp;   DENOM=642; DEF_SBOX_CPU=4; DEF_SBOX_MEM=16Gi;;
  */sbv_*)                   BENCH=sbv;   DENOM=500; DEF_SBOX_CPU=2; DEF_SBOX_MEM=8Gi;;
  */ml_*)                    BENCH=multi; DENOM=300; DEF_SBOX_CPU=4; DEF_SBOX_MEM=16Gi;;
  */sbp_*)                   BENCH=sbp;   DENOM=642; DEF_SBOX_CPU=4; DEF_SBOX_MEM=16Gi;;
  *) die "cannot detect benchmark from run path (need SWE_Bench_Verified|Multilingual|Pro in the path): $RUN";;
esac
[ -n "$POOL" ] || POOL=${SANDBOX_BASE_URL:-}
[ -n "$POOL" ] || die "set --pool or SANDBOX_BASE_URL"
case "$POOL" in http://*|https://*) ;; *) POOL=http://$POOL;; esac
[ -n "$WORKERS" ] || WORKERS=48
[ -n "$SANDBOX_CPU" ] || SANDBOX_CPU=$DEF_SBOX_CPU
[ -n "$SANDBOX_MEM" ] || SANDBOX_MEM=$DEF_SBOX_MEM
VJ="$RUN/verify_results.json"
[ "$FRESH" = 1 ] && rm -f "$VJ"

echo "=================================================================="
echo "SCORE benchmark=$BENCH run=$RUN"
echo "      pool=$POOL workers=$WORKERS sandbox_cpu=$SANDBOX_CPU sandbox_mem=$SANDBOX_MEM maxpass=$MAXPASS denom=$DENOM"
echo "=================================================================="

# ---- common env ----
export SANDBOX_API_KEY=${SANDBOX_API_KEY:?set SANDBOX_API_KEY}
export SANDBOX_BASE_URL=$POOL

GRADE_ERR='
import json,sys
def bad(s, sample):
    return bool(s.get("error")) or (sample and "exit_code" in s and s.get("exit_code") in (-1, None))
def errored(x):
    if not isinstance(x, dict):
        return False
    if "samples" in x:
        return any(bad(s, True) for s in x["samples"])
    return bad(x, False)
'
gecount(){ $PY -c "$GRADE_ERR
try: v=json.load(open(sys.argv[1]))
except Exception: print(0); raise SystemExit
print(sum(errored(x) for x in v.values()))" "$1"; }
del_errors(){ $PY -c "$GRADE_ERR
v=json.load(open(sys.argv[1]))
rm=[i for i,x in v.items() if errored(x)]
for i in rm: del v[i]
json.dump(v,open(sys.argv[1],'w'))
print('removed %d grading-error entries, re-verifying'%len(rm))" "$1"; }

cd "$EH"
case "$BENCH" in
  sbv)
    VERIFY(){ $MINI swebench-verify-orchard -c swebench_orchard -c environment.cpu=\"$SANDBOX_CPU\" -c environment.memory=\"$SANDBOX_MEM\" --subset verified --split test --workers "$WORKERS" --output "$RUN"; };;
  multi)
    VERIFY(){ $MINI swebench-multilingual-verify-orchard -c swebench_multi_orchard -c environment.cpu=\"$SANDBOX_CPU\" -c environment.memory=\"$SANDBOX_MEM\" --subset multilingual --split test -o "$RUN" -w "$WORKERS"; };;
  sbp)
    export SWE_PRO_REPO="$SELFHPO_SWE_PRO/sbp_repo"
    export SWE_PRO_CPU=$SANDBOX_CPU SWE_PRO_MEM=$SANDBOX_MEM
    VERIFY(){ $MINI swebench-pro-verify-orchard -c swebench_pro_orchard --subset swebench_pro --split test -o "$RUN" -w "$WORKERS"; };;
esac
echo "verify pass 0 $(date -u +%H:%M)"; VERIFY 2>&1 | tail -2
for p in $(seq 1 "$MAXPASS"); do
  E=$(gecount "$VJ"); echo "verify pass $p: $E grading errors $(date -u +%H:%M)"
  [ "$E" -eq 0 ] && break
  del_errors "$VJ"; VERIFY 2>&1 | tail -1
done

# ---- final tally ----
echo "=== FINAL $(date -u +%H:%M) ==="
$PY - "$VJ" "$RUN" "$DENOM" "$BENCH" <<'PY'
import json,sys,glob,collections
vj,run,denom,bench=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
try: v=json.load(open(vj))
except Exception as e: print("ERR: no verify json:",e); raise SystemExit(1)
res=sum(1 for x in v.values() if isinstance(x,dict) and (x.get("resolved") or any(s.get("resolved") for s in x.get("samples",[]))))
def bad(s, sample):
    return bool(s.get("error")) or (sample and "exit_code" in s and s.get("exit_code") in (-1, None))
ge=sum(1 for x in v.values() if isinstance(x,dict) and (any(bad(s,True) for s in x["samples"]) if "samples" in x else bad(x,False)))
c=collections.Counter(); no_model_turn=0
for f in glob.glob(run+"/*/*.traj.json"):
    try: j=json.load(open(f))
    except: c["_unreadable"]+=1; continue
    st=j.get("info",{}).get("exit_status","?"); c[st]+=1
    if st!="Submitted" and not any(m.get("role")=="assistant" for m in j.get("messages",[])): no_model_turn+=1
ise=c.get("InternalServerError",0)
print("resolved = %d/%d = %.1f%%   (graded=%d)"%(res,denom,100.0*res/denom,len(v)))
if len(v)<denom: print("WARNING: %d of %d tasks have no grade (missing trajectories count as unresolved): re-run gen_traj.sh, then score_traj.sh --run <dir> --fresh"%(denom-len(v),denom))
clean = ise==0 and ge==0 and no_model_turn==0 and len(v)>=denom
print("infra audit:  server errors=%d  no model turn=%d  grading errors=%d  %s"%(ise,no_model_turn,ge,"CLEAN" if clean else "WARNING: not clean, re-run score_traj.sh (if server errors or no-model-turn runs > 0, re-run gen_traj.sh first, then score_traj.sh --run <dir> --fresh)"))
PY
echo "SCORE_DONE $(date -u +%H:%M:%S)  run=$RUN"
echo "  summary: python3 $SELFHPO_REPO/scripts/eval/results.py -m <method> -b $BENCH"
