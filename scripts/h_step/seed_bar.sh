#!/bin/bash
# seed_bar.sh [N_RUNS] — evaluate the parent harness on the search set (N_RUNS runs, default 1).
# Writes runs/parent_seed_r<k>, the seed entry of experience/frontier.json, and data/check_ids.json.
if [ -n "${SELFHPO_REPO:-}" ]; then source "$SELFHPO_REPO/scripts/env.sh"
elif [ -f "$(dirname "$0")/../env.sh" ]; then source "$(dirname "$0")/../env.sh"
else echo "ERR: run via run_h_step.sh or source scripts/env.sh first"; exit 1; fi
source "$SELFHPO_EVAL_ENV/bin/activate"
cd "$(dirname "$0")"
N=${1:-1}
for k in $(seq 1 "$N"); do python search_eval.py parent "seed_r$k" || exit 1; done
python - <<'PY'
import glob, json, random, statistics
runs = sorted(glob.glob("runs/parent_seed_r*"))
pcts, sub = [], set()
for r in runs:
    v = json.load(open(f"{r}/verify_results_merged.json"))
    pcts.append(round(100 * sum(1 for e in v.values() if isinstance(e, dict) and e.get("resolved") is True) / len(v), 2))
for f in glob.glob("runs/parent_seed_r1/*/*/*.traj.json"):
    j = json.load(open(f))
    if (j.get("info") or {}).get("exit_status") == "Submitted":
        sub.add(j.get("instance_id"))
avg = round(statistics.mean(pcts), 2)
if not sub or avg == 0:
    raise SystemExit(f"ERR: seed bar {avg}% with {len(sub)} submitted tasks: check the serve and the sandbox pool, then rerun")
json.dump([{"name": "parent", "n_runs": len(pcts), "per_run": pcts, "avg": avg, "kept": True, "note": "seed bar"}],
          open("experience/frontier.json", "w"), indent=2)
random.seed(0)
json.dump(sorted(random.sample(sorted(sub), min(3, len(sub)))), open("data/check_ids.json", "w"))
print(f"SEED_BAR {avg}% (runs: {pcts})")
PY
