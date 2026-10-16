#!/usr/bin/env python3
"""inspect_trajs.py <candidate|seed> [n_samples] — summarize a harness's runs on the search set.
`seed` = the parent harness's runs on the search set.
"""
import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = sys.argv[1]
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
pat = "parent_seed_r[0-9]*" if NAME in ("seed", "parent") else f"{NAME}_r[0-9]*"
rundirs = sorted(d for d in glob.glob(os.path.join(HERE, "runs", pat)) if os.path.isdir(d))
inst = defaultdict(lambda: [0, 0])
for rd in rundirs:
    vj = os.path.join(rd, "verify_results_merged.json")
    if os.path.exists(vj):
        for iid, e in json.load(open(vj)).items():
            inst[iid][1] += 1
            inst[iid][0] += int(isinstance(e, dict) and e.get("resolved") is True)
consistent = sorted(i for i, (r, t) in inst.items() if t and r == 0)
print(f"candidate={NAME} runs={len(rundirs)} tasks={len(inst)}")
print(f"  never resolved (consistent failures): {len(consistent)}")
print(f"  resolved in some runs (stochastic)  : {sum(1 for r, t in inst.values() if t > 1 and 0 < r < t)}")
print(f"  resolved in all runs                : {sum(1 for r, t in inst.values() if t and r == t)}")
print("trajectories of consistent failures:")
shown = 0
for iid in consistent:
    for rd in rundirs:
        hits = glob.glob(os.path.join(rd, "*", iid, f"{iid}.traj.json"))
        if hits:
            try:
                ex = (json.load(open(hits[0])).get("info") or {}).get("exit_status", "?")
            except Exception:
                ex = "?"
            print(f"   {hits[0]}   (exit={ex})"); shown += 1
            break
    if shown >= NSAMP:
        break
