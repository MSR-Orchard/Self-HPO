#!/usr/bin/env python3
"""record_result.py <candidate> — record a candidate's search-set result in experience/frontier.json.
Also writes experience/last_candidate.json and experience/current_parent.txt.
"""
import glob, json, os, statistics, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
PROMOTE_MARGIN = 1.0
name = sys.argv[1]


def resolved(e):
    return bool(isinstance(e, dict) and e.get("resolved") is True)


def per_task(rundirs):
    tally, n = {}, 0
    for rd in rundirs:
        vj = os.path.join(rd, "verify_results_merged.json")
        if os.path.exists(vj):
            n += 1
            for iid, e in json.load(open(vj)).items():
                tally[iid] = tally.get(iid, 0) + int(resolved(e))
    return {iid: c * 2 >= n for iid, c in tally.items()} if n else {}


def run_pcts(rundirs):
    out = []
    for rd in rundirs:
        vj = os.path.join(rd, "verify_results_merged.json")
        if os.path.exists(vj):
            d = json.load(open(vj))
            out.append(round(100.0 * sum(resolved(e) for e in d.values()) / max(len(d), 1), 2))
    return out


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


fp = os.path.join(HERE, "experience", "frontier.json")
fr = json.load(open(fp))
seed_bar = next(x["avg"] for x in fr if x["name"] == "parent")
rundirs = sorted(glob.glob(os.path.join(HERE, "runs", f"{name}_r[0-9]*")))
pcts = run_pcts(rundirs)
avg = round(statistics.mean(pcts), 2) if pcts else 0.0
anchor = max(x["avg"] for x in fr if x.get("kept"))
cand = per_task(rundirs)
seed = per_task(sorted(glob.glob(os.path.join(HERE, "runs", "parent_seed_r*"))))
common = set(cand) & set(seed)
lost = sorted(i for i in common if seed[i] and not cand[i])
gained = sorted(i for i in common if cand[i] and not seed[i])
act_path = os.path.join(HERE, "agents", name, "ACT_MARKER.txt")
act_tok = (open(act_path).read().split() or [""])[0] if os.path.exists(act_path) else ""
acted = set()
for rd in rundirs:
    for f in glob.glob(os.path.join(rd, "*", "*", "*.traj.json")):
        try:
            marks = (json.load(open(f)).get("info") or {}).get("harness_marks") or []
        except Exception:
            continue
        if act_tok and any(m.get("token") == act_tok for m in marks):
            acted.add(os.path.basename(os.path.dirname(f)))
acted &= common


def split(ids):
    return {"n": len(ids), "cand_resolved": sum(cand[i] for i in ids), "seed_resolved": sum(seed[i] for i in ids),
            "gained": sorted(i for i in ids if cand[i] and not seed[i]),
            "lost": sorted(i for i in ids if seed[i] and not cand[i])}


mech = {"act_marker": act_tok, "acted": split(acted), "not_acted": split(common - acted)}
promoted = avg >= anchor + PROMOTE_MARGIN and len(gained) > len(lost) and len(acted) > 0
fr.append({"name": name, "n_runs": len(pcts), "per_run": pcts, "avg": avg, "kept": promoted, "anchor": anchor,
           "promote_margin": PROMOTE_MARGIN,
           "vs_seed": {"seed_only": len(lost), "cand_only": len(gained), "mcnemar_p": mcnemar_p(len(lost), len(gained))},
           "mechanism": {"act_marker": act_tok, "n_acted": len(acted),
                         "acted": {k: v for k, v in mech["acted"].items() if k not in ("gained", "lost")},
                         "not_acted": {k: v for k, v in mech["not_acted"].items() if k not in ("gained", "lost")}}})
json.dump(fr, open(fp, "w"), indent=2)
json.dump({"candidate": name, "avg": avg, "promoted": promoted, "regressed_vs_seed": lost, "gained_vs_seed": gained,
           "mechanism": mech,
           "note": "mechanism.acted = tasks where the mechanism acted (its ACT_MARKER was recorded); on not_acted tasks the "
                   "mechanism did not act, so gains and losses there mostly come from run-to-run sampling variation."},
          open(os.path.join(HERE, "experience", "last_candidate.json"), "w"), indent=2)
best = max((x for x in fr if x.get("kept")), key=lambda x: x["avg"])
open(os.path.join(HERE, "experience", "current_parent.txt"), "w").write(best["name"] + "\n")
with open(os.path.join(HERE, "experience", "evolution_summary.md"), "a") as h:
    a, na = mech["acted"], mech["not_acted"]
    h.write(f"{name:16s} avg={avg:6.2f}  seed_bar={seed_bar}  promoted={promoted}  parent_now={best['name']}  "
            f"| acted on {a['n']}: cand {a['cand_resolved']} vs seed {a['seed_resolved']} "
            f"| not acted on {na['n']}: cand {na['cand_resolved']} vs seed {na['seed_resolved']}\n")
print(f"RECORDED {name} avg={avg}% (runs {pcts}) promoted={promoted} anchor={anchor} "
      f"vs seed: +{len(gained)} -{len(lost)} acted={len(acted)} -> parent={best['name']}", flush=True)
