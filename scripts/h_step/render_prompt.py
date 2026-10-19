#!/usr/bin/env python3
"""render_prompt.py — fill propose_prompt.tmpl with this search's numbers -> propose_prompt.txt."""
import json, os

H = os.path.dirname(os.path.abspath(__file__))
ss = json.load(open(f"{H}/search_set.json"))
st = json.load(open(f"{H}/experience/stats.json"))
fr = json.load(open(f"{H}/experience/frontier.json"))
names = {"rebench": "SWE-rebench", "scale": "Scale-SWE"}
desc = ", ".join(f"{len(g['ids'])} {names.get(g['name'], g['name'])}" for g in ss["groups"])
vals = {"N_TASKS": ss["n_tasks"], "SEARCH_DESC": desc,
        "SEED_BAR": next(x["avg"] for x in fr if x["name"] == "parent"),
        "COVERAGE_MISS": st["coverage_miss_pct_of_failed_with_gold_files"],
        "PRECISION": st["precision_pct_of_failed_with_gold_files"],
        "SUBLOSS_N": st["submission_loss_n"], "SUBLOSS_PCT": st["submission_loss_pct_of_failed"],
        "OWN_TEST_PASS": st["own_test_passed_pct_of_failed_with_traj"],
        "OWN_TEST_NEVER": st["own_test_never_pct_of_failed_with_traj"]}
t = open(f"{H}/propose_prompt.tmpl").read()
vals["SELFHPO_REPO"] = os.environ["SELFHPO_REPO"]
for k, v in vals.items():
    t = t.replace("${" + k + "}", str(v)).replace("{" + k + "}", str(v))
open(f"{H}/propose_prompt.txt", "w").write(t)
print("wrote propose_prompt.txt")
