#!/usr/bin/env python3
"""build_search_set.py — write the search set of an H-step into <out_dir>.

Both sources draw from the RL training set (--train FILE = work/rl/data/train.jsonl from prepare_rl_data.py):
  random   --sample N        N training tasks drawn uniformly at random, or the tasks listed in --ids FILE
           [--ids FILE]      (JSON list of instance ids)
  hard     --rollouts DIR    training tasks the policy never solved in the P-step rollouts under DIR

Output (<out_dir>):
  search_set.json      {"n_tasks", "groups": [{name, runner, verifier, env_config, subset, split, ids}]}
  gold_patches.jsonl   {"instance_id", "patch"} per task
  tasks_*.jsonl        flattened task records per group
"""
import argparse, glob, json, os, re, sys

REBENCH = dict(runner="swerebench", verifier="swerebench-verify-orchard", env_config="swebench_orchard")
SCALE = dict(runner="scaleswe", verifier="scaleswe-verify-orchard", env_config="scaleswe_orchard")


def flatten(rec):
    m = dict(rec["metadata"])
    m["problem_statement"] = rec["problem_statement"]
    m["image_name"] = m.get("image_url")
    if isinstance(m.get("install_config"), dict):
        m["install_config"] = json.dumps(m["install_config"])
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(m.get(k), str):
            try:
                m[k] = json.loads(m[k])
            except Exception:
                m[k] = [m[k]]
    return m


def load_train(train):
    recs = {}
    for ln in open(train):
        r = json.loads(ln)
        recs[r["metadata"]["instance_id"]] = r
    return recs


def never_solved(rollouts, recs):
    attempted, solved = set(), set()
    pat = re.compile(r"^(.+)_(\d+)_rewards\.json$")
    for f in glob.glob(os.path.join(rollouts, "**", "*_rewards.json"), recursive=True):
        m = pat.match(os.path.basename(f))
        if not m or m.group(1) not in recs:
            continue
        attempted.add(m.group(1))
        try:
            if json.load(open(f)).get("resolved") is True:
                solved.add(m.group(1))
        except Exception:
            pass
    for f in glob.glob(os.path.join(rollouts, "**", "*_*.json"), recursive=True):
        m = re.match(r"^(.+)_(\d+)\.json$", os.path.basename(f))
        if m and m.group(1) in recs:
            attempted.add(m.group(1))
    hard = sorted(attempted - solved)
    print(f"rollouts: {len(attempted)} training tasks attempted, {len(solved)} solved at least once, {len(hard)} never solved")
    return hard


def write_groups(ids, recs, out):
    groups, gold = [], open(f"{out}/gold_patches.jsonl", "w")
    for name, is_g, spec in (("rebench", lambda d: d != "AweAI-Team/Scale-SWE", REBENCH),
                             ("scale", lambda d: d == "AweAI-Team/Scale-SWE", SCALE)):
        gids = [i for i in ids if is_g(recs[i]["metadata"].get("dataset"))]
        if not gids:
            continue
        path = f"{out}/tasks_{name}.jsonl"
        with open(path, "w") as f:
            for i in gids:
                f.write(json.dumps(flatten(recs[i])) + "\n")
                gold.write(json.dumps({"instance_id": i, "patch": recs[i]["patch"]}) + "\n")
        groups.append(dict(name=name, subset=path, split="train", ids=gids, **spec))
    gold.close()
    return groups


ap = argparse.ArgumentParser()
ap.add_argument("source", choices=["random", "hard"])
ap.add_argument("--train", required=True, help="RL training data, work/rl/data/train.jsonl")
ap.add_argument("--sample", type=int, default=300)
ap.add_argument("--ids")
ap.add_argument("--rollouts")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
recs = load_train(a.train)
if a.source == "random":
    import random
    ids = sorted(json.load(open(a.ids))) if a.ids else sorted(random.sample(sorted(recs), a.sample))
else:
    if not a.rollouts:
        sys.exit("hard: --rollouts is required")
    ids = never_solved(a.rollouts, recs)
groups = write_groups(ids, recs, a.out)
n = sum(len(g["ids"]) for g in groups)
json.dump({"n_tasks": n, "groups": groups}, open(f"{a.out}/search_set.json", "w"), indent=1)
summary = ", ".join(f"{g['name']}={len(g['ids'])}" for g in groups)
print(f"search set: {n} tasks ({summary}) -> {a.out}/search_set.json")
