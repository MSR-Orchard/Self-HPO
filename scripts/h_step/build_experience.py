#!/usr/bin/env python3
"""build_experience.py — machine-computed failure analysis of the parent harness's runs on the search set (no LLM).

Writes:
  experience/gold_archetypes.md   gold-fix shapes vs what the model produced
  experience/coverage_gap.json    per failed task: required source files vs files the model changed
  experience/precision_gap.md     failures that changed every required file and still failed
  experience/submission_loss.md   failures with a `diff --git` in the trajectory but none in the final patch
  experience/stats.json           the headline numbers (key names state each denominator)
"""
import json, os, glob, re, collections

B = os.path.dirname(os.path.abspath(__file__))
GOLD = f"{B}/data/gold_patches.jsonl"
SEEDS = sorted(glob.glob(f"{B}/runs/parent_seed_r[0-9]*"))
EXP = f"{B}/experience"
os.makedirs(EXP, exist_ok=True)

_TEST_DIR = re.compile(r'(^|/)(tests?|testing|__tests__|specs?)/', re.I)
_TEST_FILE = re.compile(r'(^test_|_tests?\.|\.(test|spec)\.|^conftest\.py$)', re.I)
IS_TEST = lambda f: bool(_TEST_DIR.search(f) or _TEST_FILE.search(f.rsplit('/', 1)[-1]))

def src_files(diff):
    fs = set(re.findall(r'^diff --git a/(\S+)', diff, re.M)) or set(re.findall(r'^\+\+\+ b/(\S+)', diff, re.M))
    return {f for f in fs if not IS_TEST(f)}

def added_removed(diff):
    add = [l[1:] for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++')]
    rem = [l[1:] for l in diff.splitlines() if l.startswith('-') and not l.startswith('---')]
    return add, rem

def classify(diff):
    add, rem = added_removed(diff)
    A, R = "\n".join(add), "\n".join(rem)
    tags = set(); nf = len(src_files(diff))
    if re.search(r'^\s*(def|class)\s+\w+', A, re.M) and not re.search(r'^\s*(def|class)\s+\w+', R, re.M): tags.add("new-callable")
    if re.search(r'^\s*def\s+(\w+)\s*\(', A, re.M) and re.search(r'^\s*def\s+(\w+)\s*\(', R, re.M): tags.add("signature-change")
    if re.search(r'\b(if|elif)\b.*\b(is None|None|not |isinstance|hasattr|len\()', A) and len(add) <= 25: tags.add("guard/edge-case")
    if re.search(r'\braise\b|\bexcept\b|\btry:', A): tags.add("error-handling")
    if re.search(r'^\s*(import|from)\s+\w', A, re.M): tags.add("new-import")
    if nf > 1: tags.add("multi-file")
    if re.search(r'@\w+', A) and not re.search(r'@\w+', R): tags.add("decorator")
    if len(add) > 60: tags.add("large-addition")
    if not tags: tags.add("in-place-logic-edit")
    return tags

def per_file_hunks(diff):
    if not diff or not isinstance(diff, str): return {}
    out, cur = {}, None
    for line in diff.splitlines():
        m = re.match(r'^diff --git a/(\S+)', line)
        if m: cur = m.group(1); out.setdefault(cur, 0); continue
        m = re.match(r'^\+\+\+ b/(\S+)', line)
        if m and cur is None: cur = m.group(1); out.setdefault(cur, 0); continue
        if line.startswith('@@ ') and cur is not None: out[cur] += 1
    return {f: h for f, h in out.items() if not IS_TEST(f)}

gold = {}
for ln in open(GOLD):
    ln = ln.strip()
    if not ln: continue
    try: d = json.loads(ln)
    except Exception: continue
    iid = d.get("instance_id"); p = d.get("patch")
    if iid and p: gold[iid] = p

resolved = {}
for sd in SEEDS:
    for iid, e in json.load(open(f"{sd}/verify_results_merged.json")).items():
        resolved[iid] = resolved.get(iid, False) or bool(isinstance(e, dict) and e.get("resolved") is True)
TEST_CMD = re.compile(r"\b(pytest|py\.test|unittest|tox|nosetests|go test|cargo test|npm (run )?test|yarn test|mvn test|gradle test|rspec|phpunit|ctest|runtests?\.py)\b")

def last_test_rc(msgs):
    pending, rc = {}, None
    for m in msgs:
        if m.get("role") == "assistant":
            for a in ((m.get("extra") or {}).get("actions") or []):
                if TEST_CMD.search(str(a.get("command", ""))):
                    pending[a.get("tool_call_id")] = True
        elif m.get("role") == "tool" and pending.pop(m.get("tool_call_id"), False):
            r = re.search(r"<returncode>(-?\d+)</returncode>", str(m.get("content")))
            if r:
                rc = int(r.group(1))
    return rc

model = {}; traj_has_diff = {}; own_test = {}
for f in glob.glob(f"{SEEDS[0]}/*/*/*.traj.json"):
    try: j = json.load(open(f))
    except Exception: continue
    iid = j.get("instance_id") or os.path.basename(os.path.dirname(f))
    sub = ((j.get("info") or {}).get("submission")) or ""
    if iid not in model: model[iid] = sub
    try: msgs_txt = json.dumps(j.get("messages") or [])
    except Exception: msgs_txt = ""
    traj_has_diff[iid] = ("diff --git" in msgs_txt)
    own_test[iid] = last_test_rc(j.get("messages") or [])

failed = sorted([i for i in gold if not resolved.get(i, False)])
failed_traj = [i for i in failed if i in own_test]
own_pass = sum(1 for i in failed_traj if own_test[i] == 0)
own_never = sum(1 for i in failed_traj if own_test[i] is None)

# ---- gold_archetypes.md ----
gold_tags = collections.Counter(); model_tags = collections.Counter(); miss = collections.Counter(); n = 0
for iid in failed:
    if iid not in gold or iid not in model: continue
    n += 1
    gt, mt = classify(gold[iid]), classify(model[iid])
    for t in gt:
        gold_tags[t] += 1
        if t not in mt: miss[t] += 1
    for t in mt: model_tags[t] += 1
lines = ["# Gold-patch archetypes vs the parent harness's patches (regex over diffs)",
         f"# Failed search-set tasks with both a gold and a model patch: {n}",
         "# in_gold / in_model: tasks whose gold / model patch has the archetype; gold_NOT_model: in the gold patch but not the model's.\n", "```",
         f"{'archetype':22s} {'in_gold':>8s} {'in_model':>9s} {'gold_NOT_model':>15s} {'miss_rate':>9s}"]
for t, c in gold_tags.most_common():
    mr = 100.0 * miss[t] / c if c else 0
    lines.append(f"{t:22s} {c:8d} {model_tags[t]:9d} {miss[t]:15d} {mr:8.0f}%")
lines += ["```", "", "The archetype classifier is a coarse regex."]
open(f"{EXP}/gold_archetypes.md", "w").write("\n".join(lines) + "\n")

# ---- coverage_gap.json ----
cov = {}
for iid in failed:
    if iid not in gold or iid not in model: continue
    gf = per_file_hunks(gold[iid]); mf = per_file_hunks(model[iid])
    gset, mset = set(gf), set(mf)
    cov[iid] = {"gold_files": sorted(gset), "model_files": sorted(mset),
                "missed_files": sorted(gset - mset), "gold_nfiles": len(gset), "model_nfiles": len(mset)}
json.dump(cov, open(f"{EXP}/coverage_gap.json", "w"), indent=1)
touched_ge1 = sum(1 for v in cov.values() if v["gold_files"] and (set(v["gold_files"]) & set(v["model_files"])))
touched_all = sum(1 for v in cov.values() if v["gold_files"] and not v["missed_files"])
tot = sum(1 for v in cov.values() if v["gold_files"])

# ---- precision_gap.md ----
pl = ["# Failed search-set tasks: required source files (from the gold patch) vs files the model changed\n", "```",
      f"failed tasks with gold source files : {tot}",
      f"  touched >=1 required file          : {touched_ge1} ({100*touched_ge1//max(tot,1)}%)",
      f"  touched ALL required files, FAILED : {touched_all} ({100*touched_all//max(tot,1)}%)",
      f"  missed >=1 required file           : {tot-touched_all} ({100*(tot-touched_all)//max(tot,1)}%)",
      "```", "",
      "# The model's own last test run on the failed tasks that have a trajectory", "```",
      f"failed tasks with a trajectory : {len(failed_traj)}",
      f"  never ran tests              : {own_never} ({100*own_never//max(len(failed_traj),1)}%)",
      f"  last test run passed         : {own_pass} ({100*own_pass//max(len(failed_traj),1)}%)",
      f"  last test run failed         : {len(failed_traj)-own_pass-own_never}", "```"]
open(f"{EXP}/precision_gap.md", "w").write("\n".join(pl) + "\n")

# ---- submission_loss.md ----
dropped = [i for i in failed if traj_has_diff.get(i) and "diff --git" not in (model.get(i) or "")]
sl = ["# Failed search-set tasks where a `diff --git` appears in the model's trajectory but the final submitted patch contains none\n",
      "```",
      f"failed search-set tasks                        : {len(failed)}",
      f"  diff in trajectory, none in final patch    : {len(dropped)} ({100*len(dropped)//max(len(failed),1)}%)",
      "```"]
open(f"{EXP}/submission_loss.md", "w").write("\n".join(sl) + "\n")

print(f"BUILT experience/ : {n} failed classified; coverage {len(cov)} tasks; "
      f"touched_all(precision)={touched_all}/{tot} ({100*touched_all//max(tot,1)}%); "
      f"missed>=1(coverage)={tot-touched_all} ({100*(tot-touched_all)//max(tot,1)}%); "
      f"submission_loss={len(dropped)}/{len(failed)}")

json.dump({"n_failed": len(failed), "n_failed_with_gold_files": tot, "n_failed_with_traj": len(failed_traj),
           "coverage_miss_pct_of_failed_with_gold_files": 100 * (tot - touched_all) // max(tot, 1),
           "precision_pct_of_failed_with_gold_files": 100 * touched_all // max(tot, 1),
           "submission_loss_n": len(dropped), "submission_loss_pct_of_failed": 100 * len(dropped) // max(len(failed), 1),
           "own_test_passed_pct_of_failed_with_traj": 100 * own_pass // max(len(failed_traj), 1),
           "own_test_never_pct_of_failed_with_traj": 100 * own_never // max(len(failed_traj), 1)},
          open(f"{EXP}/stats.json", "w"), indent=1)
