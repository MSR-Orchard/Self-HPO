#!/usr/bin/env python3
"""search_eval.py <candidate> <tag> [--ids-file FILE] [--passes 3] [--workers 48] — evaluate one harness on the search set.

<candidate> is a directory name under agents/ (or a path) holding agent.py + config.yaml.
--ids-file  run only the task ids in FILE; --passes  generation passes (default 3); --workers  concurrent tasks (default 48).
Writes runs/<candidate>_<tag>/verify_results_merged.json and prints the resolve rate.
"""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = os.environ["SELFHPO_REPO"]
FORK = f"{REPO}/mini-swe-agent"
TEMP, TOP_P, MAX_TOKENS, STEP_LIMIT, TIMEOUT = 0.95, 0.95, 16384, 250, 900
API_BASE = "http://127.0.0.1:{port}/v1"


def candidate_dir(name):
    p = Path(name)
    return p.resolve() if p.is_dir() else (HERE / "agents" / name).resolve()


def served_replicas():
    import urllib.request
    n = 0
    while n < 64:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{8000 + n}/v1/models", timeout=5).read()
        except Exception:
            break
        n += 1
    if n == 0:
        print("INFRA: no policy served on 127.0.0.1:8000 (start scripts/eval/serve_model.sh)", flush=True)
        sys.exit(3)
    return n


def eval_env(cand):
    env = dict(os.environ)
    env["SGLANG_PORT_BASE"] = "8000"
    env["SGLANG_NGPU"] = str(served_replicas())
    env["PYTHONPATH"] = ":".join([f"{FORK}/src"] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["HARNESS_AGENT_FILE"] = str(cand / "agent.py")
    env["SANDBOX_BASE_URL"] = os.environ["SANDBOX_BASE_URL"]
    env["SANDBOX_API_KEY"] = os.environ["SANDBOX_API_KEY"]
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    env["OPENAI_API_KEY"] = "EMPTY"
    return env


def id_filter(ids):
    return "^(" + "|".join(re.escape(i) for i in ids) + ")$"


def is_grading_error(e):
    if not isinstance(e, dict) or e.get("resolved") is True:
        return False
    return bool(e.get("error")) or any(s.get("error") or s.get("exit_code") in (-1, None)
                                        for s in (e.get("samples") or []))


def eval_group(cand, g, ids, out, n_pass, workers, log):
    env, filt = eval_env(cand), id_filter(ids)
    data = ["--subset", g["subset"], "--split", g["split"], "--filter", filt]
    over = ["-c", str(cand / "config.yaml"), "-c", g["env_config"],
            "-c", f"model.model_kwargs.temperature={TEMP}", "-c", f"model.model_kwargs.top_p={TOP_P}",
            "-c", f"model.model_kwargs.max_tokens={MAX_TOKENS}", "-c", "model.model_kwargs.api_key=EMPTY",
            "-c", f"model.model_kwargs.api_base={API_BASE}", "-c", f"agent.step_limit={STEP_LIMIT}",
            "-c", "environment.sandbox_timeout=1800"]
    Path(out).mkdir(parents=True, exist_ok=True)
    for p in range(1, n_pass + 1):
        log(f"[{cand.name}/{g['name']}] gen pass {p}/{n_pass} ({len(ids)} tasks)")
        subprocess.run(["mini-extra", g["runner"], *over, *data, "-o", out, "--n-samples", "1",
                        "--workers", str(workers), "--instance-timeout", str(TIMEOUT), "--no-preds"],
                       cwd=FORK, env=env, check=False)
    vj = Path(out) / "verify_results.json"
    for p in range(4):
        if p:
            d = json.loads(vj.read_text())
            bad = [k for k, e in d.items() if is_grading_error(e)]
            if not bad:
                break
            log(f"[{cand.name}/{g['name']}] re-grading {len(bad)} tasks with sandbox errors")
            vj.write_text(json.dumps({k: e for k, e in d.items() if k not in bad}))
        subprocess.run(["mini-extra", g["verifier"], "-c", g["env_config"], "-c", "environment.sandbox_timeout=1800",
                        *data, "--workers", "64", "--output", out], cwd=FORK, env=env, check=False)
        if not vj.exists():
            break
    if not vj.exists():
        log(f"INFRA: [{cand.name}/{g['name']}] the verifier wrote no results (sandbox pool down?)")
        sys.exit(3)
    d = json.loads(vj.read_text())
    left = sum(1 for e in d.values() if is_grading_error(e))
    if left:
        log(f"WARNING: [{cand.name}/{g['name']}] {left} tasks still have grading errors after 3 re-gradings (counted unresolved)")
    if len(d) < len(ids):
        log(f"WARNING: [{cand.name}/{g['name']}] {len(ids) - len(d)} of {len(ids)} tasks have no result (no trajectory; counted unresolved)")
    return d


def evaluate(name, tag, ids=None, n_pass=3, workers=48, log=print):
    cand = candidate_dir(name)
    ss = json.load(open(HERE / "search_set.json"))
    want = set(ids) if ids is not None else None
    out = HERE / "runs" / f"{cand.name}_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    merged = {}
    for g in ss["groups"]:
        gids = [i for i in g["ids"] if want is None or i in want]
        if gids:
            merged.update(eval_group(cand, g, gids, str(out / g["name"]), n_pass, workers, log))
    all_ids = sorted(want) if want is not None else [i for g in ss["groups"] for i in g["ids"]]
    merged = {i: merged.get(i, {"resolved": False}) for i in all_ids}
    (out / "verify_results_merged.json").write_text(json.dumps(merged))
    n = sum(1 for e in merged.values() if isinstance(e, dict) and e.get("resolved") is True)
    pct = 100.0 * n / max(len(all_ids), 1)
    log(f"[{cand.name}] resolve = {n}/{len(all_ids)} = {pct:.2f}%")
    return {"name": cand.name, "n_resolved": n, "n_total": len(all_ids), "resolve_pct": round(pct, 2), "out_dir": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("tag")
    ap.add_argument("--ids-file")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    ids = json.load(open(a.ids_file)) if a.ids_file else None
    r = evaluate(a.candidate, a.tag, ids, a.passes, a.workers, log=lambda *x: print(*x, flush=True))
    print(f"EVAL {r['name']}_{a.tag} resolve={r['n_resolved']}/{r['n_total']} = {r['resolve_pct']:.2f}%", flush=True)
