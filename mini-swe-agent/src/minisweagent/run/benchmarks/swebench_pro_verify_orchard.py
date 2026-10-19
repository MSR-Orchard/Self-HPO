#!/usr/bin/env python3

"""Grade SWE-bench Pro patches with the official V2 verifier (v2/tasks/<instance_id>) in the Orchard sandbox.

Based on https://github.com/scaleapi/SWE-bench_Pro-os.

Environment variables:
  SWE_PRO_REPO             local clone of github.com/scaleapi/SWE-bench_Pro-os. [REQUIRED]
  SWE_PRO_ENTRY_TIMEOUT    verifier budget in seconds (default: the task's [verifier] timeout_sec).
  SWE_PRO_CREATE_TIMEOUT   sandbox creation / image pull timeout in seconds (default 1800).
  SWE_PRO_CPU              grading sandbox cpu (default "4").
  SWE_PRO_MEM              grading sandbox memory (default "16Gi").
"""

import base64
import concurrent.futures
import json
import os
import re
import time
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.verify_common import (
    flush_results_file,
    get_existing_verified_samples,
    get_results_cache,
    update_results_file,
)
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import recursive_merge

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "swebench_pro": "ScaleAI/SWE-bench_Pro",
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_pro_orchard.yaml"

REPO = os.environ.get("SWE_PRO_REPO", "")
DEFAULT_VERIFIER_TIMEOUT = 3000


def _task_dir(instance_id: str) -> Path:
    return Path(REPO) / "v2" / "tasks" / instance_id


def _verifier_timeout(instance_id: str) -> int:
    m = re.search(r"\[verifier\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", (_task_dir(instance_id) / "task.toml").read_text())
    return int(float(m.group(1))) if m else DEFAULT_VERIFIER_TIMEOUT


# ---- Sandbox I/O helpers ----

_B64_CHUNK = 50_000


def _exec(env, command: str, timeout: int) -> dict:
    out = env.execute({"command": command}, timeout=timeout)
    if out.get("extra", {}).get("exception_type"):
        raise RuntimeError(out["extra"].get("exception") or out.get("exception_info", ""))
    return out


def _write_file_to_sandbox(env, path: str, content: str | bytes) -> None:
    encoded = base64.b64encode(content if isinstance(content, bytes) else content.encode()).decode()
    _exec(env, f"mkdir -p $(dirname {path}); : > {path}", 90)
    for i in range(0, len(encoded), _B64_CHUNK):
        _exec(env, f"echo -n '{encoded[i : i + _B64_CHUNK]}' | base64 -d >> {path}", 90)


def map_row_to_instance(row: dict) -> dict:
    return {
        "instance_id": row["instance_id"],
        "image_name": row["docker_image"],
        "patch": row.get("patch") or "",
        "repo_language": row["repo_language"],
    }


# ---- Evaluation ----


def evaluate_instance_orchard(
    instance: dict,
    patch: str,
    env_config: dict,
    *,
    logs_dir: Path | None = None,
    sample_idx: int = 0,
) -> dict:
    from minisweagent.environments.extra.orchard import OrchardEnvironment

    instance_id = instance["instance_id"]
    res = {
        "instance_id": instance_id,
        "resolved": False,
        "patch_applied": None,
        "error": None,
    }

    tests_dir = _task_dir(instance_id) / "tests"
    if not tests_dir.is_dir():
        res["error"] = f"missing_task:{tests_dir}"
        return res
    test_files = sorted(f for f in tests_dir.iterdir() if f.is_file())

    cfg = {k: v for k, v in env_config.items() if k not in ("environment_class",)}
    cfg.update(
        image=instance["image_name"],
        cwd="/",
        cpu=os.environ.get("SWE_PRO_CPU", "4"),
        memory=os.environ.get("SWE_PRO_MEM", "16Gi"),
        login_shell=False,
        env={},
        forward_env=[],
    )
    cfg["sandbox_timeout"] = int(os.environ.get("SWE_PRO_CREATE_TIMEOUT", "1800"))
    entry_timeout = int(os.environ.get("SWE_PRO_ENTRY_TIMEOUT") or _verifier_timeout(instance_id))

    try:
        env = OrchardEnvironment(**cfg)
    except Exception as e:
        msg = str(e)
        res["error"] = "timeout" if ("timeout" in msg.lower() or "timed out" in msg.lower()) else msg[:180]
        return res

    log = ""
    try:
        _write_file_to_sandbox(env, "/workspace/patch.diff", patch or "")
        for f in test_files:
            _write_file_to_sandbox(env, f"/tests/{f.name}", f.read_bytes())
        out = _exec(
            env,
            "D=/app; [ -d /app ] || D=/testbed; cd $D && git config --global --add safe.directory $D; "
            "if [ -s /workspace/patch.diff ]; then { git apply --verbose /workspace/patch.diff "
            "|| git apply --3way /workspace/patch.diff || patch --fuzz=3 -p1 -i /workspace/patch.diff; } "
            "&& echo __APPLY_OK__ || echo __APPLY_FAIL__; else echo __EMPTY__; fi",
            300,
        )
        applied = out.get("output", "")
        res["patch_applied"] = True if "__APPLY_OK__" in applied else (None if "__EMPTY__" in applied else False)
        log += applied
        _exec(env, "mkdir -p /logs/verifier; bash /tests/test.sh > /logs/verifier/stdout.log 2>&1", entry_timeout)
        log += _exec(env, "tail -c 20000 /logs/verifier/stdout.log", 60).get("output", "")
        reward = _exec(env, "cat /logs/verifier/reward.txt 2>/dev/null || echo MISSING", 60).get("output", "").strip()
        if reward == "1":
            res["resolved"] = True
        elif reward != "0":
            res["error"] = "reward:" + reward[:60]
    except Exception as e:
        msg = str(e)
        res["error"] = "timeout" if ("timeout" in msg.lower() or "timed out" in msg.lower()) else msg[:180]
    finally:
        env.cleanup()

    if logs_dir and log:
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(log)

    return res


# ---- Trajectory scanning and I/O helpers ----


def load_patch_from_traj(traj_path: Path) -> str:
    data = json.loads(traj_path.read_text())
    return data.get("info", {}).get("submission", "") or ""


def scan_trajectories(output_dir: Path) -> dict[tuple[str, int], Path]:
    trajs: dict[tuple[str, int], Path] = {}
    for traj_file in sorted(output_dir.rglob("*.traj.json")):
        name = traj_file.stem.removesuffix(".traj")
        if ".sample_" in name:
            instance_id, _, idx = name.rpartition(".sample_")
            sample_idx = int(idx)
        else:
            instance_id = name
            sample_idx = 0
        trajs[(instance_id, sample_idx)] = traj_file
    return trajs


def process_instance(
    instance: dict,
    traj_path: Path | None,
    results_path: Path,
    env_config: dict,
    logs_dir: Path,
    progress_manager: RunBatchProgressManager,
    sample_idx: int = 0,
    golden_patch: str | None = None,
) -> None:
    instance_id = instance["instance_id"]
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id
    progress_manager.on_instance_start(task_label)
    progress_manager.update_instance_status(task_label, "Starting sandbox")
    result: dict = {}
    exit_status = "error"
    try:
        patch = golden_patch if golden_patch is not None else load_patch_from_traj(traj_path)
        progress_manager.update_instance_status(task_label, "Running tests")
        result = evaluate_instance_orchard(instance, patch, env_config, logs_dir=logs_dir, sample_idx=sample_idx)
        exit_status = "resolved" if result.get("resolved") else "failed"
    except Exception as e:
        logger.error(f"Error verifying {task_label}: {e}", exc_info=True)
        result = {
            "instance_id": instance_id,
            "resolved": False,
            "patch_applied": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    progress_manager.on_instance_end(task_label, exit_status)
    update_results_file(results_path, instance_id, sample_idx, result)


def _print_verification_report(results: dict, results_path: Path) -> None:
    total_instances = len(results)
    resolved_instances = sum(1 for r in results.values() if r.get("resolved"))
    total_samples = sum(len(r.get("samples", [])) for r in results.values())
    resolved_samples = sum(sum(1 for s in r.get("samples", []) if s.get("resolved")) for r in results.values())

    console.print(f"\n[bold]{'=' * 80}[/bold]")
    console.print("[bold cyan]VERIFICATION REPORT[/bold cyan]".center(80))
    console.print(f"[bold]{'=' * 80}[/bold]")
    console.print(f"\nResults file: [bold green]{results_path}[/bold green]")
    console.print("\n[bold]Pass@k (instance-level):[/bold]")
    console.print(
        f"  Resolved: {resolved_instances}/{total_instances} ({resolved_instances / max(total_instances, 1) * 100:.1f}%)"
    )
    console.print("\n[bold]Per-sample:[/bold]")
    console.print(
        f"  Resolved: {resolved_samples}/{total_samples} ({resolved_samples / max(total_samples, 1) * 100:.1f}%)"
    )

    resolved_ids = [iid for iid, r in results.items() if r.get("resolved")]
    failed_ids = [iid for iid, r in results.items() if not r.get("resolved")]

    if resolved_ids:
        console.print(f"\n[bold green]Resolved instances ({len(resolved_ids)}):[/bold green]")
        for iid in sorted(resolved_ids)[:20]:
            console.print(f"  ✓ {iid}")
        if len(resolved_ids) > 20:
            console.print(f"  ... and {len(resolved_ids) - 20} more")

    if failed_ids:
        categories: dict[str, list[str]] = {}
        for iid in failed_ids:
            entry = results[iid]
            samples = entry.get("samples", [entry])
            for s in samples:
                if s.get("resolved"):
                    continue
                error = s.get("error") or ""
                if error:
                    cat = error[:80]
                elif s.get("patch_applied") is False:
                    cat = "patch_apply_failed"
                elif s.get("n_passed", -1) == 0:
                    cat = "no_tests_passed"
                else:
                    cat = "tests_failed"
                categories.setdefault(cat, []).append(iid)
                break

        console.print(f"\n[bold red]Failed instances ({len(failed_ids)}):[/bold red]")
        console.print("\n  [bold]Failure breakdown:[/bold]")
        for cat, ids in sorted(categories.items(), key=lambda x: -len(x[1])):
            console.print(f"    {cat}: {len(ids)}")

    console.print(f"\n[bold]{'=' * 80}[/bold]")


# ---- CLI ----


# fmt: off
@app.command()
def main(
    subset: str = typer.Option("swebench_pro", "--subset", help="SWE-bench Pro subset or dataset path", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')", rich_help_panel="Data selection"),
    language: str = typer.Option("", "--language", help="Filter by programming language (e.g., 'python')", rich_help_panel="Data selection"),
    output: str = typer.Option(..., "-o", "--output", help="Directory containing trajectory files to verify", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files/specs for orchard environment", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-verify instances already in results file", rich_help_panel="Data selection"),
    logs_dir: str = typer.Option("", "--logs-dir", help="Directory to save test logs (default: <output>/verify_logs)", rich_help_panel="Advanced"),
    verify_output: str = typer.Option("", "--verify-output", help="Output results file (default: <output>/verify_results.json)", rich_help_panel="Basic"),
    golden: bool = typer.Option(False, "--golden", help="Evaluate the dataset's gold patches instead of model patches", rich_help_panel="Advanced"),
    report_only: bool = typer.Option(False, "--report-only", help="Only print report from existing results file without running verification", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    if golden:
        default_results = output_path / "verify_results_golden.json"
        default_logs = output_path / "verify_logs_golden"
    else:
        default_results = output_path / "verify_results.json"
        default_logs = output_path / "verify_logs"
    results_path = Path(verify_output) if verify_output else default_results
    logs_path = Path(logs_dir) if logs_dir else default_logs

    if report_only:
        if not results_path.exists():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(1)
        results = json.loads(results_path.read_text())
        if filter_spec or slice_spec:
            ids = sorted(results.keys())
            if filter_spec:
                ids = [iid for iid in ids if re.match(filter_spec, iid)]
            if slice_spec:
                values = [int(x) if x else None for x in slice_spec.split(":")]
                ids = ids[slice(*values)]
            results = {iid: results[iid] for iid in ids}
        _print_verification_report(results, results_path)
        return

    if not REPO:
        console.print("[red]SWE_PRO_REPO is required: path to a local clone of scaleapi/SWE-bench_Pro-os[/red]")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent_verify.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = {row["instance_id"]: map_row_to_instance(row) for row in load_dataset(dataset_path, split=split)}

    if golden:
        instance_ids = sorted(instances.keys())
    else:
        logger.info(f"Scanning trajectories in {output_path}")
        traj_map = scan_trajectories(output_path)
        if not traj_map:
            console.print(f"[yellow]No trajectory files found in {output_path}[/yellow]")
            raise typer.Exit(1)
        logger.info(f"Found {len(traj_map)} trajectory files")

        traj_instance_ids = sorted({iid for iid, _ in traj_map})
        instance_ids = [iid for iid in traj_instance_ids if iid in instances]
        missing = [iid for iid in traj_instance_ids if iid not in instances]
        if missing:
            logger.warning(f"{len(missing)} trajectory instance_ids not found in dataset: {missing[:5]}...")

    if language:
        instance_ids = [iid for iid in instance_ids if instances[iid].get("repo_language", "").lower() == language.lower()]
    if filter_spec:
        instance_ids = [iid for iid in instance_ids if re.match(filter_spec, iid)]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instance_ids = instance_ids[slice(*values)]

    if golden:
        work_items_golden = [(iid, 0) for iid in instance_ids]
        if not redo_existing:
            existing = get_existing_verified_samples(results_path)
            before = len(work_items_golden)
            work_items_golden = [(iid, sidx) for iid, sidx in work_items_golden if (iid, sidx) not in existing]
            if before > len(work_items_golden):
                logger.info(f"Skipping {before - len(work_items_golden)} already-verified instances")
        logger.info(f"[golden] Verifying {len(work_items_golden)} instances with golden patches...")
    else:
        instance_id_set = set(instance_ids)
        work_items = [(iid, sidx, traj_map[(iid, sidx)]) for iid, sidx in sorted(traj_map) if iid in instance_id_set]

        if not redo_existing:
            existing = get_existing_verified_samples(results_path)
            before = len(work_items)
            work_items = [(iid, sidx, tp) for iid, sidx, tp in work_items if (iid, sidx) not in existing]
            if before > len(work_items):
                logger.info(f"Skipping {before - len(work_items)} already-verified samples")

        logger.info(f"Verifying {len(work_items)} samples across {len({iid for iid, _, _ in work_items})} instances...")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    config = recursive_merge(*configs)
    env_config = config.get("environment", {})

    n_jobs = len(work_items_golden) if golden else len(work_items)
    progress_manager = RunBatchProgressManager(n_jobs, output_path / f"verify_statuses_{time.time()}.yaml")

    if workers <= 1:
        live_ctx = Live(progress_manager.render_group, refresh_per_second=4)
    else:
        progress_manager._main_progress_bar.live.redirect_stdout = False
        progress_manager._main_progress_bar.live.redirect_stderr = False
        live_ctx = progress_manager._main_progress_bar

    with live_ctx:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {}
        if golden:
            for iid, s_idx in work_items_golden:
                future = executor.submit(
                    process_instance,
                    instances[iid], None, results_path, env_config, logs_path, progress_manager, s_idx,
                    golden_patch=instances[iid].get("patch", ""),
                )
                futures[future] = iid
        else:
            for iid, s_idx, traj_path in work_items:
                task_label = f"{iid}#{s_idx}" if s_idx > 0 else iid
                future = executor.submit(
                    process_instance,
                    instances[iid], traj_path, results_path, env_config, logs_path, progress_manager, s_idx,
                )
                futures[future] = task_label
        last_flush = time.time()
        try:
            done: set = set()
            while len(done) < len(futures):
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=2.0):
                        done.add(future)
                        try:
                            future.result()
                        except concurrent.futures.CancelledError:
                            pass
                        except Exception as e:
                            task_label = futures[future]
                            logger.error(f"Uncaught error for {task_label}: {e}", exc_info=True)
                            progress_manager.on_uncaught_exception(task_label, e)
                except concurrent.futures.TimeoutError:
                    pass
                if time.time() - last_flush > 300:
                    flush_results_file(results_path)
                    last_flush = time.time()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Cancelling pending tasks and waiting for in-flight tasks to finish...")
            console.print("\n[yellow]Interrupted! Waiting for in-flight tasks to save results (up to 30s)...[/yellow]")
            executor.shutdown(wait=False, cancel_futures=True)
            in_flight = [f for f in futures if not f.done()]
            if in_flight:
                concurrent.futures.wait(in_flight, timeout=30)
            flush_results_file(results_path)
            console.print(f"[green]Partial results saved to {results_path}[/green]")
            console.print("[green]Re-run the same command to resume from where you left off.[/green]")
            raise SystemExit(1)
        finally:
            executor.shutdown(wait=False)

    flush_results_file(results_path)
    if get_results_cache():
        _print_verification_report(get_results_cache(), results_path)


if __name__ == "__main__":
    app()
