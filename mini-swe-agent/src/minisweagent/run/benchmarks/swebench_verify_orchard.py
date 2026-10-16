#!/usr/bin/env python3

"""Grade SWE-bench patches in the Orchard sandbox."""

import concurrent.futures
import json
import re
import tempfile
import time
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    RESET_FAILED,
    START_TEST_OUTPUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.images import get_swebench_docker_image_name
from minisweagent.run.benchmarks.utils.verify_common import (
    _parse_traj_filename,
    _upload_file,
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
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_orchard.yaml"
DOCKER_WORKDIR = "/testbed"

# ---- Evaluation ----

def evaluate_instance_orchard(
    instance: dict, patch: str, env_config: dict, *, logs_dir: Path | None = None, sample_idx: int = 0,
) -> dict:
    from minisweagent.environments.extra.orchard import OrchardEnvironment

    instance_id = instance["instance_id"]
    image = get_swebench_docker_image_name(instance)
    test_spec = make_test_spec(instance)

    cfg = {k: v for k, v in env_config.items() if k not in ("environment_class",)}
    cfg.update(image=image, cwd=DOCKER_WORKDIR)

    env = OrchardEnvironment(**cfg)
    output = ""
    exit_code = -1
    patch_applied = False
    try:
        eval_script = test_spec.eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")
        test_timeout = cfg.get("sandbox_timeout", 600)

        _upload_file(env, "/tmp/patch.diff", patch or "", timeout=30)
        _upload_file(env, "/tmp/eval.sh", eval_script, timeout=60)

        for git_apply_cmd in [
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            res = env.execute({"command": f"cd {DOCKER_WORKDIR} && {git_apply_cmd}"}, timeout=60)
            if res["returncode"] == 0:
                patch_applied = True
                break

        if not patch_applied and patch:
            logger.warning(f"{instance_id}: model patch failed to apply")

        result = env.execute({"command": "bash /tmp/eval.sh 2>&1"}, timeout=test_timeout)
        exit_code = result["returncode"]
        output = result["output"]
    finally:
        env.cleanup()

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(output)

    bad_codes = [c for c in [APPLY_PATCH_FAIL, RESET_FAILED] if c in output]
    if bad_codes or (START_TEST_OUTPUT not in output and END_TEST_OUTPUT not in output):
        if bad_codes:
            logger.warning(f"{instance_id}: bad signals in output: {bad_codes}")
        else:
            logger.warning(f"{instance_id}: START/END test markers not found in output")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(output)
        log_path = Path(fh.name)
    try:
        report = get_eval_report(
            test_spec=test_spec,
            prediction={
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: "minisweagent",
                KEY_PREDICTION: patch or "",
            },
            test_log_path=log_path,
            include_tests_status=True,
        )
    finally:
        log_path.unlink(missing_ok=True)

    instance_report = report.get(instance_id, {})
    tests_status = instance_report.get("tests_status", {})
    f2p = tests_status.get("FAIL_TO_PASS", {})
    p2p = tests_status.get("PASS_TO_PASS", {})
    f2p_passed = sorted(f2p.get("success", []))
    f2p_failed = sorted(f2p.get("failure", []))
    p2p_passed = sorted(p2p.get("success", []))
    p2p_failed = sorted(p2p.get("failure", []))

    return {
        "instance_id": instance_id,
        "resolved": bool(instance_report.get("resolved", False)),
        "exit_code": exit_code,
        "patch_applied": patch_applied,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_failed": f2p_failed,
        "pass_to_pass_passed": len(p2p_passed),
        "pass_to_pass_failed": p2p_failed,
        "parsed_tests_count": len(f2p_passed) + len(f2p_failed) + len(p2p_passed) + len(p2p_failed),
        "error": "",
    }


# ---- Trajectory scanning and I/O helpers ----

def load_patch_from_traj(traj_path: Path) -> str:
    data = json.loads(traj_path.read_text())
    return data.get("info", {}).get("submission", "") or ""


def scan_trajectories(output_dir: Path) -> dict[tuple[str, int], Path]:
    trajs: dict[tuple[str, int], Path] = {}
    for traj_file in sorted(output_dir.rglob("*.traj.json")):
        instance_id, sample_idx = _parse_traj_filename(traj_file.stem, ".traj")
        trajs[(instance_id, sample_idx)] = traj_file
    return trajs


# ---- Instance processing ----

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
        exit_status = "resolved" if result["resolved"] else "failed"
    except Exception as e:
        logger.error(f"Error verifying {task_label}: {e}", exc_info=True)
        result = {
            "instance_id": instance_id,
            "resolved": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "fail_to_pass_passed": [],
            "fail_to_pass_failed": [],
            "pass_to_pass_passed": 0,
            "pass_to_pass_failed": [],
        }
    progress_manager.on_instance_end(task_label, exit_status)
    update_results_file(results_path, instance_id, sample_idx, result)


def _print_verification_report(results: dict, results_path: Path) -> None:
    total_instances = len(results)
    resolved_instances = sum(1 for r in results.values() if r.get("resolved"))
    total_samples = sum(len(r.get("samples", [])) for r in results.values())
    resolved_samples = sum(
        sum(1 for s in r.get("samples", []) if s.get("resolved"))
        for r in results.values()
    )

    console.print(f"\n[bold]{'=' * 80}[/bold]")
    console.print("[bold cyan]VERIFICATION REPORT[/bold cyan]".center(80))
    console.print(f"[bold]{'=' * 80}[/bold]")
    console.print(f"\nResults file: [bold green]{results_path}[/bold green]")
    console.print("\n[bold]Pass@k (instance-level):[/bold]")
    console.print(f"  Resolved: {resolved_instances}/{total_instances} ({resolved_instances / max(total_instances, 1) * 100:.1f}%)")
    console.print("\n[bold]Per-sample:[/bold]")
    console.print(f"  Resolved: {resolved_samples}/{total_samples} ({resolved_samples / max(total_samples, 1) * 100:.1f}%)")

    resolved_ids = [iid for iid, r in results.items() if r.get("resolved")]
    failed_ids = [iid for iid, r in results.items() if not r.get("resolved")]

    if resolved_ids:
        console.print(f"\n[bold green]Resolved instances ({len(resolved_ids)}):[/bold green]")
        for iid in sorted(resolved_ids)[:20]:
            console.print(f"  ✓ {iid}")
        if len(resolved_ids) > 20:
            console.print(f"  ... and {len(resolved_ids) - 20} more")

    if failed_ids:
        console.print(f"\n[bold red]Failed instances ({len(failed_ids)}):[/bold red]")
        for iid in sorted(failed_ids)[:20]:
            console.print(f"  ✗ {iid}")
        if len(failed_ids) > 20:
            console.print(f"  ... and {len(failed_ids) - 20} more")

    console.print(f"\n[bold]{'=' * 80}[/bold]")


# ---- CLI ----

# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWE-bench subset or dataset path", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')", rich_help_panel="Data selection"),
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

    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent_verify.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split)}

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
    progress_manager = RunBatchProgressManager(
        n_jobs, output_path / f"verify_statuses_{time.time()}.yaml",
    )

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
                if time.time() - last_flush > 30:
                    flush_results_file(results_path)
                    last_flush = time.time()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Cancelling pending tasks...")
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
