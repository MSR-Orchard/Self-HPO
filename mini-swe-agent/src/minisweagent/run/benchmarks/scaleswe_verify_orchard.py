#!/usr/bin/env python3

"""Grade Scale-SWE patches in the Orchard sandbox.

Based on https://github.com/AweAI-Team/AweAgent/blob/main/awe_agent/tasks/beyond_swe/evaluator.py.
"""

import base64
import concurrent.futures
import json
import re
import time
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.images import get_scaleswe_docker_image_name
from minisweagent.run.benchmarks.utils.loading import load_instances
from minisweagent.run.benchmarks.utils.verify_common import (
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
    "scaleswe": "AweAI-Team/Scale-SWE",
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "scaleswe_orchard.yaml"


# ---- Test ID parsing ----

def _parse_test_ids(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
        if isinstance(parsed, str) and parsed:
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [raw]


# ---- Pytest summary parsing ----

_COUNT_RE = re.compile(r"(\d+)\s+(\w+)")
_SUMMARY_LINE_RE = re.compile(r"\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed)\b")
_LABEL_MAP = {
    "passed": "passed", "pass": "passed",
    "failed": "failed", "fail": "failed", "failure": "failed", "failures": "failed",
    "error": "errors", "errors": "errors",
    "skipped": "skipped", "skip": "skipped",
}


def _parse_pytest_summary(output: str) -> dict[str, int]:
    summary = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    summary_line = ""
    for line in output.splitlines():
        if _SUMMARY_LINE_RE.search(line):
            summary_line = line
    if not summary_line:
        return summary
    for m in _COUNT_RE.finditer(summary_line):
        label = m.group(2).lower()
        field = _LABEL_MAP.get(label)
        if field:
            summary[field] = int(m.group(1))
    return summary


# ---- JUnit XML parsing ----

def _normalize_for_match(s: str) -> str:
    return s.replace(".py", "").replace("/", ".").replace("::", ".").strip(".")


def _parse_junit_xml(xml_content: str, expected_tests: list[str]) -> tuple[bool, dict]:
    import xml.etree.ElementTree as ET

    exact_set = set(expected_tests)
    norm_map = {_normalize_for_match(t): t for t in expected_tests}
    fp_map = {re.sub(r"\s+", "", _normalize_for_match(t)): t for t in expected_tests}

    matched: dict[str, str] = {}
    found = set()

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return False, {"xml_error": str(e)}

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        file_attr = tc.get("file", "")
        if tc.find("skipped") is not None:
            continue
        status = "failed" if tc.find("failure") is not None or tc.find("error") is not None else "passed"

        for candidate in [
            f"{file_attr}::{name}" if file_attr else None,
        ]:
            if candidate and candidate in exact_set:
                matched[candidate] = status
                found.add(candidate)
                break
        else:
            norm = _normalize_for_match(f"{classname}.{name}")
            if norm in norm_map:
                matched[norm_map[norm]] = status
                found.add(norm_map[norm])
            else:
                fp = re.sub(r"\s+", "", norm)
                if fp in fp_map:
                    matched[fp_map[fp]] = status
                    found.add(fp_map[fp])
                else:
                    fallback = f"{classname.replace('.', '/')}.py::{name}"
                    if fallback in exact_set:
                        matched[fallback] = status
                        found.add(fallback)

    unmatched = [t for t in expected_tests if t not in found]
    all_passed = len(found) > 0 and all(v == "passed" for v in matched.values()) and len(unmatched) == 0
    return all_passed, {"matched": matched, "unmatched": unmatched, "total_matched": len(matched)}


# ---- Injected pytest runner script ----

PYTEST_RUNNER_SCRIPT = '''\
import json, sys, os
import pytest

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    test_ids = config["test_ids"]
    xml_path = config.get("xml_path", "/tmp/_awe_test_results.xml")
    sys.path.insert(0, os.getcwd())
    sys.argv = ["pytest"]
    args = ["-vv", f"--junitxml={xml_path}", "-o", "addopts=", "--rootdir=."] + test_ids
    ret = pytest.main(args)
    print("<pytest>true</pytest>" if ret == 0 else "<pytest>false</pytest>")
'''


# ---- Evaluation ----

def evaluate_instance_orchard(
    instance: dict, patch: str, env_config: dict, *, logs_dir: Path | None = None, sample_idx: int = 0,
) -> dict:
    from minisweagent.environments.extra.orchard import OrchardEnvironment

    instance_id = instance["instance_id"]
    image = get_scaleswe_docker_image_name(instance)
    workdir = instance.get("workdir", "/")
    task_type = instance.get("task_type", "domainfix")

    f2p_patch = instance.get("f2p_patch") or instance.get("test_patch", "")
    f2p_script = instance.get("f2p_script", "")
    f2p_ids = _parse_test_ids(instance.get("FAIL_TO_PASS"))
    p2p_ids = _parse_test_ids(instance.get("PASS_TO_PASS"))

    cfg = {k: v for k, v in env_config.items() if k not in ("environment_class",)}
    cfg.update(image=image, cwd=workdir)

    env = OrchardEnvironment(**cfg)
    output = ""
    exit_code = -1
    patch_applied = False
    f2p_patch_applied = False

    try:
        test_timeout = cfg.get("sandbox_timeout", 600)

        pre_commands = instance.get("pre_commands")
        if pre_commands:
            if isinstance(pre_commands, list):
                pre_commands = " && ".join(pre_commands)
            env.execute({"command": pre_commands}, cwd=workdir, timeout=300)

        if patch and patch.strip():
            _upload_file(env, "/tmp/patch.diff", patch, timeout=30)
            for git_apply_cmd in [
                "git apply --verbose /tmp/patch.diff",
                "git apply --verbose --reject /tmp/patch.diff",
                "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
            ]:
                res = env.execute({"command": f"cd {workdir} && {git_apply_cmd}"}, timeout=60)
                if res["returncode"] == 0:
                    patch_applied = True
                    break
            if not patch_applied:
                logger.warning(f"{instance_id}: model patch failed to apply")

        if task_type == "doc2repo":
            result = _eval_doc2repo(instance, env, workdir, test_timeout)
            exit_code = result.pop("exit_code", -1)
            output = result.pop("output", "")
        else:
            if f2p_patch:
                _upload_file(env, "/tmp/f2p_patch.diff", f2p_patch, timeout=30)
                for git_apply_cmd in [
                    "git apply --verbose /tmp/f2p_patch.diff",
                    "git apply --verbose --reject /tmp/f2p_patch.diff",
                    "patch --batch --fuzz=5 -p1 -i /tmp/f2p_patch.diff",
                ]:
                    res = env.execute({"command": f"cd {workdir} && {git_apply_cmd}"}, timeout=60)
                    if res["returncode"] == 0:
                        f2p_patch_applied = True
                        break
                if not f2p_patch_applied:
                    logger.warning(f"{instance_id}: f2p_patch failed to apply")

            if f2p_script:
                _upload_file(env, f"{workdir}/test_fail_to_pass.py", f2p_script, timeout=30)

            all_tests = f2p_ids + p2p_ids
            result = _run_tests_with_runner(env, workdir, all_tests, test_timeout)
            exit_code = result.pop("exit_code", -1)
            output = result.pop("output", "")
    finally:
        env.cleanup()

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(output)

    if task_type == "doc2repo":
        return {
            "instance_id": instance_id,
            "exit_code": exit_code,
            "patch_applied": patch_applied,
            "error": "",
            **result,
        }

    all_passed = result.get("all_passed", False)
    details = result.get("details", {})
    matched = details.get("matched", {})

    f2p_passed_list = sorted(t for t in f2p_ids if matched.get(t) == "passed")
    f2p_failed_list = sorted(t for t in f2p_ids if matched.get(t) != "passed")
    p2p_passed_list = sorted(t for t in p2p_ids if matched.get(t) == "passed")
    p2p_failed_list = sorted(t for t in p2p_ids if matched.get(t) != "passed")

    resolved = all_passed or (len(f2p_ids) > 0 and not f2p_failed_list and not p2p_failed_list)

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "exit_code": exit_code,
        "patch_applied": patch_applied,
        "f2p_patch_applied": f2p_patch_applied,
        "fail_to_pass_passed": f2p_passed_list,
        "fail_to_pass_failed": f2p_failed_list,
        "pass_to_pass_passed": len(p2p_passed_list),
        "pass_to_pass_failed": p2p_failed_list,
        "parsed_tests_count": len(matched),
        "error": "",
    }


def _run_tests_with_runner(env, workdir: str, test_ids: list[str], timeout: int) -> dict:
    if not test_ids:
        return {"all_passed": False, "output": "", "exit_code": -1, "details": {"error": "no_test_ids"}}

    _upload_file(env, "/tmp/_awe_pytest_runner.py", PYTEST_RUNNER_SCRIPT, timeout=30)
    config_data = json.dumps({"test_ids": test_ids, "xml_path": "/tmp/_awe_test_results.xml"})
    _upload_file(env, "/tmp/_awe_test_config.json", config_data, timeout=30)

    result = env.execute(
        {"command": f"cd {workdir} && python /tmp/_awe_pytest_runner.py /tmp/_awe_test_config.json"},
        timeout=timeout,
    )
    raw_output = result["output"]
    exit_code = result["returncode"]

    if "<pytest>true</pytest>" in raw_output:
        return {
            "all_passed": True, "output": raw_output, "exit_code": exit_code,
            "details": {"matched": {t: "passed" for t in test_ids}, "source": "marker"},
        }

    xml_result = env.execute({"command": "cat /tmp/_awe_test_results.xml"}, timeout=30)
    if xml_result["returncode"] == 0 and xml_result["output"].strip():
        all_passed, xml_details = _parse_junit_xml(xml_result["output"], test_ids)
        xml_details["source"] = "junit_xml"
        xml_details["exit_code"] = exit_code
        return {"all_passed": all_passed, "output": raw_output, "exit_code": exit_code, "details": xml_details}

    summary = _parse_pytest_summary(raw_output)
    all_passed = summary["passed"] > 0 and summary["failed"] == 0 and summary["errors"] == 0
    return {
        "all_passed": all_passed, "output": raw_output, "exit_code": exit_code,
        "details": {"source": "pytest_summary", **summary},
    }


def _eval_doc2repo(instance: dict, env, workdir: str, timeout: int) -> dict:
    test_suite_name = instance.get("test_suite", "")
    test_suite_path = instance.get("test_suite_path", "")
    test_suite_num = instance.get("test_suite_num", 0)

    env.execute({"command": f"cd {workdir} && pip install -e ."}, timeout=300)

    if not test_suite_name or not test_suite_path:
        return {"resolved": False, "error": "missing test_suite or test_suite_path", "output": "", "exit_code": -1}

    local_path = Path(test_suite_path) / test_suite_name
    if not local_path.exists():
        return {"resolved": False, "error": f"test suite zip not found: {local_path}", "output": "", "exit_code": -1}

    zip_b64 = base64.b64encode(local_path.read_bytes()).decode()
    env.execute(
        {"command": f'python3 -c "import base64; open(\'/tmp/_awe_test_suite.zip\',\'wb\').write(base64.b64decode(\'{zip_b64}\'))"'},
        timeout=60,
    )
    env.execute({"command": f"cd {workdir} && unzip -o /tmp/_awe_test_suite.zip"}, timeout=600)

    result = env.execute({"command": f"cd {workdir} && python realswe_eval_script.py"}, timeout=timeout)
    output = result["output"]
    exit_code = result["returncode"]

    all_passed = "<pytest>true</pytest>" in output
    summary = _parse_pytest_summary(output)
    effective_total = test_suite_num if test_suite_num > 0 else (summary["passed"] + summary["failed"] + summary["errors"])
    pass_rate = summary["passed"] / effective_total if effective_total > 0 else 0.0

    return {
        "resolved": all_passed,
        "pass_rate": pass_rate,
        "test_suite_num": test_suite_num,
        "passed": summary["passed"],
        "failed": summary["failed"],
        "errors": summary["errors"],
        "effective_total": effective_total,
        "output": output,
        "exit_code": exit_code,
    }


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
    traj_path: Path,
    results_path: Path,
    env_config: dict,
    logs_dir: Path,
    progress_manager: RunBatchProgressManager,
    sample_idx: int = 0,
) -> None:
    instance_id = instance["instance_id"]
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id
    progress_manager.on_instance_start(task_label)
    progress_manager.update_instance_status(task_label, "Starting sandbox")
    result: dict = {}
    exit_status = "error"
    try:
        patch = load_patch_from_traj(traj_path)
        progress_manager.update_instance_status(task_label, "Running tests")
        result = evaluate_instance_orchard(instance, patch, env_config, logs_dir=logs_dir, sample_idx=sample_idx)
        exit_status = "resolved" if result.get("resolved") else "failed"
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
    subset: str = typer.Option("scaleswe", "--subset", help="Scale-SWE subset or dataset path", rich_help_panel="Data selection"),
    split: str = typer.Option("train", "--split", help="Dataset split", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')", rich_help_panel="Data selection"),
    language: str = typer.Option("", "--language", help="Filter by programming language (e.g., 'python')", rich_help_panel="Data selection"),
    output: str = typer.Option(..., "-o", "--output", help="Directory containing trajectory files to verify", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files/specs for orchard environment", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-verify instances already in results file", rich_help_panel="Data selection"),
    logs_dir: str = typer.Option("", "--logs-dir", help="Directory to save test logs (default: <output>/verify_logs)", rich_help_panel="Advanced"),
    verify_output: str = typer.Option("", "--verify-output", help="Output results file (default: <output>/verify_results.json)", rich_help_panel="Basic"),
    report_only: bool = typer.Option(False, "--report-only", help="Only print report from existing results file without running verification", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    results_path = Path(verify_output) if verify_output else output_path / "verify_results.json"
    logs_path = Path(logs_dir) if logs_dir else output_path / "verify_logs"

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
        if language:
            results = {iid: r for iid, r in results.items()
                       if any(s.get("language", "").lower() == language.lower() for s in r.get("samples", [r]))}
        _print_verification_report(results, results_path)
        return

    add_file_handler(output_path / "minisweagent_verify.log")
    logger.info(f"Scanning trajectories in {output_path}")

    traj_map = scan_trajectories(output_path)
    if not traj_map:
        console.print(f"[yellow]No trajectory files found in {output_path}[/yellow]")
        raise typer.Exit(1)
    logger.info(f"Found {len(traj_map)} trajectory files")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    _ds = load_instances(dataset_path, split)
    instances = {inst["instance_id"]: inst for inst in _ds}

    traj_instance_ids = sorted({iid for iid, _ in traj_map})
    instance_ids = [iid for iid in traj_instance_ids if iid in instances]
    missing = [iid for iid in traj_instance_ids if iid not in instances]
    if missing:
        logger.warning(f"{len(missing)} trajectory instance_ids not found in dataset: {missing[:5]}...")

    if language:
        instance_ids = [iid for iid in instance_ids if instances[iid].get("language", "").lower() == language.lower()]
    if filter_spec:
        instance_ids = [iid for iid in instance_ids if re.match(filter_spec, iid)]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instance_ids = instance_ids[slice(*values)]

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

    progress_manager = RunBatchProgressManager(
        len(work_items), output_path / f"verify_statuses_{time.time()}.yaml"
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
