#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-rebench instances in batch mode."""

import concurrent.futures
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path
from queue import Empty

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.images import get_swebench_docker_image_name
from minisweagent.run.benchmarks.utils.loading import load_harness_agent_class, load_instances
from minisweagent.run.benchmarks.utils.runner_common import (
    ProgressTrackingAgent,
    _NoopProgressManager,
    _drain_result,
    _resolve_per_instance_api_base,
    filter_instances,
    get_existing_samples,
    get_existing_samples_from_trajs,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWE-rebench instances (or another SWE-bench-format dataset).

Each instance uses its own image and workdir when the dataset provides them (default workdir /testbed).

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
    "rebench_v2": "nebius/SWE-rebench-V2",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal", "orchard"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name
    env_config["cwd"] = instance.get("workdir") or "/testbed"

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def _run_instance_subprocess(
    instance: dict,
    output_dir: Path,
    config: dict,
    sample_idx: int,
    instance_timeout: int,
    result_queue: multiprocessing.Queue,
) -> None:
    import copy

    globals()["ProgressTrackingAgent"] = load_harness_agent_class(ProgressTrackingAgent)

    config = copy.deepcopy(config)
    instance_id = instance["instance_id"]
    _resolve_per_instance_api_base(config, instance_id)
    instance_dir = output_dir / instance_id
    sample_suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
    traj_filename = f"{instance_id}{sample_suffix}.traj.json"
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id
    deadline = 0

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    progress_manager = _NoopProgressManager()

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        deadline = time.monotonic() + instance_timeout if instance_timeout > 0 else 0
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=task_label,
            deadline=deadline,
            **config.get("agent", {}),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {task_label}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / traj_filename
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        "sample_idx": sample_idx,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        result_queue.put(
            {
                "exit_status": exit_status,
                "result": result,
                "model_name": model.config.model_name,
                "extra_info": extra_info,
            }
        )


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    sample_idx: int = 0,
    no_preds: bool = False,
    instance_timeout: int = 0,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    sample_suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
    traj_filename = f"{instance_id}{sample_suffix}.traj.json"
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id

    progress_manager.on_instance_start(task_label)
    progress_manager.update_instance_status(task_label, "Pulling/starting environment")

    if not no_preds:
        remove_from_preds_file(output_dir / "preds.json", instance_id, sample_idx)
    traj_path = instance_dir / traj_filename
    traj_path.unlink(missing_ok=True)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_instance_subprocess,
        args=(instance, output_dir, config, sample_idx, instance_timeout, result_queue),
        daemon=True,
    )
    proc.start()

    child_result = _drain_result(proc, result_queue, (instance_timeout + int(os.environ.get("MSWEA_PULL_ALLOWANCE", "1800"))) if instance_timeout > 0 else None)
    if child_result is not None and proc.is_alive():
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(timeout=5)

    if child_result is None and proc.is_alive():
        logger.warning(f"Instance {task_label} exceeded wall-clock timeout ({instance_timeout}s), skipping.")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(timeout=5)
        exit_status = "WallClockTimeout"
        result = ""
        model_name = config.get("model", {}).get("model_name", "unknown")
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(
            json.dumps(
                {
                    "info": {"exit_status": exit_status, "submission": result, "sample_idx": sample_idx},
                    "instance_id": instance_id,
                },
                indent=2,
            )
        )
    else:
        if child_result is None:
            try:
                child_result = result_queue.get_nowait()
            except Empty:
                child_result = None

        if child_result is None:
            exit_status = "MissingResult"
            result = ""
            model_name = config.get("model", {}).get("model_name", "unknown")
            logger.warning(f"No result returned for {task_label}")
        else:
            exit_status = child_result.get("exit_status")
            result = child_result.get("result", "")
            model_name = child_result.get("model_name") or config.get("model", {}).get("model_name", "unknown")

    if not no_preds:
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result, sample_idx)
    progress_manager.on_instance_end(task_label, exit_status)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="Dataset subset (a key of DATASET_MAPPING, e.g. rebench) or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    n_samples: int = typer.Option(1, "-n", "--n-samples", help="Number of samples (runs) per instance for pass@k evaluation", rich_help_panel="Basic"),
    no_preds: bool = typer.Option(False, "--no-preds", help="Disable writing to preds.json", rich_help_panel="Advanced"),
    instance_timeout: int = typer.Option(0, "--instance-timeout", help="Hard timeout in seconds per instance (0=no limit)", rich_help_panel="Advanced"),
    include_jsonl: str = typer.Option("", "--include-jsonl", help="Path to a JSONL file; only instances with matching metadata.instance_id will be included", rich_help_panel="Data selection"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_instances(dataset_path, split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if include_jsonl:
        include_entries = load_dataset("json", data_files=include_jsonl, split="train")
        include_ids = set()
        for entry in include_entries:
            iid = entry.get("instance_id") or (entry.get("metadata") or {}).get("instance_id")
            if iid:
                include_ids.add(iid)
        before = len(instances)
        instances = [inst for inst in instances if inst["instance_id"] in include_ids]
        logger.info(f"Include JSONL filter: {before} -> {len(instances)} instances")

    work_items: list[tuple[dict, int]] = []
    if not redo_existing:
        existing_samples = get_existing_samples(output_path / "preds.json")
        existing_samples |= get_existing_samples_from_trajs(output_path)
    else:
        existing_samples = set()
    for instance in instances:
        for s_idx in range(n_samples):
            if (instance["instance_id"], s_idx) not in existing_samples:
                work_items.append((instance, s_idx))
    if len(work_items) < len(instances) * n_samples:
        logger.info(f"Skipping {len(instances) * n_samples - len(work_items)} existing samples")
    logger.info(f"Running {len(work_items)} jobs ({len(instances)} instances x {n_samples} samples)...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(work_items), output_path / f"exit_statuses_{time.time()}.yaml")

    if workers <= 1:
        live_ctx = Live(progress_manager.render_group, refresh_per_second=4)
    else:
        progress_manager._main_progress_bar.live.redirect_stdout = False
        progress_manager._main_progress_bar.live.redirect_stderr = False
        live_ctx = progress_manager._main_progress_bar
    with live_ctx:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {}
        for instance, s_idx in work_items:
            iid = instance["instance_id"]
            task_label = f"{iid}#{s_idx}" if s_idx > 0 else iid
            future = executor.submit(process_instance, instance, output_path, config, progress_manager, s_idx, no_preds, instance_timeout)
            futures[future] = task_label
        try:
            done = set()
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
                            logger.error(f"Error in future for {task_label}: {e}", exc_info=True)
                            progress_manager.on_uncaught_exception(task_label, e)
                except concurrent.futures.TimeoutError:
                    pass
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down immediately...")
            executor.shutdown(wait=False, cancel_futures=True)
            os._exit(1)
        finally:
            executor.shutdown(wait=False)


if __name__ == "__main__":
    app()
