"""Helpers shared by the batch runners (swebench, swerebench, scaleswe, swebench_pro)."""

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from queue import Empty

from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import logger

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    def __init__(
        self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", deadline: float = 0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id
        self.deadline = deadline

    def step(self) -> dict:
        if self.deadline and time.monotonic() > self.deadline:
            raise TimeoutError("Instance exceeded wall-clock deadline")
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


class _NoopProgressManager:
    def update_instance_status(self, instance_id: str, status: str) -> None:
        return


def _drain_result(proc, result_queue, timeout):
    deadline = (time.monotonic() + timeout) if timeout else None
    while True:
        try:
            return result_queue.get(timeout=1)
        except Empty:
            pass
        if not proc.is_alive():
            try:
                return result_queue.get_nowait()
            except Empty:
                return None
        if deadline is not None and time.monotonic() > deadline:
            return None


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


def get_existing_samples(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    output_data = json.loads(output_path.read_text())
    existing = set()
    for instance_id, entry in output_data.items():
        for s in entry.get("samples", []):
            existing.add((instance_id, s["sample_idx"]))
        if "samples" not in entry:
            existing.add((instance_id, 0))
    return existing


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str, sample_idx: int = 0):
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        samples = output_data.get(instance_id, {}).get("samples", [])
        samples = [s for s in samples if s.get("sample_idx") != sample_idx]
        samples.append({"sample_idx": sample_idx, "model_patch": result})
        samples.sort(key=lambda s: s["sample_idx"])
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": samples[0]["model_patch"],
            "samples": samples,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str, sample_idx: int | None = None):
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id not in output_data:
            return
        if sample_idx is None:
            del output_data[instance_id]
        else:
            samples = output_data[instance_id].get("samples", [])
            samples = [s for s in samples if s.get("sample_idx") != sample_idx]
            if not samples:
                del output_data[instance_id]
            else:
                output_data[instance_id]["samples"] = samples
                output_data[instance_id]["model_patch"] = samples[0]["model_patch"]
        output_path.write_text(json.dumps(output_data, indent=2))


_COMPLETED_EXIT_STATUSES = {"Submitted"}


def get_existing_samples_from_trajs(output_dir: Path) -> set[tuple[str, int]]:
    existing = set()
    for traj_file in output_dir.rglob("**/*.traj.json"):
        try:
            if (json.loads(traj_file.read_text()).get("info") or {}).get("exit_status") not in _COMPLETED_EXIT_STATUSES:
                continue
        except (json.JSONDecodeError, OSError):
            continue
        name = traj_file.stem.removesuffix(".traj")
        if ".sample_" in name:
            instance_id, _, idx = name.rpartition(".sample_")
            existing.add((instance_id, int(idx)))
        else:
            existing.add((name, 0))
    return existing


def _resolve_per_instance_api_base(config: dict, instance_id: str) -> None:
    model_kwargs = config.setdefault("model", {}).setdefault("model_kwargs", {})
    api_base = model_kwargs.get("api_base")
    if not isinstance(api_base, str) or "{port}" not in api_base:
        return
    port_base = int(os.environ.get("SGLANG_PORT_BASE", "8000"))
    n_ports = int(os.environ.get("SGLANG_NGPU", "1"))
    ports = list(range(port_base, port_base + n_ports))
    h = int(hashlib.md5(instance_id.encode()).hexdigest(), 16)
    model_kwargs["api_base"] = api_base.format(port=ports[h % len(ports)])
