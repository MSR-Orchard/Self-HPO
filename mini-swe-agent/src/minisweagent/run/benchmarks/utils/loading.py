"""Shared loaders for the benchmark runners: harness agent class and instance datasets."""

import importlib.util
import os
from pathlib import Path


def load_harness_agent_class(default: type) -> type:
    path = os.environ.get("HARNESS_AGENT_FILE", "").strip()
    if not path:
        return default
    spec = importlib.util.spec_from_file_location("_harness_agent_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = module.AgentHarness
    try:
        (Path(path).parent / ".hook_fired").write_text(agent_class.__name__)
    except Exception:
        pass
    return agent_class


def load_instances(dataset_path: str, split: str):
    from datasets import load_dataset

    if Path(dataset_path).is_file() and dataset_path.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=dataset_path, split="train")
    return load_dataset(dataset_path, split=split)
