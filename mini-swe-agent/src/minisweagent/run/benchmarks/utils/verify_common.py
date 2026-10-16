"""Helpers shared by the *_verify_orchard verifiers: incremental results-file cache
(update_results_file / flush_results_file / get_results_cache), resume scan, trajectory filename parsing,
chunked file upload into a sandbox (_upload_file)."""

import base64
import json
import threading
from pathlib import Path

_OUTPUT_FILE_LOCK = threading.Lock()
_RESULTS_CACHE: dict | None = None
_RESULTS_DIRTY = False


def update_results_file(output_path: Path, instance_id: str, sample_idx: int, result: dict):
    global _RESULTS_CACHE, _RESULTS_DIRTY
    with _OUTPUT_FILE_LOCK:
        if _RESULTS_CACHE is None:
            _RESULTS_CACHE = (
                json.loads(output_path.read_text()) if output_path.exists() and output_path.stat().st_size > 0 else {}
            )
        entry = _RESULTS_CACHE.get(instance_id, {"instance_id": instance_id, "samples": []})
        samples = [s for s in entry.get("samples", []) if s.get("sample_idx") != sample_idx]
        result["sample_idx"] = sample_idx
        samples.append(result)
        samples.sort(key=lambda s: s["sample_idx"])
        entry["samples"] = samples
        entry["resolved"] = any(s.get("resolved") for s in samples)
        _RESULTS_CACHE[instance_id] = entry
        _RESULTS_DIRTY = True


def flush_results_file(output_path: Path):
    global _RESULTS_DIRTY
    with _OUTPUT_FILE_LOCK:
        if _RESULTS_CACHE is not None and _RESULTS_DIRTY:
            output_path.write_text(json.dumps(_RESULTS_CACHE, indent=2))
            _RESULTS_DIRTY = False


def get_existing_verified_samples(results_path: Path) -> set[tuple[str, int]]:
    if not results_path.exists():
        return set()
    data = json.loads(results_path.read_text())
    existing = set()
    for instance_id, entry in data.items():
        for s in entry.get("samples", []):
            existing.add((instance_id, s["sample_idx"]))
        if "samples" not in entry:
            existing.add((instance_id, 0))
    return existing


def get_results_cache() -> dict | None:
    return _RESULTS_CACHE


def _parse_traj_filename(name: str, suffix: str) -> tuple[str, int]:
    name = name.removesuffix(suffix)
    if ".sample_" in name:
        instance_id, _, idx = name.rpartition(".sample_")
        return instance_id, int(idx)
    return name, 0


# ---- File upload ----

_B64WRITE_CHUNK_SIZE = 48000


def _b64write_cmd(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f"python3 -c \"import base64; open({path!r},'wb').write(base64.b64decode({b64!r}))\""


def _check_upload(res: dict, path: str) -> None:
    if res.get("returncode") != 0:
        raise RuntimeError(f"upload {path} failed: {res.get('exception_info') or res.get('output', '')[:200]}")


def _upload_file(env, path: str, content: str, *, timeout: int = 30) -> None:
    raw = content.encode()
    if len(raw) <= _B64WRITE_CHUNK_SIZE:
        _check_upload(env.execute({"command": _b64write_cmd(path, content)}, timeout=timeout), path)
    else:
        _check_upload(env.execute({"command": f"rm -f {path}"}, timeout=10), path)
        for offset in range(0, len(raw), _B64WRITE_CHUNK_SIZE):
            chunk = raw[offset:offset + _B64WRITE_CHUNK_SIZE]
            b64 = base64.b64encode(chunk).decode()
            cmd = f"python3 -c \"import base64; open({path!r},'ab').write(base64.b64decode({b64!r}))\""
            _check_upload(env.execute({"command": cmd}, timeout=timeout), path)
