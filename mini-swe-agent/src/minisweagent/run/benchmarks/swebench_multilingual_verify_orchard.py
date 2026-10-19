#!/usr/bin/env python3

"""Grade SWE-bench Multilingual patches in the Orchard sandbox."""

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
from minisweagent.run.benchmarks.utils.images import get_swebench_docker_image_name
from minisweagent.run.benchmarks.utils.log_parsers import (
    _ansi_escape,
    _get_changed_files,
    _normalize_test_name,
    _parse_log_ant,
    _parse_log_cargo,
    _parse_log_django,
    _parse_log_doctest,
    _parse_log_googletest,
    _parse_log_gotest,
    _parse_log_gradle,
    _parse_log_jekyll,
    _parse_log_jest,
    _parse_log_jest_json,
    _parse_log_jq,
    _parse_log_marked,
    _parse_log_matplotlib,
    _parse_log_micropython,
    _parse_log_phpunit,
    _parse_log_pytest,
    _parse_log_pytest_options,
    _parse_log_pytest_v2,
    _parse_log_react_pdf,
    _parse_log_redis,
    _parse_log_rspec_json,
    _parse_log_ruby_unit,
    _parse_log_seaborn,
    _parse_log_sympy,
    _parse_log_tap,
    _parse_log_vitest,
)
from minisweagent.run.benchmarks.utils.verify_common import (
    _check_upload,
    _parse_traj_filename,
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
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_multi_orchard.yaml"

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
DOCKER_WORKDIR = "/testbed"
FAIL_ONLY_REPOS = {"chartjs/Chart.js", "processing/p5.js", "markedjs/marked"}

NON_TEST_EXTS = [
    ".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg",
    ".pkl", ".zip", ".csv", ".parquet", ".feather", ".svg",
    ".npy", ".yml", ".yaml", ".toml",
]


# ---- Inlined log parsers ----


def _parse_log_maven(log: str) -> dict[str, str]:
    sm = {}
    pending_tests: list[str] = []
    unmatched_results: list[str] = []
    name_pat = re.compile(r"^.*-Dtest=(\S+).*$")
    result_pat = re.compile(r"^.*BUILD (SUCCESS|FAILURE)$")
    for line in log.split("\n"):
        m = name_pat.match(line.strip())
        if m:
            pending_tests.append(m.group(1))
        m = result_pat.match(line.strip())
        if m:
            status = m.group(1)
            if pending_tests:
                test_name = pending_tests.pop(0)
                sm[test_name] = "PASSED" if status == "SUCCESS" else "FAILED"
            else:
                unmatched_results.append(status)
    while pending_tests and unmatched_results:
        test_name = pending_tests.pop(0)
        status = unmatched_results.pop(0)
        sm[test_name] = "PASSED" if status == "SUCCESS" else "FAILED"
    return sm


def _parse_log_karma(log: str) -> dict[str, str]:
    sm = {}
    current_suite: list[str] = []
    current_indent = -1
    started = False
    pat = re.compile(r"^(\s*)?([✔✖])?\s(.*)$")
    for line in log.split("\n"):
        if line.startswith("SUMMARY:"):
            return sm
        if "Starting browser" in line:
            started = True
            continue
        if not started:
            continue
        m = pat.match(line)
        if m:
            indent, status, name = m.groups()
            if indent and not status:
                new_indent = len(indent)
                if new_indent > current_indent:
                    current_indent = new_indent
                    current_suite.append(name)
                elif new_indent < current_indent:
                    current_indent = new_indent
                    if current_suite:
                        current_suite.pop()
                    continue
            if status in ("✔", "✖"):
                full = " > ".join(current_suite + [name])
                sm[full] = "PASSED" if status == "✔" else "FAILED"
    return sm


def _parse_log_p5js(log: str) -> dict[str, str]:
    log = _ansi_escape(log)
    sm = {}
    fail_pat = re.compile(r"^\s*(\d+)\)(.{0,1000}?):", re.MULTILINE | re.DOTALL)
    for m in fail_pat.finditer(log):
        names = [n.strip() for n in m.group(2).split("\n") if n.strip()]
        if names:
            sm[":".join(names)] = "FAILED"
    return sm


def _parse_log_chart_js(log: str) -> dict[str, str]:
    log = _ansi_escape(log)
    sm = {}
    for m in re.findall(r"Chrome\s[\d\.]+[^\S\r\n]\(.+?\)[^\S\r\n](.*)FAILED$", log, re.MULTILINE):
        sm[m] = "FAILED"
    return sm


def _parse_log_calypso(log: str) -> dict[str, str]:
    sm = {}
    suite: list = []
    for chunk in log.split(" ./node_modules/.bin/jest ")[1:]:
        for line in chunk.split("\n"):
            if any(line.startswith(x) for x in ["Test Suites", "  ● "]):
                break
            stripped = line.strip()
            if stripped.startswith("✓"):
                m = re.match(r"^\s+✓\s(.*?)(?:\(\d+ms\))?$", line)
                if m:
                    name = " - ".join([" - ".join(s[0] for s in suite), m.group(1).strip()]).strip()
                    sm[name] = "PASSED"
            elif stripped.startswith("✕"):
                m = re.match(r"^\s+✕\s(.*?)(?:\(\d+ms\))?$", line)
                if m:
                    name = " - ".join([" - ".join(s[0] for s in suite), m.group(1).strip()]).strip()
                    sm[name] = "FAILED"
            elif len(line) - len(line.lstrip()) > 0:
                indent = len(line) - len(line.lstrip())
                if not suite:
                    suite = [(stripped, indent)]
                else:
                    while suite and suite[-1][1] >= indent:
                        suite.pop()
                    suite.append((stripped, indent))
    return sm


def _parse_log_immutable_js(log: str, instance_id: str) -> dict[str, str]:
    pr = instance_id.split("-")[-1]
    if pr == "2005":
        return _parse_log_jest_json(log)
    return _parse_log_jest(log)


# ---- Repo-to-parser mapping ----

_REPO_TO_PARSER: dict[str, object] = {
    "astropy/astropy": _parse_log_pytest_v2,
    "django/django": _parse_log_django,
    "marshmallow-code/marshmallow": _parse_log_pytest,
    "matplotlib/matplotlib": _parse_log_matplotlib,
    "mwaskom/seaborn": _parse_log_seaborn,
    "pallets/flask": _parse_log_pytest,
    "psf/requests": _parse_log_pytest_options,
    "pvlib/pvlib-python": _parse_log_pytest,
    "pydata/xarray": _parse_log_pytest,
    "pydicom/pydicom": _parse_log_pytest_options,
    "pylint-dev/astroid": _parse_log_pytest,
    "pylint-dev/pylint": _parse_log_pytest_options,
    "pytest-dev/pytest": _parse_log_pytest,
    "pyvista/pyvista": _parse_log_pytest,
    "scikit-learn/scikit-learn": _parse_log_pytest_v2,
    "sqlfluff/sqlfluff": _parse_log_pytest,
    "sphinx-doc/sphinx": _parse_log_pytest_v2,
    "sympy/sympy": _parse_log_sympy,
    "caddyserver/caddy": _parse_log_gotest,
    "hashicorp/terraform": _parse_log_gotest,
    "prometheus/prometheus": _parse_log_gotest,
    "gohugoio/hugo": _parse_log_gotest,
    "gin-gonic/gin": _parse_log_gotest,
    "google/gson": _parse_log_maven,
    "apache/druid": _parse_log_maven,
    "javaparser/javaparser": _parse_log_maven,
    "projectlombok/lombok": _parse_log_ant,
    "apache/lucene": _parse_log_gradle,
    "reactivex/rxjava": _parse_log_gradle,
    "Automattic/wp-calypso": _parse_log_calypso,
    "babel/babel": _parse_log_jest,
    "vuejs/core": _parse_log_vitest,
    "facebook/docusaurus": _parse_log_jest,
    "mrdoob/three.js": _parse_log_tap,
    "preactjs/preact": _parse_log_karma,
    "axios/axios": _parse_log_tap,
    "markedjs/marked": _parse_log_marked,
    "diegomura/react-pdf": _parse_log_react_pdf,
    "chartjs/Chart.js": _parse_log_chart_js,
    "processing/p5.js": _parse_log_p5js,
    "phpoffice/phpspreadsheet": _parse_log_phpunit,
    "laravel/framework": _parse_log_phpunit,
    "php-cs-fixer/php-cs-fixer": _parse_log_phpunit,
    "briannesbitt/carbon": _parse_log_phpunit,
    "fluent/fluentd": _parse_log_ruby_unit,
    "fastlane/fastlane": _parse_log_rspec_json,
    "jordansissel/fpm": _parse_log_rspec_json,
    "faker-ruby/faker": _parse_log_ruby_unit,
    "rubocop/rubocop": _parse_log_rspec_json,
    "burntsushi/ripgrep": _parse_log_cargo,
    "sharkdp/bat": _parse_log_cargo,
    "astral-sh/ruff": _parse_log_cargo,
    "tokio-rs/tokio": _parse_log_cargo,
    "uutils/coreutils": _parse_log_cargo,
    "nushell/nushell": _parse_log_cargo,
    "tokio-rs/axum": _parse_log_cargo,
    "redis/redis": _parse_log_redis,
    "jqlang/jq": _parse_log_jq,
    "nlohmann/json": _parse_log_doctest,
    "micropython/micropython": _parse_log_micropython,
    "valkey-io/valkey": _parse_log_redis,
    "fmtlib/fmt": _parse_log_googletest,
}


# ---- Specs loading helpers ----

_SWEBENCH_IMPORT_MSG = (
    "swebench package is required for multilingual evaluation. "
    "Install it with: pip install swebench"
)


def _get_instance_specs(repo: str, version: str) -> dict:
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except ImportError:
        raise ImportError(_SWEBENCH_IMPORT_MSG)
    return MAP_REPO_VERSION_TO_SPECS[repo][version]


def _get_repo_language(repo: str) -> str:
    try:
        from swebench.harness.constants import MAP_REPO_TO_EXT
    except ImportError:
        raise ImportError(_SWEBENCH_IMPORT_MSG)
    return MAP_REPO_TO_EXT.get(repo, "py")


# ---- File upload utilities ----

_B64WRITE_CHUNK_SIZE = 48000


def _b64write_cmd(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f"echo '{b64}' | base64 -d > {path}"


def _upload_file(env, path: str, content: str, *, timeout: int = 30) -> None:
    raw = content.encode()
    if len(raw) <= _B64WRITE_CHUNK_SIZE:
        _check_upload(env.execute({"command": _b64write_cmd(path, content)}, timeout=timeout), path)
    else:
        _check_upload(env.execute({"command": f"rm -f {path}"}, timeout=10), path)
        for offset in range(0, len(raw), _B64WRITE_CHUNK_SIZE):
            chunk = raw[offset:offset + _B64WRITE_CHUNK_SIZE]
            b64 = base64.b64encode(chunk).decode()
            cmd = f"echo '{b64}' | base64 -d >> {path}"
            _check_upload(env.execute({"command": cmd}, timeout=timeout), path)


# ---- Eval script building ----

def _get_modified_files(patch: str) -> list[str]:
    return [line[6:] for line in patch.split("\n") if line.startswith("--- a/")]


def _get_test_directives(instance: dict) -> list[str]:
    repo = instance.get("repo", "")
    test_patch = instance.get("test_patch", "")
    directives = [d for d in _get_changed_files(test_patch)
                  if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]
    if repo == "django/django":
        transformed = []
        for d in directives:
            d = d[:-3] if d.endswith(".py") else d
            d = d[6:] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            transformed.append(d)
        directives = transformed
    return directives


_DEFAULT_PYTEST_CMD = "python -m pytest --no-header -rA --tb=no -p no:cacheprovider"

_REPO_TEST_CMD: dict[str, str] = {
    "django/django": "./tests/runtests.py --verbosity 2",
    "sympy/sympy": "bin/test -C --verbose",
}


def _build_eval_script(instance: dict) -> str:
    repo = instance["repo"]
    version = instance["version"]
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]

    specs = _get_instance_specs(repo, version)
    language = _get_repo_language(repo)

    test_files = _get_modified_files(test_patch)
    reset_cmd = f"git checkout {base_commit} {' '.join(test_files)}" if test_files else 'echo "No test files to reset"'
    apply_test_patch = "git apply -v /tmp/test_patch.diff"

    if language == "py":
        test_cmd = _REPO_TEST_CMD.get(repo, _DEFAULT_PYTEST_CMD)
        directives = _get_test_directives(instance)
        test_command = " ".join([test_cmd, *directives]).strip()

        lines = [
            "#!/bin/bash",
            "set -o pipefail",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"cd {DOCKER_WORKDIR}",
            f"git config --global --add safe.directory {DOCKER_WORKDIR}",
            f"cd {DOCKER_WORKDIR}",
            "git status",
            "git show",
            f"git -c core.fileMode=false diff {base_commit}",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            reset_cmd,
            apply_test_patch,
            f"echo '{START_TEST_OUTPUT}'",
            test_command,
            f"echo '{END_TEST_OUTPUT}'",
            reset_cmd,
        ]
    else:
        test_cmds = specs.get("test_cmd", [])
        if isinstance(test_cmds, str):
            test_cmds = [test_cmds]
        build_cmds = list(specs.get("build", []))

        test_lines = []
        for cmd in test_cmds:
            safe_cmd = cmd.replace("'", "'\\''")
            test_lines.append(f"echo '+ {safe_cmd}'")
            test_lines.append(cmd)

        lines = [
            "#!/bin/bash",
            "set -uo pipefail",
            'export PATH="/usr/local/cargo/bin:/usr/local/go/bin:$PATH"',
            '[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"',
            f"cd {DOCKER_WORKDIR}",
            f"git config --global --add safe.directory {DOCKER_WORKDIR}",
            f"cd {DOCKER_WORKDIR}",
            reset_cmd,
            "git apply -v /tmp/test_patch.diff",
            *build_cmds,
            f"echo '{START_TEST_OUTPUT}'",
            *test_lines,
            f"echo '{END_TEST_OUTPUT}'",
            reset_cmd,
        ]

    return "\n".join(lines) + "\n"


# ---- Test output parsing ----


def _parse_test_output(log: str, instance: dict) -> dict[str, str]:
    repo = instance["repo"]
    instance_id = instance["instance_id"]

    if repo == "jekyll/jekyll":
        def parser(log):
            return _parse_log_jekyll(log, instance_id)
    elif repo == "immutable-js/immutable-js":
        def parser(log):
            return _parse_log_immutable_js(log, instance_id)
    else:
        parser = _REPO_TO_PARSER.get(repo)

    if parser is None:
        logger.warning(f"No log parser for repo {repo!r}, falling back to pytest_v2")
        parser = _parse_log_pytest_v2

    if START_TEST_OUTPUT in log and END_TEST_OUTPUT in log:
        test_content = log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    else:
        test_content = log

    status_map = parser(test_content)
    if not status_map:
        status_map = parser(log)
    return status_map


# ---- Evaluation ----

def evaluate_instance_orchard(
    instance: dict, patch: str, env_config: dict, *, logs_dir: Path | None = None, sample_idx: int = 0,
) -> dict:
    from minisweagent.environments.extra.orchard import OrchardEnvironment

    instance_id = instance["instance_id"]
    repo = instance.get("repo", "")
    image = get_swebench_docker_image_name(instance)

    fail_to_pass = [_normalize_test_name(n) for n in instance.get("FAIL_TO_PASS", [])]
    pass_to_pass = [_normalize_test_name(n) for n in instance.get("PASS_TO_PASS", [])]
    fail_only = repo in FAIL_ONLY_REPOS

    cfg = {k: v for k, v in env_config.items() if k not in ("environment_class",)}
    cfg.update(image=image, cwd=DOCKER_WORKDIR)

    env = OrchardEnvironment(**cfg)
    output = ""
    exit_code = -1
    patch_applied = False
    try:
        eval_script = _build_eval_script(instance)
        test_timeout = cfg.get("sandbox_timeout", 600)

        _upload_file(env, "/tmp/patch.diff", patch or "", timeout=30)
        _upload_file(env, "/tmp/test_patch.diff", instance.get("test_patch", ""), timeout=60)
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

        result = env.execute({"command": "bash /tmp/eval.sh"}, timeout=test_timeout)
        exit_code = result["returncode"]
        output = result["output"]
    finally:
        env.cleanup()

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(output)

    bad_codes = [c for c in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT] if c in output]
    if bad_codes or (START_TEST_OUTPUT not in output and END_TEST_OUTPUT not in output):
        if bad_codes:
            logger.warning(f"{instance_id}: bad signals in output: {bad_codes}")
        else:
            logger.warning(f"{instance_id}: START/END test markers not found in output")
        parsed: dict[str, str] = {}
    else:
        parsed = _parse_test_output(output, instance)
        parsed = {_normalize_test_name(k): v for k, v in parsed.items()}

    passed = {k for k, v in parsed.items() if v in ("PASSED", "XFAIL")}

    def _check(case: str) -> bool:
        if fail_only:
            return case not in parsed or parsed.get(case) != "FAILED"
        return case in parsed and parsed[case] in ("PASSED", "XFAIL")

    f2p_passed = sorted(t for t in fail_to_pass if _check(t))
    f2p_failed = sorted(t for t in fail_to_pass if not _check(t))
    p2p_passed = sorted(passed & set(pass_to_pass))
    p2p_failed = sorted(set(pass_to_pass) - passed)

    if fail_only:
        resolved = len(fail_to_pass) > 0 and not f2p_failed
    else:
        resolved = len(fail_to_pass) > 0 and not f2p_failed and not p2p_failed

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "exit_code": exit_code,
        "patch_applied": patch_applied,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_failed": f2p_failed,
        "pass_to_pass_passed": len(p2p_passed),
        "pass_to_pass_failed": p2p_failed,
        "parsed_tests_count": len(parsed),
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


# ---- Instance processing and reporting ----

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
    subset: str = typer.Option("multilingual", "--subset", help="Dataset subset or path", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
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
