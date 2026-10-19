"""
Reward computation for SWE-bench tasks (see ``reward_func`` for the reward modes).
"""

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import tempfile
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from slime.utils.types import Sample

from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.constants import LATEST, SWEbenchInstance
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
)

from .swe_agent import EnvironmentUnavailable, _is_environment_error
from .swe_env import (
    create_environment,
    load_config,
    stop_environment,
)
from .swe_grading import GradingTestSpec, get_eval_report

logger = logging.getLogger(__name__)

SWE_TIMEOUT_REWARD_CREATE_ENV = int(os.environ["SWE_TIMEOUT_CREATE_ENV"])
SWE_TIMEOUT_REWARD_EXECUTE = int(os.environ["SWE_TIMEOUT_REWARD_EXECUTE"])
SWE_TIMEOUT_REWARD_TOTAL = int(os.environ["SWE_TIMEOUT_REWARD_TOTAL"])

# ---- constants and enums ----

class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"


# ---- utility functions ----

def extract_patch_from_output(text: str) -> str:
    diff_pattern = r"```diff\s*\n(.*?)\n```"
    diff_match = re.search(diff_pattern, text, re.DOTALL)
    if diff_match:
        return _with_final_newline(diff_match.group(1).strip())

    patch_pattern = r"```patch\s*\n(.*?)\n```"
    patch_match = re.search(patch_pattern, text, re.DOTALL)
    if patch_match:
        return _with_final_newline(patch_match.group(1).strip())

    code_pattern = r"```\s*\n(.*?)\n```"
    code_match = re.search(code_pattern, text, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        if content.startswith("diff ") or "@@" in content or content.startswith("---"):
            return _with_final_newline(content)

    if text.startswith("diff ") or "@@" in text or text.startswith("---"):
        return _with_final_newline(text.lstrip())

    return ""


def _with_final_newline(patch: str) -> str:
    return patch if patch.endswith("\n") else patch + "\n"


def is_valid_patch(patch: str) -> bool:
    if not patch:
        return False

    has_diff_markers = any(
        marker in patch
        for marker in ["diff --git", "---", "+++", "@@", "Index:", "==="]
    )

    return has_diff_markers


# ---- test execution ----


async def run_tests_in_docker(
    instance_id: str,
    patch: str,
    repo: str,
    base_commit: str,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    instance: dict | None = None,
    config: dict | None = None,
) -> dict:
    result = {
        "passed": False,
        "resolved": False,
        "resolution": ResolvedStatus.NO.value,
        "fail_to_pass_rate": 0.0,
        "pass_to_pass_rate": 0.0,
        "tests_run": 0,
        "tests_passed": 0,
        "f2p_total": 0,
        "f2p_passed": 0,
        "p2p_total": 0,
        "p2p_passed": 0,
        "error": "",
        "output": "",
    }

    if not is_valid_patch(patch):
        result["error"] = "Invalid patch format"
        return result

    if fail_to_pass is None:
        fail_to_pass = []
    if pass_to_pass is None:
        pass_to_pass = []

    if instance is None:
        instance = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
        }

    if config is None:
        config_path = os.environ["SWE_CONFIG_PATH"]
        try:
            config = load_config(config_path)
        except Exception as e:
            result["error"] = f"Failed to load config: {e}"
            return result

    MAX_ENV_UNAVAILABLE_RETRIES = int(os.environ["SWE_MAX_ENV_RETRIES"])
    ENV_UNAVAILABLE_RETRY_WAIT = int(os.environ["SWE_ENV_RETRY_WAIT"])

    resource_level = 1
    for env_attempt in range(1, MAX_ENV_UNAVAILABLE_RETRIES + 1):
        attempt_result = dict(result)
        try:
            return await _run_tests_with_swebench_harness(
                instance=instance,
                patch=patch,
                config=config,
                result=attempt_result,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                resource_level=resource_level,
            )
        except EnvironmentUnavailable as e:
            logger.error(
                f"[SWE_REWARD] EnvironmentUnavailable during reward evaluation "
                f"(attempt {env_attempt}/{MAX_ENV_UNAVAILABLE_RETRIES}): {e}"
            )
            resource_level += 1
            if env_attempt < MAX_ENV_UNAVAILABLE_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying reward evaluation in {ENV_UNAVAILABLE_RETRY_WAIT}s..."
                )
                await asyncio.sleep(ENV_UNAVAILABLE_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_ENV_UNAVAILABLE_RETRIES} retry attempts exhausted"
                )
                attempt_result["error"] = (
                    f"Environment unavailable after {MAX_ENV_UNAVAILABLE_RETRIES} attempts: {e}"
                )
                return attempt_result
        except Exception as e:
            logger.error(
                f"[SWE_REWARD] Unexpected error during reward evaluation for "
                f"instance_id={instance.get('instance_id', 'unknown')}: {e}\n{traceback.format_exc()}"
            )
            if env_attempt < MAX_ENV_UNAVAILABLE_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying reward evaluation in {ENV_UNAVAILABLE_RETRY_WAIT}s..."
                )
                await asyncio.sleep(ENV_UNAVAILABLE_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_ENV_UNAVAILABLE_RETRIES} retry attempts exhausted"
                )
                attempt_result["error"] = f"Unexpected error: {e}"
                return attempt_result


async def _execute(env, command: str) -> dict:
    try:
        result = await asyncio.wait_for(env.execute(command), timeout=SWE_TIMEOUT_REWARD_EXECUTE)
        output_text = result.get("output", "")
        if result.get("returncode") != 0 and _is_environment_error(output_text):
            raise EnvironmentUnavailable(output_text)
        return result
    except asyncio.TimeoutError:
        cmd_preview = command[:200] + ("..." if len(command) > 200 else "")
        logger.error(
            "[SWE_REWARD] _execute TIMED OUT after %ds for command: %s",
            SWE_TIMEOUT_REWARD_EXECUTE,
            cmd_preview,
        )
        return {"output": f"Command timed out after {SWE_TIMEOUT_REWARD_EXECUTE}s", "returncode": -1, "timed_out": True}


_B64_CHUNK_LIMIT = 32_000

async def _write_file_to_sandbox(env, path: str, content: str | bytes) -> dict:
    if isinstance(content, str):
        raw = content.encode()
    else:
        raw = content
    encoded = base64.b64encode(raw).decode()

    quoted_path = shlex.quote(path)
    quoted_b64 = shlex.quote(path + ".b64")

    await _execute(env, f"rm -f {quoted_path} {quoted_b64}")

    last_result: dict = {}
    for i in range(0, len(encoded), _B64_CHUNK_LIMIT):
        chunk = encoded[i:i + _B64_CHUNK_LIMIT]
        last_result = await _execute(
            env,
            f"cat >> {quoted_b64} << 'B64EOF'\n{chunk}\nB64EOF",
        )
        if last_result.get("returncode", 0) != 0:
            return last_result

    return await _execute(
        env,
        f"base64 -d {quoted_b64} > {quoted_path} && rm -f {quoted_b64}",
    )


def make_test_spec_scaleswe(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        if key not in instance:
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    test_patch = instance.get("f2p_patch") or instance.get("test_patch", "")

    repo_directory = instance.get("workdir", "/testbed")

    base_test_cmd = "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider -W ignore::DeprecationWarning"
    all_test_ids = fail_to_pass + pass_to_pass
    if all_test_ids:
        test_id_str = " ".join(shlex.quote(tid) for tid in all_test_ids)
        test_command = f"{base_test_cmd} {test_id_str}"
    else:
        test_command = base_test_cmd

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": "",
        "log_parser": "parse_log_pytest_scaleswe",
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.11",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = {}
    repo_script_list = []
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    eval_script_list = [
        f"cd {repo_directory}",
    ]
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/f2p_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            f"cd {repo_directory} && "
            f"(git apply --verbose /tmp/f2p_patch.diff || "
            f"git apply --verbose --reject /tmp/f2p_patch.diff || "
            f"patch --batch --forward --fuzz=5 -p1 -i /tmp/f2p_patch.diff || "
            f"true)"
        )
    eval_script_list.extend([
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ])

    arch = "x86_64"

    image_name = instance.get("image_url") or instance.get("image_name")

    return GradingTestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language='py',
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=image_name,
    )


def make_test_spec_swerebench_v2(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"

    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        if key not in instance:
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    test_patch = instance.get("test_patch", "")
    repo_directory = instance.get("workdir", "/testbed")

    inst_install_config = instance.get("install_config", {})
    test_command = inst_install_config.get("test_cmd")
    log_parser = inst_install_config.get("log_parser", "parse_log_pytest")

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": inst_install_config.get("install", ""),
        "log_parser": log_parser,
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.10",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = inst_install_config.get("docker_specs", {})
    repo_script_list = [install_config.get("install", "")]
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    _NON_TEST_EXTS = [".json", ".png", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".zip", ".csv", ".parquet", ".feather", ".svg", ".npy", ".yml", ".yaml"]
    _base_commit = instance.get("base_commit", "HEAD")
    _repo = instance.get("repo", "")
    def _changed_files(pp):
        return [l[6:] for l in (pp or "").split("\n") if l.startswith("+++ b/") and l[6:] not in ("", "/dev/null")]
    def _modified_files(pp):
        return [l[6:] for l in (pp or "").split("\n") if l.startswith("--- a/")]
    _directives = [d for d in _changed_files(test_patch) if not any(d.endswith(e) for e in _NON_TEST_EXTS)]
    if "pytest" in (test_command or ""):
        _directives = [d for d in _directives if d.endswith(".py")]
    if _repo == "django/django":
        _dt = []
        for d in _directives:
            d = d[:-3] if d.endswith(".py") else d
            d = d[6:] if d.startswith("tests/") else d
            _dt.append(d.replace("/", "."))
        _directives = _dt
    _test_full = " ".join([test_command, *_directives]).strip()
    _tfiles = _modified_files(test_patch)
    _reset = ("git checkout %s %s" % (_base_commit, " ".join(_tfiles))) if _tfiles else 'echo no-reset'
    _install = (install_config.get("install") or "").strip() or ": no-install"
    _pathprep = "export PATH=/opt/conda/envs/testbed/bin:/opt/miniconda3/envs/testbed/bin:$PATH"
    _act = "source /opt/miniconda3/bin/activate 2>/dev/null; source /opt/conda/bin/activate 2>/dev/null; conda activate testbed 2>/dev/null || true"
    _pybin = "PYBIN=$(ls /opt/conda/envs/testbed/bin/python /opt/miniconda3/envs/testbed/bin/python 2>/dev/null | head -1); [ -n \"$PYBIN\" ] || PYBIN=$(command -v python3 || command -v python)"
    _install_run = ("$PYBIN -m " + _install) if _install.startswith("pip ") else _install
    _test_run = ("$PYBIN -m " + _test_full) if _test_full.startswith("pytest") else _test_full
    eval_script_list = [
        "set -o pipefail",
        _pybin, _pathprep, _act,
        f"cd {repo_directory}",
        "export PYTHONPATH=$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}",
        f"git config --global --add safe.directory {repo_directory} 2>/dev/null || true",
        _install_run,
        _reset,
    ]
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/test_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            "git apply -v /tmp/test_patch.diff || git apply -v --reject /tmp/test_patch.diff || patch --batch --fuzz=5 -p1 -i /tmp/test_patch.diff || true"
        )
    eval_script_list += [
        f": '{START_TEST_OUTPUT}'",
        _test_run,
        f": '{END_TEST_OUTPUT}'",
        _reset,
    ]
    arch = "x86_64"

    image_name = instance.get("image_url") or instance.get("image_name")

    return GradingTestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language="py",
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=image_name,
    )


async def _run_tests_with_swebench_harness(
    instance: dict,
    patch: str,
    config: dict,
    result: dict,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    resource_level: int = 1,
) -> dict:
    import time
    reward_start_time = time.time()
    instance_id = instance.get("instance_id", "unknown")
    logger.info(f"[SWE_REWARD] Starting _run_tests_with_swebench_harness for instance_id={instance_id}")

    _is_ml = False
    if instance.get("dataset", "") == "AweAI-Team/Scale-SWE":
        test_spec = make_test_spec_scaleswe(instance)
    elif instance.get("dataset", "") == "nebius/SWE-rebench-V2":
        test_spec = make_test_spec_swerebench_v2(instance)
    elif (not instance.get("dataset")) and instance.get("install_config"):
        test_spec = make_test_spec_swerebench_v2(instance)
    elif instance.get("dataset", "") == "swe-bench/SWE-Bench_Multilingual":
        try:
            from . import swe_multilingual as _SM_build
            class _MLSpec:
                instance_id = instance.get("instance_id", "")
                repo = instance.get("repo", "")
                version = str(instance.get("version", ""))
            _MLSpec.eval_script = _SM_build._build_eval_script(instance)
            test_spec = _MLSpec()
            _is_ml = True
        except Exception as _mle:
            logger.warning(f"[SWE_REWARD] multilingual eval-script build failed for {instance.get('instance_id')}, falling back to swebench's make_test_spec: {_mle}")
            test_spec = make_test_spec(instance)
    else:
        try:
            test_spec = make_test_spec(instance)
        except Exception as e:
            result["error"] = f"Failed to create TestSpec: {e}"
            logger.error(f"[SWE_REWARD] Failed to create TestSpec for instance_id={instance_id}: {e}")
            return result


    env_config = config.get("environment", {}).copy()
    env_config["environment_class"] = env_config.get("environment_class", "orchard_docker")

    env_config["exe_timeout"] = max(env_config.get("exe_timeout", 60) + 300, SWE_TIMEOUT_REWARD_EXECUTE)

    env_config["cpu"] = str(resource_level * 2)
    env_config["memory"] = f"{resource_level * 8}Gi"
    logger.info(f"[SWE_REWARD] Environment resources for resource_level {resource_level}: cpu={env_config['cpu']}, memory={env_config['memory']}")

    workdir = instance.get("workdir", "/testbed")
    env_config.setdefault("cwd", workdir)
    logger.info(f"[SWE_REWARD] Using workdir={workdir} for instance_id={instance_id}")

    startup_command = config.get("run", {}).get("env_startup_command")

    MAX_CREATE_ENV_RETRIES = 1
    RETRY_WAIT_SECONDS = 5
    env = None

    logger.info(f"[SWE_REWARD] Creating test environment for instance_id={instance_id}")
    _env_create_start = time.time()
    for attempt in range(1, MAX_CREATE_ENV_RETRIES + 1):
        logger.info(f"[SWE_REWARD] Attempt {attempt}/{MAX_CREATE_ENV_RETRIES} to create environment for instance_id={instance_id}")
        try:
            env = await asyncio.wait_for(
                create_environment(
                    env_config,
                    instance_id=f"test-{instance_id}",
                    instance=instance,
                    startup_command=startup_command,
                ),
                timeout=SWE_TIMEOUT_REWARD_CREATE_ENV,
            )
            break
        except asyncio.TimeoutError:
            logger.error(
                f"[SWE_REWARD] TIMEOUT creating test environment for instance_id={instance_id} "
                f"after {SWE_TIMEOUT_REWARD_CREATE_ENV}s (attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying in {RETRY_WAIT_SECONDS}s for instance_id={instance_id}"
                )
                await asyncio.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}"
                )
                result["error"] = (
                    f"Timed out creating test environment after {SWE_TIMEOUT_REWARD_CREATE_ENV}s "
                    f"({MAX_CREATE_ENV_RETRIES} attempts)"
                )
                return result
        except Exception as e:
            logger.error(
                f"[SWE_REWARD] Failed to create test environment for instance_id={instance_id}: {e} "
                f"(attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying in {RETRY_WAIT_SECONDS}s for instance_id={instance_id}"
                )
                await asyncio.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}"
                )
                result["error"] = f"Failed to create test environment: {e} ({MAX_CREATE_ENV_RETRIES} attempts)"
                return result
    assert env is not None, "Environment creation failed without raising an exception"

    logger.info(f"[SWE_REWARD] Test environment created for instance_id={instance_id}, elapsed={time.time() - _env_create_start:.2f}s")

    test_output_path = None
    try:
        logger.info(f"[SWE_REWARD] [{instance_id}] Writing patch to container...")
        await _write_file_to_sandbox(env, "/tmp/patch.diff", patch)
        if _is_ml:
            await _write_file_to_sandbox(env, "/tmp/test_patch.diff", instance.get("test_patch", ""))
        logger.info(f"[SWE_REWARD] [{instance_id}] Patch written, applying with git apply...")

        patch_applied = False
        for git_apply_cmd in [
            f"cd {workdir} && git apply --verbose /tmp/patch.diff",
            f"cd {workdir} && git apply --verbose --reject /tmp/patch.diff",
            f"cd {workdir} && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            apply_result = await _execute(env, git_apply_cmd)
            if apply_result.get("returncode", 0) == 0:
                patch_applied = True
                break
            logger.warning("[SWE_REWARD] [%s] Patch apply failed with: %s", instance_id, git_apply_cmd)

        if not patch_applied:
            result["error"] = f"Failed to apply patch: {apply_result.get('output', '')}"
            return result

        f2p_script = instance.get("f2p_script", "")
        if f2p_script:
            write_f2p_cmd = f"cat > {workdir}/test_fail_to_pass.py << 'SWEAGENT_F2P_SCRIPT_EOF'\n{f2p_script}\nSWEAGENT_F2P_SCRIPT_EOF"
            await _execute(env, write_f2p_cmd)

        eval_script = test_spec.eval_script
        eval_script = eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")

        write_eval_cmd = f"cat > /root/eval.sh << 'EVAL_EOF'\n{eval_script}\nEVAL_EOF"
        await _execute(env, write_eval_cmd)
        await _execute(env, "chmod +x /root/eval.sh")

        run_command = f"cd {workdir}"
        if "pylint" in test_spec.instance_id:
            run_command += " && PYTHONPATH="
        if not _is_ml:
            run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
        run_command += " && /bin/bash /root/eval.sh 2>&1"

        _eval_start = time.time()
        logger.info(f"[SWE_REWARD] [{instance_id}] Running eval script...")

        test_result = await _execute(env, run_command)
        logger.info(f"[SWE_REWARD] [{instance_id}] Eval script finished in {time.time() - _eval_start:.2f}s")
        output = test_result.get("output", "")
        result["output"] = output

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(output)
            test_output_path = Path(f.name)

        pred = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "slime-swe-agent",
            KEY_PREDICTION: patch,
        }

        if _is_ml:
            from . import swe_multilingual as _SM_grade
            _ml = _SM_grade.grade_multilingual(instance, output, exit_code=test_result.get("returncode", 0), patch_applied=patch_applied)
            _r = bool(_ml.get("resolved"))
            result["passed"] = _r
            result["resolved"] = _r
            result["resolution"] = ResolvedStatus.FULL.value if _r else ResolvedStatus.NO.value
            _f2pp = len(_ml.get("fail_to_pass_passed", []))
            _f2pf = len(_ml.get("fail_to_pass_failed", []))
            _p2pp = _ml.get("pass_to_pass_passed", 0)
            _p2pf = len(_ml.get("pass_to_pass_failed", []))
            result["f2p_total"] = _f2pp + _f2pf
            result["f2p_passed"] = _f2pp
            result["p2p_total"] = _p2pp + _p2pf
            result["p2p_passed"] = _p2pp
            result["tests_passed"] = _f2pp + _p2pp
            result["tests_run"] = result["f2p_total"] + result["p2p_total"]
            result["fail_to_pass_rate"] = (_f2pp / result["f2p_total"]) if result["f2p_total"] > 0 else 1.0
            result["pass_to_pass_rate"] = (_p2pp / result["p2p_total"]) if result["p2p_total"] > 0 else 1.0
            logger.info(f"[SWE_REWARD] multilingual grade for {instance_id}: resolved={_r} tests_run={result['tests_run']} f2p={result['f2p_passed']}/{result['f2p_total']}")
            return result
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        instance_report = report.get(instance_id, {})
        resolved = instance_report.get("resolved", False)

        result["passed"] = resolved
        result["resolved"] = resolved
        result["resolution"] = ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value

        tests_status = instance_report.get("tests_status", {})
        f2p_status = tests_status.get(FAIL_TO_PASS, {})
        p2p_status = tests_status.get(PASS_TO_PASS, {})

        f2p_passed = len(f2p_status.get("success", []))
        f2p_failed = len(f2p_status.get("failure", []))
        p2p_passed = len(p2p_status.get("success", []))
        p2p_failed = len(p2p_status.get("failure", []))

        result["f2p_total"] = f2p_passed + f2p_failed
        result["f2p_passed"] = f2p_passed
        result["p2p_total"] = p2p_passed + p2p_failed
        result["p2p_passed"] = p2p_passed
        result["tests_passed"] = f2p_passed + p2p_passed
        result["tests_run"] = result["f2p_total"] + result["p2p_total"]

        if result["f2p_total"] > 0:
            result["fail_to_pass_rate"] = f2p_passed / result["f2p_total"]
        else:
            result["fail_to_pass_rate"] = 1.0

        if result["p2p_total"] > 0:
            result["pass_to_pass_rate"] = p2p_passed / result["p2p_total"]
        else:
            result["pass_to_pass_rate"] = 1.0

    except EnvironmentUnavailable:
        raise
    except Exception as e:
        result["error"] = f"Test execution error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[SWE_REWARD] Test execution error for instance_id={instance_id}: {e}")
    finally:
        if env is not None:
            logger.info(f"[SWE_REWARD] Stopping test environment for instance_id={instance_id}")
            try:
                await asyncio.wait_for(stop_environment(env), timeout=SWE_TIMEOUT_REWARD_EXECUTE)
                logger.info(f"[SWE_REWARD] Test environment stopped for instance_id={instance_id}")
            except asyncio.TimeoutError:
                logger.error(f"[SWE_REWARD] TIMEOUT stopping test environment for instance_id={instance_id}")
            except Exception as e:
                logger.error(f"[SWE_REWARD] Failed to stop test environment for instance_id={instance_id}: {e}")
        else:
            logger.warning(f"[SWE_REWARD] No environment to stop for instance_id={instance_id} (env was None)")
        if test_output_path is not None and test_output_path.exists():
            try:
                test_output_path.unlink()
            except Exception:
                pass

    result["reward_eval_time"] = time.time() - reward_start_time
    logger.info(
        "[SWE_REWARD] Test results for %s: resolved=%s f2p=%.3f p2p=%.3f tests=%s error=%s total_time=%.2fs",
        instance_id,
        result.get("resolved", False),
        result.get("fail_to_pass_rate", 0.0),
        result.get("pass_to_pass_rate", 0.0),
        result.get("tests_run", 0),
        bool(result.get("error")),
        result["reward_eval_time"],
    )
    return result


# ---- reward computation functions ----

async def compute_simple_reward(args, sample: Sample) -> float:
    final_output = sample.metadata.get("final_output", "")

    if not final_output:
        return -1.0

    patch = extract_patch_from_output(final_output)

    if not is_valid_patch(patch):
        return -1.0

    instance_id = sample.metadata.get("instance_id", "unknown")
    repo = sample.metadata.get("repo", "")
    base_commit = sample.metadata.get("base_commit", "")

    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = []

    test_results = await run_tests_in_docker(
        instance_id=instance_id,
        patch=patch,
        repo=repo,
        base_commit=base_commit,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        instance=sample.metadata,
    )

    unresolved_reward = float(os.environ.get("SWE_UNRESOLVED_REWARD", "-1.0"))
    final_reward = 1.0 if test_results["resolved"] else unresolved_reward

    test_results["final_reward"] = final_reward
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return final_reward


SUBMITTED_EXIT_STATUS = "Submitted"


async def compute_simple_truncated_zero_reward(args, sample: Sample) -> float:
    exit_status = (
        sample.metadata.get("exit_status", "Unknown") if sample.metadata else "Unknown"
    )

    if exit_status != SUBMITTED_EXIT_STATUS:
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        logger.info(
            f"[SWE_REWARD] Trajectory not submitted (exit_status={exit_status}) "
            f"for instance_id={instance_id}, assigning reward=-2.0"
        )
        truncated_results = {
            "truncated": True,
            "exit_status": exit_status,
            "final_reward": -2.0,
        }
        trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(truncated_results, f)
                    logger.info(f"test_results: {json.dumps(truncated_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
        return -2.0

    simple_reward = await compute_simple_reward(args, sample)
    return 1.0 if simple_reward > 0 else 0.0


async def reward_func(args, sample: Sample, **kwargs) -> float:
    instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
    logger.info(f"[SWE_REWARD] Starting reward_func for instance_id={instance_id}")

    reward_mode = getattr(args, "swe_reward_mode", None)
    if reward_mode is None:
        reward_mode = os.environ.get("SWE_REWARD_MODE", "simple")

    is_eval = bool(kwargs.get("evaluation", False))
    if is_eval:
        eval_mode = os.environ.get("SWE_EVAL_REWARD_MODE", "simple_truncated_zero")
        if eval_mode != reward_mode:
            logger.info(
                f"[SWE_REWARD] {instance_id}: eval rollout -> overriding "
                f"reward_mode '{reward_mode}' -> '{eval_mode}'"
            )
        reward_mode = eval_mode

    try:
        if reward_mode == "simple":
            result = await asyncio.wait_for(compute_simple_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_truncated_zero":
            result = await asyncio.wait_for(compute_simple_truncated_zero_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        logger.info(f"[SWE_REWARD] Completed reward_func for instance_id={instance_id}, reward={result}")
        assert isinstance(result, (float, int)), f"Reward function must return a scalar float or int, got {type(result)}"
        return result
    except asyncio.TimeoutError:
        logger.error(
            f"[SWE_REWARD] reward_func TIMED OUT after {SWE_TIMEOUT_REWARD_TOTAL}s for instance_id={instance_id}. "
        )
        return -1.0
    except Exception as e:
        logger.error(f"[SWE_REWARD] reward_func ERROR for instance_id={instance_id}: {e}\n{traceback.format_exc()}")
        return -1.0
