"""SWE-bench grading for SWE-rebench and Scale-SWE instances."""

import re
from dataclasses import dataclass
from typing import Any

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.test_spec import TestSpec

_STATUSES = ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]


@dataclass
class GradingTestSpec(TestSpec):
    install_config: dict = None
    image_name: str = None


def parse_log_pytest(log, test_spec=None):
    status = {}
    for line in log.split("\n"):
        if any(line.startswith(x) for x in _STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                status[parts[1]] = parts[0]
    return status


def parse_log_pytest_v2(log, test_spec=None):
    status = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(str.maketrans("", "", escapes))
        if any(line.startswith(x) for x in _STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                status[parts[1]] = parts[0]
        elif any(line.endswith(x) for x in _STATUSES):
            parts = line.split()
            if len(parts) >= 2:
                status[parts[0]] = parts[1]
    return status


def parse_log_pytest_scaleswe(log, test_spec=None):
    status = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            parts = line.split(None, 1)
            if len(parts) <= 1:
                continue
            status[parts[1].strip()] = parts[0]
    return status


LOG_PARSERS = {
    "parse_log_pytest": parse_log_pytest,
    "parse_log_pytest_options": parse_log_pytest,
    "parse_log_pytest_v2": parse_log_pytest_v2,
    "parse_log_pytest_scaleswe": parse_log_pytest_scaleswe,
}


def resolve_log_parser(test_spec):
    install_config = getattr(test_spec, "install_config", None) or {}
    name = install_config.get("log_parser") if isinstance(install_config, dict) else None
    if name in LOG_PARSERS:
        return LOG_PARSERS[name]
    if test_spec.repo in MAP_REPO_TO_PARSER:
        return MAP_REPO_TO_PARSER[test_spec.repo]
    return parse_log_pytest_v2


def get_logs_eval(test_spec, log_fp: str) -> tuple[dict[str, str], bool]:
    log_parser = resolve_log_parser(test_spec)
    with open(log_fp) as f:
        content = f.read()
    if any(x in content for x in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]):
        return {}, False
    if not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        return {}, False
    test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    status_map = log_parser(test_content, test_spec)
    if not status_map:
        status_map = log_parser(content, test_spec)
    return status_map, True


def get_eval_report(test_spec, prediction: dict[str, str], test_log_path: str, include_tests_status: bool) -> dict[str, Any]:
    instance_id = prediction[KEY_INSTANCE_ID]
    report_map = {instance_id: {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
    }}
    if prediction[KEY_PREDICTION] is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    eval_status_map, found = get_logs_eval(test_spec, test_log_path)
    if not found:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True
    if include_tests_status:
        report_map[instance_id]["tests_status"] = report
    return report_map
