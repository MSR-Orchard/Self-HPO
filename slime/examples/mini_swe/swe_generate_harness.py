"""
RL rollout with an arbitrary harness: loads `$SWE_HARNESS_DIR/agent.py` (the same file used for evaluation)
into the slime rollout agent, and the prompt templates from `$SWE_HARNESS_DIR/config.yaml`.

Use with --custom-generate-function-path examples.mini_swe.swe_generate_harness.generate
"""

import asyncio
import importlib.util
import logging
import os
import sys
import time
import types

import yaml

from minisweagent.exceptions import Submitted

from . import swe_generate
from .swe_agent import SWEAgent

logger = logging.getLogger(__name__)


REPO_DIFF_COMMAND = (
    'tmp=$(mktemp) && GIT_INDEX_FILE="$tmp" git read-tree HEAD && GIT_INDEX_FILE="$tmp" git add -A'
    ' && GIT_INDEX_FILE="$tmp" git diff --cached HEAD; rc=$?; rm -f "$tmp"; [ $rc -eq 0 ]'
)


class _EnvBridge:
    def __init__(self, env, loop, timeout):
        self._env, self._loop, self._timeout = env, loop, timeout

    def execute(self, action, cwd="", timeout=None):
        command = action["command"]
        if cwd:
            command = f"cd {cwd} && {command}"
        fut = asyncio.run_coroutine_threadsafe(self._env.execute(command), self._loop)
        return fut.result(timeout or self._timeout)

    def get_template_vars(self):
        return self._env.get_template_vars() if hasattr(self._env, "get_template_vars") else {}


class _ModelFacade:
    @staticmethod
    def format_message(**kwargs):
        return dict(kwargs)


class _EvalAgentFacade:
    def __init__(self, *args, **kwargs):
        self._rl = None
        self._message = None
        self._outputs = []
        self._submission = None

    @property
    def messages(self):
        return self._rl.messages

    @messages.setter
    def messages(self, value):
        self._rl.messages = value

    @property
    def config(self):
        return self._rl.config

    @property
    def n_calls(self):
        return self._rl.n_calls

    @property
    def cost(self):
        return self._rl.cost

    model = _ModelFacade()

    @property
    def deadline(self):
        limit = getattr(self._rl.config, "time_limit", 0) or 0
        return time.monotonic() + (self._rl._created_at + limit - time.time()) if limit > 0 else 0

    @property
    def repo_dir(self):
        return getattr(getattr(self._rl.env, "config", None), "cwd", "") or "/testbed"

    def repo_diff(self):
        out = self.env.execute({"command": REPO_DIFF_COMMAND}, cwd=self.repo_dir)
        return out.get("output", "") if out.get("returncode", 0) == 0 else ""

    def add_messages(self, *messages):
        return self._rl.add_messages(*messages)

    def mark(self, token):
        pass

    def get_template_vars(self, **kwargs):
        return self._rl.get_template_vars(**kwargs)

    def step(self):
        return self.execute_actions(self._message)

    def execute_actions(self, message):
        if self._submission is not None:
            sub = self._submission
            raise Submitted({"role": "exit", "content": sub, "extra": {"exit_status": "Submitted", "submission": sub}})
        return self.add_messages(*self._rl.format_observation_messages(message, self._outputs))


def _load_harness_class(harness_dir):
    runners = ["swebench", "swerebench", "scaleswe", "utils.runner_common"]
    parents = ["minisweagent.run", "minisweagent.run.benchmarks", "minisweagent.run.benchmarks.utils"]
    names = parents + [f"minisweagent.run.benchmarks.{r}" for r in runners]
    saved = {n: sys.modules.get(n) for n in names}
    for n in parents:
        sys.modules.setdefault(n, types.ModuleType(n))
    for n in names[len(parents):]:
        stub = types.ModuleType(n)
        stub.ProgressTrackingAgent = _EvalAgentFacade
        sys.modules[n] = stub
    try:
        spec = importlib.util.spec_from_file_location("_rl_harness_agent", os.path.join(harness_dir, "agent.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    cls = mod.AgentHarness
    if not issubclass(cls, _EvalAgentFacade):
        raise TypeError(f"{harness_dir}/agent.py: AgentHarness must subclass ProgressTrackingAgent")
    return cls


def _load_harness_prompts(harness_dir):
    path = os.path.join(harness_dir, "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    agent, model = cfg.get("agent") or {}, cfg.get("model") or {}
    prompts = {k: agent[k] for k in ("system_template", "instance_template") if agent.get(k)}
    for k in ("observation_template", "format_error_template"):
        if agent.get(k) or model.get(k):
            prompts[k] = agent.get(k) or model.get(k)
    return prompts


_HARNESS_ERRORS = [0]
HARNESS_DIR = os.environ.get("SWE_HARNESS_DIR", "")
HARNESS_CLS = _load_harness_class(HARNESS_DIR) if HARNESS_DIR else None
HARNESS_PROMPTS = _load_harness_prompts(HARNESS_DIR) if HARNESS_DIR else {}
OBS_TIMEOUT = float(os.environ.get("SWE_TIMEOUT_GET_OBSERVATION", "90"))


class HarnessSWEAgent(SWEAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, **HARNESS_PROMPTS})
        self._harness = HARNESS_CLS() if HARNESS_CLS is not None else None
        if self._harness is not None:
            self._harness._rl = self

    def setup(self, task, **kwargs):
        if "workdir" in kwargs:
            kwargs.setdefault("cwd", kwargs["workdir"])
        super().setup(task, **kwargs)

    async def execute_actions(self, actions, timeout=None):
        outputs, submission_info = await super().execute_actions(actions, timeout=timeout)
        if self._harness is None:
            return outputs, submission_info
        h = self._harness
        h.env = _EnvBridge(self.env, asyncio.get_running_loop(), OBS_TIMEOUT)
        h._message = self.messages[-1]
        h._outputs = outputs
        h._submission = submission_info["submission"] if submission_info else None

        def run_step():
            try:
                h.step()
                return None
            except Submitted as e:
                return e

        try:
            exc = await asyncio.to_thread(run_step)
        except Exception as e:
            _HARNESS_ERRORS[0] += 1
            logger.error("harness step failed (%s); ending this rollout as a failure [%d harness errors so far]",
                         e, _HARNESS_ERRORS[0], exc_info=True)
            self.add_messages({"role": "exit", "content": f"harness error: {e!r}",
                               "extra": {"exit_status": "HarnessError", "submission": ""}})
            tool_call_id = submission_info.get("tool_call_id") if submission_info else None
            return outputs, {"submission": "", "tool_call_id": tool_call_id}
        if exc is None:
            if submission_info is not None:
                self._answer_dangling_tool_calls(h._message)
            return [], None
        extra = (exc.messages[0].get("extra") or {}) if getattr(exc, "messages", None) else {}
        submission = extra.get("submission", "")
        tool_call_id = submission_info.get("tool_call_id") if submission_info else None
        return outputs, {"submission": submission, "tool_call_id": tool_call_id}


    _HELD = ("<returncode>0</returncode>\n<output>\n[submission held by the harness; see the next message]\n</output>")

    def _answer_dangling_tool_calls(self, message):
        try:
            i = next(k for k in range(len(self.messages) - 1, -1, -1) if self.messages[k] is message)
        except StopIteration:
            return
        answered = {m.get("tool_call_id") for m in self.messages[i + 1:] if m.get("role") == "tool"}
        held = [{"role": "tool", "tool_call_id": a["tool_call_id"], "content": self._HELD, "extra": {"returncode": 0}}
                for a in (message.get("extra") or {}).get("actions", [])
                if a.get("tool_call_id") and a["tool_call_id"] not in answered]
        self.messages[i + 1:i + 1] = held


swe_generate.SWEAgent = HarnessSWEAgent

from .swe_generate import generate  # noqa: E402,F401
