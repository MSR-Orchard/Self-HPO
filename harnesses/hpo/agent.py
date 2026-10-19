import base64
import time

from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.swebench import ProgressTrackingAgent


class AgentHarness(ProgressTrackingAgent):
    MARGIN = 30
    RESCUE_STEP_MARGIN = 15
    RESCUE_TIME_SECONDS = 150.0
    FB_WORKDIR = "/testbed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grounded = False
        self._empty_guarded = False
        self._rescued = False
        self._submitted_once = False
        self._fallback_done = False
        self._fallback_fired = False

    def _answer_dangling_toolcalls(self, note: str) -> None:
        if not self.messages:
            return
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        placeholder = f"<returncode>0</returncode>\n<output>\n[{note}]\n</output>"
        for tc in (last.get("tool_calls") or []):
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if not tcid:
                continue
            self.add_messages({
                "role": "tool",
                "tool_call_id": tcid,
                "content": placeholder,
                "extra": {"returncode": 0, "synthetic": "harness_mechanism"},
            })

    def _sh(self, command: str, timeout: int = 120) -> dict:
        try:
            return self.env.execute({"command": command}, cwd=self.FB_WORKDIR, timeout=timeout)
        except Submitted:
            return {"output": "", "returncode": -1}
        except Exception as e:
            return {"output": "", "returncode": -1, "exception_info": str(e)}

    def _sub_applies_clean(self, sub: str) -> bool:
        b64 = base64.b64encode(sub.encode("utf-8", "surrogatepass")).decode("ascii")

        self._sh("rm -f /tmp/fbsub.b64 /tmp/fbsub.patch")
        for i in range(0, len(b64), 60000):
            chunk = b64[i:i + 60000]
            self._sh(f"printf %s '{chunk}' >> /tmp/fbsub.b64")

        cmd = (
            "cd /testbed || exit 0; "
            "base64 -d /tmp/fbsub.b64 > /tmp/fbsub.patch 2>/dev/null || { echo FBRC=200; exit 0; }; "
            ""
            "git stash push --include-untracked -q >/dev/null 2>&1 || { echo FBRC=202; exit 0; }; "
            "git apply --check /tmp/fbsub.patch >/dev/null 2>&1; RC=$?; "
            "git stash pop -q >/dev/null 2>&1; "
            "echo FBRC=$RC"
        )
        out = self._sh(cmd)
        text = out.get("output", "") or ""
        if "FBRC=" not in text:
            return True
        try:
            rc = int(text.rsplit("FBRC=", 1)[1].split()[0])
        except Exception:
            return True

        return rc == 0 or rc >= 200

    def _authoritative_diff(self) -> str:
        cmd = (
            "cd /testbed; "
            "u=$(git ls-files --others --exclude-standard 2>/dev/null); "
            "[ -n \"$u\" ] && git add -N $u >/dev/null 2>&1 || true; "
            "git --no-pager diff"
        )
        out = self._sh(cmd)
        return out.get("output", "") or ""

    def _apply_fallback(self, e) -> None:
        if self._fallback_done:
            return
        self._fallback_done = True
        try:
            extra = (e.messages[0].get("extra", {}) or {}) if e.messages else {}
        except Exception:
            return
        sub = extra.get("submission", "") or ""
        if self._is_empty_submission(sub):
            return
        if self._sub_applies_clean(sub):
            return
        auth = self._authoritative_diff()
        if auth and auth.strip() and not self._is_empty_submission(auth):
            e.messages[0]["extra"]["submission"] = auth
            e.messages[0]["content"] = auth
            self._fallback_fired = True

    def _ground_text(self) -> str:
        return (
            "[GROUND YOUR DIFF IN THE REPO'S EXISTING TESTS — one-time check; your "
            "submission is NOT finalized yet]\n\n"
            "The most common way this task is failed is submitting a patch that APPLIES "
            "cleanly but implements the WRONG contract — a guessed function/option name, a "
            "signature the codebase never uses, a rename or new spelling that no caller "
            "expects, or an error/return shape different from what the code around it uses. "
            "Re-reading the PR will NOT catch this, because your reading of the PR is what "
            "produced the wrong contract. The codebase itself is the authority. The repo "
            "ships EXISTING tests that already exercise the code you touched and the API "
            "next to it — they show the EXACT names, signatures, argument spellings, and "
            "error/return types the code is expected to have. Use them as ground truth now.\n\n"
            "1. Print your changes and identify the concrete symbols you added or changed:\n"
            "   git --no-pager diff\n"
            "   List every NEW or RENAMED symbol/option/message string in your diff.\n"
            "2. For each such symbol, search the repo's tests and existing call sites for how "
            "that symbol — and the nearest EXISTING sibling API — is actually used:\n"
            "   grep -rn --include='*.py' -e '<symbol>' -e '<sibling_api>' . | head -40\n"
            "   Then OPEN and READ the specific test files that reference the area you changed "
            "(e.g. tests/ or test_*.py near your file) and note the exact spelling, call "
            "signature, and expected result/exception the codebase uses.\n"
            "3. Confront your diff against what the tests/call sites actually expect. If you "
            "introduced a name, option spelling, signature, or error shape that NO existing "
            "test or caller uses — while an existing convention does — change your diff to "
            "MATCH the codebase's convention. Do not invent a new API; conform to the one "
            "already there. Also confirm you did not MISS a behavior the tests imply is "
            "required.\n"
            "4. If you changed anything, regenerate the patch:\n"
            "   git diff -- <the source files you modified> > patch.txt\n"
            "   and re-inspect patch.txt so it contains ONLY your intended source changes.\n"
            "5. When your diff conforms to the codebase's real contract, re-issue the EXACT "
            "submit command:\n"
            "   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "This check happens only once — after you re-submit, it goes through."
        )

    def _empty_guard_text(self) -> str:
        return (
            "[YOUR SUBMISSION IS EMPTY — it contains no source-code changes, so it cannot fix "
            "anything. Not finalized.]\n\n"
            "You have not edited any source file yet (patch.txt is empty or has no diff hunks). "
            "Do NOT give up. Re-localize the bug:\n"
            "1. Re-read the PR and grep for the exact symbol/message/flag it names.\n"
            "2. Open the file where that symbol lives and make the concrete edit the PR requires.\n"
            "3. IMPORTANT — did your fix CREATE a NEW file (a brand-new module)? A plain "
            "`git diff` does NOT show untracked new files, so your patch looks empty even though "
            "the file exists. Stage new files first so they appear as additions:\n"
            "   cd /testbed && git add -N <the new file(s) you created>\n"
            "4. Verify with `cd /testbed && git --no-pager diff` that a real change (or a "
            "`new file mode` hunk) now shows.\n"
            "5. Then `cd /testbed && git diff -- <files> > patch.txt` and submit with:\n"
            "   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        )

    def _rescue_text(self) -> str:
        return (
            "[BUDGET ALMOST EXHAUSTED — STOP EXPLORING AND SUBMIT NOW]\n\n"
            "You are nearly out of steps/time and have not submitted a patch. An empty result "
            "scores zero, but a focused best-effort edit can pass. Right now:\n"
            "1. Decide the single most likely fix location from what you already know — do NOT "
            "start new open-ended exploration.\n"
            "2. Make that concrete edit to the source file (the smallest change that matches the "
            "PR's literal request).\n"
            "3. Immediately create and submit the patch, in SEPARATE commands:\n"
            "   git diff -- <the source files you modified> > patch.txt\n"
            "   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n"
            "Do this in your next one or two commands."
        )

    @staticmethod
    def _is_empty_submission(sub: str) -> bool:
        if not sub:
            return True
        s = sub.strip()
        if not s:
            return True

        return not any(marker in s for marker in
                       ("diff --git", "@@ ", "--- a/", "+++ b/", "new file mode"))

    def step(self) -> list[dict]:
        if not self._rescued and not self._submitted_once:
            near_step = self.n_calls >= self.config.step_limit - self.RESCUE_STEP_MARGIN
            near_time = bool(self.deadline) and (
                time.monotonic() > self.deadline - self.RESCUE_TIME_SECONDS)
            if near_step or near_time:
                self._rescued = True
                self.add_messages(self.model.format_message(role="user", content=self._rescue_text()))

        try:
            return super().step()
        except Submitted as e:
            self._submitted_once = True
            sub = ""
            try:
                sub = (e.messages[0].get("extra", {}) or {}).get("submission", "") if e.messages else ""
            except Exception:
                sub = ""

            if not self._empty_guarded and self._is_empty_submission(sub):
                if self.n_calls < self.config.step_limit - self.MARGIN:
                    self._empty_guarded = True
                    self._answer_dangling_toolcalls(
                        "submission held — it contains no diff; see the next message")
                    self.add_messages(self.model.format_message(role="user", content=self._empty_guard_text()))
                    return []

                self._apply_fallback(e)
                raise

            if self._grounded or self.n_calls >= self.config.step_limit - self.MARGIN:
                self._apply_fallback(e)
                raise
            self._grounded = True
            self._answer_dangling_toolcalls(
                "submission held — a one-time test-contract grounding check was triggered; "
                "see the next message")
            self.add_messages(self.model.format_message(role="user", content=self._ground_text()))
            return []

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"harness_mechanisms": {
            "grounded": self._grounded,
            "empty_guarded": self._empty_guarded,
            "rescued": self._rescued,
            "fallback_fired": self._fallback_fired,
        }}}, *extra_dicts)
