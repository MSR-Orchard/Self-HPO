import re
from minisweagent.run.benchmarks.swebench import ProgressTrackingAgent
from minisweagent.exceptions import Submitted


class AgentHarness(ProgressTrackingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._touched_files: set[str] = set()
        self._first_submit = None

    def step(self):
        if self._first_submit is None:
            self._first_submit = True
        self._touched_files = set()
        return super().step()

    def _extract_touched_files_from_message(self, message: dict) -> set[str]:
        touched = set()

        actions = message.get("extra", {}).get("actions", [])
        for action in actions:
            cmd = action.get("command", "")
            if not cmd:
                continue

            read_patterns = [
                r'cat\s+([^\n|;&]+)',
                r'head\s+(?:-\d+\s+)?([^\n|;&]+)',
                r'tail\s+(?:-\d+\s+)?([^\n|;&]+)',
                r'grep\s+(?:-[a-zA-Z]+\s+)*["\'][^"\']+["\']\s+([^\n|;&]+)',
                r'grep\s+(?:-[a-zA-Z]+\s+)*-l\s+["\'][^"\']+["\']\s+([^\n|;&]+)',
                r'rg\s+(?:-[a-zA-Z]+\s+)*["\'][^"\']+["\']\s+([^\n|;&]+)',
                r'rg\s+(?:-[a-zA-Z]+\s+)*-l\s+["\'][^"\']+["\']\s+([^\n|;&]+)',
                r'less\s+([^\n|;&]+)',
                r'tac\s+([^\n|;&]+)',
            ]

            write_patterns = [
                r'sed\s+-i\s+([^\n|;&]+)',
                r'echo\s+[^>]+\s*>+\s+([^\n|;&]+)',
                r'>>\s+([^\n|;&]+)',
                r'>\s+([^\n|;&]+)',
            ]

            for pat in read_patterns + write_patterns:
                for m in re.finditer(pat, cmd):
                    f = m.group(1).strip()

                    f = f.replace('"', '').replace("'", '').strip()

                    if f and len(f) > 2 and not f.startswith('-'):
                        touched.add(f)

            if 'diff --git' in cmd or 'git diff' in cmd:
                for m in re.finditer(r'diff --git a/([a-zA-Z0-9_/.-]+)', cmd):
                    touched.add(m.group(1))

        return touched

    def _get_submitted_files_from_sandbox(self) -> set[str]:
        try:
            result = self.env.execute({"command": "cd /testbed && git diff --cached HEAD 2>/dev/null || git diff HEAD 2>/dev/null"})
            output = result.get("output", "")
            if not output:
                return set()

            submitted = set()
            for m in re.finditer(r'diff --git a/([a-zA-Z0-9_/.-]+) b/', output):
                submitted.add(m.group(1))

            return submitted

        except Exception:
            return set()

    def execute_actions(self, message: dict) -> list[dict]:
        touched = self._extract_touched_files_from_message(message)
        self._touched_files.update(touched)

        try:
            return super().execute_actions(message)
        except Submitted as e:
            submitted_files = self._get_submitted_files_from_sandbox()
            submitted_count = len(submitted_files)
            touched_count = len(self._touched_files)

            missing = self._touched_files - submitted_files

            audit = f"""
[Full Files Auditor - Coverage Check]
----------------------------------------
You touched {touched_count} files during exploration: {sorted(self._touched_files)}
Your submitted patch has {submitted_count} files: {sorted(submitted_files) if submitted_files else '(none)'}

{'FOUND GAP: ' + str(len(missing)) + ' touched files NOT in your patch:' if missing else 'Coverage check: all touched files appear in patch.'}
"""
            if missing:
                audit += "\nPotential missing files:\n"
                for f in sorted(missing)[:10]:
                    audit += f"  - {f}\n"
                if len(missing) > 10:
                    audit += f"  ... and {len(missing) - 10} more"

            self.add_messages({
                "role": "user",
                "content": audit
            })

            decorators_used = self._check_decorator_imports_in_patch()

            if decorators_used:
                decorator_msg = f"""

[Decorator Reminder]
----------------------------------------
Your patch uses these decorator(s) which may need importing:
{chr(10).join('  - ' + d for d in decorators_used)}

Common patterns:
  - @abstractmethod: from abc import abstractmethod
  - @lru_cache/@cache: from functools import lru_cache
  - @singledispatch: from functools import singledispatch
  - @override: from typing import override

Make sure these are imported before submitting."""
                self.add_messages({
                    "role": "user",
                    "content": decorator_msg
                })

            pr_terms = self._extract_pr_intent_terms()
            patch_terms_found = self._check_patch_for_pr_terms(pr_terms)

            if pr_terms and not patch_terms_found:
                pr_msg = f"""

[PR Intent Reminder]
----------------------------------------
Your patch may not address the core PR intent. Key terms from PR:
{chr(10).join('  - ' + t for t in pr_terms[:5])}

Your submitted patch does not mention these key terms.
Consider: Does your edit actually address the issue described?
"""
                self.add_messages({
                    "role": "user",
                    "content": pr_msg
                })

            if self._first_submit:
                self._first_submit = False
                return []

            raise e

    def _check_decorator_imports_in_patch(self) -> list[str]:
        try:
            result = self.env.execute(
                {"command": "cd /testbed && git diff --cached HEAD 2>/dev/null || git diff HEAD 2>/dev/null || true"}
            )
            patch = result.get("output", "")
            if not patch:
                return []

            decorators_used = []

            builtins = {'property', 'setter', 'deleter', 'classmethod', 'staticmethod',
                       'abstractclassmethod', 'abstractstaticmethod', 'cached_property',
                       'lru_cache', 'cache', 'singledispatch', 'singledispatchmethod',
                       'final', 'no_type_check', 'override'}

            for line in patch.split('\n'):
                if line.startswith('+ @'):
                    match = re.match(r'\+ *@([a-zA-Z_][a-zA-Z0-9_]*)', line)
                    if match:
                        dec_name = match.group(1)
                        if dec_name not in builtins:
                            decorators_used.append(dec_name)

            return list(set(decorators_used))
        except Exception:
            return []

    def _extract_pr_intent_terms(self) -> list[str]:
        task_content = self._get_task_description()
        if not task_content:
            return []

        terms = []

        error_patterns = [
            r'\b([A-Z][A-Za-z]*Error)\b',
            r'\b([a-z]+[A-Z][A-Za-z]+)\b',
        ]
        for pat in error_patterns:
            terms.extend(re.findall(pat, task_content))

        version_terms = ['semver', 'semantic', 'version', 'compatibility', 'incompatibility',
                        'match', 'spec', 'wrappe', 'typ', 'unwrapp']
        task_lower = task_content.lower()
        for term in version_terms:
            if term in task_lower:
                terms.append(term)

        func_patterns = [
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'method\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]
        for pat in func_patterns:
            terms.extend(re.findall(pat, task_content, re.I))

        quoted_patterns = [
            r'"([^"]{5,50})"',
            r"'([^']{5,50})'",
        ]
        for pat in quoted_patterns:
            matches = re.findall(pat, task_content)
            terms.extend(matches)

        return list(set(terms))[:10]

    def _get_task_description(self) -> str:
        if len(self.messages) < 2:
            return ""

        task_msg = self.messages[1]
        content = task_msg.get("content", "")

        pr_match = re.search(r'<pr_description>\s*\n*(.*?)\s*\n*<\/pr_description>', content, re.DOTALL)
        if pr_match:
            return pr_match.group(1)

        return content

    def _check_patch_for_pr_terms(self, terms: list[str]) -> bool:
        if not terms:
            return False

        try:
            result = self.env.execute(
                {"command": "cd /testbed && git diff --cached HEAD 2>/dev/null || git diff HEAD 2>/dev/null || true"}
            )
            patch = result.get("output", "")
            if not patch:
                return False

            patch_lower = patch.lower()
            for term in terms:
                if term.lower() in patch_lower:
                    return True

            return False

        except Exception:
            return False
