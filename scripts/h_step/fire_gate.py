#!/usr/bin/env python3
"""fire_gate.py <candidate> — check that a candidate's mechanism actually runs.

Prints FIRED=yes ... (exit 0), or FIRED=no REASON=<...> (exit 1). Exit 3 means the server or the
sandbox service failed, not the candidate.
"""
import sys, json, glob, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def fail(reason):
    print(f"FIRED=no REASON={reason}")
    sys.exit(1)

if len(sys.argv) < 2:
    print("usage: fire_gate.py <candidate_name>"); sys.exit(2)
name = sys.argv[1]
cand = HERE / "agents" / name
if not cand.is_dir():
    fail(f"agents/{name} does not exist")
agent_py = cand / "agent.py"
if not agent_py.exists():
    fail(f"agents/{name}/agent.py missing")
code = agent_py.read_text()
mk_path = cand / "MARKER.txt"
if not mk_path.exists():
    fail(f"write your unique MARKER token to agents/{name}/MARKER.txt")
marker = mk_path.read_text().strip()
if not marker:
    fail("MARKER.txt is empty -- put your unique firing token in it")

# ---- static check ----
if "class AgentHarness" not in code:
    fail("your class MUST be named exactly 'class AgentHarness(ProgressTrackingAgent):' (literal name).")
uses_env = ("self.env" in code) or (".env.execute" in code)
bad = []
if re.search(r'\bimport\s+subprocess\b|\bsubprocess\.', code): bad.append("subprocess (a LOCAL process; the repo is NOT local)")
if "patch.txt" in code: bad.append("patch.txt (there is NO local patch file -- the submission is in the Submitted exception)")
if re.search(r'Path\(\s*["\']/testbed["\']\s*\)\s*\.\s*(exists|read_text|glob|iterdir|is_dir|open)', code): bad.append("local Path('/testbed').<fs op>")
if re.search(r'open\(\s*["\'][^"\']*(?:testbed|patch\.txt)', code): bad.append("local open() on the repo/patch")
if re.search(r'os\.(listdir|walk|scandir)\(', code) and "/testbed" in code: bad.append("os.listdir/walk on /testbed (local)")
if bad and not uses_env:
    fail("LOCAL FILE ACCESS -> your mechanism will silently no-op on EVERY task. Found: " + "; ".join(bad) +
         ". FIX: the repo runs in a REMOTE sandbox. Read the submission from the Submitted exception "
         "(inspect the exception object), and run any repo command inside the sandbox via self.env.execute. "
         "No subprocess, no patch.txt, no local Path.")

# ---- run on 3 tasks ----
import search_eval
ids = json.load(open(HERE / "data" / "check_ids.json"))
try:
    (cand / ".hook_fired").unlink()
except Exception:
    pass
out = search_eval.evaluate(name, "firegate", ids, n_pass=1, workers=3, log=lambda *a: None)["out_dir"]

hf = cand / ".hook_fired"
hookval = hf.read_text().strip() if hf.exists() else "(absent)"
trajs = glob.glob(f"{out}/*/*/*.traj.json")
submitted = 0; fired = 0
for f in trajs:
    try: j = json.load(open(f))
    except Exception: continue
    if (j.get("info") or {}).get("exit_status") == "Submitted": submitted += 1
    if any(m.get("token") == marker for m in (j.get("info") or {}).get("harness_marks") or []): fired += 1

if hookval != "AgentHarness":
    fail(f"hook loaded '{hookval}', not 'AgentHarness' -> your class is not named/subclassed correctly "
         f"(must be 'class AgentHarness(ProgressTrackingAgent):').")
if fired == 0:
    if submitted == 0:
        _ran = 0
        for f in trajs:
            try: j = json.load(open(f))
            except Exception: continue
            if any(m.get("role") == "assistant" for m in j.get("messages") or []): _ran += 1
        if _ran == 0:
            print(f"INFRA: none of {len(trajs)} gate tasks produced a model turn (serve or sandbox pool down)")
            sys.exit(3)
        fail(f"none of {len(trajs)} gate tasks reached a submit and your MARKER was not recorded on any of them; "
             f"self.mark(MARKER) must run on the first line of your handler.")
    fail(f"MARKER '{marker}' was not recorded in any of {submitted} submitted trajs -> your mechanism DID NOT RUN. "
         f"The class loaded but your handler never fired. Call self.mark(MARKER) UNCONDITIONALLY on the FIRST line of "
         f"your handler; the submission is in the Submitted exception; any repo access must go through "
         f"self.env.execute (NOT local files).")
# ---- submit check ----
from collections import Counter as _Counter
_exits = _Counter(); _raw = ""
for _f in trajs:
    try: _j = json.load(open(_f))
    except Exception: continue
    _ex = (_j.get("info") or {}).get("exit_status")
    _exits[_ex] += 1
    if _ex and _ex not in ("Submitted","LimitsExceeded","WallClockTimeout","TimeoutError") and not _raw:
        _tb = (_j.get("info") or {}).get("traceback") or json.dumps(_j)
        _m = re.search(r'(System message must[^"\\]{0,80}|BadRequestError[^"\\]{0,160}|[A-Za-z]*Error[^"\\]{0,120})', _tb)
        _raw = (_m.group(1) if _m else str(_ex))[:200]
_nsub = _exits.get("Submitted", 0)
_loops = _exits.get("LimitsExceeded", 0) + _exits.get("WallClockTimeout", 0) + _exits.get("TimeoutError", 0)
_others = ", ".join(f"{k}={v}" for k,v in _exits.items() if k != "Submitted")
if _raw and ("BadRequest" in _raw or "System message" in _raw):
    fail("your mechanism BREAKS the model call [" + _raw + "] on the check tasks (" + _others + "). "
         "A working harness must NOT make the request invalid. HINT: if you inject a message use role:'user' "
         "(NOT 'system' -- a system message must be the FIRST message only); and if the submit turn raised Submitted "
         "BEFORE its tool result was appended, answer that dangling tool_call (add a role:'tool' msg with the matching "
         "tool_call_id) BEFORE you swallow, or the next model call is malformed.")
if _nsub < 2:
    _hint = ""
    if _loops >= 2:
        _hint = (" HINT: the runs ended at the step/time limit instead of submitting -> a common cause is code that swallows "
                 "EVERY submit: the run ends only when a submission goes through, so the swallow needs a condition under which "
                 "it lets the submission through (e.g. a per-instance flag). And to re-raise the ORIGINAL submission use a BARE "
                 "`raise` -- never `raise Submitted()` with no args (that submits an EMPTY patch).")
    fail("your mechanism PREVENTS submission: only " + str(_nsub) + "/" + str(len(trajs)) + " check tasks SUBMITTED (" +
         _others + "). raw: " + (_raw or "(none)") + ". The starting harness submitted on each of these tasks in its seed "
         "run, so the candidate is keeping the model from submitting." + _hint)

# ---- swallow check ----
def _has_inblock_swallow(_src):
    import ast as _ast
    try: _tree = _ast.parse(_src)
    except Exception: return False
    for _fn in _ast.walk(_tree):
        if not isinstance(_fn, _ast.FunctionDef): continue
        _has_sub = any(isinstance(_n, _ast.ExceptHandler) and _n.type is not None and "Submitted" in _ast.dump(_n.type) for _n in _ast.walk(_fn))
        _has_ret = any(isinstance(_n, _ast.Return) and isinstance(_n.value, _ast.List) and len(_n.value.elts) == 0 for _n in _ast.walk(_fn))
        if _has_sub and _has_ret: return True
    return False
if "return []" in code and not _has_inblock_swallow(code):
    _cont = 0
    for _f in trajs:
        try: _j = json.load(open(_f))
        except Exception: continue
        _msgs = _j.get("messages") or []
        _first = next((m.get("at_message") for m in (_j.get("info") or {}).get("harness_marks") or []
                       if m.get("token") == marker), None)
        if _first is None: continue
        if any((_m.get("role") == "assistant") for _m in _msgs[_first:]):
            _cont += 1
    if _cont == 0:
        fail("INERT SWALLOW: your code has `return []` (you intend to SWALLOW the submission and give the model "
             "another turn) but on ALL " + str(len(trajs)) + " check tasks the run ENDED right at the submit where "
             "your MARKER was recorded -- there is NO model turn AFTER the marker, so the model NEVER saw your message and the "
             "swallow had no effect. The usual cause: your `return []` is inside a HELPER method, "
             "but execute_actions()/step() calls the helper then does `raise`, which DISCARDS the `return []` and "
             "re-raises the ORIGINAL Submitted. FIX: execute_actions()/step() must RETURN the swallow value ITSELF -- "
             "`return self._your_helper(...)` (helper returns [] to swallow, or None/re-raise to let through), or put "
             "`return []` directly in the `except Submitted` block. Never `helper(); raise`. (If you did NOT intend to "
             "swallow, remove the `return []`.)")
print(f"FIRED=yes  marker='{marker}' fired in {fired}/{len(trajs)} trajs ({submitted} submitted); hook={hookval}")
