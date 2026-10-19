"""parent_check.py <candidate> — reject a candidate that dropped or rewrote its parent's mechanism."""
import os
import sys, ast, re
name = sys.argv[1]
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
try:
    parent = open(f"{base}/{name}/parent").read().strip()
except Exception:
    print("PARENT_CHECK=skip (no parent file)"); sys.exit(0)
if not parent or parent == name:
    print("PARENT_CHECK=skip (no distinct parent)"); sys.exit(0)

def read(p):
    try: return open(p).read()
    except Exception: return None
psrc = read(f"{base}/{parent}/agent.py")
csrc = read(f"{base}/{name}/agent.py")
if psrc is None or csrc is None:
    print("PARENT_CHECK=fail (cannot read agent.py)"); sys.exit(1)

def methods(src):
    try:
        t = ast.parse(src)
        return set(n.name for n in ast.walk(t) if isinstance(n, ast.FunctionDef))
    except Exception:
        return set(re.findall(r'def\s+(\w+)', src))

# ---- parent methods ----
pm, cm = methods(psrc), methods(csrc)
if not pm:
    print("PARENT_CHECK=skip (the parent has no mechanism yet)"); sys.exit(0)
missing = pm - cm
if missing:
    print(f"PARENT_CHECK=fail DROPPED_PARENT_METHODS={sorted(missing)} -- you REWROTE the file. Keep ALL of {parent}'s methods verbatim and only APPEND one new method.")
    sys.exit(1)

# ---- line coverage ----
def codelines(src):
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"): continue
        out.append(s)
    return out
pl = codelines(psrc)
cset = set(codelines(csrc))
present = sum(1 for s in pl if s in cset)
cov = present / max(1, len(pl))
if cov < 0.85:
    print(f"PARENT_CHECK=fail LINE_COVERAGE={cov:.2f} ({present}/{len(pl)} of {parent}'s code lines present) -- you GUTTED/REWROTE the parent's mechanism. Keep its code verbatim and only APPEND your new method + a few wiring lines.")
    sys.exit(1)

print(f"PARENT_CHECK=ok (all {len(pm)} parent methods + {cov:.2f} line-coverage of {parent})")
sys.exit(0)
