"""Test-log parsers shared by swerebench_verify_orchard and swebench_multilingual_verify_orchard:
each _parse_log_* maps raw test output to {test_name: status}."""

import re
import xml.etree.ElementTree as ET

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _ansi_escape(text: str) -> str:
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


def _parse_log_pytest(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = parts[0]
    return sm


def _parse_log_pytest_options(log: str) -> dict[str, str]:
    opt_re = re.compile(r"(.*?)\[(.*)\]")
    sm = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                m = opt_re.search(parts[1])
                if m:
                    main, opt = m.groups()
                    if opt.startswith("/") and not opt.startswith("//") and "*" not in opt:
                        opt = "/" + opt.split("/")[-1]
                    name = f"{main}[{opt}]"
                else:
                    name = parts[1]
                sm[name] = parts[0]
    return sm


def _parse_log_pytest_v2(log: str) -> dict[str, str]:
    sm = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(str.maketrans("", "", escapes))
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = parts[0]
        elif any(line.endswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[0]] = parts[1]
    return sm


def _parse_log_django(log: str) -> dict[str, str]:
    sm = {}
    prev_test = None
    for line in log.split("\n"):
        line = line.strip()
        if "--version is equivalent to version" in line:
            sm["--version is equivalent to version"] = "PASSED"
        if " ... " in line:
            prev_test = line.split(" ... ")[0]
        for suffix in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suffix):
                sm[line.rsplit(suffix, 1)[0]] = "PASSED"
                break
        if " ... skipped" in line:
            sm[line.split(" ... skipped")[0]] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            sm[line.split(" ... FAIL")[0]] = "FAILED"
        if line.startswith("FAIL:"):
            sm[line.split()[1].strip()] = "FAILED"
        if line.endswith(" ... ERROR"):
            sm[line.split(" ... ERROR")[0]] = "ERROR"
        if line.startswith("ERROR:"):
            sm[line.split()[1].strip()] = "ERROR"
        if line.lstrip().startswith("ok") and prev_test is not None:
            sm[prev_test] = "PASSED"
    return sm


def _parse_log_seaborn(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        if line.startswith("FAILED"):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = "FAILED"
        elif " PASSED " in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "PASSED":
                sm[parts[0]] = "PASSED"
        elif line.startswith("PASSED"):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = "PASSED"
    return sm


def _parse_log_sympy(log: str) -> dict[str, str]:
    sm = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        sm[f"{match[1]}.py:{match[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                sm[line.split()[0]] = "ERROR"
            elif line.endswith(" F"):
                sm[line.split()[0]] = "FAILED"
            elif line.endswith(" ok"):
                sm[line.split()[0]] = "PASSED"
    return sm


def _parse_log_matplotlib(log: str) -> dict[str, str]:
    log = log.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3")
    return _parse_log_pytest(log)


def _parse_log_gotest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^--- (PASS|FAIL|SKIP): (.+) \(.+\)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = {"PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED"}[status]
    return sm


def _parse_log_ant(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*\[junit\]\s+\[(PASS|FAIL|ERR)\]\s+(.*)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = "PASSED" if status == "PASS" else "FAILED"
    return sm


def _parse_log_gradle(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^([^>].+)\s+(PASSED|FAILED)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, status = m.groups()
            sm[name] = status
    return sm


def _parse_log_jest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(✓|✕|○)\s(.+?)(?:\s\((\d+\s*m?s)\))?$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            sym, name, _ = m.groups()
            sm[name] = {"✓": "PASSED", "✕": "FAILED", "○": "SKIPPED"}[sym]
    return sm


def _parse_log_jest_json(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\[(PASSED|FAILED)\]\s(.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = status
    return sm


def _parse_log_vitest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(✓|×|↓)\s(.+?)(?:\s(\d+\s*m?s?|\[skipped\]))?$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            sym, name, _ = m.groups()
            sm[name] = {"✓": "PASSED", "×": "FAILED", "↓": "SKIPPED"}[sym]
    return sm


def _parse_log_tap(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^(ok|not ok) (\d+) (.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, _, name = m.groups()
            sm[name] = "PASSED" if status == "ok" else "FAILED"
    return sm


def _parse_log_marked(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        m = re.search(r"^\d+\)\s(.*)", line)
        if m:
            sm[m.group(1).strip()] = "FAILED"
    return sm


def _parse_log_react_pdf(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        for pattern, status in [
            (r"^PASS\s(.*)\s\([\d\.]+ ?[ms]+\)", "PASSED"),
            (r"^PASS\s(.*)", "PASSED"),
            (r"^FAIL\s(.*)\s\([\d\.]+ ?[ms]+\)", "FAILED"),
            (r"^FAIL\s(.*)", "FAILED"),
        ]:
            m = re.match(pattern, line)
            if m:
                sm[m.group(1)] = status
                break
    return sm


def _parse_log_phpunit(log: str) -> dict[str, str]:
    sm = {}
    suite = None
    suite_pat = re.compile(r"^(\w.+) \(.+\)$")
    test_pat = re.compile(r"^\s*([✔✘↩])\s*(.*)$")
    for line in log.split("\n"):
        m = suite_pat.match(line)
        if m:
            suite = m.group(1)
            continue
        m = test_pat.match(line)
        if m:
            sym, name = m.groups()
            full = f"{suite} > {name}"
            sm[full] = {"✔": "PASSED", "✘": "FAILED", "↩": "SKIPPED"}[sym]
    return sm


def _parse_log_ruby_unit(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(?:test: )?(.+):\s+(\.|E\b|F\b|O\b)")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            sm[name] = {".": "PASSED", "E": "FAILED", "F": "FAILED", "O": "SKIPPED"}[outcome]
    return sm


def _parse_log_rspec_json(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"(.+) - (passed|failed|pending)")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            sm[name] = {"passed": "PASSED", "failed": "FAILED", "pending": "SKIPPED"}[outcome]
    return sm


def _parse_log_minitest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^(.+)\. .*=.*(\.|F|E).*$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            sm[name] = {".": "PASSED", "F": "FAILED", "E": "FAILED"}[outcome]
    return sm


def _parse_log_jekyll(log: str, instance_id: str) -> dict[str, str]:
    pr = instance_id.split("-")[1]
    if pr in ["9141", "8047", "8167"]:
        return _parse_log_minitest(log)
    elif pr in ["8761", "8771"]:
        sm = {}
        pat = re.compile(r"^(.*) \.+(\.|F)")
        for line in log.split("\n"):
            m = pat.match(line.strip())
            if m:
                name, outcome = m.groups()
                sm[name] = "PASSED" if outcome == "." else "FAILED"
        return sm
    return {}


def _parse_log_cargo(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(\w+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            if outcome == "ok":
                sm[name] = "PASSED"
            elif outcome == "FAILED":
                sm[name] = "FAILED"
    return sm


def _parse_log_redis(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\[(ok|err|skip|ignore)\]:\s(.+?)(?:\s\([\d\s\w]+\))?$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            if status == "ok":
                sm[name] = "PASSED"
            elif status == "err":
                name = re.sub(r"\s+in\s+\S+$", "", name)
                sm[name] = "FAILED"
            elif status in ("skip", "ignore"):
                sm[name] = "SKIPPED"
    return sm


def _parse_log_jq(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(PASS|FAIL):\s(.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = "PASSED" if status == "PASS" else "FAILED"
    return sm


def _parse_log_doctest(log: str) -> dict[str, str]:
    sm = {}
    start = log.find("<doctest")
    end_tag = "</doctest>"
    end = log.find(end_tag, start) + len(end_tag) if start != -1 else -1
    if start != -1 and end != -1:
        try:
            root = ET.fromstring(log[start:end])
            for tc in root.findall(".//TestCase"):
                tc_name = tc.get("name")
                for sc in tc.findall(".//SubCase"):
                    sc_name = sc.get("name")
                    name = f"{tc_name} > {sc_name}"
                    exprs = sc.findall(".//Expression")
                    sm[name] = "PASSED" if all(e.get("success") == "true" for e in exprs) else "FAILED"
        except ET.ParseError:
            pass
    return sm


def _parse_log_micropython(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^(pass|FAIL|skip)\s+(.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = {"pass": "PASSED", "FAIL": "FAILED", "skip": "SKIPPED"}[status]
    return sm


def _parse_log_googletest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^.*\[\s*(OK|FAILED)\s*\]\s(.*)\s\(.*\)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = "PASSED" if status == "OK" else "FAILED"
    return sm


def _get_changed_files(patch: str) -> list[str]:
    result = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            path = line[6:]
            if path and path != "/dev/null":
                result.append(path)
    return result


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()
