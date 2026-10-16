"""SWE-bench Multilingual grading for in-training rewards (used by swe_reward.py): per-language log
parsers, the repo-to-parser map, the eval-script builder and the grader."""
import json, re, logging
import xml.etree.ElementTree as ET
logger = logging.getLogger("swe_multilingual")


START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
DOCKER_WORKDIR = "/testbed"
FAIL_ONLY_REPOS = {"chartjs/Chart.js", "processing/p5.js", "markedjs/marked"}

NON_TEST_EXTS = [
    ".json", ".png", ".txt", ".md", ".jpg", ".jpeg",
    ".pkl", ".zip", ".csv", ".parquet", ".feather", ".svg",
    ".npy", ".yml", ".yaml", ".toml",
]

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


# ---- log parsers ----


def _ansi_escape(text: str) -> str:
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


# ---- python parsers ----

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


# ---- go parsers ----

def _parse_log_gotest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^--- (PASS|FAIL|SKIP): (.+) \(.+\)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = {"PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED"}[status]
    return sm


# ---- java parsers ----

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


# ---- javascript parsers ----

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


# ---- php parsers ----

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


# ---- ruby parsers ----

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


# ---- rust parsers ----

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


# ---- c parsers ----

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


# ---- repo-to-parser mapping ----

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


# ---- specs loading helpers ----


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


# ---- eval script building ----

def _get_modified_files(patch: str) -> list[str]:
    return [line[6:] for line in patch.split("\n") if line.startswith("--- a/")]


def _get_changed_files(patch: str) -> list[str]:
    result = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            path = line[6:]
            if path and path != "/dev/null":
                result.append(path)
    return result


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


# ---- test output parsing ----

def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _parse_test_output(log: str, instance: dict) -> dict[str, str]:
    repo = instance["repo"]
    instance_id = instance["instance_id"]

    if repo == "jekyll/jekyll":
        parser = lambda l: _parse_log_jekyll(l, instance_id)
    elif repo == "immutable-js/immutable-js":
        parser = lambda l: _parse_log_immutable_js(l, instance_id)
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


# ---- evaluation ----




def grade_multilingual(instance, output, exit_code=0, patch_applied=True):
    instance_id = instance.get("instance_id", "unknown")
    repo = instance.get("repo", "")
    f2p_raw = instance.get("FAIL_TO_PASS", []) or []
    p2p_raw = instance.get("PASS_TO_PASS", []) or []
    if isinstance(f2p_raw, str): f2p_raw = json.loads(f2p_raw)
    if isinstance(p2p_raw, str): p2p_raw = json.loads(p2p_raw)
    fail_to_pass = [_normalize_test_name(n) for n in f2p_raw]
    pass_to_pass = [_normalize_test_name(n) for n in p2p_raw]
    fail_only = repo in FAIL_ONLY_REPOS
    output = output or ""
    bad_codes = [c for c in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT] if c in output]
    if bad_codes or (START_TEST_OUTPUT not in output and END_TEST_OUTPUT not in output):
        parsed = {}
    else:
        parsed = _parse_test_output(output, instance)
        parsed = {_normalize_test_name(k): v for k, v in parsed.items()}
    passed = {k for k, v in parsed.items() if v in ("PASSED", "XFAIL")}
    def _check(case):
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
    return {"instance_id": instance_id, "resolved": resolved, "exit_code": exit_code,
            "patch_applied": patch_applied, "fail_to_pass_passed": f2p_passed,
            "fail_to_pass_failed": f2p_failed, "pass_to_pass_passed": len(p2p_passed),
            "pass_to_pass_failed": p2p_failed, "parsed_tests_count": len(parsed), "error": ""}
