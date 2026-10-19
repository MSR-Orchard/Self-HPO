#!/bin/bash
# setup.sh <env|serve|swe_pro|rl_data|all> — evaluation setup.
#   env      eval environment: this repo's mini-swe-agent + Orchard sandbox client + swebench
#   serve    serving environment: SGLang built from commit 3be5a7ec8
#   swe_pro  ScaleAI SWE-bench Pro per-task run scripts
#   rl_data  RL training tasks for the H-step search set and the P-step (scripts/p_step/prepare_rl_data.py)
#   all      env + serve + swe_pro + rl_data
# Set PYTHON=python3.12 if your default python3 is another version.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PY=${PYTHON:-python3}

setup_env() {
  if command -v uv >/dev/null; then
    uv venv --python "$PY" "$SELFHPO_EVAL_ENV"; PIP="uv pip install --python $SELFHPO_EVAL_ENV/bin/python"
  else
    "$PY" -m venv "$SELFHPO_EVAL_ENV"; "$SELFHPO_EVAL_ENV/bin/python" -m pip install -U pip
    PIP="$SELFHPO_EVAL_ENV/bin/python -m pip install"
  fi
  $PIP -e "$SELFHPO_REPO/mini-swe-agent"
  $PIP "orchard-env @ git+https://github.com/microsoft/Orchard-Agentic.git#subdirectory=orchard_env"
  $PIP "swebench==4.1.0" "litellm==1.89.2" "datasets>=4.5.0" unidiff pyyaml
  "$SELFHPO_EVAL_ENV/bin/python" -c "import minisweagent, orchard_env, swebench; print('eval env OK:', orchard_env.__version__)"
}

setup_serve() {
  command -v uv >/dev/null || { echo "please install uv (https://docs.astral.sh/uv/)"; exit 1; }
  uv venv --python "$PY" "$SELFHPO_SERVE_ENV"
  uv pip install --python "$SELFHPO_SERVE_ENV/bin/python" --prerelease=allow \
    "sglang @ git+https://github.com/sgl-project/sglang.git@3be5a7ec8976278196e7022a15671cfed83e8782#subdirectory=python"
  "$SELFHPO_SERVE_ENV/bin/python" -c "import sglang, torch; print('serve env OK:', sglang.__version__, torch.__version__)"
}

setup_swe_pro() {
  mkdir -p "$SELFHPO_SWE_PRO"
  [ -d "$SELFHPO_SWE_PRO/sbp_repo/.git" ] || git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os "$SELFHPO_SWE_PRO/sbp_repo"
}

setup_rl_data() {
  "$SELFHPO_EVAL_ENV/bin/python" "$(dirname "$0")/../p_step/prepare_rl_data.py"
}

case "${1:-}" in
  env) setup_env;; serve) setup_serve;; swe_pro) setup_swe_pro;; rl_data) setup_rl_data;;
  all) setup_env; setup_serve; setup_swe_pro; setup_rl_data;;
  *) echo "usage: $0 <env|serve|swe_pro|rl_data|all>"; exit 1;;
esac
