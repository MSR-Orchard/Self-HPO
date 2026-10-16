#!/bin/bash
# setup_container.sh — run once inside the slimerl/slime:v0.3.0 container before the P-step.
# Installs the rollout / reward dependencies.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
pip freeze --exclude-editable | grep -v "@ file" \
  | grep -viE "^(swebench|mini-swe-agent|unidiff|orchard-env)( |==|@)" > /tmp/selfhpo_image_constraints.txt
PIP="pip install --break-system-packages -q -c /tmp/selfhpo_image_constraints.txt"
$PIP "swebench==4.1.0" "mini-swe-agent==2.3.0" unidiff \
     "orchard-env @ git+https://github.com/microsoft/Orchard-Agentic.git#subdirectory=orchard_env"
cd "$HERE/../../slime"
SWE_TIMEOUT_CREATE_ENV=1 SWE_TIMEOUT_REWARD_EXECUTE=1 SWE_TIMEOUT_REWARD_TOTAL=1 python3 -c "import examples.mini_swe.swe_reward as r, orchard_env; print('rollout/reward modules OK')"
