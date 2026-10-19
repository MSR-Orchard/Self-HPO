#!/bin/bash
# setup.sh — proposer tooling for the H-step: litellm proxy (work/envs/proposer) + Claude Code 2.1.250.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p "$SELFHPO_WORK/logs"
${PYTHON:-python3} -m venv "$SELFHPO_WORK/envs/proposer"
"$SELFHPO_WORK/envs/proposer/bin/pip" install -q --upgrade pip
"$SELFHPO_WORK/envs/proposer/bin/pip" install -q "litellm[proxy]==1.89.1"
command -v claude >/dev/null && echo "claude found: $(claude --version)" || curl -fsSL https://claude.ai/install.sh | bash -s 2.1.250
echo "proposer tooling ready (Claude Code: $(claude --version 2>/dev/null || echo 'add ~/.local/bin to PATH'))"
