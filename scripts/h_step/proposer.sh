#!/bin/bash
# proposer.sh — run Claude Code as the harness proposer.
#   PROPOSER=self (default): Claude Code driven by the served policy through the litellm proxy
#                            (start_proxy.sh); this is Self-HPO.
#   PROPOSER=api           : Claude Code with an API model (needs ANTHROPIC_API_KEY; PROPOSER_MODEL optional).
if [ "${PROPOSER:-self}" = self ]; then
  export ANTHROPIC_BASE_URL=http://127.0.0.1:4001
  export ANTHROPIC_AUTH_TOKEN=sk-local
  export ANTHROPIC_API_KEY=sk-local
  for v in ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL \
           ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL; do export $v=policy; done
  export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
  curl -sf http://127.0.0.1:4001/health/liveliness >/dev/null || {
    echo "ERR: no proxy on 127.0.0.1:4001 (run scripts/h_step/start_proxy.sh)" >&2; exit 1; }
elif [ -n "${PROPOSER_MODEL:-}" ]; then
  export ANTHROPIC_MODEL=$PROPOSER_MODEL
fi
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 DISABLE_ERROR_REPORTING=1
exec claude "$@"
