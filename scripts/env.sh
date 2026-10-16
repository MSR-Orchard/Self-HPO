# Common environment for all Self-HPO scripts. Source it:  source scripts/env.sh
#
# Required (export before sourcing, or put them in scripts/env.local.sh):
#   SANDBOX_BASE_URL   Orchard sandbox service, e.g. http://<orchard-endpoint>
#   SANDBOX_API_KEY    its API key
# Optional:
#   SELFHPO_WORK       work dir for envs, data, models, runs   (default: <repo>/work)

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SELFHPO_REPO="$(cd "$_ENV_DIR/.." && pwd)"
[ -f "$_ENV_DIR/env.local.sh" ] && source "$_ENV_DIR/env.local.sh"
export SELFHPO_WORK="${SELFHPO_WORK:-$SELFHPO_REPO/work}"

# ---- python environments ----
export SELFHPO_EVAL_ENV="${SELFHPO_EVAL_ENV:-$SELFHPO_WORK/envs/eval}"
export SELFHPO_SERVE_ENV="${SELFHPO_SERVE_ENV:-$SELFHPO_WORK/envs/serve}"

# ---- shared caches / data ----
export HF_HOME="${HF_HOME:-$SELFHPO_WORK/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SELFHPO_SWE_PRO="$SELFHPO_WORK/swe_pro"
export MSWEA_COST_TRACKING=ignore_errors
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$SELFHPO_WORK" "$HF_HOME"
