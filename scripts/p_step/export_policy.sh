#!/bin/bash
# export_policy.sh <p_step out dir> [iteration]
# Export a P-step checkpoint as a HF policy: <out>/policy.
# iteration defaults to the latest checkpoint (<out>/ckpt/latest_checkpointed_iteration.txt).
set -euo pipefail
OUT=$(realpath "${1:?usage: export_policy.sh <p_step out dir> [iteration]}")
if [ -n "${2:-}" ]; then IT=$2
else
  LATEST="$OUT/ckpt/latest_checkpointed_iteration.txt"
  [ -f "$LATEST" ] || { echo "no $LATEST: pass the iteration explicitly"; exit 1; }
  IT=$(tr -d '[:space:]' < "$LATEST")
fi
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$OUT/run_inputs.txt" ] || { echo "no $OUT/run_inputs.txt (run run_p_step.sh first)"; exit 1; }
INIT=$(cut -d" " -f1 "$OUT/run_inputs.txt")
SLIME_ROOT=${SLIME_ROOT:-$HERE/../../slime}
D=$(printf "%s/ckpt/iter_%07d" "$OUT" "$IT")
[ -f "$D/.metadata" ] || [ -f "$D/metadata.json" ] || { echo "no complete checkpoint at $D"; exit 1; }
cd "$SLIME_ROOT"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/root/Megatron-LM:$SLIME_ROOT python3 tools/convert_torch_dist_to_hf.py \
  --input-dir "$D" --output-dir "$OUT/policy_export" --origin-hf-dir "$INIT" -a -f
BASE_HF=$INIT python3 "$HERE/fuse_experts.py" "$OUT/policy_export" "$OUT/policy"
BASE_HF=$INIT python3 "$HERE/verify_fused.py" "$OUT/policy" "$OUT/policy_export"
rm -rf "$OUT/policy_export"
echo "policy -> $OUT/policy"
