#!/bin/bash
# prepare_ckpt.sh <policy HF dir> <out torch_dist dir>
# Convert a HF policy into a Megatron torch_dist checkpoint.
# Run inside the slime container on one GPU.
set -euo pipefail
SRC=$(realpath "${1:?usage: prepare_ckpt.sh <policy HF dir> <out torch_dist dir>}"); OUT=${2:?out dir}
SLIME_ROOT=${SLIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../slime" && pwd)}
mkdir -p "$(dirname "$OUT")"; OUT=$(realpath "$OUT")
if grep -q '"model.language_model.layers.0.mlp.experts.0.gate_proj.weight"' "$SRC/model.safetensors.index.json"; then
  echo "$SRC stores per-expert weights (a raw slime export); build a fused HF first: python3 scripts/p_step/fuse_experts.py <in> <out>"; exit 1
fi
cd "$SLIME_ROOT"
source scripts/models/qwen3.5-35B-A3B.sh
PYTHONPATH=/root/Megatron-LM:$SLIME_ROOT WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
  --bf16 --no-gradient-accumulation-fusion --hf-checkpoint "$SRC" --save "$OUT"
echo "wrote $OUT"
