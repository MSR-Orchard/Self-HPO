#!/bin/bash
# run_p_step.sh --policy <HF dir> --harness <harness dir> --out <dir> [--init-dist <torch_dist dir>] [--name <run name>]
# P-step: GRPO on the RL training set, with the agent running under the given harness.
#   --policy     initial policy: a HF model dir or a Hugging Face model id, e.g. Qwen/Qwen3.5-35B-A3B for the first P-step
#   --harness    harness dir (agent.py + config.yaml), e.g. the output of the preceding H-step
#   --out        output dir: ckpt/ (Megatron checkpoints), trajectories/ (rollouts), and policy/
#                (the exported policy, written by export_policy.sh)
#   --init-dist  torch_dist of the policy (default <out>/init_torch_dist, built by prepare_ckpt.sh)
#   --name       run name (default p_step): W&B group/run names and the rollout subdir under trajectories/
# Before training it evaluates the initial policy once on SWE-bench Verified (--eval-interval 100000,
# so there is no further in-training evaluation).
# Single node (8 GPUs), inside the slimerl/slime:v0.3.0 container (setup_container.sh first).
# Required env: SELFHPO_WORK (holds rl/data from prepare_rl_data.py), SANDBOX_BASE_URL, SANDBOX_API_KEY
# Optional env: WANDB_API_KEY (W&B logging; offline if unset), WANDB_ENTITY,
#               SLIME_ROOT (slime checkout, default <repo>/slime)
set -e
NAME=p_step
while [ $# -gt 0 ]; do case "$1" in
  --policy) POLICY=$2; shift 2;; --harness) HARNESS=$(realpath "$2"); shift 2;;
  --out) OUT=$2; shift 2;; --init-dist) INIT_DIST=$(realpath "$2"); shift 2;; --name) NAME=$2; shift 2;;
  *) echo "unknown arg $1"; exit 1;; esac; done
: "${POLICY:?--policy}" "${HARNESS:?--harness}" "${OUT:?--out}"
source "$(dirname "${BASH_SOURCE[0]}")/../resolve_model.sh"
POLICY=$(resolve_model "$POLICY")
[ -f "$HARNESS/agent.py" ] || { echo "no agent.py in $HARNESS"; exit 1; }
mkdir -p "$OUT"; OUT=$(realpath "$OUT"); INIT_DIST=${INIT_DIST:-$OUT/init_torch_dist}
if [ -f "$OUT/run_inputs.txt" ] && [ "$(cat "$OUT/run_inputs.txt")" != "$POLICY $HARNESS" ]; then
  echo "ERR: $OUT was used for: $(cat "$OUT/run_inputs.txt"); use a new --out for each P-step"; exit 1
fi
echo "$POLICY $HARNESS" > "$OUT/run_inputs.txt"
SELFHPO_WORK=$(realpath "${SELFHPO_WORK:?set SELFHPO_WORK}")
SLIME_ROOT=${SLIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../slime" && pwd)}
[ -f "$INIT_DIST/latest_checkpointed_iteration.txt" ] || bash "$(dirname "${BASH_SOURCE[0]}")/prepare_ckpt.sh" "$POLICY" "$INIT_DIST"
export SWE_HARNESS_DIR=$HARNESS
export SANDBOX_API_KEY=${SANDBOX_API_KEY:?set SANDBOX_API_KEY}
export SANDBOX_BASE_URL=${SANDBOX_BASE_URL}
export WANDB_MODE=$([ -n "${WANDB_API_KEY:-}" ] && echo online || echo offline)
mkdir -p $OUT/ckpt $OUT/trajectories
cd ${SLIME_ROOT}

export WANDB_BASE_URL=https://api.wandb.ai
export PYTHONUNBUFFERED=1
export SWE_TRAJECTORY_DIR=$OUT/trajectories

export PYTHONPATH=${SLIME_ROOT}:/root/Megatron-LM/:${PYTHONPATH}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

source "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

# ---- swe ----
export SWE_REWARD_MODE="simple"
export SWE_UNRESOLVED_REWARD="-1.0"
export SWE_CONFIG_PATH="examples/mini_swe/configs/swe_agent_qwen3.5.yaml"
export SWE_TIMEOUT_CREATE_ENV=480
export SWE_TIMEOUT_LLM_INFERENCE=60
export SWE_TIMEOUT_GET_OBSERVATION=90
export SWE_TIMEOUT_STOP_ENV=60
export SWE_TIMEOUT_REWARD_EXECUTE=900
export SWE_TIMEOUT_REWARD_TOTAL=1200
export SWE_MAX_ENV_RETRIES=2
export SWE_ENV_RETRY_WAIT=10
export SWE_MAX_CREATE_ENV_RETRIES=1
export SWE_CREATE_ENV_RETRY_WAIT=10
export SWE_ENV_CREATE_JITTER_MAX=5

# ---- hyperparameters ----
OVER_SAMPLING_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=32
N_SAMPLES=4
N_SAMPLES_MAX=8
N_SAMPLES_STRIDE=8
POS_RATIO_MIN=0.25
POS_RATIO_MAX=0.75
GLOBAL_BATCH_SIZE=128
ADVANTAGE_ESTIMATOR="grpo"
LR="1e-6"
NUM_NODES=1
PPO_EPOCHS=1

RUN_TAG="qwen3.5-35b-a3b_${NAME}"
export WANDB_RANDOM_SUFFIX="${RUN_TAG}"
WANDB_GROUP_NAME="${NAME}-grpo-pe${PPO_EPOCHS}_r${ROLLOUT_BATCH_SIZE}_n${N_SAMPLES}_max${N_SAMPLES_MAX}_s${N_SAMPLES_STRIDE}_pmin${POS_RATIO_MIN}_pmax${POS_RATIO_MAX}_g${GLOBAL_BATCH_SIZE}_${ADVANTAGE_ESTIMATOR}_lr${LR}_${NUM_NODES}nodes_${RUN_TAG}"
export WANDB_RUN_ID="${WANDB_GROUP_NAME}_$(cat /proc/sys/kernel/random/uuid | cut -c1-6)"

CKPT_ARGS=(
   --hf-checkpoint ${POLICY}
   --ref-load ${INIT_DIST}
   --load ${OUT}/ckpt
   --save ${OUT}/ckpt
   --save-interval 5
)

ROLLOUT_ARGS=(
   --prompt-data ${SELFHPO_WORK}/rl/data/train.jsonl
   --input-key problem_statement
   --label-key patch
   --multimodal-keys '{}'
   --rollout-shuffle

   --custom-generate-function-path examples.mini_swe.swe_generate_harness.generate
   --custom-rm-path examples.mini_swe.swe_reward.reward_func
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted_nonzero_std_and_pos_reward

   --eval-prompt-data swe_val ${SELFHPO_WORK}/rl/data/swebench_verified_500.jsonl
   --eval-interval 100000
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-temperature 0.95
   --eval-top-p 0.95
   --num-rollout 50
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES}
   --n-samples-per-prompt-max ${N_SAMPLES_MAX}
   --n-samples-per-prompt-stride ${N_SAMPLES_STRIDE}
   --progressive-rollout-pos-ratio-min ${POS_RATIO_MIN}
   --progressive-rollout-pos-ratio-max ${POS_RATIO_MAX}
   --rollout-max-response-len 16384
   --rollout-temperature 0.95
   --rollout-top-p 0.95

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)


PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator ${ADVANTAGE_ESTIMATOR}
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 1e-4
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
WANDB_ARGS=(
   --use-wandb
   --wandb-host ${WANDB_BASE_URL}
   --wandb-project self-hpo
   ${WANDB_ENTITY:+--wandb-team ${WANDB_ENTITY}}
   --wandb-group ${WANDB_GROUP_NAME}
   --wandb-run-id ${WANDB_RUN_ID}
   --disable-wandb-random-suffix
)
fi

SGLANG_ARGS=(
   --sglang-moe-runner-backend triton
   --rollout-num-gpus-per-engine 2
   --sglang-enable-dp-attention
   --sglang-data-parallel-size 2
   --sglang-expert-parallel-size 2
   --sglang-server-concurrency 128
   --sglang-max-running-requests 128
   --sglang-chunked-prefill-size 16384
   --sglang-mem-fraction-static 0.7
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

# ---- ray ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
pkill -TERM sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 -f 'train.py' || true
sleep 3
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265
echo "Waiting for Ray GCS on ${MASTER_ADDR}:6379..."
for i in $(seq 1 30); do
  if python3 -c "import socket; s=socket.create_connection(('${MASTER_ADDR}', 6379), timeout=2); s.close()" 2>/dev/null; then echo "GCS reachable after ${i}s."; break; fi
  if [ "$i" -eq 30 ]; then echo "ERROR: GCS not reachable after 30s."; exit 1; fi
  sleep 1
done
sleep 5

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"SWE_REWARD_MODE\": \"${SWE_REWARD_MODE}\",
    \"SWE_UNRESOLVED_REWARD\": \"${SWE_UNRESOLVED_REWARD}\",
    \"SWE_CONFIG_PATH\": \"${SWE_CONFIG_PATH}\",
    \"SWE_HARNESS_DIR\": \"${SWE_HARNESS_DIR}\",
    \"SWE_TIMEOUT_CREATE_ENV\": \"${SWE_TIMEOUT_CREATE_ENV}\",
    \"SWE_TIMEOUT_LLM_INFERENCE\": \"${SWE_TIMEOUT_LLM_INFERENCE}\",
    \"SWE_TIMEOUT_GET_OBSERVATION\": \"${SWE_TIMEOUT_GET_OBSERVATION}\",
    \"SWE_TIMEOUT_STOP_ENV\": \"${SWE_TIMEOUT_STOP_ENV}\",
    \"SWE_TIMEOUT_REWARD_EXECUTE\": \"${SWE_TIMEOUT_REWARD_EXECUTE}\",
    \"SWE_TIMEOUT_REWARD_TOTAL\": \"${SWE_TIMEOUT_REWARD_TOTAL}\",
    \"SWE_MAX_ENV_RETRIES\": \"${SWE_MAX_ENV_RETRIES}\",
    \"SWE_ENV_RETRY_WAIT\": \"${SWE_ENV_RETRY_WAIT}\",
    \"SWE_MAX_CREATE_ENV_RETRIES\": \"${SWE_MAX_CREATE_ENV_RETRIES}\",
    \"SWE_CREATE_ENV_RETRY_WAIT\": \"${SWE_CREATE_ENV_RETRY_WAIT}\",
    \"SWE_ENV_CREATE_JITTER_MAX\": \"${SWE_ENV_CREATE_JITTER_MAX}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"SWE_TRAJECTORY_DIR\": \"${SWE_TRAJECTORY_DIR}\",
    \"WANDB_RANDOM_SUFFIX\": \"${WANDB_RANDOM_SUFFIX}\",
    \"SANDBOX_BASE_URL\": \"${SANDBOX_BASE_URL}\",
    \"SANDBOX_API_KEY\": \"${SANDBOX_API_KEY}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

echo "P-step finished: $OUT/ckpt"
