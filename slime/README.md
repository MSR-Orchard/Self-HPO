# slime (vendored)

Snapshot of [slime](https://github.com/THUDM/slime) (Apache-2.0, see `LICENSE`) used for the Self-HPO P-step.
It is based on slime v0.3.0 and adds:

- progressive group sampling in `slime/rollout/sglang_rollout.py`. A group is sampled in strides up to
  `--n-samples-per-prompt-max`, and a training group with a positive-reward fraction in
  `[--progressive-rollout-pos-ratio-min, --progressive-rollout-pos-ratio-max]` is kept.
  The four arguments (with `--n-samples-per-prompt-stride`) are added to `slime/utils/arguments.py`.
  `slime/rollout/data_source.py` reserves `--n-samples-per-prompt-max` sample indices per group.
- deterministic sampling seeds for all `--n-samples-per-prompt-max` samples of a group (`slime/rollout/sglang_rollout.py`)
- a per-group info writer in `slime/rollout/sglang_rollout.py`. It appends one jsonl line per sample
  (exit status, reward, status, trajectory path, sample index, token length) to the sample's `group_info_path`.
- eval sample indices offset by the rollout id, so evaluations at different steps get distinct indices
  (`slime/rollout/sglang_rollout.py`)
- the `check_no_aborted` and `check_no_aborted_nonzero_std_and_pos_reward` dynamic-sampling filters
  (`slime/rollout/filter_hub/dynamic_sampling_filters.py`)
- handling of samples without a reward (aborted rollouts) in training and eval logging (`slime/ray/rollout.py`)
- `--ppo-epochs` (`slime/backends/megatron_utils/model.py`) and `--grpo-per-step-discount`
  (`slime/backends/megatron_utils/loss.py`, `slime/utils/ppo_utils.py`). Both are off by default.
  Their arguments are in `slime/utils/arguments.py`.
- retries of transient connection errors on SGLang engine requests (`slime/backends/sglang_utils/sglang_engine.py`)
- progress logging while waiting for the rollout engines to offload (`train.py`)
- `examples/mini_swe/`: mini-swe-agent rollout and SWE-task reward for agentic RL

Upstream parts that Self-HPO does not use were removed.
These are `docs/`, `docker/`, `tests/`, `imgs/`, `.github/`, `.claude/`, `.agents/`, `train_async.py`,
`build_conda.sh`, `README_zh.md`, `CONTRIBUTING.md`, `.gitignore` and `.pre-commit-config.yaml`.
All upstream examples other than `examples/mini_swe/` were removed, and so was `slime_plugins/rollout_buffer/`.
In `scripts/`, the `run-*.sh` launchers and `low_precision/` were removed, and `scripts/models/` keeps only `qwen3.5-35B-A3B.sh`.
`tools/` keeps only `convert_hf_to_torch_dist.py`, `convert_torch_dist_to_hf.py` and `fp8_cast_bf16.py`.

Run it inside the `slimerl/slime:v0.3.0` image (Megatron-LM at `/root/Megatron-LM`).
