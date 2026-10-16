import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = [
    "check_reward_nonzero_std",
    "check_no_aborted",
    "check_no_aborted_nonzero_std_and_pos_reward",
]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_no_aborted(args, samples: list[Sample], **kwargs):
    has_aborted = any(s.status == Sample.Status.ABORTED for s in samples)
    return DynamicFilterOutput(
        keep=not has_aborted,
        reason=None if not has_aborted else "aborted",
    )


def check_no_aborted_nonzero_std_and_pos_reward(args, samples: list[Sample], **kwargs):
    result = check_no_aborted(args, samples, **kwargs)
    if not result.keep:
        return result
    result = check_reward_nonzero_std(args, samples, **kwargs)
    if not result.keep:
        return result
    rewards = [sample.get_reward_value(args) for sample in samples]
    has_positive = any(r > 0 for r in rewards)
    return DynamicFilterOutput(
        keep=has_positive,
        reason=None if has_positive else f"no_positive_reward_max_{round(max(rewards), 1)}",
    )
