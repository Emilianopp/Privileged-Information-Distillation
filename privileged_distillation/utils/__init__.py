"""Utility functions for privileged information distillation."""

from privileged_distillation.utils.logprob_utils import (
    get_sample_log_probs,
    get_log_probs_for_positions,
)
from privileged_distillation.utils.importance_sampling import compute_snis_weights
from privileged_distillation.utils.reward_utils import (
    rescale_rewards,
    apply_control_variate_flip,
)
from privileged_distillation.utils.memory_utils import process_logits_memory_efficient

__all__ = [
    "get_sample_log_probs",
    "get_log_probs_for_positions",
    "compute_snis_weights",
    "rescale_rewards",
    "apply_control_variate_flip",
    "process_logits_memory_efficient",
]
