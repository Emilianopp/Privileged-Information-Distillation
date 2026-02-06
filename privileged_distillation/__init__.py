"""
Privileged Information Distillation

A PyTorch library for training models with privileged information using
various reinforcement learning and distillation techniques.

Main Components:
- losses: Novel loss functions (PPO, TOPR, Distillation, KL)
- utils: Supporting utilities for log-probs, importance sampling, rewards
"""

__version__ = "0.1.0"

from privileged_distillation.losses import (
    calculate_ppo_clipped_importance_sampling,
    calculate_topr_loss,
    on_policy_distillation,
    compute_kl_divergence_rao_blackwellized,
)

from privileged_distillation.utils import (
    get_sample_log_probs,
    compute_snis_weights,
    rescale_rewards,
    process_logits_memory_efficient,
)

__all__ = [
    # Losses
    "calculate_ppo_clipped_importance_sampling",
    "calculate_topr_loss",
    "on_policy_distillation",
    "compute_kl_divergence_rao_blackwellized",
    # Utils
    "get_sample_log_probs",
    "compute_snis_weights",
    "rescale_rewards",
    "process_logits_memory_efficient",
]
