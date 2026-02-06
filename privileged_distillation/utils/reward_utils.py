"""
Reward Processing Utilities

This module provides utilities for processing and normalizing rewards,
including min-max scaling and control variate transformations.
"""

from typing import List
import torch


def rescale_rewards(rewards: List[float]) -> List[float]:
    """
    Min-max scale a list of rewards to the range [-1, 1].
    
    This normalization helps stabilize training by ensuring rewards are in
    a consistent range across different tasks or environments.
    
    Args:
        rewards: List of reward values.
    
    Returns:
        List of scaled rewards in range [-1, 1].
        - Empty list returns []
        - All identical rewards return all zeros
    
    Example:
        >>> rewards = [0.0, 0.5, 1.0, -0.3, 0.8]
        >>> scaled = rescale_rewards(rewards)
        >>> print(f"Original range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        >>> print(f"Scaled range: [{min(scaled):.2f}, {max(scaled):.2f}]")
        >>> # Scaled range: [-1.00, 1.00]
    """
    if not rewards:
        return []
    
    try:
        min_r = float(min(rewards))
        max_r = float(max(rewards))
    except TypeError:
        # If values are not comparable/numeric, return as-is
        return rewards
    
    # All rewards identical - return zeros
    if max_r == min_r:
        return [0.0 for _ in rewards]
    
    # Min-max scale to [-1, 1]
    denom = max_r - min_r
    scaled = [2.0 * ((float(r) - min_r) / denom) - 1.0 for r in rewards]
    
    # Clamp for numerical safety
    return [max(-1.0, min(1.0, s)) for s in scaled]


def apply_control_variate_flip(rewards: List[float]) -> List[float]:
    """
    Apply control variate to flip reward signs and scale to [-1, 1] or [0, 1].
    
    This transformation is useful for converting rewards from one range to another
    while maintaining relative ordering. It uses the minimum reward as a baseline
    (control variate) and then applies min-max scaling.
    
    The transformation is:
        1. Subtract negative of min reward: r' = r - (-min_r) = r + min_r
        2. Min-max scale to appropriate range
    
    Args:
        rewards: List of original reward values.
    
    Returns:
        List of flipped and scaled rewards.
        - If all flipped rewards are non-negative: scaled to [0, 1]
        - Otherwise: scaled to [-1, 1]
        - Empty list returns []
        - All identical returns all zeros
    
    Example:
        >>> # Rewards with negative values
        >>> rewards = [-1.0, -0.5, 0.0, 0.5, 1.0]
        >>> flipped = apply_control_variate_flip(rewards)
        >>> print(f"Original: {rewards}")
        >>> print(f"Flipped: {[f'{x:.2f}' for x in flipped]}")
        >>> # Flipped rewards maintain relative ordering but in new range
    """
    if not rewards:
        return rewards
    
    min_reward = min(rewards)
    control_variate_baseline = -min_reward
    flipped_rewards = [control_variate_baseline + reward for reward in rewards]
    
    min_flipped = min(flipped_rewards)
    max_flipped = max(flipped_rewards)
    
    # If all rewards are positive, scale to [0, 1]
    if min_flipped >= 0:
        if max_flipped == min_flipped:
            scaled_rewards = [0.0 for _ in flipped_rewards]
        else:
            scaled_rewards = [
                (r - min_flipped) / (max_flipped - min_flipped)
                for r in flipped_rewards
            ]
    else:
        # Scale to [-1, 1]
        if max_flipped == min_flipped:
            scaled_rewards = [0.0 for _ in flipped_rewards]
        else:
            scaled_rewards = [
                2 * (r - min_flipped) / (max_flipped - min_flipped) - 1
                for r in flipped_rewards
            ]
    
    return scaled_rewards


def normalize_rewards(
    rewards: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize rewards to have zero mean and unit variance.
    
    This is a common preprocessing step in reinforcement learning to
    stabilize training and make the algorithm less sensitive to reward scale.
    
    Args:
        rewards: Tensor of rewards. Shape: [N].
        epsilon: Small constant for numerical stability. Default: 1e-8.
    
    Returns:
        Normalized rewards with mean ≈ 0 and std ≈ 1. Shape: [N].
    
    Example:
        >>> rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = normalize_rewards(rewards)
        >>> print(f"Mean: {normalized.mean():.6f}")  # ~0.0
        >>> print(f"Std: {normalized.std():.6f}")    # ~1.0
    """
    mean = rewards.mean()
    std = rewards.std()
    
    # Avoid division by zero
    if std < epsilon:
        return rewards - mean
    
    return (rewards - mean) / (std + epsilon)


def compute_advantages(
    rewards: List[float],
    group_ids: List[int],
) -> List[float]:
    """
    Compute advantages as reward minus group mean.
    
    This is useful for Group Relative Policy Optimization (GRPO) where
    we want to compare trajectories within the same group (e.g., same task).
    
    Args:
        rewards: List of reward values.
        group_ids: List of group IDs for each reward.
            Rewards with the same group_id are compared against each other.
    
    Returns:
        List of advantages (reward - group_mean).
    
    Example:
        >>> rewards = [1.0, 2.0, 0.5, 1.5, 3.0, 2.5]
        >>> groups  = [0,   0,   1,   1,   0,   1  ]  # Three groups
        >>> advantages = compute_advantages(rewards, groups)
        >>> # Group 0: mean = (1.0 + 2.0 + 3.0) / 3 = 2.0
        >>> # Group 1: mean = (0.5 + 1.5 + 2.5) / 3 = 1.5
        >>> print(advantages)
        >>> # [-1.0, 0.0, -1.0, 0.0, 1.0, 1.0]
    """
    if len(rewards) != len(group_ids):
        raise ValueError(
            f"rewards and group_ids must have same length: "
            f"{len(rewards)} vs {len(group_ids)}"
        )
    
    # Compute group means
    group_sums = {}
    group_counts = {}
    
    for reward, group_id in zip(rewards, group_ids):
        if group_id not in group_sums:
            group_sums[group_id] = 0.0
            group_counts[group_id] = 0
        group_sums[group_id] += reward
        group_counts[group_id] += 1
    
    group_means = {
        gid: group_sums[gid] / group_counts[gid]
        for gid in group_sums
    }
    
    # Compute advantages
    advantages = [
        reward - group_means[group_id]
        for reward, group_id in zip(rewards, group_ids)
    ]
    
    return advantages
