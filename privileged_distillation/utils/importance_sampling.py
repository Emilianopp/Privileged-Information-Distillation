"""
Importance Sampling Utilities

This module provides utilities for computing importance sampling weights,
including self-normalized importance sampling (SNIS) with reward tilting.
"""

from typing import Optional
import torch


def compute_snis_weights(
    log_p_ref: torch.Tensor,
    log_q_proposal: torch.Tensor,
    rewards: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Compute self-normalized importance sampling (SNIS) weights.
    
    The SNIS weights are used in the surrogate loss:
        L(θ) = -Σ_k w̃_k log π_θ(τ_k, z_k | x)
    
    where:
        w̃_k = w_k / Σ_j w_j  (normalized weights)
        w_k = π_ref(τ_k, z_k | x) * exp(R(τ_k, x) / β) / π_φ^h(τ_k, z_k | x, h)
    
    This is useful for off-policy evaluation and training, especially with
    exponential reward tilting to focus on high-reward trajectories.
    
    Args:
        log_p_ref: Log probability under reference policy π_ref(τ,z|x).
            Shape: [K], where K is the number of samples.
            This is the full trajectory: log p_θ(z|x) + log p_θ(a|x,z).
        log_q_proposal: Log probability under proposal policy π_φ^h(τ,z|x,h).
            Shape: [K].
            This is the full trajectory: log q_φ(z|x,h) + log q_φ(a|x,z,h).
        rewards: Task rewards R(τ,x) for each sample.
            Shape: [K].
        beta: Temperature parameter for reward tilting.
            Higher values reduce the influence of rewards.
            Lower values increase focus on high-reward trajectories.
    
    Returns:
        Normalized importance weights w̃_k that sum to 1.
        Shape: [K].
        When K=1, returns tensor([1.0]) without normalization.
    
    Example:
        >>> K = 4
        >>> log_p_ref = torch.tensor([-2.3, -1.8, -2.1, -2.5])
        >>> log_q_proposal = torch.tensor([-2.0, -2.2, -1.9, -2.3])
        >>> rewards = torch.tensor([0.8, 0.3, 0.9, 0.1])
        >>> beta = 0.1
        >>> weights = compute_snis_weights(log_p_ref, log_q_proposal, rewards, beta)
        >>> print(weights.sum())  # Should be ~1.0
        >>> 
        >>> # Visualize how high-reward samples get higher weights
        >>> for i in range(K):
        >>>     print(f"Sample {i}: reward={rewards[i]:.2f}, weight={weights[i]:.4f}")
    
    Note:
        - When K=1, returns weight of 1.0 without normalization
        - Uses logsumexp for numerical stability
        - Supports gradient flow through log_p_ref for policy learning
    """
    K = log_p_ref.shape[0]
    
    # Skip normalization when K=1 - single sample gets weight 1.0
    if K == 1:
        return torch.ones(1, device=log_p_ref.device, dtype=log_p_ref.dtype)
    
    # Compute log unnormalized weights:
    # log w_k = log π_ref(τ,z|x) - log π_φ^h(τ,z|x,h) + R(τ,x) / β
    log_weights = log_p_ref - log_q_proposal + rewards / beta
    
    # Self-normalize using logsumexp for numerical stability:
    # log w̃_k = log w_k - log(Σ_j w_j) = log w_k - logsumexp(log w)
    log_normalizer = torch.logsumexp(log_weights, dim=0)
    log_normalized_weights = log_weights - log_normalizer
    
    # Convert back from log space to get normalized weights
    normalized_weights = torch.exp(log_normalized_weights)
    
    return normalized_weights


def compute_importance_ratio(
    log_policy: torch.Tensor,
    log_behavior: torch.Tensor,
    epsilon_low: Optional[float] = None,
    epsilon_high: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute importance sampling ratio with optional clipping.
    
    The importance ratio is:
        ρ = π(a|s) / μ(a|s)
    
    where π is the target policy and μ is the behavior policy.
    
    Args:
        log_policy: Log probability under target policy. Shape: [N].
        log_behavior: Log probability under behavior policy. Shape: [N].
        epsilon_low: Optional lower clipping bound (e.g., 0.8). Default: None.
        epsilon_high: Optional upper clipping bound (e.g., 1.2). Default: None.
    
    Returns:
        Importance ratios, optionally clipped. Shape: [N].
    
    Example:
        >>> log_policy = torch.tensor([-1.2, -0.8, -1.5])
        >>> log_behavior = torch.tensor([-1.0, -1.1, -1.3])
        >>> 
        >>> # Unclipped ratios
        >>> ratio = compute_importance_ratio(log_policy, log_behavior)
        >>> 
        >>> # Clipped ratios (PPO-style)
        >>> ratio_clipped = compute_importance_ratio(
        ...     log_policy, log_behavior, epsilon_low=0.8, epsilon_high=1.2
        ... )
    """
    # Compute ratio in log space then exponentiate
    ratio = torch.exp(log_policy - log_behavior)
    
    # Apply clipping if bounds are provided
    if epsilon_low is not None or epsilon_high is not None:
        ratio = torch.clamp(ratio, min=epsilon_low, max=epsilon_high)
    
    return ratio
