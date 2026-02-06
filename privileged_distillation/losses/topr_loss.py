"""
TOPR (Tapered Off-Policy Reinforcement Learning) Loss

This module implements the TOPR loss function which combines:
- Supervised Fine-Tuning (SFT) for positive advantages
- Truncated Importance Sampling (TIS) for negative advantages

References:
    "Tapered Off-Policy Reinforcement Learning" (2025)
    https://arxiv.org/pdf/2503.14286
"""

from typing import List, Optional
import torch
import torch.nn.functional as F


def calculate_topr_loss(
    logits: List[torch.Tensor],
    labels: torch.Tensor,
    advantages: torch.Tensor,
    ref_logits: Optional[List[torch.Tensor]] = None,
    og_reward: Optional[torch.Tensor] = None,
    use_sft_positive: bool = False,
    ignore_index: int = -100,
    epsilon_low_neg: float = 0.8,
    epsilon_high_neg: float = 1.2,
) -> torch.Tensor:
    """
    Calculate TOPR (Tapered Off-Policy Reinforcement Learning) loss or SFT loss.
    
    The TOPR objective is defined as:
        ∇J_topr(π) = Σ_{τ∈T+} μ(τ)R(τ)∇log π(τ) 
                   + Σ_{τ∈T-} μ(τ) * clip(π/μ, ε_low, ε_high) * R(τ) * ∇log π(τ)
    
    where:
        T+ are trajectories with positive advantages (A > 0)
        T- are trajectories with negative advantages (A ≤ 0)
        μ(τ) is the behavior policy
        π(τ) is the current policy
    
    Modes:
        Standard TOPR (use_sft_positive=False):
            - For A > 0: SFT loss weighted by advantage
            - For A ≤ 0: Truncated Importance Sampling (TIS)
        
        SFT on positive trajectories (use_sft_positive=True):
            - If og_reward == 1: Apply SFT loss (weight = 1.0)
            - Otherwise: Zero out the loss (weight = 0.0)
            - Keeps DDP synchronized while only training on positive samples
    
    Args:
        logits: Current policy logits (list of chunks).
            Each chunk has shape [batch_size, seq_len_chunk, vocab_size].
        labels: Ground truth labels. Shape: [batch_size, seq_len].
        advantages: Advantage values per sample. Shape: [batch_size] or scalar.
        ref_logits: Optional reference policy logits for TIS (used for negative advantages).
            List of chunks with same shape as logits.
        og_reward: Original reward signal (0 or 1) for SFT positive mode.
            Shape: [batch_size] or scalar.
        use_sft_positive: If True, only train on positive trajectories (og_reward == 1).
        ignore_index: Label value to ignore. Default: -100.
        epsilon_low_neg: Lower clipping bound for negative samples. Default: 0.8.
        epsilon_high_neg: Upper clipping bound for negative samples. Default: 1.2.
    
    Returns:
        Combined TOPR/SFT loss (summed over valid tokens). Shape: scalar.
    
    Example:
        >>> # Standard TOPR mode
        >>> logits = [torch.randn(1, 10, 1000), torch.randn(1, 10, 1000)]
        >>> labels = torch.randint(0, 1000, (1, 20))
        >>> advantages = torch.tensor([0.5])  # Positive advantage
        >>> loss = calculate_topr_loss(logits, labels, advantages)
        
        >>> # With negative advantage and reference
        >>> advantages_neg = torch.tensor([-0.3])
        >>> ref_logits = [torch.randn(1, 10, 1000), torch.randn(1, 10, 1000)]
        >>> loss = calculate_topr_loss(logits, labels, advantages_neg, ref_logits=ref_logits)
        
        >>> # SFT positive mode
        >>> og_reward = torch.tensor([1.0])
        >>> loss = calculate_topr_loss(
        ...     logits, labels, advantages, og_reward=og_reward, use_sft_positive=True
        ... )
    """
    device = logits[0].device if isinstance(logits, list) else logits.device
    
    # Process chunks
    num_chunks = len(logits)
    labels_chunks = labels.view(1, -1).chunk(num_chunks, dim=1)
    
    policy_log_ps_list = []
    ref_log_ps_list = []
    
    # Gather log-probs chunk by chunk
    for i, logit_chunk in enumerate(logits):
        labels_flat = labels_chunks[i].view(-1)
        valid_mask = labels_flat != ignore_index
        
        if not valid_mask.any():
            continue
        
        # Filter to valid tokens
        logit_valid = logit_chunk[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        # Current policy log-probs
        log_ps = F.log_softmax(logit_valid, dim=-1)
        vocab_size = log_ps.size(-1)
        valid_indices = labels_valid.clamp(0, vocab_size - 1).unsqueeze(-1)
        selected_log_ps = torch.gather(log_ps, dim=-1, index=valid_indices).squeeze(-1)
        policy_log_ps_list.append(selected_log_ps)
        
        # Reference policy log-probs (for negative advantages)
        if ref_logits is not None:
            ref_logit_chunk = ref_logits[i]
            ref_logit_valid = ref_logit_chunk[valid_mask]
            ref_log_ps = F.log_softmax(ref_logit_valid, dim=-1)
            ref_selected_log_ps = torch.gather(
                ref_log_ps, dim=-1, index=valid_indices
            ).squeeze(-1)
            ref_log_ps_list.append(ref_selected_log_ps)
    
    # Handle empty case
    if not policy_log_ps_list:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Concatenate all valid tokens
    policy_log_ps = torch.cat(policy_log_ps_list)
    
    # ===== SFT on Positive Trajectories Mode =====
    if use_sft_positive:
        # Determine weight: 1.0 if positive (og_reward == 1), else 0.0
        if og_reward is not None:
            weight = torch.where(
                og_reward == 1.0,
                torch.ones_like(og_reward, dtype=policy_log_ps.dtype),
                torch.zeros_like(og_reward, dtype=policy_log_ps.dtype),
            )
        else:
            weight = torch.ones_like(advantages, dtype=policy_log_ps.dtype)
        
        # Standard SFT loss: -log π(τ), weighted by trajectory quality
        # This ensures DDP stays synchronized even when weight is 0
        weight = weight.to(policy_log_ps.device)
        sft_loss = -(weight * policy_log_ps).sum()
        
        return sft_loss
    
    # ===== Standard TOPR Mode =====
    # Separate positive and negative advantages
    positive_mask = advantages > 0
    negative_mask = advantages <= 0
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # ===== Positive samples: SFT loss weighted by advantage =====
    # L_positive = -Σ A * log π(τ) for A > 0
    if positive_mask.any():
        positive_loss = -(advantages * policy_log_ps).sum()
        total_loss = total_loss + positive_loss
    
    # ===== Negative samples: Truncated Importance Sampling =====
    # L_negative = -Σ clip(π/μ, ε_low, ε_high) * A * log π(τ) for A ≤ 0
    if negative_mask.any() and ref_logits is not None:
        ref_log_ps = torch.cat(ref_log_ps_list)
        
        # Importance weight: π(τ) / μ(τ)
        # CRITICAL: Detach policy_log_ps for unbiased gradient estimation
        importance_weight = torch.exp(policy_log_ps - ref_log_ps.detach())
        
        # Clip importance weight for negative samples
        clipped_weight = torch.clamp(
            importance_weight, min=epsilon_low_neg, max=epsilon_high_neg
        )
        
        # TIS loss: -clipped_weight * advantage * log π(τ)
        # The gradient flows only through policy_log_ps, not through clipped_weight
        negative_loss = -(clipped_weight * advantages * policy_log_ps).sum()
        total_loss = total_loss + negative_loss
    
    elif negative_mask.any():
        # Fallback: treat as SFT if no reference available
        negative_loss = -(advantages * policy_log_ps).sum()
        total_loss = total_loss + negative_loss
    
    return total_loss
