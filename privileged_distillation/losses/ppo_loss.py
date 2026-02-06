"""
PPO (Proximal Policy Optimization) Loss with Clipped Importance Sampling

This module implements PPO-style clipped importance sampling for privileged
information distillation. It handles sequences of different lengths and uses
memory-efficient chunk processing.

References:
    Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
    https://arxiv.org/abs/1707.06347
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


def calculate_ppo_clipped_importance_sampling(
    policy_with_priv_logits: List[torch.Tensor],
    policy_without_priv_logits: List[torch.Tensor],
    labels_with_priv: torch.Tensor,
    labels_without_priv: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
    epsilon_low: float = 0.8,
    epsilon_high: float = 1.2,
    return_stats: bool = False,
) -> torch.Tensor:
    """
    Calculate PPO-style clipped importance sampling loss.
    
    This implementation handles logits of different lengths by finding the intersection
    of valid (non-ignored) tokens. Memory-optimized through chunk-by-chunk processing.
    
    The PPO loss is defined as:
        L^CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    
    where:
        r_t = π(a_t|s_t) / π_old(a_t|s_t)  is the importance ratio
        A_t is the advantage estimate
        ε is the clipping parameter
    
    Args:
        policy_with_priv_logits: Logits from policy with privilege (list of chunks).
            Each chunk has shape [batch_size, seq_len_chunk, vocab_size].
        policy_without_priv_logits: Logits from policy without privilege (list of chunks).
            Each chunk has shape [batch_size, seq_len_chunk, vocab_size].
        labels_with_priv: Ground truth labels for privileged model.
            Shape: [batch_size, seq_len].
        labels_without_priv: Ground truth labels for non-privileged model.
            Shape: [batch_size, seq_len].
        advantages: Advantage values for each sample.
            Shape: [batch_size] or scalar.
        ignore_index: Label value to ignore in loss computation. Default: -100.
        epsilon_low: Lower clipping bound (e.g., 0.8 for 1-0.2). Default: 0.8.
        epsilon_high: Upper clipping bound (e.g., 1.2 for 1+0.2). Default: 1.2.
        return_stats: If True, return tuple of (loss, stats_dict). Default: False.
    
    Returns:
        If return_stats=False:
            PPO clipped loss (summed over all valid tokens). Shape: scalar.
        If return_stats=True:
            Tuple of (loss, stats_dict) where stats_dict contains:
                - total_tokens: Number of valid tokens
                - clipped_tokens: Number of clipped tokens
                - clipped_low: Number of tokens clipped below epsilon_low
                - clipped_high: Number of tokens clipped above epsilon_high
                - mean_ratio: Mean importance ratio
    
    Example:
        >>> student_logits = [torch.randn(1, 10, 1000), torch.randn(1, 10, 1000)]
        >>> teacher_logits = [torch.randn(1, 10, 1000), torch.randn(1, 10, 1000)]
        >>> labels = torch.randint(0, 1000, (1, 20))
        >>> advantages = torch.tensor([0.5])
        >>> loss = calculate_ppo_clipped_importance_sampling(
        ...     teacher_logits, student_logits, labels, labels, advantages
        ... )
    """
    device = (
        policy_with_priv_logits[0].device
        if isinstance(policy_with_priv_logits, list)
        else policy_with_priv_logits.device
    )
    
    policy_with_priv_log_ps_list = []
    policy_without_priv_log_ps_list = []
    
    # Process each chunk individually to save memory
    num_chunks = len(policy_with_priv_logits)
    labels_with_priv_chunks = labels_with_priv.view(1, -1).chunk(num_chunks, dim=1)
    labels_without_priv_chunks = labels_without_priv.view(1, -1).chunk(num_chunks, dim=1)
    
    for i, (priv_chunk, no_priv_chunk) in enumerate(
        zip(policy_with_priv_logits, policy_without_priv_logits)
    ):
        # For each chunk, we expect labels to be 1D
        labels_priv_flat = labels_with_priv_chunks[i].view(-1)
        labels_no_priv_flat = labels_without_priv_chunks[i].view(-1)
        
        # Find valid indices within the current chunk's scope
        valid_mask_priv = labels_priv_flat != ignore_index
        valid_mask_no_priv = labels_no_priv_flat != ignore_index
        
        # Ensure we are comparing the same tokens
        assert (
            valid_mask_priv.sum() == valid_mask_no_priv.sum()
        ), f"Valid token counts do not match: {valid_mask_priv.sum()} vs {valid_mask_no_priv.sum()}"
        
        # Filter logits and labels for valid tokens
        priv_logits_valid = priv_chunk[valid_mask_priv]
        no_priv_logits_valid = no_priv_chunk[valid_mask_no_priv]
        labels_valid = labels_priv_flat[valid_mask_priv]
        
        if labels_valid.numel() == 0:
            continue
        
        # Compute log softmax with epsilon clamping to prevent NaN in ratio computation
        eps = 1e-8
        priv_log_ps = F.log_softmax(priv_logits_valid, dim=-1).clamp(
            min=torch.finfo(priv_logits_valid.dtype).min + eps
        )
        no_priv_log_ps = F.log_softmax(no_priv_logits_valid, dim=-1).clamp(
            min=torch.finfo(no_priv_logits_valid.dtype).min + eps
        )
        
        # Gather log-probabilities for the correct tokens
        vocab_size = priv_log_ps.size(-1)
        valid_indices = labels_valid.clamp(0, vocab_size - 1).unsqueeze(-1)
        
        priv_selected_log_ps = torch.gather(
            priv_log_ps, dim=-1, index=valid_indices
        ).squeeze(-1)
        no_priv_selected_log_ps = torch.gather(
            no_priv_log_ps, dim=-1, index=valid_indices
        ).squeeze(-1)
        
        policy_with_priv_log_ps_list.append(priv_selected_log_ps)
        policy_without_priv_log_ps_list.append(no_priv_selected_log_ps)
    
    # Handle case with no valid tokens
    if not policy_with_priv_log_ps_list:
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if return_stats:
            stats = {
                "total_tokens": 0,
                "clipped_tokens": 0,
                "clipped_low": 0,
                "clipped_high": 0,
                "mean_ratio": 0.0,
            }
            return zero_loss, stats
        return zero_loss
    
    # Concatenate results from all chunks
    priv_selected_log_ps_final = torch.cat(policy_with_priv_log_ps_list)
    no_priv_selected_log_ps_final = torch.cat(policy_without_priv_log_ps_list)
    
    # Calculate the importance ratio: π_new / π_old
    ratio = torch.exp(no_priv_selected_log_ps_final - priv_selected_log_ps_final)
    
    # Clip the ratio to [epsilon_low, epsilon_high]
    clipped_ratio = torch.clamp(ratio, epsilon_low, epsilon_high)
    
    # Calculate clipping statistics
    total_tokens = ratio.numel()
    clipped_mask = (ratio < epsilon_low) | (ratio > epsilon_high)
    clipped_tokens = clipped_mask.sum().item()
    clipped_low = (ratio < epsilon_low).sum().item()
    clipped_high = (ratio > epsilon_high).sum().item()
    mean_ratio = ratio.mean().item()
    
    # PPO policy loss with advantages
    # L^CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # Sum over all tokens
    loss = policy_loss.sum()
    
    if return_stats:
        stats = {
            "total_tokens": total_tokens,
            "clipped_tokens": clipped_tokens,
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "mean_ratio": mean_ratio,
        }
        return loss, stats
    
    return loss
