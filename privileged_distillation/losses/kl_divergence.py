"""
Rao-Blackwellized KL Divergence Estimator

This module implements a Rao-Blackwellized estimator for KL divergence between
token distributions, which reduces variance compared to standard Monte Carlo
estimation.

The estimator is particularly useful for comparing thought tokens between
privileged and non-privileged models.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def compute_kl_divergence_rao_blackwellized(
    logits_p: Union[torch.Tensor, List[torch.Tensor]],
    logits_q: Union[torch.Tensor, List[torch.Tensor]],
    labels_p: Optional[torch.Tensor] = None,
    labels_q: Optional[torch.Tensor] = None,
    mask_p: Optional[torch.Tensor] = None,
    mask_q: Optional[torch.Tensor] = None,
    return_stats: bool = False,
    ignore_index: int = -100,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Computes the Rao-Blackwellized KL divergence estimate between two token distributions.
    
    The Rao-Blackwellized estimator uses the full distribution for KL computation:
        KL(p||q) = Σ_x p(x) * [log p(x) - log q(x)]
                 = Σ_x exp(log p(x)) * (log p(x) - log q(x))
    
    This is more stable than sampling-based estimation and has lower variance.
    
    Args:
        logits_p: Logits from the first model (p distribution).
            Can be a single tensor [B, T, V] or list of chunks [B, T_i, V].
        logits_q: Logits from the second model (q distribution).
            Must have same structure as logits_p.
        labels_p: Optional labels for p to create validity mask.
            Shape: [B, T]. If provided, only non-ignore tokens are used.
        labels_q: Optional labels for q to create validity mask.
            Shape: [B, T]. If provided, only non-ignore tokens are used.
        mask_p: Optional boolean mask for p distribution.
            Shape: [B, T]. True indicates valid tokens.
        mask_q: Optional boolean mask for q distribution.
            Shape: [B, T]. True indicates valid tokens.
        return_stats: Whether to return detailed statistics.
            If True, returns (kl, stats_dict). Default: False.
        ignore_index: Label value to ignore when creating masks from labels.
            Default: -100.
    
    Returns:
        If return_stats=False:
            Mean KL divergence per valid token. Shape: scalar.
        If return_stats=True:
            Tuple of (kl, stats) where stats contains:
                - kl_min: Minimum per-token KL
                - kl_max: Maximum per-token KL
                - kl_p25: 25th percentile
                - kl_p50: 50th percentile (median)
                - kl_p75: 75th percentile
                - kl_values: All per-token KL values (CPU tensor)
    
    Example:
        >>> # Simple KL computation on all tokens
        >>> logits_p = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> logits_q = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> kl = compute_kl_divergence_rao_blackwellized(logits_p, logits_q)
        >>> 
        >>> # KL with masking via labels
        >>> labels = torch.randint(-100, 1000, (2, 20))
        >>> kl = compute_kl_divergence_rao_blackwellized(
        ...     logits_p, logits_q, labels_p=labels, labels_q=labels
        ... )
        >>> 
        >>> # KL with custom masks and statistics
        >>> mask = torch.ones(2, 20, dtype=torch.bool)
        >>> mask[:, :5] = False  # Ignore first 5 tokens
        >>> kl, stats = compute_kl_divergence_rao_blackwellized(
        ...     logits_p, logits_q, mask_p=mask, mask_q=mask, return_stats=True
        ... )
        >>> print(f"Median KL: {stats['kl_p50']:.4f}")
    """
    device = (
        logits_p[0].device if isinstance(logits_p, list) else logits_p.device
    )
    
    # Convert to chunked format
    if isinstance(logits_p, torch.Tensor):
        num_chunks = 1
        logits_p = [logits_p]
    else:
        num_chunks = len(logits_p)
    
    if isinstance(logits_q, torch.Tensor):
        logits_q = [logits_q]
    
    # Concatenate chunks for easier processing
    logits_p_concat = torch.cat(logits_p, dim=1)  # [B, T_total, V]
    logits_q_concat = torch.cat(logits_q, dim=1)  # [B, T_total, V]
    
    batch_size = logits_p_concat.size(0)
    seq_len = logits_p_concat.size(1)
    
    # Build validity masks
    if mask_p is None and labels_p is not None:
        mask_p = labels_p != ignore_index
    if mask_q is None and labels_q is not None:
        mask_q = labels_q != ignore_index
    
    # Default to all valid if no masks provided
    if mask_p is None:
        mask_p = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if mask_q is None:
        mask_q = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Combine masks - both must be valid
    valid_mask = mask_p & mask_q
    
    # Select only valid tokens
    logits_p_valid = logits_p_concat[valid_mask]  # [N_valid, V]
    logits_q_valid = logits_q_concat[valid_mask]  # [N_valid, V]
    
    if logits_p_valid.numel() == 0:
        # No valid tokens
        zero = torch.tensor(0.0, device=device)
        if return_stats:
            stats = {
                "kl_min": zero,
                "kl_max": zero,
                "kl_p25": zero,
                "kl_p50": zero,
                "kl_p75": zero,
                "kl_values": torch.tensor([], device="cpu"),
            }
            return zero, stats
        return zero
    
    # Compute log probabilities with numerical stability
    eps = 1e-8
    logp = F.log_softmax(logits_p_valid, dim=-1).clamp(
        min=torch.finfo(logits_p_valid.dtype).min + eps
    )
    logq = F.log_softmax(logits_q_valid, dim=-1).clamp(
        min=torch.finfo(logits_q_valid.dtype).min + eps
    )
    
    # KL(p||q) = Σ_x exp(log p(x)) * (log p(x) - log q(x))
    # For numerical stability, use exp(logq) since we're computing reverse KL in some contexts
    # But for standard KL(p||q), we use p as the base distribution
    kl_per_token = torch.sum(torch.exp(logp) * (logp - logq), dim=-1).clamp(min=0.0)
    
    # Compute mean KL
    mean_kl = kl_per_token.mean()
    
    if return_stats:
        # Move to CPU for statistics to save GPU memory
        kl_values_cpu = kl_per_token.detach().cpu()
        
        stats = {
            "kl_min": kl_values_cpu.min().to(device),
            "kl_max": kl_values_cpu.max().to(device),
            "kl_p25": torch.quantile(kl_values_cpu, 0.25).to(device),
            "kl_p50": torch.quantile(kl_values_cpu, 0.50).to(device),
            "kl_p75": torch.quantile(kl_values_cpu, 0.75).to(device),
            "kl_values": kl_values_cpu,
        }
        return mean_kl, stats
    
    return mean_kl
