"""
Log Probability Computation Utilities

This module provides utilities for computing log probabilities from model logits,
supporting various modes including:
- Per-sample mean log-probs
- Per-token log-probs
- Entropy computation
- Position-specific log-probs
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def get_sample_log_probs(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    labels: torch.Tensor,
    start_pos: Optional[torch.Tensor] = None,
    end_pos: Optional[torch.Tensor] = None,
    compute_entropy: bool = False,
    return_per_token: bool = False,
    ignore_index: int = -100,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Computes log probability statistics for each sample, handling chunked logits.
    
    This function supports three modes:
    1. Mean log-prob per sample (default)
    2. Mean entropy per sample (compute_entropy=True)
    3. Per-token log-probs (return_per_token=True)
    
    Args:
        logits: Model logits. Can be:
            - Single tensor of shape [B, T, V]
            - List of chunked tensors, each [B, T_i, V]
        labels: Target labels. Shape: [B, T]. Should NOT be pre-shifted;
            this function handles shifting internally.
        start_pos: Optional start positions for each sample. Shape: [B].
            If provided with end_pos, only tokens in [start, end) are used.
        end_pos: Optional end positions for each sample. Shape: [B].
            If provided with start_pos, only tokens in [start, end) are used.
        compute_entropy: If True, compute entropy instead of log-probs.
            Entropy is H(p) = -Σ p(x) * log p(x). Default: False.
        return_per_token: If True, return per-token values instead of per-sample mean.
            Returns list of tensors (one per chunk) with shape [B, T_i].
            Default: False.
        ignore_index: Label value to ignore. Default: -100.
    
    Returns:
        If return_per_token=False:
            Mean value per sample over valid tokens. Shape: [B].
        If return_per_token=True:
            List of per-chunk tensors with shape [B, T_i], containing
            per-token values. Positions outside the range (if specified)
            are zeroed.
    
    Example:
        >>> logits = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> labels = torch.randint(0, 1000, (2, 20))
        >>> 
        >>> # Mean log-prob per sample
        >>> mean_logp = get_sample_log_probs(logits, labels)
        >>> print(mean_logp.shape)  # [2]
        >>> 
        >>> # Mean entropy per sample
        >>> entropy = get_sample_log_probs(logits, labels, compute_entropy=True)
        >>> 
        >>> # Per-token log-probs
        >>> per_token = get_sample_log_probs(logits, labels, return_per_token=True)
        >>> print(len(per_token))  # 2 chunks
        >>> 
        >>> # Log-probs for specific range
        >>> start = torch.tensor([5, 3])
        >>> end = torch.tensor([15, 18])
        >>> range_logp = get_sample_log_probs(logits, labels, start_pos=start, end_pos=end)
    """
    if isinstance(logits, torch.Tensor):
        logits = [logits]
    
    device = logits[0].device
    batch_size = labels.shape[0]
    seq_len = labels.shape[1]
    
    # Shift labels for next-token prediction
    # Create padding with ignore_index
    ignore_padding = torch.full(
        (batch_size, 1), ignore_index, dtype=labels.dtype, device=device
    )
    labels_shifted = torch.cat([labels[:, 1:], ignore_padding], dim=1)
    
    # Create range mask if positions are provided
    range_mask = torch.ones_like(labels_shifted, dtype=torch.bool)
    if start_pos is not None and end_pos is not None:
        range_mask = torch.zeros_like(labels_shifted, dtype=torch.bool)
        for i in range(batch_size):
            s, e = start_pos[i].item(), end_pos[i].item()
            if s < e:
                range_mask[i, s:e] = True
    
    # If caller requests per-token outputs, return per-chunk gathered logprobs
    if return_per_token:
        per_token_chunks: List[torch.Tensor] = []
        num_chunks = len(logits)
        labels_chunks = labels_shifted.chunk(num_chunks, dim=1)
        mask_chunks = range_mask.chunk(num_chunks, dim=1)
        
        for logit_chunk, label_chunk, mask_chunk in zip(
            logits, labels_chunks, mask_chunks
        ):
            log_probs_full = F.log_softmax(logit_chunk.float(), dim=-1)
            vocab_size = log_probs_full.shape[-1]
            # Clamp labels to prevent gather crash on ignore_index
            valid_indices = label_chunk.clamp(0, vocab_size - 1)
            gathered = torch.gather(
                log_probs_full, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)
            # Zero out positions outside range
            if start_pos is not None and end_pos is not None:
                gathered = gathered * mask_chunk
            per_token_chunks.append(gathered)
        return per_token_chunks
    
    # Compute per-sample statistics
    chunk_values = []
    chunk_token_counts = []
    
    num_chunks = len(logits)
    labels_chunks = labels_shifted.chunk(num_chunks, dim=1)
    mask_chunks = range_mask.chunk(num_chunks, dim=1)
    
    for logit_chunk, label_chunk, mask_chunk in zip(
        logits, labels_chunks, mask_chunks
    ):
        # Skip chunks with no valid tokens
        if not mask_chunk.any():
            chunk_values.append(
                torch.zeros(batch_size, device=device, dtype=logits[0].dtype)
            )
            chunk_token_counts.append(torch.zeros(batch_size, device=device))
            continue
        
        if compute_entropy:
            # Entropy: H(p) = -Σ p(x) * log p(x)
            probs = F.softmax(logit_chunk, dim=-1)
            log_probs = F.log_softmax(logit_chunk, dim=-1)
            value_per_token_2d = -torch.sum(probs * log_probs, dim=-1)
            # Apply range mask
            masked_values = value_per_token_2d * mask_chunk
            chunk_values.append(masked_values.sum(dim=1))
        else:
            # Log probability of target tokens
            log_probs_full = F.log_softmax(logit_chunk, dim=-1)
            vocab_size = log_probs_full.shape[-1]
            
            # Mask for valid tokens (within vocab range and not ignore_index)
            padding_mask = (label_chunk >= 0) & (label_chunk < vocab_size)
            
            # Replace invalid indices with dummy (0) to prevent gather crash
            labels_for_gather = torch.where(
                padding_mask, label_chunk, torch.zeros_like(label_chunk)
            )
            
            gathered_log_probs = torch.gather(
                log_probs_full, -1, labels_for_gather.unsqueeze(-1)
            ).squeeze(-1)
            
            # Combine validity and range masks
            final_mask = padding_mask & mask_chunk
            
            masked_log_probs = gathered_log_probs * final_mask
            chunk_values.append(masked_log_probs.sum(dim=1))
        
        chunk_token_counts.append(mask_chunk.sum(dim=1))
    
    # Aggregate across chunks
    total_value_per_sample = torch.stack(chunk_values).sum(dim=0)
    num_tokens_per_sample = torch.stack(chunk_token_counts).sum(dim=0)
    
    # Calculate mean value per sample
    mean_value = torch.where(
        num_tokens_per_sample > 0,
        total_value_per_sample / num_tokens_per_sample,
        torch.zeros_like(total_value_per_sample),
    )
    
    return mean_value


def get_log_probs_for_positions(
    logits: List[torch.Tensor],
    labels: torch.Tensor,
    positions: List[List[Tuple[int, int]]],
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Extract and compute mean log probabilities for tokens at specified positions.
    
    This is useful for computing log-probs only over specific segments like
    "thought tokens" or "action tokens" in a sequence.
    
    Args:
        logits: List of logit tensors (one per chunk).
            Each chunk has shape [B, T_i, V].
        labels: Label tensor. Shape: [B, T]. NOT pre-shifted.
        positions: List of position ranges per sample in batch.
            Format: List[List[Tuple[start, end]]] where:
                - Outer list has length B (batch size)
                - Inner list contains tuples of (start, end) positions
                - Multiple ranges can be specified per sample
        ignore_index: Label value to ignore. Default: -100.
    
    Returns:
        Mean log probability over all specified positions per sample.
        Shape: [B].
    
    Example:
        >>> logits = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> labels = torch.randint(0, 1000, (2, 20))
        >>> # Sample 0: positions 5-10 and 15-18
        >>> # Sample 1: positions 3-12
        >>> positions = [[(5, 10), (15, 18)], [(3, 12)]]
        >>> logp = get_log_probs_for_positions(logits, labels, positions)
        >>> print(logp.shape)  # [2]
    """
    if not positions:
        device = logits[0].device if isinstance(logits, list) else logits.device
        return torch.tensor(0.0, device=device)
    
    device = logits[0].device
    batch_size = labels.shape[0]
    
    # Get per-token log-probs for the entire sequence
    per_token_chunks = get_sample_log_probs(
        logits, labels, return_per_token=True, ignore_index=ignore_index
    )
    
    # Concatenate chunks
    per_token_logprobs = torch.cat(per_token_chunks, dim=1)  # [B, T]
    
    # Extract and average log-probs for specified positions
    mean_logprobs = []
    
    for sample_idx in range(batch_size):
        sample_positions = positions[sample_idx] if sample_idx < len(positions) else []
        
        if not sample_positions:
            mean_logprobs.append(torch.tensor(0.0, device=device))
            continue
        
        total_logprob = torch.tensor(0.0, device=device)
        total_tokens = 0
        
        for start, end in sample_positions:
            if start >= end or end > per_token_logprobs.size(1):
                continue
            
            # Sum log-probs for this range
            segment_logprob = per_token_logprobs[sample_idx, start:end].sum()
            total_logprob = total_logprob + segment_logprob
            total_tokens += (end - start)
        
        # Compute mean
        if total_tokens > 0:
            mean_logprobs.append(total_logprob / total_tokens)
        else:
            mean_logprobs.append(torch.tensor(0.0, device=device))
    
    return torch.stack(mean_logprobs)
