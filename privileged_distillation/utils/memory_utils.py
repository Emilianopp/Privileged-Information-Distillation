"""
Memory-Efficient Utilities

This module provides utilities for memory-efficient processing of logits
and other large tensors, which is crucial for training large language models.
"""

from typing import List, Tuple, Union
import torch


def process_logits_memory_efficient(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
    """
    Memory-efficient logits processing that filters out ignored tokens.
    
    This function processes logits and labels to remove all tokens marked
    with ignore_index, which reduces memory usage during loss computation.
    Processing is done chunk-by-chunk to minimize peak memory.
    
    Args:
        logits: Model logits. Can be:
            - Single tensor of shape [B, T, V]
            - List of chunked tensors, each [B, T_i, V]
        labels: Target labels. Shape: [B, T].
        ignore_index: Label value to ignore. Default: -100.
    
    Returns:
        Tuple of (filtered_logits, filtered_labels):
            If input was a list:
                - filtered_logits: List of chunks containing only valid tokens
                - filtered_labels: Tensor of shape [1, N_valid] where N_valid is
                  the total number of non-ignored tokens
            If input was a tensor:
                - filtered_logits: Tensor of shape [N_valid, V]
                - filtered_labels: Tensor of shape [N_valid]
    
    Example:
        >>> # With chunked logits
        >>> logits = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> labels = torch.randint(-100, 1000, (2, 20))
        >>> filtered_logits, filtered_labels = process_logits_memory_efficient(
        ...     logits, labels
        ... )
        >>> 
        >>> # With single tensor
        >>> logits = torch.randn(2, 20, 1000)
        >>> filtered_logits, filtered_labels = process_logits_memory_efficient(
        ...     logits, labels
        ... )
    """
    if isinstance(logits, list):
        # Process chunks sequentially to minimize peak memory
        all_valid_logits = []
        all_valid_labels = []
        
        labels_chunks = labels.chunk(len(logits), dim=1)
        
        for i, (logit_chunk, label_chunk) in enumerate(zip(logits, labels_chunks)):
            # Process one chunk at a time
            logit_2d = logit_chunk.view(-1, logit_chunk.size(-1))
            label_1d = label_chunk.view(-1)
            
            # Get valid mask
            valid_mask = label_1d != ignore_index
            
            if valid_mask.any():
                # Use the most memory-efficient selection
                all_valid_logits.append(logit_2d[valid_mask])
                all_valid_labels.append(label_1d[valid_mask])
            
            # Free intermediate tensors immediately
            del logit_2d, label_1d, valid_mask
        
        if all_valid_logits:
            concatenated_logits = torch.cat(all_valid_logits, dim=0)
            concatenated_labels = torch.cat(all_valid_labels, dim=0)
            
            # Re-chunk if originally was a list
            original_chunks = len(logits)
            chunk_size = (
                concatenated_logits.size(0) + original_chunks - 1
            ) // original_chunks
            rechunked_logits = list(concatenated_logits.split(chunk_size, dim=0))
            
            return rechunked_logits, concatenated_labels.unsqueeze(0)
        else:
            return [], torch.empty(0, dtype=labels.dtype, device=labels.device)
    else:
        # Handle tensor case
        logit_2d = logits.view(-1, logits.size(-1))
        label_1d = labels.view(-1)
        valid_mask = label_1d != ignore_index
        
        if valid_mask.any():
            return logit_2d[valid_mask], label_1d[valid_mask]
        else:
            return (
                torch.empty((0, logits.size(-1)), dtype=logits.dtype, device=logits.device),
                torch.empty(0, dtype=labels.dtype, device=labels.device),
            )


def chunk_tensor(
    tensor: torch.Tensor,
    num_chunks: int,
    dim: int = 1,
) -> List[torch.Tensor]:
    """
    Split a tensor into approximately equal chunks along a dimension.
    
    This is useful for processing large tensors in smaller pieces to
    reduce memory usage.
    
    Args:
        tensor: Tensor to chunk. Shape: [d0, d1, ..., dn].
        num_chunks: Number of chunks to create.
        dim: Dimension to chunk along. Default: 1.
    
    Returns:
        List of chunked tensors.
    
    Example:
        >>> tensor = torch.randn(2, 100, 1000)
        >>> chunks = chunk_tensor(tensor, num_chunks=4, dim=1)
        >>> print(len(chunks))  # 4
        >>> print(chunks[0].shape)  # [2, 25, 1000]
    """
    return list(tensor.chunk(num_chunks, dim=dim))


def unchunk_tensor(
    chunks: List[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    """
    Concatenate a list of tensor chunks back into a single tensor.
    
    Args:
        chunks: List of tensor chunks to concatenate.
        dim: Dimension to concatenate along. Default: 1.
    
    Returns:
        Concatenated tensor.
    
    Example:
        >>> chunks = [torch.randn(2, 25, 1000) for _ in range(4)]
        >>> tensor = unchunk_tensor(chunks, dim=1)
        >>> print(tensor.shape)  # [2, 100, 1000]
    """
    return torch.cat(chunks, dim=dim)
