"""
On-Policy Distillation with Reverse KL Divergence

This module implements on-policy distillation using reverse KL divergence,
enabling knowledge transfer from a teacher model to a student model on
sampled trajectories.

The reverse KL objective encourages the student to focus on modes of the teacher
distribution, which is useful for behavior cloning from a privileged teacher.
"""

from typing import Dict, List, Optional, Tuple, Union
from bisect import bisect_right
import torch
import torch.nn.functional as F


def on_policy_distillation(
    student_logits: Union[List[torch.Tensor], torch.Tensor],
    teacher_logits: Union[List[torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    student_action_positions: Optional[List[List[Tuple[int, int]]]] = None,
    teacher_action_positions: Optional[List[List[Tuple[int, int]]]] = None,
    entropy_coef: float = 1.0,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Compute reverse-KL distillation statistics on on-policy samples.
    
    The reverse KL divergence is defined as:
        D_KL(π_student || π_teacher) = E_π_student[log π_student - log π_teacher]
                                      = H(π_student, π_teacher) - H(π_student)
    
    where:
        H(p, q) is the cross-entropy between p and q
        H(p) is the entropy of p
    
    The distillation loss with entropy regularization is:
        L = H(π_student, π_teacher) - α * H(π_student)
    
    where α is the entropy coefficient. Setting α < 1 encourages entropy
    maximization relative to pure reverse KL.
    
    Args:
        student_logits: Logits from the student (non-privileged) policy evaluated
            on sampled trajectories. Accepts a tensor or list of chunked tensors
            with shape [B, T, V] per chunk.
        teacher_logits: Logits from the teacher (privileged) policy evaluated on
            the same prompts and continuations. Must mirror student_logits structure.
        labels: Token labels prior to shifting for next-token prediction
            (shape [B, T]). Internally shifted to align with logits and
            ignore_index tokens are masked.
        student_action_positions: Optional per-sample list of (start, end) spans
            identifying the generated/action tokens within student sequences.
            When provided, the reverse KL loss is restricted to those positions.
            Format: List[List[Tuple[int, int]]] where outer list is batch dimension.
        teacher_action_positions: Optional per-sample list of spans for the teacher
            sequences. Use this when prompt lengths differ so that only matching
            action tokens are compared. If omitted, assumes teacher and student
            action tokens are aligned.
        entropy_coef: Scaling factor applied to the student entropy term.
            Setting this below 1.0 encourages entropy maximization relative to
            pure reverse KL objective. Default: 1.0.
        ignore_index: Label value to ignore in loss computation. Default: -100.
    
    Returns:
        Dictionary containing:
            - loss: Reverse-KL loss with entropy bonus applied (ready for backprop).
                    Shape: scalar.
            - reverse_kl: Detached KL estimate. Shape: scalar.
            - cross_entropy: Detached cross-entropy term. Shape: scalar.
            - entropy: Detached student entropy. Shape: scalar.
            - token_count: Number of tokens included in the estimate. Shape: scalar.
    
    Example:
        >>> student_logits = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> teacher_logits = [torch.randn(2, 10, 1000), torch.randn(2, 10, 1000)]
        >>> labels = torch.randint(0, 1000, (2, 20))
        >>> 
        >>> # Simple distillation on all tokens
        >>> result = on_policy_distillation(student_logits, teacher_logits, labels)
        >>> loss = result["loss"]
        >>> loss.backward()
        >>> 
        >>> # Distillation on specific action positions
        >>> action_pos = [[(5, 15)], [(3, 18)]]  # Different for each sample
        >>> result = on_policy_distillation(
        ...     student_logits, teacher_logits, labels,
        ...     student_action_positions=action_pos
        ... )
    """
    
    def _ensure_chunk_list(
        logits: Union[List[torch.Tensor], torch.Tensor]
    ) -> List[torch.Tensor]:
        """Convert logits to list of chunks."""
        if isinstance(logits, torch.Tensor):
            if logits.dim() != 3:
                raise ValueError("Expected tensor logits to have shape [B, T, V]")
            return [logits]
        chunks: List[torch.Tensor] = []
        for chunk in logits:
            if chunk.dim() == 3:
                chunks.append(chunk)
            elif chunk.dim() == 2:
                chunks.append(chunk.unsqueeze(0))
            else:
                raise ValueError(
                    f"Logit chunks must have 2 or 3 dims (found dim={chunk.dim()})."
                )
        return chunks
    
    student_chunks = _ensure_chunk_list(student_logits)
    teacher_chunks = _ensure_chunk_list(teacher_logits)
    
    student_batch = student_chunks[0].size(0)
    teacher_batch = teacher_chunks[0].size(0)
    
    if student_batch != teacher_batch:
        raise ValueError(
            f"Student and teacher logits must have the same batch dimension: "
            f"{student_batch} vs {teacher_batch}"
        )
    
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    
    device = student_chunks[0].device
    labels = labels.to(device)
    
    # Shift labels for next-token prediction
    # Create ignore padding tensor
    ignore_padding = torch.full(
        (labels.shape[0], 1), ignore_index, dtype=labels.dtype, device=device
    )
    shifted_labels = torch.cat([labels[:, 1:], ignore_padding], dim=1)
    
    # Mask for valid tokens
    target_mask = shifted_labels != ignore_index
    
    # Apply action position mask if provided
    if student_action_positions is not None:
        if shifted_labels.size(0) != len(student_action_positions):
            raise ValueError(
                f"action_positions length must equal batch dimension: "
                f"{len(student_action_positions)} vs {shifted_labels.size(0)}"
            )
        action_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        for sample_idx, spans in enumerate(student_action_positions):
            if not spans:
                continue
            for span in spans:
                if not span or len(span) != 2:
                    continue
                start, end = int(span[0]), int(span[1])
                if end <= start:
                    continue
                start = max(start, 0)
                end = min(end, action_mask.size(1))
                if start < end:
                    action_mask[sample_idx, start:end] = True
        if action_mask.any():
            target_mask = target_mask & action_mask
    
    token_count = target_mask.sum()
    if token_count.item() == 0:
        zero = torch.zeros((), device=device, dtype=student_chunks[0].dtype)
        return {
            "loss": zero,
            "reverse_kl": zero.detach(),
            "cross_entropy": zero.detach(),
            "entropy": zero.detach(),
            "token_count": torch.tensor(0, device=device),
        }
    
    # Calculate sequence lengths and offsets for chunked processing
    student_lengths = [chunk.size(1) for chunk in student_chunks]
    teacher_lengths = [chunk.size(1) for chunk in teacher_chunks]
    
    student_offsets = [0]
    for length in student_lengths:
        student_offsets.append(student_offsets[-1] + length)
    
    teacher_offsets = [0]
    for length in teacher_lengths:
        teacher_offsets.append(teacher_offsets[-1] + length)
    
    student_total_seq = student_offsets[-1]
    teacher_total_seq = teacher_offsets[-1]
    
    # Calculate valid tokens per sample
    token_count_per_sample = target_mask.sum(dim=1).to(torch.float32)
    
    # Helper to convert span lists to sorted index lists
    def spans_to_indices(
        spans: Optional[List[Tuple[int, int]]],
        seq_len: int,
    ) -> List[int]:
        indices: List[int] = []
        if not spans:
            return indices
        for span in spans:
            if not span or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if end <= start:
                continue
            start = max(start, 0)
            end = min(end, seq_len)
            if start >= end:
                continue
            for idx in range(start, end):
                indices.append(idx)
        return indices
    
    if (
        teacher_action_positions is not None
        and len(teacher_action_positions) != labels.size(0)
    ):
        raise ValueError(
            f"teacher_action_positions length must equal batch dimension: "
            f"{len(teacher_action_positions)} vs {labels.size(0)}"
        )
    
    seq_delta = teacher_total_seq - student_total_seq
    
    # Accumulate entropy and cross-entropy per sample
    entropy_sum = torch.zeros(student_batch, device=device, dtype=torch.float32)
    cross_entropy_sum = torch.zeros(student_batch, device=device, dtype=torch.float32)
    
    for sample_idx in range(student_batch):
        sample_mask = target_mask[sample_idx]
        if not sample_mask.any():
            continue
        
        # Determine which positions to include
        if student_action_positions is not None:
            student_indices = spans_to_indices(
                student_action_positions[sample_idx], student_total_seq
            )
        else:
            student_indices = sample_mask.nonzero(as_tuple=False).view(-1).tolist()
        
        if not student_indices:
            continue
        
        # Map to teacher indices
        if teacher_action_positions is not None:
            teacher_indices = spans_to_indices(
                teacher_action_positions[sample_idx], teacher_total_seq
            )
        else:
            teacher_indices = [idx + seq_delta for idx in student_indices]
        
        teacher_indices = [
            idx for idx in teacher_indices if 0 <= idx < teacher_total_seq
        ]
        
        if len(teacher_indices) != len(student_indices):
            raise ValueError(
                f"Mismatch between student and teacher action token counts "
                f"({len(student_indices)} vs {len(teacher_indices)}); "
                f"ensure teacher_action_positions is provided when prompts differ"
            )
        
        entropy_term = torch.tensor(0.0, device=device, dtype=torch.float32)
        cross_entropy_term = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # Process each token position
        for s_idx, t_idx in zip(student_indices, teacher_indices):
            # Find which chunk this position belongs to
            s_chunk = bisect_right(student_offsets, s_idx) - 1
            t_chunk = bisect_right(teacher_offsets, t_idx) - 1
            
            if s_chunk < 0 or t_chunk < 0:
                continue
            
            # Local position within chunk
            s_local = s_idx - student_offsets[s_chunk]
            t_local = t_idx - teacher_offsets[t_chunk]
            
            # Extract logits for this position
            student_vec = student_chunks[s_chunk][sample_idx, s_local, :]
            teacher_vec = teacher_chunks[t_chunk][sample_idx, t_local, :].detach()
            
            # Compute log probabilities
            student_log_vec = F.log_softmax(student_vec, dim=-1)
            teacher_log_vec = F.log_softmax(teacher_vec, dim=-1)
            student_prob_vec = student_log_vec.exp()
            
            # Entropy: H(p) = -Σ p(x) * log p(x)
            entropy_term = entropy_term - torch.sum(
                student_prob_vec * student_log_vec
            )
            
            # Cross-entropy: H(p, q) = -Σ p(x) * log q(x)
            cross_entropy_term = cross_entropy_term - torch.sum(
                student_prob_vec * teacher_log_vec
            )
        
        entropy_sum[sample_idx] = entropy_term
        cross_entropy_sum[sample_idx] = cross_entropy_term
    
    # Calculate averages over valid samples and tokens
    valid_mask = token_count_per_sample > 0
    total_tokens = token_count_per_sample[valid_mask].sum().clamp_min(1.0)
    
    entropy = entropy_sum[valid_mask].sum() / total_tokens
    cross_entropy = cross_entropy_sum[valid_mask].sum() / total_tokens
    reverse_kl = cross_entropy - entropy
    
    # Loss with entropy regularization
    loss = cross_entropy - entropy_coef * entropy
    
    return {
        "loss": loss,
        "reverse_kl": reverse_kl.detach(),
        "cross_entropy": cross_entropy.detach(),
        "entropy": entropy.detach(),
        "token_count": token_count.to(device=device, dtype=torch.float32),
    }
