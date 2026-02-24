"""
MoA (Mixture of Sparse Attention, arXiv 2406.14909): per-head/layer sparse patterns.
Training-free: Window, Global, Dilated patterns extend effective context (8K->32K+)
without full dense attention. Used in Attention when config.use_moa is True.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def make_moa_sparse_mask(
    seq_len: int,
    n_heads: int,
    patterns: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build causal sparse attention mask (B=1, n_heads, T, T) for MoA.
    patterns: e.g. ["window_512", "global_8", "dilated_2"] cycled over heads.
    True = attend, False = mask out (-inf in attention).
    """
    if patterns is None:
        patterns = ["window_512", "global_8", "dilated_2"]
    mask = torch.zeros(n_heads, seq_len, seq_len, dtype=torch.bool, device=device)
    for h in range(n_heads):
        p = patterns[h % len(patterns)]
        if p.startswith("window_"):
            w = int(p.split("_")[1])
            for i in range(seq_len):
                start = max(0, i - w)
                mask[h, i, start : i + 1] = True
        elif p.startswith("global_"):
            g = int(p.split("_")[1])
            for i in range(seq_len):
                mask[h, i, : min(g, i + 1)] = True
                mask[h, i, max(0, i - g) : i + 1] = True
        elif p.startswith("dilated_"):
            d = int(p.split("_")[1])
            for i in range(seq_len):
                for j in range(0, i + 1, d):
                    mask[h, i, j] = True
                mask[h, i, i] = True
        else:
            mask[h, :, :] = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0)


def apply_moa_mask(
    attn_weights: torch.Tensor,
    seq_len: int,
    n_heads: int,
    patterns: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Apply MoA sparse mask to attention logits (before softmax).
    attn_weights: (B, n_heads, T, T). Masked positions get -1e9.
    """
    device = attn_weights.device
    moa_mask = make_moa_sparse_mask(seq_len, n_heads, patterns=patterns, device=device)
    expand = attn_weights.shape[0]
    moa_mask = moa_mask.expand(expand, -1, -1, -1)
    attn_weights = attn_weights.masked_fill(~moa_mask, -1e9)
    return attn_weights
