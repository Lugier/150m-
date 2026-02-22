"""
Single source of truth for device selection (Plan: RunPod RTX 3090, MPS, CPU).
Used by train.py, sft_train.py, rl_train.py, inference scripts.
"""

from __future__ import annotations

import torch


def resolve_device(override: str | None = None) -> torch.device:
    """
    Resolve training/inference device: cuda → mps → cpu.
    Override wins if provided (e.g. from config or CLI).
    """
    if override is not None and override.strip():
        return torch.device(override.strip())
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
