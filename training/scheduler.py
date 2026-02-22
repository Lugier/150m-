"""
WSD (Warmup-Stable-Decay) + three Stage-LRs (IMU-1 style).
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_wsd_lambda(
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
):
    """Returns a multiplier for LR: warmup -> 1.0 -> stable -> linear decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        step = step - warmup_steps
        if step < stable_steps:
            return 1.0
        step = step - stable_steps
        if step >= decay_steps:
            return 0.01
        return 1.0 - 0.99 * (float(step) / float(decay_steps))

    return lr_lambda


def get_stage_lr(
    step: int,
    stage_boundaries: list[int],
    stage_lrs: list[float],
) -> float:
    """Return LR for current step based on stage boundaries. stage_boundaries = [end_stage1, end_stage2, ...]."""
    for i, end in enumerate(stage_boundaries):
        if step < end:
            return stage_lrs[min(i, len(stage_lrs) - 1)]
    return stage_lrs[-1] if stage_lrs else 1e-5


class WSDScheduler(LambdaLR):
    """Warmup-Stable-Decay per stage."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 500,
        stable_steps: int = 10000,
        decay_steps: int = 20000,
    ) -> None:
        fn = get_wsd_lambda(warmup_steps, stable_steps, decay_steps)
        super().__init__(optimizer, fn)


class StageLRScheduler(LambdaLR):
    """Stage-dependent LR: base_lr * get_stage_lr(step)."""

    def __init__(
        self,
        optimizer: Optimizer,
        stage_boundaries: list[int],
        stage_lrs: list[float],
        warmup_steps: int = 500,
    ) -> None:
        self.stage_boundaries = stage_boundaries
        self.stage_lrs = stage_lrs
        self.warmup_steps = warmup_steps
        base_lr0 = stage_lrs[0] if stage_lrs else 1e-4

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            idx = 0
            s = step - warmup_steps
            for i, end in enumerate(stage_boundaries):
                if s < end:
                    idx = i
                    break
                idx = i + 1
            lr = stage_lrs[min(idx, len(stage_lrs) - 1)]
            return lr / base_lr0

        super().__init__(optimizer, lr_lambda)

    def get_current_lr_value(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.stage_lrs[0] * step / max(1, self.warmup_steps)
        idx = 0
        s = step - self.warmup_steps
        for i, end in enumerate(self.stage_boundaries):
            if s < end:
                idx = i
                break
            idx = i + 1
        return self.stage_lrs[min(idx, len(self.stage_lrs) - 1)]
