"""
BitNet b1.58: ternary weights (-1, 0, +1), ~1.58 bit.
Straight-Through Estimator for rounding; layer-wise scaling (TernaryLM).
Optional: use for native low-bit training on 100-150M.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def round_ste_clip(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """Straight-through: forward = clip+round, backward = identity."""
    rounded = torch.clamp(torch.round(x), min_val, max_val)
    return x + (rounded - x).detach()


class BitLinear(nn.Module):
    """
    Linear layer with ternary weights (BitNet b1.58).
    W_quant in {-1, 0, +1}; scaling per output channel via beta.
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        layer_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        # per-layer scaling (TernaryLM-style)
        self.layer_scale = layer_scale if layer_scale is not None else 1.0 / math.sqrt(in_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _quantize_weight(self) -> torch.Tensor:
        # beta = mean abs for scaling
        beta = self.weight.data.abs().mean().clamp(min=1e-8)
        w_norm = self.weight / beta
        w_quant = round_ste_clip(w_norm, -1.0, 1.0)
        return w_quant * beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._quantize_weight()
        out = nn.functional.linear(x, w, self.bias)
        return out * self.layer_scale


def replace_linear_with_bitlinear(module: nn.Module, layer_scale: Optional[float] = None) -> nn.Module:
    """Replace nn.Linear with BitLinear in-place (for model conversion)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            bit = BitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                layer_scale=layer_scale,
            )
            bit.weight.data = child.weight.data
            if child.bias is not None:
                bit.bias.data = child.bias.data
            setattr(module, name, bit)
        else:
            replace_linear_with_bitlinear(child, layer_scale)
    return module
