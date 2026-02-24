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
    use_median_scaling: beta = median(|W|) for small models (more robust than mean).
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        layer_scale: Optional[float] = None,
        use_median_scaling: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_median_scaling = use_median_scaling
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        # per-layer scaling (TernaryLM-style); median recommended for small models (Plan §2)
        self.layer_scale = layer_scale if layer_scale is not None else 1.0 / math.sqrt(in_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _quantize_weight(self) -> torch.Tensor:
        abs_w = self.weight.data.abs()
        beta = (
            abs_w.median().clamp(min=1e-8)
            if self.use_median_scaling
            else abs_w.mean().clamp(min=1e-8)
        )
        w_norm = self.weight / beta
        w_quant = round_ste_clip(w_norm, -1.0, 1.0)
        return w_quant * beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._quantize_weight()
        out = nn.functional.linear(x, w, self.bias)
        return out * self.layer_scale


def replace_linear_with_bitlinear(
    module: nn.Module,
    layer_scale: Optional[float] = None,
    use_median_scaling: bool = False,
) -> nn.Module:
    """Replace nn.Linear with BitLinear in-place (for model conversion)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            bit = BitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                layer_scale=layer_scale,
                use_median_scaling=use_median_scaling,
            )
            bit.weight.data = child.weight.data
            if child.bias is not None:
                bit.bias.data = child.bias.data
            setattr(module, name, bit)
        else:
            replace_linear_with_bitlinear(child, layer_scale, use_median_scaling)
    return module
