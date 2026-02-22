"""
Mamba-2-Hybrid – optional/2R: 43% Mamba-2, 7% Attention, 50% MLP.
Requires mamba-ssm or equivalent; this module provides the layer router and stub.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig


class MambaBlockStub(nn.Module):
    """
    Stub when mamba_ssm is not installed. Replace with real Mamba-2 block when available.
    Fallback: linear projection to preserve shape.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.proj(x)

class Mamba2Wrapper(nn.Module):
    """Wrap Mamba2 to ignore attention-specific kwargs like attention_mask/past_kv."""
    def __init__(self, d_model: int):
        super().__init__()
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(d_model=d_model, d_state=128, d_conv=4, expand=2)
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        # returns output and a dummy cache
        return self.mamba(x), None


def make_mamba_hybrid_layer(config: ModelConfig, layer_idx: int) -> nn.Module:
    """
    Return one hybrid layer: Mamba, Attention, or MLP by ratio.
    Layer type determined by (layer_idx / n_layer) vs mamba_ratio, attention_ratio, mlp_ratio.
    """
    n = config.n_layer
    r = layer_idx / max(1, n)
    if r < config.mamba_ratio:
        try:
            return Mamba2Wrapper(config.d_model)
        except ImportError:
            pass
        except Exception:
            pass
        return MambaBlockStub(config.d_model)
    if r < config.mamba_ratio + config.attention_ratio:
        from .gpt import TransformerBlock
        return TransformerBlock(config)
    from .gpt import MLP
    return nn.Sequential(
        nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        MLP(config),
    )
