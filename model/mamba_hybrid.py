"""
Mamba-2-Hybrid (Plan §2): 43% Mamba-2, 7% Attention, 50% MLP per layer index.
Used in gpt.py when config.use_mamba_hybrid is True. Requires mamba-ssm for real Mamba2;
otherwise MambaBlockStub preserves shape. All layer types return (x, cache) for unified forward.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig


class MambaBlockStub(nn.Module):
    """
    Stub wenn mamba_ssm nicht installiert ist. Shape-erhaltende lineare Projektion
    (kein Konfig-Dummy: Abhängigkeit von mamba_ssm-Paket).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        return x + self.proj(x), None

class Mamba2Wrapper(nn.Module):
    """Wrap Mamba2 to ignore attention-specific kwargs like attention_mask/past_kv.
    Isolates Mamba C++ kernel in bfloat16 to avoid Autocast/BitNet precision crashes (Error 18).
    Avoids native PyTorch gradient checkpointing crashes (Error 24)."""
    def __init__(self, d_model: int):
        super().__init__()
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(d_model=d_model, d_state=128, d_conv=4, expand=2)
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        # Error 23: Mamba has no QK-Norm native. We don't artificially inject it.
        # Error 18, 24: Isolate Mamba C++ kernel in bfloat16 and protect from external AMP conflicts.
        device_type = x.device.type if x.device.type in ["cuda", "cpu"] else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = self.mamba(x.to(torch.bfloat16))
        return out, None


def make_mamba_hybrid_layer(config: ModelConfig, layer_idx: int) -> nn.Module:
    """
    Return one hybrid layer: Mamba, Attention, or MLP by ratio (Plan §2).
    r = layer_idx/n_layer; r < mamba_ratio → Mamba, else r < mamba+attention → TransformerBlock, else MLP-only.
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
    # MLP-only block (50%): same (x, cache) API for gpt.py
    class MLPOnlyBlock(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
            self.mlp = MLP(cfg)
        def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = False, past_kv: Optional[tuple] = None) -> tuple[torch.Tensor, None]:
            return x + self.mlp(self.ln(x)), None
    return MLPOnlyBlock(config)
