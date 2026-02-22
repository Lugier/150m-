"""
Decoder-only Code GPT (100-150M).
Deep-Thin, optional BitLinear, QK-Norm, value residual, L-MTP.
Mamba-2-Hybrid and LEAM++ are optional extension points (see mamba_hybrid.py, leam.py).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .bitnet import BitLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed


def _linear(in_features: int, out_features: int, bias: bool, use_bitnet: bool) -> nn.Module:
    if use_bitnet:
        return BitLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


class Attention(nn.Module):
    """Multi-head self-attention with QK-norm, value residual, optional per-head gating."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim or (config.d_model // config.n_head)
        self.d_model = config.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qk_norm = config.qk_norm
        self.use_value_residual = config.use_value_residual
        self.use_per_head_gating = config.use_per_head_gating
        use_bit = config.use_bitnet

        self.q_proj = _linear(config.d_model, self.n_head * self.head_dim, bias=False, use_bitnet=use_bit)
        self.k_proj = _linear(config.d_model, self.n_head * self.head_dim, bias=False, use_bitnet=use_bit)
        self.v_proj = _linear(config.d_model, self.n_head * self.head_dim, bias=False, use_bitnet=use_bit)
        self.o_proj = _linear(self.n_head * self.head_dim, config.d_model, bias=False, use_bitnet=use_bit)
        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.resid_drop = nn.Dropout(config.resid_dropout)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.layer_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if config.use_per_head_gating:
            self.gate = nn.Parameter(torch.ones(1, self.n_head, 1, 1))
        else:
            self.gate = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        att = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        if self.gate is not None:
            att = att * self.gate

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        if self.use_value_residual:
            out = out + self.v_proj(x)
        out = self.o_proj(out)
        out = self.resid_drop(out)

        cache = (k, v) if use_cache else None
        return out, cache


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        use_bit = config.use_bitnet
        self.fc1 = _linear(config.d_model, config.intermediate_size, bias=False, use_bitnet=use_bit)
        self.fc2 = _linear(config.intermediate_size, config.d_model, bias=False, use_bitnet=use_bit)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.attn = Attention(config)
        self.ln2 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a, cache = self.attn(self.ln1(x), attention_mask, use_cache=use_cache, past_kv=past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, cache


class CodeGPTLMHeadModel(nn.Module):
    """
    Code-only GPT with optional L-MTP (multiple next-token heads).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.mtp_n = max(1, config.mtp_n)

        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model, config.layer_norm_eps)

        # LM head: shared trunk, mtp_n prediction heads
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.mtp_n > 1:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(config.mtp_n - 1)
            ])
        else:
            self.mtp_heads = nn.ModuleList([])

        self.embed.weight.data.normal_(std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mtp_labels: Optional[list[torch.Tensor]] = None,
        use_cache: bool = False,
        past_kv_list: Optional[list[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        x = self.embed(input_ids) + self.pos_embed[:, :T, :]
        causal = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)
        if attention_mask is not None:
            causal = causal & attention_mask.unsqueeze(1).bool()

        cache_list: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, block in enumerate(self.blocks):
            past = past_kv_list[i] if past_kv_list is not None else None
            x, cache = block(x, attention_mask=causal, use_cache=use_cache, past_kv=past)
            if use_cache:
                cache_list.append(cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            if self.mtp_n > 1 and mtp_labels is not None:
                for offset, head in enumerate(self.mtp_heads, start=1):
                    if offset >= len(mtp_labels):
                        break
                    logits_t = head(x)
                    shift_t = logits_t[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                    labels_t = mtp_labels[offset][..., 1:].contiguous().view(-1)
                    loss = loss + F.cross_entropy(shift_t, labels_t, ignore_index=-100)
                loss = loss / max(1, min(len(mtp_labels), self.mtp_n))

        out: dict[str, torch.Tensor] = {"logits": logits}
        if loss is not None:
            out["loss"] = loss
        if use_cache:
            out["past_key_values"] = cache_list
        return out
