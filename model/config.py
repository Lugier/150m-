"""
Model config for 100-150M Code-Only-LLM (Giant-Killer).
Deep-Thin (40-48 layers, d_model=384), optional Mamba-2-Hybrid, BitNet b1.58, L-MTP.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Single config for 100-150M target; IMU-1 / Deep-Thin compatible."""

    # Size (Shallow-Wide target ~300M)
    d_model: int = 1024
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int | None = None
    head_dim: int | None = None  # default d_model // n_head
    intermediate_size: int | None = None  # default 4 * d_model (ffn)
    vocab_size: int = 16384
    max_seq_len: int = 4096

    # Architecture (Plan §2: Giant-Killer – alle Säulen konfigurierbar)
    use_bitnet: bool = False   # BitLinear b1.58
    use_bitnet_median_scaling: bool = False  # Beta = median(|W|) for small models (Plan §2)
    use_mamba_hybrid: bool = False  # 43% Mamba, 7% Attn, 50% MLP
    use_blt: bool = False       # Byte Latent Transformer: byte input, 256 vocab head
    use_leam: bool = False      # LEAM++ grammar constrainer in Inferenz (run_chat/run_torch)
    mamba_ratio: float = 0.43
    attention_ratio: float = 0.07
    mlp_ratio: float = 0.50
    use_masa: bool = False
    use_mohd: bool = False
    use_moa: bool = False  # MoA: Mixture of Sparse Attention (arXiv 2406.14909), extends effective context
    moa_patterns: list[str] | None = None  # e.g. ["window_512", "global_8", "dilated_2"]; cycled over heads

    # L-MTP (Leap Multi-Token Prediction)
    mtp_n: int = 1  # 1 = next-token only; 2-4 = multi-token

    # Regularization
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    # QK-Norm, gating (IMU-1)
    qk_norm: bool = True
    use_value_residual: bool = True
    use_per_head_gating: bool = True

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.d_model

    @property
    def num_parameters_approx(self) -> int:
        """Rough parameter count (embedding + layers)."""
        emb = self.vocab_size * self.d_model
        # per layer: attn ~ 4*d^2, ffn ~ 3*4*d^2 = 12*d^2
        per_layer = 4 * self.d_model * self.d_model + 3 * self.d_model * self.intermediate_size
        return int(emb + self.n_layer * per_layer)


# Preset configs from plan
EARLY_ITERATION_CONFIG = ModelConfig(
    d_model=384,
    n_layer=28,
    n_head=6,
    vocab_size=16384,
    max_seq_len=512,
    mtp_n=1,
)

TARGET_100_150M_CONFIG = ModelConfig(
    d_model=1024,
    n_layer=24,
    n_head=16,
    vocab_size=16384,
    max_seq_len=4096,
    use_bitnet=False,
    use_mamba_hybrid=False,
    mtp_n=1,
)

TARGET_100_150M_BITNET_CONFIG = ModelConfig(
    d_model=1024,
    n_layer=24,
    n_head=16,
    vocab_size=16384,
    max_seq_len=4096,
    use_bitnet=True,
    use_mamba_hybrid=False,
    mtp_n=4,
)
