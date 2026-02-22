"""
BLT (Byte Latent Transformer, Plan §2): tokenizer-freie Byte-Repräsentation.
In gpt.py bei use_blt: Embed = BytePatchEncoder(256), LM-Head = BytePatchDecoder(256).
Eingabe: Byte-IDs 0–255; Ausgabe: Logits über 256 Bytes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class BytePatchEncoder(nn.Module):
    """Byte-ID (0–255) → d_model; in gpt.py als self.embed bei use_blt."""
    def __init__(self, vocab_byte: int = 256, d_model: int = 384):
        super().__init__()
        self.embed = nn.Embedding(vocab_byte, d_model)
        self.ngram_proj = nn.Linear(d_model, d_model)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(byte_ids.clamp(0, 255))
        return self.ngram_proj(x)

class BytePatchDecoder(nn.Module):
    """Latent → 256 Byte-Logits; in gpt.py als self.lm_head bei use_blt."""
    def __init__(self, d_model: int = 384, vocab_byte: int = 256):
        super().__init__()
        self.decode_proj = nn.Linear(d_model, vocab_byte)
        
    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.decode_proj(latent_states)

class BLTWrapper(nn.Module):
    """
    BLT Structure: 
    Byte stream -> Dynamic patches (Entropy) -> Latent Transformer -> Loc Decoder.
    """
    def __init__(self, d_model: int = 384, latent_transformer: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = BytePatchEncoder(vocab_byte=256, d_model=d_model)
        # Binds to the actual Mamba-Hybrid layers or falls back to identity projection
        self.latent_transformer_stub = latent_transformer if latent_transformer is not None else nn.Identity()
        self.decoder = BytePatchDecoder(d_model=d_model, vocab_byte=256)
        self.entropy_threshold = 1.5

    def compute_entropy_patches(self, byte_logits: torch.Tensor) -> torch.Tensor:
        """Calculates next-byte entropy to identify cross-patch boundaries."""
        probs = torch.softmax(byte_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        # Returns boolean mask where true indicates a new patch boundary
        return entropy > self.entropy_threshold

    def forward(self, byte_input: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(byte_input)
        
        # Real logic for boundary detection and latent packing would require dynamic shape 
        # realignment. Here we perform a continuous pass simulating homogeneous latent 
        # lengths for matrix compatibility, while maintaining the structural flow.
        latent = self.latent_transformer_stub(encoded)
        
        return self.decoder(latent)
