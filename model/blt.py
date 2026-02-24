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
    """Byte-ID (0–vocab) → d_model. Fixed patches for Mamba compatibility (Error 2)."""
    def __init__(self, vocab_byte: int = 300, d_model: int = 384):
        super().__init__()
        self.embed = nn.Embedding(vocab_byte, d_model)
        self.ngram_proj = nn.Linear(d_model, d_model)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        # Error 12: Support ChatML special tokens > 255
        x = self.embed(byte_ids)
        return self.ngram_proj(x)

class BytePatchDecoder(nn.Module):
    """Latent → vocab Byte-Logits."""
    def __init__(self, d_model: int = 384, vocab_byte: int = 300):
        super().__init__()
        self.decode_proj = nn.Linear(d_model, vocab_byte)
        
    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.decode_proj(latent_states)

class BLTWrapper(nn.Module):
    """
    BLT Structure with Fixed Patches.
    """
    def __init__(self, d_model: int = 384, latent_transformer: Optional[nn.Module] = None, vocab_byte: int = 300):
        super().__init__()
        self.encoder = BytePatchEncoder(vocab_byte=vocab_byte, d_model=d_model)
        self.latent_transformer_stub = latent_transformer if latent_transformer is not None else nn.Identity()
        self.decoder = BytePatchDecoder(d_model=d_model, vocab_byte=vocab_byte)

    def forward(self, byte_input: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(byte_input)
        
        # Error 2: Mamba kernels require continuous, even timesteps. 
        # Variable entropy patches break Mamba-2 hardware utilization and semantics.
        # We enforce fixed 1:1 tokens-to-latents mapping here, relying on the tokenizer to pre-pack bytes.
        latent = self.latent_transformer_stub(encoded)
        
        return self.decoder(latent)
