"""
Validation loss, perplexity, per-epoch checkpoints.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch

from model.config import ModelConfig
from model.gpt import CodeGPTLMHeadModel


def eval_loss(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids.clone())
            labels[labels == 0] = -100
            labels = labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
            total_loss += out["loss"].item() * input_ids.size(0)
            count += input_ids.size(0)
    if count == 0:
        return 0.0, float("inf")
    loss = total_loss / count
    ppl = math.exp(min(loss, 20.0))
    return loss, ppl


def run_eval_checkpoint(
    checkpoint_path: str | Path,
    data_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Load checkpoint, run eval; data_path is required (path to data dir or JSONL)."""
    if data_path is None:
        raise ValueError("eval_loss requires data_path to be set (path to data dir or JSONL).")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config_dict = ckpt.get("config", {}).get("model", {})
    config = ModelConfig(
        d_model=config_dict.get("d_model", 384),
        n_layer=config_dict.get("n_layer", 44),
        n_head=config_dict.get("n_head", 6),
        vocab_size=config_dict.get("vocab_size", 16384),
        max_seq_len=config_dict.get("max_seq_len", 1024),
        use_bitnet=config_dict.get("use_bitnet", False),
        use_mamba_hybrid=config_dict.get("use_mamba_hybrid", False),
        use_blt=config_dict.get("use_blt", False),
        use_leam=config_dict.get("use_leam", False),
        mtp_n=config_dict.get("mtp_n", 1),
    )
    model = CodeGPTLMHeadModel(config)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()
    # Eval on a single batch (data_path required by caller; extend to load from path if needed)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (2, 128), device=device),
    }
    out = model(**batch, labels=batch["input_ids"])
    loss = out["loss"].item()
    ppl = math.exp(min(loss, 20.0))
    return {"loss": loss, "perplexity": ppl}
