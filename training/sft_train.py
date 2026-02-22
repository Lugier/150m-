"""
SFT Training Script.
Loads a Pre-Trained checkpoint and Fine-Tunes on Instruction Data using Assistant-Only loss.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml

# Project root for imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.config import ModelConfig
from model.gpt import CodeGPTLMHeadModel
from training.train import get_optimizer_grouped, ema_update, ema_copy_to_model
from training.device_utils import resolve_device

try:
    from data.sft_dataloader import get_sft_dataloader
except ImportError:
    get_sft_dataloader = None

def sft_train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    use_amp = scaler is not None
    with torch.autocast(device_type="cuda", enabled=use_amp):
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()

def run_sft_training(
    config_path: str = "training/config_sft.yaml",
    checkpoint: str = "checkpoints/final.pt",
    data_dir: str = "data/processed",
    instruction_data: Optional[str] = None,
    output_dir: str = "sft_checkpoints",
    tokenizer_path: str = "data/tokenizer",
    max_steps_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> None:
    
    config_path = _ROOT / config_path if not os.path.isabs(config_path) else Path(config_path)
    with open(config_path) as f:
        train_cfg = yaml.safe_load(f)

    m = train_cfg["model"]
    model_cfg = ModelConfig(
        d_model=m["d_model"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        vocab_size=m["vocab_size"],
        max_seq_len=m["max_seq_len"],
        use_bitnet=m.get("use_bitnet", False),
        use_mamba_hybrid=m.get("use_mamba_hybrid", False),
        use_blt=m.get("use_blt", False),
        use_leam=m.get("use_leam", False),
        mtp_n=m.get("mtp_n", 1),
    )

    device = resolve_device(device_override or train_cfg.get("device"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # RTX 3090

    print(f"Loading Pre-Trained Checkpoint from {checkpoint}")
    ckpt = torch.load(_ROOT / checkpoint, map_location="cpu")
    
    # We might need to resize embeddings if we added special tokens for ChatML
    pt_vocab_size = ckpt["model"]["embed.weight"].shape[0]
    if pt_vocab_size != model_cfg.vocab_size:
        print(f"Resizing vocab from {pt_vocab_size} to {model_cfg.vocab_size} to accommodate ChatML tokens.")
        # Only instantiate, load, then resize
        model = CodeGPTLMHeadModel(model_cfg)
        # Load matched shapes first
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        model = CodeGPTLMHeadModel(model_cfg)
        model.load_state_dict(ckpt["model"])
        
    model.to(device)

    opt_cfg = train_cfg["optimizer"]
    use_normuon = opt_cfg.get("use_normuon", True)
    base_lr = train_cfg["scheduler"].get("lr", 2e-5)
    
    if use_normuon:
        from training.normuon import HybridNorMuonAdamW
        optimizer = HybridNorMuonAdamW(
            model,
            lr=base_lr,
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=opt_cfg.get("betas", [0.9, 0.95]),
            normuon_momentum=opt_cfg.get("momentum", 0.95)
        )
    else:
        grouped = get_optimizer_grouped(model, weight_decay=opt_cfg.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(grouped, lr=base_lr, betas=opt_cfg.get("betas", [0.9, 0.95]))

    warmup_steps = train_cfg["scheduler"].get("warmup_steps", 500)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps))
    )

    tr_cfg = train_cfg["training"]
    batch_size = tr_cfg["batch_size"]
    seq_len = tr_cfg["seq_len"]
    grad_accum = tr_cfg.get("gradient_accumulation_steps", 4)
    max_steps = max_steps_override if max_steps_override is not None else tr_cfg["max_steps"]
    save_every = tr_cfg.get("save_every", 2000)
    use_amp = tr_cfg.get("mixed_precision") == "bf16"
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    if tr_cfg.get("gradient_checkpointing"):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = ema_cfg.get("decay", 0.999)
    ema_params = [p.clone().detach() for p in model.parameters() if p.requires_grad] if ema_enabled else []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if get_sft_dataloader is None:
        raise RuntimeError("SFT dataloader not available. Ensure data.sft_dataloader is importable.")
    instruction_path = (Path(instruction_data) if instruction_data else None)
    if instruction_path and not instruction_path.is_absolute():
        instruction_path = _ROOT / instruction_path
    data_loader = get_sft_dataloader(
        data_dir=str(_ROOT / data_dir),
        tokenizer_path=str(_ROOT / tokenizer_path),
        vocab_path="",
        vocab_size=model_cfg.vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        instruction_data_path=str(instruction_path) if instruction_path else None,
    )
    data_iter = data_loader.iter_forever()
    first_batch = next(data_iter)
    print(f"Data check: first batch input_ids {first_batch['input_ids'].shape}, labels {first_batch['labels'].shape} (batch_size, seq_len)")

    def get_batch() -> dict[str, torch.Tensor]:
        nonlocal first_batch
        if first_batch is not None:
            out, first_batch = first_batch, None
            return out
        return next(data_iter)

    global_step = 0
    running_loss = 0.0
    
    print("Starting SFT Training...")
    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            batch = get_batch()
            if batch["input_ids"].device != device:
                non_blocking = device.type == "cuda"  # SFT DataLoader nutzt pin_memory=True
                batch = {k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = sft_train_step(model, batch, device, scaler)
            running_loss += loss
            
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        global_step += 1

        if ema_enabled and ema_params:
            ema_update(ema_params, [p for p in model.parameters() if p.requires_grad], ema_decay)

        if global_step % 100 == 0:
            print(f"SFT Step {global_step} | Loss {running_loss / (100 * grad_accum):.4f}")
            running_loss = 0.0

        if save_every and global_step % save_every == 0:
            ckpt_path = Path(output_dir) / f"sft_ckpt_{global_step}.pt"
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "config": train_cfg,
            }
            if ema_enabled and ema_params:
                save_dict["ema"] = [e.cpu() for e in ema_params]
            torch.save(save_dict, ckpt_path)
            print(f"Saved {ckpt_path}")

    if ema_enabled and ema_params:
        ema_copy_to_model(ema_params, [p for p in model.parameters() if p.requires_grad])
        
    final_path = Path(output_dir) / "final_sft.pt"
    torch.save({"model": model.state_dict(), "step": global_step, "config": train_cfg}, final_path)
    print(f"SFT Done. Saved to {final_path}")

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config_sft.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--data_dir", default="data/processed", help="Directory containing instruction_sft.jsonl unless --instruction_data is set")
    parser.add_argument("--instruction_data", default=None, help="Path to instruction SFT JSONL (overrides data_dir/instruction_sft.jsonl)")
    parser.add_argument("--output_dir", default="sft_checkpoints")
    parser.add_argument("--tokenizer_path", default="data/tokenizer")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_sft_training(
        config_path=args.config,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        instruction_data=args.instruction_data,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        max_steps_override=args.max_steps,
        device_override=args.device,
    )

if __name__ == "__main__":
    main()
