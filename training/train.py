"""
Training loop: three Stage-LRs, WSD, NorMuon-AdamW hybrid, Checkpoint-EMA.
Target: RunPod RTX 3090 (24 GB); gradient checkpointing, bf16.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
import yaml

# Project root for imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.config import ModelConfig
from model.gpt import CodeGPTLMHeadModel
from training.scheduler import StageLRScheduler
from training.device_utils import resolve_device

try:
    from data.dataloader import get_training_dataloader
except ImportError:
    get_training_dataloader = None


def get_optimizer_grouped(
    model: nn.Module,
    weight_decay: float = 0.01,
    use_normuon_2d: bool = True,
) -> list[dict[str, Any]]:
    """
    NorMuon-AdamW hybrid: NorMuon for 2D (Linear weight matrices), AdamW for 1D (embed, bias, norm).
    If NorMuon not available, all AdamW.
    """
    decay = set()
    no_decay = {"bias", "LayerNorm.weight", "norm.weight", "embed.weight"}
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            decay.add(n)
        else:
            no_decay.add(n)

    grouped: list[dict[str, Any]] = []
    decay_params = [p for n, p in model.named_parameters() if n in decay and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if n not in decay and p.requires_grad]

    if decay_params:
        grouped.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        grouped.append({"params": no_decay_params, "weight_decay": 0.0})

    return grouped


def ema_update(
    ema_params: list[torch.Tensor],
    model_params: list[torch.Tensor],
    decay: float = 0.999,
) -> None:
    for ema, p in zip(ema_params, model_params):
        ema.mul_(decay).add_(p.data, alpha=1 - decay)


def ema_copy_to_model(ema_params: list[torch.Tensor], model_params: list[torch.Tensor], is_bitnet: bool = False) -> None:
    for ema, p in zip(ema_params, model_params):
        if is_bitnet and p.dim() >= 2:
            beta = ema.abs().mean().clamp(min=1e-8)
            w_norm = ema / beta
            w_quant = torch.clamp(torch.round(w_norm), -1.0, 1.0) * beta
            p.data.copy_(w_quant)
        else:
            p.data.copy_(ema)


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch.get("labels")
    if labels is None:
        labels = input_ids.clone()
        labels[labels == 0] = -100
    else:
        labels = labels.to(device)

    use_amp = scaler is not None  # bf16 only on CUDA (Plan §6)
    
    mtp_labels = None
    if getattr(model, "mtp_n", 1) > 1:
        mtp_labels = [labels]
        for offset in range(1, model.mtp_n):
            shifted = torch.roll(labels, shifts=-offset, dims=1)
            shifted[:, -offset:] = -100
            mtp_labels.append(shifted)

    with torch.autocast(device_type="cuda", enabled=use_amp):
        out = model(input_ids=input_ids, labels=labels, mtp_labels=mtp_labels)
        loss = out["loss"]

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


def run_training(
    config_path: str = "training/config_train.yaml",
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> None:
    config_path = _ROOT / config_path if not os.path.isabs(config_path) else Path(config_path)
    with open(config_path) as f:
        train_cfg = yaml.safe_load(f)
    if device_override is not None:
        train_cfg["device"] = device_override

    m = train_cfg["model"]
    # Plan §2: use_bitnet, use_mamba_hybrid, use_blt, use_leam aus YAML (config_train.yaml).
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
        torch.backends.cudnn.benchmark = True  # RTX 3090: feste Input-Shapes → schneller
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    elif device.type == "mps":
        print("Using MPS (Apple Silicon)")
    else:
        print("Running on CPU. Set device=cuda on RunPod for full training (Plan §12).")
    model = CodeGPTLMHeadModel(model_cfg).to(device)

    opt_cfg = train_cfg["optimizer"]
    use_normuon = opt_cfg.get("use_normuon", True)
    
    if use_normuon:
        from training.normuon import HybridNorMuonAdamW
        optimizer = HybridNorMuonAdamW(
            model,
            lr=train_cfg["scheduler"]["stage_lrs"]["stage1"],
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=opt_cfg.get("betas", [0.9, 0.95]),
            normuon_momentum=opt_cfg.get("momentum", 0.95)
        )
    else:
        grouped = get_optimizer_grouped(model, weight_decay=opt_cfg.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(grouped, lr=train_cfg["scheduler"]["stage_lrs"]["stage1"], betas=opt_cfg.get("betas", [0.9, 0.95]))

    stages = train_cfg.get("stages", {})
    stage_boundaries = [
        stages.get("stage1_steps", 40000),
        stages.get("stage1_steps", 40000) + stages.get("stage2_steps", 35000),
        stages.get("stage1_steps", 40000) + stages.get("stage2_steps", 35000) + stages.get("stage3_steps", 25000),
    ]
    stage_lrs = [
        train_cfg["scheduler"]["stage_lrs"]["stage1"],
        train_cfg["scheduler"]["stage_lrs"]["stage2"],
        train_cfg["scheduler"]["stage_lrs"]["stage3"],
    ]
    scheduler = StageLRScheduler(
        optimizer,
        stage_boundaries=stage_boundaries,
        stage_lrs=stage_lrs,
        warmup_steps=train_cfg["scheduler"].get("warmup_steps", 500),
    )

    tr_cfg = train_cfg["training"]
    batch_size = tr_cfg["batch_size"]
    seq_len = tr_cfg["seq_len"]
    grad_accum = tr_cfg.get("gradient_accumulation_steps", 4)
    max_steps = max_steps_override if max_steps_override is not None else tr_cfg["max_steps"]
    eval_every = tr_cfg.get("eval_every", 2000)
    save_every = tr_cfg.get("save_every", 5000)
    use_amp = tr_cfg.get("mixed_precision") == "bf16"
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    if tr_cfg.get("gradient_checkpointing"):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = ema_cfg.get("decay", 0.999)
    ema_params = [p.clone().detach() for p in model.parameters() if p.requires_grad] if ema_enabled else []

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    start_step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)

    data_dir_resolved = _ROOT / data_dir if not os.path.isabs(data_dir) else Path(data_dir)
    tokenizer_path_resolved = (tokenizer_path and (_ROOT / tokenizer_path if not os.path.isabs(tokenizer_path) else Path(tokenizer_path))) or _ROOT / "data" / "tokenizer"
    vocab_path_resolved = (vocab_path and (_ROOT / vocab_path if not os.path.isabs(vocab_path) else Path(vocab_path))) or _ROOT / "model" / "vocab"
    if get_training_dataloader is None:
        raise RuntimeError("Training dataloader not available. Ensure data.dataloader is importable.")
    data_loader = get_training_dataloader(
        data_dir=data_dir_resolved,
        tokenizer_path=tokenizer_path_resolved,
        vocab_path=vocab_path_resolved,
        vocab_size=model_cfg.vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=train_cfg.get("seed", 42),
        pin_memory=(device.type == "cuda"),
    )
    if data_loader is None:
        raise FileNotFoundError("Dataloader returned None. Check data_dir and tokenizer path; train tokenizer and prepare JSONL data.")
    data_iter = data_loader.iter_forever()
    print("Using data from", data_dir_resolved)
    first_batch = next(data_iter)
    print(f"Data check: first batch input_ids shape {first_batch['input_ids'].shape} (batch_size, seq_len)")

    def get_batch() -> dict[str, torch.Tensor]:
        nonlocal first_batch
        if first_batch is not None:
            out, first_batch = first_batch, None
            return out
        return next(data_iter)

    global_step = start_step
    running_loss = 0.0
    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            batch = get_batch()
            if batch["input_ids"].device != device:
                non_blocking = (device.type == "cuda" and batch["input_ids"].is_pinned())
                batch = {k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = train_step(model, batch, device, scaler)
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
            print(f"step {global_step} loss {running_loss / (100 * grad_accum):.4f}")
            running_loss = 0.0

        if save_every and global_step % save_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"ckpt_{global_step}.pt"
            save_dict: dict[str, Any] = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "config": train_cfg,
            }
            if ema_enabled and ema_params:
                save_dict["ema"] = [e.cpu() for e in ema_params]
            torch.save(save_dict, ckpt_path)
            # Also save safetensors for inference
            try:
                from safetensors.torch import save_file
                st_path = Path(checkpoint_dir) / f"ckpt_{global_step}.safetensors"
                save_file({k: v.cpu() for k, v in model.state_dict().items()}, st_path)
                # Save config for inference
                import json
                config_path = Path(checkpoint_dir) / f"ckpt_{global_step}_config.json"
                with open(config_path, "w") as f:
                    json.dump({"model": train_cfg["model"]}, f, indent=2)
            except Exception:
                pass
            print(f"Saved {ckpt_path}")

    if ema_enabled and ema_params:
        ema_copy_to_model(ema_params, [p for p in model.parameters() if p.requires_grad], is_bitnet=model_cfg.use_bitnet)
    final_path = Path(checkpoint_dir) / "final.pt"
    torch.save({"model": model.state_dict(), "step": global_step, "config": train_cfg}, final_path)
    try:
        from safetensors.torch import save_file
        save_file({k: v.cpu() for k, v in model.state_dict().items()}, Path(checkpoint_dir) / "final.safetensors")
        import json
        with open(Path(checkpoint_dir) / "final_config.json", "w") as f:
            json.dump({"model": train_cfg["model"]}, f, indent=2)
    except Exception:
        pass
    print(f"Training done. Final checkpoint: {final_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config_train.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tokenizer_path", default=None, help="e.g. data/tokenizer or model/vocab")
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Override config max_steps (e.g. 20 for smoke test)")
    parser.add_argument("--device", default=None, help="Override device (cuda, mps, cpu). Default from config.")
    args = parser.parse_args()
    run_training(
        config_path=args.config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        tokenizer_path=args.tokenizer_path,
        vocab_path=args.vocab_path,
        max_steps_override=args.max_steps,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
