# High-Efficiency Design (Best-in-Class & Instruction SFT + RL Plans)

## Data setup (damit alles läuft)

- **Pre-Train:** `data/processed/` mit mindestens einer JSONL (z. B. `stage1.jsonl`, `train.jsonl`) und Tokenizer unter `data/tokenizer` (z. B. `python data/tokenizer_train.py --output data/tokenizer`).
- **SFT:** `data/processed/instruction_sft.jsonl` (z. B. `python data/generate_instruction_data.py --output data/processed/instruction_sft.jsonl`).
- **RL:** Gleiche JSONL wie SFT oder eigene mit `--rl-data` angeben.
- **Prüfen:** `python scripts/verify_data.py` zeigt vorhandene Daten und Configs.

This document summarizes how the codebase implements a **clean, highly efficient** design aligned with both plans.

## 1. Single Source of Truth

| Concern | Location | Usage |
|--------|----------|--------|
| **Device** | `training/device_utils.resolve_device(override)` | train.py, sft_train.py, rl_train.py; cuda → mps → cpu |
| **Model config** | YAML: `training/config_train.yaml`, `training/config_sft.yaml` | Pre-train, SFT; model/config.py dataclass |
| **Parameter cap** | Config comments + ModelConfig | Max 250M (Plan §6): n_layer≤48, d_model≤448 |
| **Plan §2 Säulen** | use_blt, use_mamba_hybrid, use_leam, use_bitnet, mtp_n in YAML | BLT, Mamba-2-Hybrid, LEAM++, BitNet, L-MTP; CODA/CodeDenoise in data; CodeRL+ in RL; S*/PoT in inference/test_time_evolution.py |

## 2. Training Efficiency (Plan §6, RTX 3090 24 GB)

- **Mixed precision:** bf16 in pre-train and SFT (`mixed_precision: bf16`), autocast in RL policy forward/loss.
- **Gradient checkpointing:** In Config aktiviert; reduziert Peak-VRAM (Pflicht für 150M @ 24 GB).
- **Gradient accumulation:** Pre-train eff. Batch 32 (8×4), SFT eff. Batch 36 (6×6); bei OOM `batch_size` in YAML reduzieren.
- **Optimizer:** NorMuon-AdamW hybrid für weniger Optimizer-State und schnellere Konvergenz.
- **Stage-LRs + EMA:** Pre-train StageLRScheduler + Checkpoint-EMA; SFT Warmup + EMA.

**RTX 3090 (24 GB) – empfohlene Config-Werte:**

| Phase     | batch_size | seq_len | grad_accum | Hinweis |
|----------|------------|---------|------------|---------|
| Pre-train| 8          | 512     | 4          | OOM → 6 oder seq_len 384 |
| SFT      | 6          | 1024    | 6          | OOM → batch 4 |
| RL       | —          | —       | num_candidates=4 | Policy + Ref auf GPU |

**CUDA-Optimierungen (automatisch bei device=cuda):** `torch.backends.cudnn.benchmark = True`; Pre-train DataLoader mit `pin_memory=True` und `non_blocking=True` beim Transfer.

## 3. Data Pipeline Efficiency

- **SFT:** Streaming JSONL, DataLoader mit `pin_memory=True`; Batch-Transfer mit `non_blocking=True` auf CUDA.
- **Pre-train:** `CodeDataLoader` mit optionalem `pin_memory` (bei CUDA aktiv); `get_training_dataloader(..., pin_memory=True)` für async CPU→GPU.
- **RL:** Prompts einmal aus JSONL; pro Prompt K Kandidaten (num_candidates=4), Policy + Ref auf GPU.

## 4. RL Efficiency

- **GRPO:** Single prompt → K candidates → rewards → advantage → one policy backward; ref model in `no_grad`, policy in `autocast(device_type="cuda")` when on GPU.
- **Execution reward:** Syntax check first (cheap), then sandbox only when needed; `run_tests_in_sandbox` returns `(passed, output)` once.

## 5. No Redundant Work

- No fallbacks: tokenizer, data paths, and RL data are required; missing resources raise immediately.
- Tokenizer loaded once per process (SFT/RL/pre-train); no per-batch reload.
- Config read once at start; model config built from YAML only.

## 6. Module Boundaries

- **data/** – formats, tokenizer load, dataloaders (instruction, pretrain).
- **model/** – config, gpt (with gradient checkpointing), optional bitnet/mamba_hybrid/leam.
- **training/** – device_utils, scheduler, normuon, train/sft_train/rl_train.
- **post_training/** – execution_reward, distill.
- **inference/** – run_chat, run_torch, run_mlx.
- **evaluation/** – eval_repair, eval_loss, eval_humaneval, eval_livecode.

## 7. Plan Compliance

- **Instruction SFT + RL:** ChatML, Assistant-only labels, Evol-Instruct/Teacher data, SFT then RL with execution reward, GRPO; inferenz with ChatML (run_chat). All implemented; no placeholders.
- **Best-in-class (core):** Deep-Thin (44L, d_model=384), NorMuon-AdamW, Stage-LRs, EMA, gradient checkpointing, bf16, streaming-capable data path; optional BitNet/Mamba/LEAM in config and modules.
