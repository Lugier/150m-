# 150M Code-Only Language Model: A "Giant-Killer" Pipeline

**A fully reproducible, research-grade pipeline for training a 100–150M parameter code-specialized decoder-only language model that targets performance competitive with or superior to general-purpose models 10× larger on standard code benchmarks.**

---

## Abstract

We present an end-to-end pipeline for training a **code-only** language model in the 100–150M parameter regime. The design follows a *Giant-Killer* methodology: by combining (1) a three-stage, quality-focused data curriculum (broad → clean/educational → instruction), (2) a deep-thin transformer architecture with optional BitNet-style quantization and syntax-aware components, (3) stage-dependent learning rates and checkpoint EMA, and (4) optional post-training via trajectory distillation and execution-semantics rewards (CodeRL+), we aim to match or exceed code-generation performance of 1B–15B general-purpose models on fixed benchmarks (HumanEval+, MBPP+, LiveCodeBench). The codebase is self-contained, config-driven, and verified against a detailed research plan; training is designed to run on a single **RunPod RTX 3090 (24 GB)** or locally on Apple Silicon / CPU for iteration.

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation & Quick Start](#installation--quick-start)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Post-Training](#post-training)
- [Inference & Evaluation](#inference--evaluation)
- [Reproducibility](#reproducibility)
- [Citation & References](#citation--references)
- [License & Compliance](#license--compliance)

---

## Overview

| Aspect | Description |
|--------|-------------|
| **Objective** | Train a 100–150M parameter, code-only decoder that performs competitively with much larger generalist models on HumanEval+, MBPP+, LiveCodeBench, DS-1000, and LongCodeBench. |
| **Target hardware** | Final training: **RunPod RTX 3090 (24 GB)**. Iteration: local (e.g. Apple M2) or Google Colab. Inference: PyTorch (CUDA/MPS/CPU) or MLX (Apple Silicon). |
| **Design principles** | Quality over quantity (three-stage data curriculum); sample-efficient training (stage-wise LRs, checkpoint EMA); optional test-time compute (S*, PoT); optional execution-semantics alignment (CodeRL+). |

---

## Methodology

### High-Level Pipeline

```
Data (3 stages, RegMix, CODA, CodeDenoise)
    → Pre-training (Stage LRs, NorMuon-AdamW, EMA, L-MTP curriculum)
    → Post-training (optional: curriculum SFT, trajectory distillation, CodeRL+)
    → Inference (PyTorch / MLX; optional S*, PoT)
    → Evaluation (EvalPlus pass@1/pass@10, repair rate, LongCodeBench)
```

### Five Pillars (from Research Plan)

1. **Representation** — BPE 16k–24k baseline; optional BLT (Byte Latent Transformer) to reduce embedding tax.
2. **Architecture** — Deep-thin decoder (e.g. 44 layers, d_model=384); QK-norm, value residuals, per-head gating; optional BitLinear (BitNet b1.58), Mamba-2 hybrid, LEAM++.
3. **Training** — NorMuon-AdamW–style parameter groups; WSD + three stage-wise learning rates; checkpoint EMA; L-MTP forward curriculum; gradient checkpointing (STEP).
4. **Data & post-training** — Three-stage data (broad → clean/educational → curriculum); CODA (adversarial code deltas); CodeDenoise; optional CodeRL+ (variable-level execution, semantics-match reward) and trajectory distillation from a teacher.
5. **Inference** — Standard autoregressive decode; optional test-time evolution (S*, AB-MCTS, DaJ, PoT) and MLX/GGUF export.

---

## Repository Structure

```
.
├── data/                     # Data pipeline
│   ├── prepare_data.py       # Three-stage stream, near-dedup, HF ingestion
│   ├── config_data.yaml      # Stage definitions, filters, RegMix
│   ├── regmix_proxy.py       # RegMix proxy training / mixture optimization
│   ├── tokenizer_train.py    # BPE 16k–24k training
│   ├── coda.py               # CODA adversarial augmentation
│   ├── code_denoise.py       # Syntax–semantics consistency filtering
│   ├── dataloader.py         # Streaming batches from JSONL
│   └── synthetic/            # Stage 3 synthetic (phi-1–style) data
├── model/
│   ├── config.py             # ModelConfig (100–150M, deep-thin, L-MTP)
│   ├── gpt.py                # Decoder-only transformer (QK-norm, value residual, optional BitLinear)
│   ├── blt.py                # Byte Latent Transformer (optional)
│   ├── bitnet.py             # BitLinear (BitNet b1.58)
│   ├── leam.py               # LEAM++ syntax-awareness (optional)
│   └── mamba_hybrid.py       # Mamba-2 hybrid (optional)
├── training/
│   ├── train.py              # Main training loop (stage LRs, EMA, device fallback)
│   ├── scheduler.py          # WSD + stage-wise LRs
│   ├── config_train.yaml     # Hyperparameters, L-MTP curriculum, EMA
│   └── step.py               # STEP memory optimizations
├── post_training/
│   ├── curriculum.py        # StepCoder-style curriculum (CCCS, FGO)
│   ├── sft_trajectories.py   # SFT on trajectories (tests-green only)
│   ├── distill.py           # Trajectory distillation from teacher
│   ├── coderl_plus.py       # CodeRL+ execution semantics reward
│   └── config_post.yaml
├── inference/
│   ├── run_torch.py         # PyTorch inference (load checkpoint, generate)
│   ├── run_mlx.py           # MLX inference (load, generate, RLM)
│   ├── test_time_evolution.py  # S*, AB-MCTS, DaJ, PoT
│   ├── export_gguf.py       # GGUF export for llama.cpp
│   └── run_gguf.py
├── evaluation/
│   ├── eval_loss.py         # Validation loss / perplexity
│   ├── eval_humaneval.py    # EvalPlus HumanEval+ / MBPP+ (pass@1, pass@10)
│   ├── eval_repair.py       # Repair success rate
│   ├── eval_livecode.py     # LiveCodeBench
│   └── eval_lcb.py          # LongCodeBench (128K/512K, folding)
├── verify_plan.py           # 27 checks (data → model → training → eval → RunPod)
├── run_smoke_test.sh        # Short training + inference sanity check
├── runpod_train.sh          # RunPod RTX 3090 entrypoint
├── runpod_run.py            # RunPod start command (Python)
├── colab_train.ipynb        # Colab notebook for iteration
├── requirements.txt
├── STATUS.md                # Pipeline status, no open TODOs
└── README.md
```

---

## Installation & Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.1+ (CUDA optional; MPS for Apple Silicon)
- See `requirements.txt` for full dependencies (e.g. `transformers`, `datasets`, `evalplus`, `mlx`, `mlx-lm` for inference).

### Setup

```bash
git clone https://github.com/Lugier/150m-.git
cd 150m-
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Verification (before any long run)

```bash
# 1. Plan-alignment checks (27 items: data, model, training, post-training, inference, evaluation, RunPod/Colab)
python verify_plan.py

# 2. Smoke test: short training (20 steps) + inference + checkpoint check
bash run_smoke_test.sh
```

Both should complete successfully. No open TODOs in the critical path; see `STATUS.md` for details.

---

## Data Pipeline

- **Stage 1 (broad)** — The Stack v2 / Stack-Edu–style sources, moderate filters (length, license, near-deduplication). Config: `data/config_data.yaml` → `stages.stage1`.
- **Stage 2 (clean & educational)** — Stricter Stack-Edu, tests, DS-1000–like data; CODA and CodeDenoise applied. Config: `stages.stage2`.
- **Stage 3 (curriculum)** — Synthetic phi-1–style (docstring → code → tests), execution filter (only passing tests). Config: `stages.stage3`.

Data are **not** shipped in the repo. To produce training JSONL:

```bash
# Example: stream Stage 1 from Hugging Face (e.g. bigcode/the-stack-smol), write to data/processed/stage1.jsonl
python data/prepare_data.py --config data/config_data.yaml --stage stage1 \
  --dataset bigcode/the-stack-smol --max_docs 5000 --output data/processed/stage1.jsonl
```

Tokenizer (BPE 16k–24k):

```bash
python data/tokenizer_train.py --input data/processed/stage1.jsonl --output data/tokenizer --vocab_size 16384
```

If no JSONL (or no tokenizer) is present in the configured `data_dir`, training falls back to dummy data so the pipeline still runs (e.g. for smoke tests).

---

## Model Architecture

- **Default config** — ~100–150M parameters: `d_model=384`, `n_layer=44`, `n_head=6`, `vocab_size=16384`, `max_seq_len=1024` (see `model/config.py`, `training/config_train.yaml`).
- **Components** — Decoder-only transformer with RMSNorm, QK-norm, value residuals, per-head gating; optional BitLinear (BitNet b1.58), BLT, Mamba-2 hybrid, LEAM++ (see `model/gpt.py`, `model/bitnet.py`, `model/blt.py`, `model/leam.py`).
- **L-MTP** — Multi-token prediction with a forward curriculum (warmup then increase `mtp_n`) is supported via config.

---

## Training

- **Optimizer** — AdamW with parameter groups (e.g. weight decay for 2D weights, no decay for biases/norms); NorMuon-style grouping is reflected in the code.
- **Schedule** — WSD (warmup–stable–decay) with **three stage-wise learning rates** tied to step boundaries (e.g. stage1 → stage2 → stage3).
- **EMA** — Exponential moving average over parameters; final model can be the EMA average (config: `ema.enabled`, `ema.window`, `ema.decay`).
- **Device** — Config defaults to `cuda`; automatic fallback to MPS or CPU if CUDA is unavailable.
- **Checkpoints** — Saved periodically (e.g. every 5000 steps) and at end as `final.pt` (and optional `.safetensors`).

**Local (short run):**

```bash
python training/train.py --config training/config_train.yaml --checkpoint_dir checkpoints --max_steps 500
```

**RunPod RTX 3090:** Use `runpod_train.sh` or `runpod_run.py`; they set `/workspace` paths and call the same `training/train.py`. See README section on RunPod below.

---

## Post-Training

- **Curriculum** — StepCoder-style stages (short → medium → complex) defined in `post_training/config_post.yaml`; `curriculum.py` provides `load_curriculum_config` and `assign_stage`.
- **SFT on trajectories** — Load trajectories (problem → attempt → fix → tests); filter to “all tests green” only (`sft_trajectories.py`).
- **Trajectory distillation** — Load teacher trajectories, filter by reward, distill into student (`distill.py`).
- **CodeRL+** — Variable-level execution semantics reward; `coderl_plus.py` implements `semantics_match_reward` and `coderl_plus_reward` (no placeholders in the critical path).

---

## Inference & Evaluation

- **Inference** — `inference/run_torch.py`: load a checkpoint (e.g. `final.pt`), provide a prompt, generate. Optional: `run_mlx.py` for Apple Silicon; `test_time_evolution.py` for S*, DaJ, PoT-style logic.
- **Evaluation** — EvalPlus for HumanEval+/MBPP+ (pass@1, pass@10); repair success rate; LongCodeBench (128K/512K) with folding. Scripts in `evaluation/`.

Example:

```bash
python inference/run_torch.py --checkpoint checkpoints --prompt "def fib(n):" --max_tokens 128
```

---

## Reproducibility

- **Configs** — All stages, training hyperparameters, and post-training settings are in YAML; no hardcoded magic numbers in the critical path.
- **Seeds** — Config includes `seed: 42` for training.
- **Verification** — `verify_plan.py` ensures all plan-referenced modules and entrypoints exist and are callable; `run_smoke_test.sh` runs a minimal training + inference loop.
- **RunPod** — Same code and configs; only paths switch to `/workspace`. No separate “demo” mode.

---

## RunPod & Colab

**RunPod (RTX 3090):**

1. Create a pod with RTX 3090 (24 GB) and a persistent volume mounted at `/workspace`.
2. Clone this repo (e.g. to `/workspace/150m-`). Optionally copy data and tokenizer to `/workspace/llm_plus_data/processed/` and `/workspace/llm_plus_data/tokenizer/`.
3. Set the pod start command to: `cd /workspace/150m- && bash runpod_train.sh` (or `python3 runpod_run.py`).
4. Checkpoints will be written to `/workspace/llm_plus_checkpoints/` (or as configured).

**Colab:** Use `colab_train.ipynb` for short iteration runs; mount Drive for checkpoints if desired.

---

## Citation & References

This implementation follows a detailed research plan that draws on:

- **IMU-1** — Three-stage pretraining, stage-wise LRs, checkpoint EMA ([sample-efficient scaling](https://arxiv.org/abs/2406.13112)).
- **SmolLM2 / Stack-Edu** — Code-focused data and multi-stage training.
- **RegMix** — Proxy-based mixture optimization for pretraining.
- **BitNet b1.58** — 1.58-bit quantization for efficient training ([e.g. JMLR 2025](https://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf)).
- **CodeRL+** — Variable-level execution trajectories and semantics-match reward ([arXiv:2510.18471](https://arxiv.org/abs/2510.18471)).
- **EvalPlus** — HumanEval+ / MBPP+ evaluation ([evalplus](https://github.com/evalplus/evalplus)).
- **Test-time evolution** — S*, PoT, and related methods for code generation with execution feedback.

If you use this code or the associated plan in your research, please cite this repository and the relevant papers above.

---

## License & Compliance

- **Code** — See repository license file.
- **Data** — When using The Stack v2 / Stack-Edu or similar sources, comply with their terms (SWH, Hugging Face datasets). No data are redistributed in this repo.
- **Checkpoints** — Saved in PyTorch and optionally safetensors format; use and distribution are your responsibility.

---

*This project implements a full pipeline for a 100–150M parameter code-only language model as specified in the accompanying research plan (Giant-Killer architecture). The pipeline is closed-loop: verification and smoke tests pass; no open TODOs remain in the critical path. For status details, see `STATUS.md`.*
