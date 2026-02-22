# Giant-Killer: A 100–150M Parameter Code-Only Language Model

<div align="center">

| Python | PyTorch | License |
|--------|---------|---------|
| 3.10+  | 2.1+    | See repo |

**Code-only decoder. 100–150M parameters. Single-GPU (RTX 3090, 24 GB). Instruction SFT + execution-based RL. HumanEval+ / MBPP+ / LiveCodeBench.**

</div>

---

## Abstract

We present **Giant-Killer**, a code-only language model in the 100–150M parameter regime, designed for **single-GPU training** (e.g. one RTX 3090 with 24 GB VRAM) and competitive performance on code-generation benchmarks. The pipeline combines: (1) **three-stage pretraining** on code corpora with stage-wise learning rates, checkpoint EMA, and optional **CODA** (code-delta augmentation) and **CodeDenoise** (syntax–semantics filtering); (2) **instruction tuning** with ChatML and assistant-only loss; (3) **reinforcement learning** with GRPO and execution-based rewards, optionally blended with **CodeRL+** semantics-match reward. The architecture supports **extensible research components**: Byte Latent Transformer (BLT), Mamba-2-Hybrid, BitNet b1.58, L-MTP (multi-token prediction), LEAM++ (grammar-constrained decoding), and **test-time evolution** (S*, AB-MCTS, DaJ, PoT). All settings are config-driven (YAML); reproducibility is enforced via fixed seeds, plan-alignment checks, and documented RTX 3090 optimisations.

---

## 1. Research Context and Objectives

### 1.1 Setting

- **Parameter scale:** 100–150M (default: 44 layers, `d_model`=384; cap ≤250M for instruction plan).
- **Hardware:** Single GPU, 24 GB VRAM (RTX 3090); no multi-node or custom kernels required.
- **Domain:** Code generation only (no general-purpose text); training and evaluation target functional correctness and repair under feedback.

### 1.2 Objectives

- **Primary:** Maximise **pass@1** and **pass@10** (EvalPlus) on HumanEval+ and MBPP+, and repair success rate (fix-after-feedback), while remaining an order of magnitude smaller than typical 1B+ baselines.
- **Secondary:** Long-context code (LongCodeBench 128K/512K) and LiveCodeBench where applicable; optional test-time evolution (S*, PoT) for improved robustness at inference.

### 1.3 Metrics and Reproducibility

- **Metrics:** pass@1 / pass@10 (EvalPlus), repair success rate, LongCodeBench accuracy; optional LiveCodeBench.
- **Reproducibility:** All hyperparameters and data-stage definitions in YAML; fixed seed in config; `verify_plan.py` and `run_smoke_test.sh` lock the pipeline to a single, documented plan. No dummy fallbacks in the critical path: tokenizer, data paths, and RL data are required.

---

## 2. Methodology

### 2.1 Architecture

| Component | Default | Extensions |
|-----------|---------|------------|
| Layers | 44 | Mamba-2-Hybrid (43% Mamba, 7% Attention, 50% MLP) when `use_mamba_hybrid` |
| d_model | 384 | — |
| Heads | 6 | — |
| Vocab | 16k BPE | BLT: byte-level, 256 |
| Max sequence length | 1024 | — |
| Extras | QK-norm, value residual, per-head gating | BitLinear (BitNet b1.58), L-MTP (mtp_n), LEAM++ (inference) |

Defined in `model/config.py` and `training/config_train.yaml` / `config_sft.yaml`. Optional components are toggled via YAML (`use_bitnet`, `use_mamba_hybrid`, `use_blt`, `use_leam`, `mtp_n`).

### 2.2 Data Pipeline

- **Stage 1 (broad):** The Stack v2 / Stack-Edu, moderate filters (length, license, near-dedup).  
- **Stage 2 (clean + adversarial):** Stricter filters; **CodeDenoise** (syntax–semantics consistency); **CODA** (code-difference-guided adversarial augmentation, e.g. `<` → `<=`, `+` → `-`) at configurable rate.  
- **Stage 3 (curriculum):** Docstring→Code→Tests, execution filter (phi-1-style).  

Configuration: `data/config_data.yaml` (stages, `data_pipeline`, `adversarial.coda_mutation_rate`, `run_code_denoise`). RegMix proxy available for mixture tuning.

### 2.3 Training Pipeline

1. **Pre-training** (`training/train.py`): Stage-wise learning rates (WSD), NorMuon-AdamW hybrid, checkpoint EMA, gradient checkpointing, bf16. L-MTP forward curriculum optional. **RTX 3090:** `batch_size=8`, `seq_len=512`, `gradient_accumulation_steps=4` (effective batch 32); see `training/config_train.yaml`.
2. **Instruction SFT** (`training/sft_train.py`): ChatML format; assistant-only labels; Evol-Instruct/teacher data → `instruction_sft.jsonl`. Loads pre-trained checkpoint; `batch_size=6`, `seq_len=1024`, `gradient_accumulation_steps=6` (effective 36) on 24 GB.
3. **RL** (`training/rl_train.py`): GRPO with execution-based reward (syntax / runtime / test pass); optional **CodeRL+** blend when RL JSONL contains `reference_solution` or `target_code` (semantics-match reward). SFT checkpoint required; `--rl-data` required.

### 2.4 Post-Training and Inference

- **Execution reward:** `post_training/execution_reward.py` — syntax (−1.0), runtime/timeout (−0.5), tests (+1.0 / −0.5); sandbox via `evaluation/eval_repair.run_tests_in_sandbox`.
- **CodeRL+:** `post_training/coderl_plus.py` — variable-level execution trajectories and semantics-match reward; wired into `rl_train.py` when reference code is provided.
- **Inference:** `inference/run_chat.py` (ChatML, interactive), `inference/run_torch.py` (prompt → completion). **LEAM++** grammar constraint applied when `use_leam` is set.
- **Test-time evolution:** `inference/test_time_evolution.py` — S* (parallel candidates + differentiating tests + sandbox selection), AB-MCTS, DaJ (judge), PoT (transient LoRA/GRPO stub). Sandbox-backed; can be composed with `run_chat` or evaluation scripts.

---

## 3. Experimental Setup (RTX 3090, 24 GB)

| Phase | batch_size | seq_len | gradient_accumulation | Note |
|-------|------------|---------|------------------------|------|
| Pre-train | 8 | 512 | 4 | OOM → 6 or seq_len 384 |
| SFT | 6 | 1024 | 6 | OOM → batch 4 |
| RL | — | — | num_candidates=4 | Policy + reference on GPU |

**CUDA optimisations** (automatic when `device=cuda`): `torch.backends.cudnn.benchmark = True`; pre-train DataLoader with `pin_memory=True` and `non_blocking=True` for CPU→GPU transfer. Gradient checkpointing is required for 150M on 24 GB. See `DESIGN.md` for details.

---

## 4. Repository Layout

```
├── data/                 prepare_data, config_data.yaml, coda, code_denoise, dataloader, tokenizer_train, sft_dataloader, instruction_data, chat_format
├── model/                config, gpt, blt, leam, mamba_hybrid; BitLinear in gpt
├── training/             train.py, sft_train.py, rl_train.py, scheduler, device_utils, config_train.yaml, config_sft.yaml
├── post_training/        execution_reward, coderl_plus, distill
├── inference/            run_chat.py, run_torch.py, run_mlx.py, test_time_evolution.py, export_gguf
├── evaluation/           eval_repair, eval_humaneval, eval_livecode, eval_loss
├── scripts/              verify_data.py
├── verify_plan.py       Plan-alignment checks
├── run_smoke_test.sh    Short train + inference + checkpoint check
├── DESIGN.md            Efficiency and RTX 3090 notes
└── requirements.txt
```

---

## 5. Quick Start

### 5.1 Environment

```bash
git clone https://github.com/Lugier/150m-.git
cd 150m-
python -m venv venv && source venv/bin/activate   # or Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Verification

```bash
python verify_plan.py      # Plan-alignment and module checks
bash run_smoke_test.sh     # Short train + inference + final.pt check
```

### 5.3 Data (required for full pipeline)

- **Pre-train:** JSONL under `data/processed/` (e.g. `stage1.jsonl`); tokenizer under `data/tokenizer` (e.g. `python data/tokenizer_train.py --input data/processed/stage1.jsonl --output data/tokenizer --vocab_size 16384`).
- **SFT:** `data/processed/instruction_sft.jsonl` (e.g. from `data/generate_instruction_data.py`).
- **RL:** Same or dedicated JSONL with `prompt`, `tests`, and optionally `reference_solution` / `target_code` for CodeRL+.

Check: `python scripts/verify_data.py`.

### 5.4 Training

```bash
# Pre-train (short run)
python training/train.py --config training/config_train.yaml --checkpoint_dir checkpoints --max_steps 500

# SFT (after pre-train checkpoint)
python training/sft_train.py --config training/config_sft.yaml --checkpoint checkpoints/final.pt --instruction_data data/processed/instruction_sft.jsonl --output_dir sft_checkpoints

# RL (after SFT)
python training/rl_train.py --sft-checkpoint sft_checkpoints/final_sft.pt --rl-data data/processed/instruction_sft.jsonl --output-dir rl_checkpoints
```

### 5.5 Inference

```bash
python inference/run_torch.py --checkpoint rl_checkpoints --prompt "def fib(n):" --max_tokens 128
python inference/run_chat.py --checkpoint rl_checkpoints --tokenizer data/tokenizer
```

Apple Silicon: `inference/run_mlx.py` (requires MLX export). GGUF: `inference/export_gguf.py`.

---

## 6. Evaluation

| Benchmark | Script |
|-----------|--------|
| HumanEval+ / MBPP+ | `evaluation/eval_humaneval.py` (EvalPlus pass@1, pass@10) |
| Repair success rate | `evaluation/eval_repair.py` |
| LongCodeBench | `evaluation/eval_lcb.py` |
| LiveCodeBench | `evaluation/eval_livecode.py` |

---

## 7. References and Related Work

| Work | Role in this codebase |
|------|------------------------|
| **IMU-1** | Three-stage pretraining, stage-wise LRs, checkpoint EMA |
| **SmolLM2 / Stack-Edu** | Code-centric data and stage design |
| **RegMix** | Proxy-based mixture optimisation for pretraining data |
| **CODA / CodeDenoise** | Adversarial code deltas and syntax–semantics filtering (Stage 2) |
| **BitNet b1.58** | Optional 1.58-bit linear layers (BitLinear) |
| **CodeRL+** | Variable-level execution semantics and semantics-match reward in RL |
| **LEAM++** | Grammar-constrained decoding at inference |
| **S* / PoT** | Test-time evolution (parallel candidates, differentiating tests, transient updates) |
| **EvalPlus** | HumanEval+ / MBPP+ pass@k evaluation |

When using this repository in publications, please cite this repo and the corresponding papers for the methods you build upon.

---

## 8. License and Data

- **Code:** See `LICENSE` in the repository.
- **Data:** Not distributed. Use of The Stack v2 / Stack-Edu must comply with their terms (Hugging Face, SWH). Checkpoints: user responsibility for use and distribution.

---

<div align="center">

**Giant-Killer** — 100–150M code-only decoder, single-GPU training, instruction SFT + execution RL, plan-aligned and reproducible.

</div>
