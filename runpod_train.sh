#!/usr/bin/env bash
# RunPod RTX 3090 (24 GB): final 100-150M training.
# Volume is at /workspace; use it for checkpoints and data so they persist.
# Set in RunPod template "Start Command" to: bash runpod_train.sh

set -e

# RunPod: /workspace is persistent; RUNPOD_POD_ID is set
export RUNPOD_WORKSPACE="${RUNPOD_WORKSPACE:-/workspace}"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "RunPod detected: POD_ID=$RUNPOD_POD_ID GPU_COUNT=${RUNPOD_GPU_COUNT:-1}"
  export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUNPOD_WORKSPACE/llm_plus_checkpoints}"
  export DATA_DIR="${DATA_DIR:-$RUNPOD_WORKSPACE/llm_plus_data/processed}"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-$RUNPOD_WORKSPACE/llm_plus_data/tokenizer}"
else
  export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
  export DATA_DIR="${DATA_DIR:-./data/processed}"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizer}"
fi

# Single GPU (RTX 3090)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_ROOT="${PROJECT_ROOT:-.}"
cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINT_DIR" "$DATA_DIR"

echo "Checkpoints: $CHECKPOINT_DIR"
echo "Data:        $DATA_DIR"
echo "Tokenizer:   $TOKENIZER_PATH"

# Install deps if not present
if ! python3 -c "import torch" 2>/dev/null; then
  pip install -q torch transformers tokenizers datasets safetensors pyyaml tqdm accelerate
fi

# --- PHASE 1: PRE-TRAINING ---
echo "==================================="
echo "PHASE 1: PRE-TRAINING (100-300M)..."
echo "==================================="
python3 training/train.py \
  --config training/config_train.yaml \
  --data_dir "$DATA_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --vocab_path "$PROJECT_ROOT/model/vocab" \
  "$@"


# --- PHASE 2: INSTRUCTION DATA GENERATION & SFT ---
echo "==================================="
echo "PHASE 2: SFT INSTRUCTION TUNING..."
echo "==================================="
# Generate 5,000 synthetic instruction pairs using Evol-Instruct (simulated API fallback or real endpoint)
python3 data/generate_instruction_data.py \
  --num_samples 5000 \
  --evolve \
  --output_file "$DATA_DIR/instruction_sft.jsonl" || echo "SFT Data Gen failed, checking if fallback data exists..."

# Run SFT Training from the Final Pre-Trained Checkpoint
python3 training/sft_train.py \
  --config training/config_sft.yaml \
  --checkpoint "$CHECKPOINT_DIR/final.pt" \
  --instruction_data "$DATA_DIR/instruction_sft.jsonl" \
  --output_dir "$CHECKPOINT_DIR/sft" \
  --max_steps 5000 \
  "$@"


# --- PHASE 3: RL GRPO (SELF-REPAIR) ---
echo "==================================="
echo "PHASE 3: RL GRPO (SELF-REPAIR)..."
echo "==================================="
# Fine-tune the SFT model using GRPO and Sandbox code execution rewards
python3 training/rl_train.py \
  --config training/config_rl.yaml \
  --checkpoint "$CHECKPOINT_DIR/sft/final_sft.pt" \
  --output_dir "$CHECKPOINT_DIR/rl" \
  --max_steps 2000 \
  "$@"

echo "==================================="
echo "ALL TRAINING PHASES COMPLETE."
echo "Final Model: $CHECKPOINT_DIR/rl/final_rl.pt"
echo "==================================="
