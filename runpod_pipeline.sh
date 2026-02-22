#!/usr/bin/env bash
# RunPod RTX 3090 (24 GB): Full End-to-End Pipeline
# Automates Base Pre-Training -> SFT Data Gen -> Instruction Tuning -> RL
# Volume is at /workspace; use it for checkpoints and data so they persist.
# Set in RunPod template "Start Command" to: bash runpod_pipeline.sh

set -e

# RunPod Path Resolution
export RUNPOD_WORKSPACE="${RUNPOD_WORKSPACE:-/workspace}"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "RunPod detected: POD_ID=$RUNPOD_POD_ID GPU_COUNT=${RUNPOD_GPU_COUNT:-1}"
  export WORKSPACE_DIR="$RUNPOD_WORKSPACE/llm_plus"
else
  export WORKSPACE_DIR="./"
fi

export CHECKPOINT_DIR="$WORKSPACE_DIR/checkpoints"
export SFT_CHECKPOINT_DIR="$WORKSPACE_DIR/sft_checkpoints"
export RL_CHECKPOINT_DIR="$WORKSPACE_DIR/rl_checkpoints"
export DATA_DIR="$WORKSPACE_DIR/data/processed"
export TOKENIZER_PATH="$WORKSPACE_DIR/data/tokenizer"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_ROOT="${PROJECT_ROOT:-.}"
cd "$PROJECT_ROOT"

# Setup environments
mkdir -p "$CHECKPOINT_DIR" "$SFT_CHECKPOINT_DIR" "$RL_CHECKPOINT_DIR" "$DATA_DIR"
if ! python3 -c "import torch" 2>/dev/null; then
  pip install -q torch transformers tokenizers datasets safetensors pyyaml tqdm accelerate requests
fi

echo "=========================================================="
echo " Starting Full Giant-Killer Pipeline (Base -> SFT -> RL) "
echo "=========================================================="

echo "[1/4] Starting Base Pre-Training..."
python3 training/train.py \
  --config training/config_train.yaml \
  --data_dir "$DATA_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --vocab_path "$PROJECT_ROOT/model/vocab"

echo "[2/4] Generating Instruction SFT Dataset..."
# Warning: This assumes a teacher model API is reachable. Adjust endpoint via ENV vars if needed.
python3 data/generate_instruction_data.py --evolve --output_file "$DATA_DIR/instruction_sft.jsonl" || true

echo "[3/4] Starting Instruction Fine-Tuning (SFT)..."
python3 training/sft_train.py \
  --config training/config_sft.yaml \
  --checkpoint "$CHECKPOINT_DIR/final.pt" \
  --data_dir "$DATA_DIR" \
  --output_dir "$SFT_CHECKPOINT_DIR" \
  --tokenizer_path "$TOKENIZER_PATH"

echo "[4/4] Starting Execution-Based Reinforcement Learning (GRPO)..."
python3 training/rl_train.py

echo "=========================================================="
echo " Pipeline Complete. Final checkpoint at: $RL_CHECKPOINT_DIR/final_rl.pt "
echo " Use 'python3 inference/run_chat.py --checkpoint $RL_CHECKPOINT_DIR/final_rl.pt' to interact."
echo "=========================================================="
