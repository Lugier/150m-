#!/bin/bash
set -e

echo "=========================================="
echo " Giant-Killer LLM Pipeline Smoke Test     "
echo "=========================================="

if [ ! -d "venv" ]; then
    echo "[!] Virtual environment 'venv' not found. Please create one with 'python3 -m venv venv' and install requirements."
    exit 1
fi

source venv/bin/activate
export PYTHONPATH=.

echo "[1/4] Generating Dummy Smoke Config..."
cat <<EOF > testing_smoke_config.yaml
model:
  config_path: model/config.py
  d_model: 64
  n_layer: 2
  n_head: 2
  vocab_size: 1024
  max_seq_len: 128
  use_bitnet: false
  mtp_n: 1

optimizer:
  name: normuon_adamw_hybrid
  weight_decay: 0.01

scheduler:
  type: wsd
  warmup_steps: 5
  stage_lrs:
    stage1: 3.0e-4
    stage2: 1.5e-4
    stage3: 5.0e-5

training:
  batch_size: 2
  seq_len: 32
  gradient_accumulation_steps: 1
  max_steps: 20
  eval_every: 10
  save_every: 20
  gradient_checkpointing: false
  mixed_precision: 'no'

ema:
  enabled: true
  window: 5
  decay: 0.99

stages:
  stage1_steps: 10
  stage2_steps: 5
  stage3_steps: 5

device: cpu
seed: 42
EOF

echo "[2/4] Running Training Loop (20 steps)..."
rm -rf smoke_checkpoints
python training/train.py --config testing_smoke_config.yaml --checkpoint_dir smoke_checkpoints --data_dir data/processed --device cpu

echo ""
echo "[3/4] Running Inference (run_torch.py)..."
python inference/run_torch.py --checkpoint smoke_checkpoints --prompt "def fib(n):" --max_tokens 10

echo ""
echo "[4/4] Validation..."
if [ -f "smoke_checkpoints/final.pt" ]; then
    echo "✅ SUCCESS: smoke_checkpoints/final.pt found!"
else
    echo "❌ ERROR: Training did not produce final.pt"
    exit 1
fi

echo "All tests passed. Pipeline is structurally sound."
