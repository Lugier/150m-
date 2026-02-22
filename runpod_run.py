#!/usr/bin/env python3
"""
RunPod Start Command entrypoint.
Use this as the Pod Start Command so training starts automatically with GPU + /workspace.
Example in RunPod template: python3 runpod_run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    # RunPod: /workspace is persistent
    workspace = os.environ.get("RUNPOD_WORKSPACE", "/workspace")
    if os.environ.get("RUNPOD_POD_ID"):
        checkpoint_dir = os.path.join(workspace, "llm_plus_checkpoints")
        data_dir = os.path.join(workspace, "llm_plus_data", "processed")
        tokenizer_path = os.path.join(workspace, "llm_plus_data", "tokenizer")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = str(ROOT / "checkpoints")
        data_dir = str(ROOT / "data" / "processed")
        tokenizer_path = str(ROOT / "data" / "tokenizer")

    cmd = [
        sys.executable,
        str(ROOT / "training" / "train.py"),
        "--config", str(ROOT / "training" / "config_train.yaml"),
        "--data_dir", data_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--tokenizer_path", tokenizer_path,
        "--vocab_path", str(ROOT / "model" / "vocab"),
    ]
    # Optional: limit steps for testing (e.g. RUNPOD_MAX_STEPS=100)
    max_steps = os.environ.get("RUNPOD_MAX_STEPS")
    if max_steps:
        cmd.extend(["--max_steps", max_steps])
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
