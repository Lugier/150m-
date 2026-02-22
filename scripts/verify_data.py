#!/usr/bin/env python3
"""
Verify data paths and show first batch shapes (no training).
Run from repo root: python scripts/verify_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    print("=== Data verification ===\n")

    # 1. Instruction SFT JSONL
    sft_path = ROOT / "data" / "processed" / "instruction_sft.jsonl"
    if sft_path.exists():
        import json
        lines = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line)
        if lines:
            ex = json.loads(lines[0])
            msgs = ex.get("messages", [])
            print(f"[SFT] {sft_path.name}: {len(lines)} examples, first has {len(msgs)} messages")
        else:
            print(f"[SFT] {sft_path.name}: empty")
    else:
        print(f"[SFT] {sft_path} not found. Run: python data/generate_instruction_data.py --output {sft_path}")

    # 2. Tokenizer
    tok_path = ROOT / "data" / "tokenizer"
    tok_json = tok_path / "tokenizer.json" if tok_path.is_dir() else tok_path
    if tok_path.exists() or tok_json.exists():
        print(f"[Tokenizer] {tok_path} exists")
    else:
        print(f"[Tokenizer] {tok_path} not found. Run: python data/tokenizer_train.py --output data/tokenizer")

    # 3. Pre-train JSONL (any)
    processed = ROOT / "data" / "processed"
    if processed.exists():
        jsonl_files = list(processed.glob("*.jsonl"))
        if jsonl_files:
            print(f"[Pre-train] data/processed: {[f.name for f in jsonl_files]}")
        else:
            print("[Pre-train] data/processed: no .jsonl files")
    else:
        print("[Pre-train] data/processed directory not found")

    # 4. Configs
    for name in ["training/config_train.yaml", "training/config_sft.yaml", "data/chat_template.yaml"]:
        p = ROOT / name
        print(f"[Config] {name}: {'OK' if p.exists() else 'MISSING'}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
