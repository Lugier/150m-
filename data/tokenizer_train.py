"""
Train BPE tokenizer (16k-24k) for baseline. Supports JSONL input (content/code) and model/vocab output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def iter_jsonl_text(path: str | Path, text_key: str = "content", max_lines: int = 0) -> Iterator[str]:
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get(text_key) or doc.get("code", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def train_bpe_from_jsonl(
    jsonl_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
    text_key: str = "content",
    max_lines: int = 0,
    also_save_model_vocab: bool = True,
) -> None:
    """Train BPE from JSONL (content/code); save tokenizer.json to output_dir; optionally model/vocab."""
    if special_tokens is None:
        special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_jsonl_text(jsonl_path, text_key=text_key, max_lines=max_lines),
        trainer=trainer,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    print(f"Saved tokenizer to {output_dir / 'tokenizer.json'}")

    if also_save_model_vocab:
        model_vocab = Path("model/vocab")
        model_vocab.mkdir(parents=True, exist_ok=True)
        try:
            tokenizer.model.save(str(model_vocab), name="vocab")
            print(f"Saved model/vocab for dataloader: {model_vocab}")
        except TypeError:
            try:
                tokenizer.model.save(str(model_vocab))
            except Exception:
                pass


def train_baseline_bpe(dataset_path: str, vocab_size: int = 24000) -> None:
    """Legacy: train from directory with batch_*.jsonl; saves model/vocab."""
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    files = [f"{dataset_path}/batch_{i}.jsonl" for i in range(1)]
    if not os.path.exists(dataset_path):
        print(f"Skipping: Path {dataset_path} does not exist.")
        return
    tokenizer.train(files, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    ])
    os.makedirs("model/vocab", exist_ok=True)
    tokenizer.save_model("model/vocab")
    print("Baseline tokenizer saved to model/vocab/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL file (content/code) or dir for legacy")
    p.add_argument("--output", default="data/tokenizer", help="Output dir for tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=16384)
    p.add_argument("--max_lines", type=int, default=0)
    p.add_argument("--text_key", default="content")
    p.add_argument("--no_model_vocab", action="store_true", help="Do not write model/vocab")
    args = p.parse_args()
    if Path(args.input).is_file():
        train_bpe_from_jsonl(
            args.input,
            args.output,
            vocab_size=args.vocab_size,
            text_key=args.text_key,
            max_lines=args.max_lines,
            also_save_model_vocab=not args.no_model_vocab,
        )
    else:
        train_baseline_bpe(args.input, args.vocab_size)
