"""
Streaming dataloader for pretraining: reads JSONL (content/code), tokenizes, yields batches.
Supports stage1/stage2/stage3 files or single train.jsonl; tokenizer from model/vocab or data/tokenizer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator, Optional

import torch


def load_tokenizer_for_training(
    tokenizer_path: Optional[str | Path] = None,
    vocab_path: Optional[str | Path] = None,
    vocab_size: int = 16384,
) -> Any:
    """Load tokenizer: PreTrainedTokenizerFast from tokenizer.json, or ByteLevelBPE from model/vocab."""
    # 1) data/tokenizer/tokenizer.json (from tokenizers library save)
    tp = Path(tokenizer_path) if tokenizer_path else None
    if tp is not None:
        j = tp if tp.suffix == ".json" else tp / "tokenizer.json"
        if Path(j).exists():
            try:
                from transformers import PreTrainedTokenizerFast
                return PreTrainedTokenizerFast(tokenizer_file=str(j))
            except Exception:
                pass
    # 2) model/vocab (vocab.json + merges.txt)
    vp = Path(vocab_path) if vocab_path else Path("model/vocab")
    if (vp / "vocab.json").exists() and (vp / "merges.txt").exists():
        try:
            from tokenizers import ByteLevelBPETokenizer
            tok = ByteLevelBPETokenizer(str(vp / "vocab.json"), str(vp / "merges.txt"))
            # Wrap so we have encode/decode returning list and str
            class Wrapper:
                def encode(self, text: str) -> list[int]:
                    return tok.encode(text).ids
                def decode(self, ids: list[int]) -> str:
                    return tok.decode(ids)
                @property
                def vocab_size(self) -> int:
                    return len(tok.get_vocab()) if hasattr(tok, "get_vocab") else 65536
            return Wrapper()
        except Exception:
            pass
    raise FileNotFoundError(
        "No tokenizer found. Train with: python data/tokenizer_train.py --input data/processed/stage1.jsonl --output data/tokenizer"
    )


def iter_jsonl(path: str | Path, text_key: str = "content", max_docs: int = 0) -> Iterator[str]:
    path = Path(path)
    if not path.exists():
        return
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get(text_key) or doc.get("code", "")
                if text:
                    yield text
                    count += 1
                    if max_docs and count >= max_docs:
                        return
            except json.JSONDecodeError:
                continue


def chunk_sequence(ids: list[int], seq_len: int, overlap: int = 0) -> Iterator[list[int]]:
    """Yield consecutive chunks of seq_len; overlap tokens between chunks if overlap > 0."""
    start = 0
    while start < len(ids):
        chunk = ids[start : start + seq_len]
        if len(chunk) < seq_len and len(chunk) < 16:
            break
        if len(chunk) >= 16:
            yield chunk
        start += seq_len - overlap


class CodeDataLoader:
    """Streaming batches from JSONL; tokenizes and chunks to seq_len."""

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: Any,
        seq_len: int = 512,
        batch_size: int = 4,
        text_key: str = "content",
        shuffle: bool = True,
        seed: Optional[int] = None,
        stage_files: Optional[list[str]] = None,
        pin_memory: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.text_key = text_key
        self.shuffle = shuffle
        self.seed = seed
        self.stage_files = stage_files or ["stage1.jsonl", "stage2.jsonl", "stage3.jsonl", "train.jsonl"]
        self.pin_memory = pin_memory  # RTX 3090: async CPU→GPU mit non_blocking=True

    def _collect_chunks(self) -> list[list[int]]:
        chunks: list[list[int]] = []
        for name in self.stage_files:
            path = self.data_dir / name
            if not path.exists():
                continue
            for text in iter_jsonl(path, self.text_key):
                ids = self.tokenizer.encode(text) if hasattr(self.tokenizer, "encode") else self.tokenizer(text, return_tensors=None).get("input_ids", [])
                if isinstance(ids, list):
                    pass
                else:
                    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                for ch in chunk_sequence(ids, self.seq_len):
                    chunks.append(ch)
        return chunks

    def _iter_batches(self) -> Iterator[dict[str, torch.Tensor]]:
        chunks = self._collect_chunks()
        if not chunks:
            return
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(chunks)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", 0) or 0
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            max_len = max(len(c) for c in batch_chunks)
            max_len = min(max_len, self.seq_len)
            padded = []
            for c in batch_chunks:
                c = c[:max_len]
                c = c + [pad_id] * (max_len - len(c))
                padded.append(c)
            input_ids = torch.tensor(padded, dtype=torch.long)
            if self.pin_memory:
                input_ids = input_ids.pin_memory()
            yield {"input_ids": input_ids}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self._iter_batches()

    def iter_forever(self) -> Iterator[dict[str, torch.Tensor]]:
        """Infinite iteration over batches (re-iterate each epoch)."""
        while True:
            got_any = False
            for batch in self._iter_batches():
                got_any = True
                yield batch
            if not got_any:
                break


def get_training_dataloader(
    data_dir: str | Path,
    tokenizer_path: Optional[str | Path] = None,
    vocab_path: Optional[str | Path] = None,
    vocab_size: int = 16384,
    seq_len: int = 512,
    batch_size: int = 4,
    seed: Optional[int] = None,
    pin_memory: bool = False,
) -> Optional[CodeDataLoader]:
    """Build dataloader if data_dir has JSONL and tokenizer exists. pin_memory=True für CUDA (RTX 3090)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None
    any_jsonl = list(data_dir.glob("*.jsonl"))
    if not any_jsonl:
        return None
    try:
        tok = load_tokenizer_for_training(tokenizer_path, vocab_path, vocab_size)
    except FileNotFoundError:
        return None
    return CodeDataLoader(
        data_dir=data_dir,
        tokenizer=tok,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=pin_memory,
    )
