import ast
import json
import torch
from pathlib import Path
from math import ceil
from data.chat_format import parse_chat_to_message_spans

class SFTDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset for JSONL-based chat histories.
    """
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 1024, pad_token_id: int = 0):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        # Für Truncation-Fix: EOS-ID aus Tokenizer (Fallback 2 = typisch für BPE)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 2)
        
    def __iter__(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"SFT JSONL not found: {self.data_path}. Run python data/generate_instruction_data.py --output {self.data_path}.")

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    if not messages:
                        continue
                        
                    input_ids, labels = parse_chat_to_message_spans(self.tokenizer, messages)
                    
                    if len(input_ids) > self.max_seq_len:
                        input_ids = input_ids[:self.max_seq_len]
                        labels = labels[:self.max_seq_len]
                        # SOTA-Fix: Nach hartem Abschneiden fehlt sonst <|eos|>/<|im_end|>.
                        # Ohne EOS am Ende lernt das Modell, mitten im Code abzubrechen ("Babbler");
                        # mit EOS am letzten Position lernt es immer ein klares Stopp-Signal.
                        input_ids[-1] = self.eos_token_id
                        labels[-1] = self.eos_token_id
                    elif len(input_ids) < self.max_seq_len:
                        # Pad with PAD tokens, and -100 for labels
                        pad_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.pad_token_id] * pad_len)
                        labels.extend([-100] * pad_len)
                        
                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long)
                    }
                except json.JSONDecodeError:
                    pass

class SFTDataLoader:
    def __init__(self, data_dir: str, tokenizer_path: str,
                 vocab_path: str, vocab_size: int,
                 seq_len: int, batch_size: int, seed: int = 42,
                 instruction_data_path: str | None = None):
        from transformers import PreTrainedTokenizerFast
        self.batch_size = batch_size

        _tp = Path(tokenizer_path)
        if not _tp.exists() and not (_tp / "tokenizer.json").exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}. Train tokenizer (e.g. python data/tokenizer_train.py --output data/tokenizer).")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        jsonl_path = Path(instruction_data_path) if instruction_data_path else Path(data_dir) / "instruction_sft.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Instruction SFT data not found: {jsonl_path}. Run python data/generate_instruction_data.py --output {jsonl_path}.")
        self.dataset = SFTDataset(jsonl_path, self.tokenizer, max_seq_len=seq_len, pad_token_id=0)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    def iter_forever(self):
        while True:
            for batch in self.loader:
                yield batch

def get_sft_dataloader(data_dir: str, tokenizer_path: str, vocab_path: str, vocab_size: int, seq_len: int, batch_size: int, seed: int = 42, instruction_data_path: str | None = None):
    """Instantiate SFT Dataloader. If instruction_data_path is set, use it as JSONL path; else data_dir/instruction_sft.jsonl."""
    return SFTDataLoader(data_dir, tokenizer_path, vocab_path, vocab_size, seq_len, batch_size, seed, instruction_data_path)
