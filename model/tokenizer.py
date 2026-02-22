import os
from tokenizers import ByteLevelBPETokenizer

class GiantKillerTokenizer:
    """
    Wrapper handling either the primary BLT byte-patch strategy, or 
    the fallback 16k-24k BPE baseline.
    """
    def __init__(self, use_blt: bool = True, vocab_path: str = None):
        self.use_blt = use_blt
        
        if not use_blt and vocab_path and os.path.exists(vocab_path):
            self.tokenizer = ByteLevelBPETokenizer(
                f"{vocab_path}/vocab.json", 
                f"{vocab_path}/merges.txt"
            )
        else:
            self.tokenizer = None
            
    def encode(self, text: str):
        if self.use_blt:
            # BLT encodes raw bytes
            return list(text.encode("utf-8"))
        elif self.tokenizer:
            return self.tokenizer.encode(text).ids
        else:
            raise ValueError("Tokenizer not initialized for BPE fallback.")
            
    def decode(self, ids: list[int]) -> str:
        if self.use_blt:
            return bytes(ids).decode("utf-8", errors="replace")
        elif self.tokenizer:
            return self.tokenizer.decode(ids)
        else:
            return ""

if __name__ == "__main__":
    tk = GiantKillerTokenizer(use_blt=True)
    out = tk.encode("def test(): pass")
    print("BLT Encoded Bytes:", out)
    print("BLT Decoded String:", tk.decode(out))
