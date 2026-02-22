import argparse
import sys
import torch
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.gpt import CodeGPTLMHeadModel
from model.config import ModelConfig
from model.leam import LEAMGrammarConstrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory or .pt file")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer", help="Path to tokenizer")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None, help="Override: cuda, mps, cpu (Plan §8)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_file = ckpt_path / "final.pt"
        if not ckpt_file.exists():
            # grab the latest
            ckpts = list(ckpt_path.glob("ckpt_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_path}")
            ckpt_file = max(ckpts, key=lambda p: int(p.stem.split('_')[1]))
    else:
        ckpt_file = ckpt_path

    print(f"Loading checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    train_cfg = ckpt.get("config", {})
    if not train_cfg:
        with open("training/config_train.yaml") as f:
            train_cfg = yaml.safe_load(f)

    m = train_cfg.get("model", {})
    model_cfg = ModelConfig(
        d_model=m.get("d_model", 384),
        n_layer=m.get("n_layer", 44),
        n_head=m.get("n_head", 6),
        vocab_size=m.get("vocab_size", 16384),
        max_seq_len=m.get("max_seq_len", 1024),
        use_bitnet=m.get("use_bitnet", False),
        use_mamba_hybrid=m.get("use_mamba_hybrid", False),
        use_blt=m.get("use_blt", False),
        use_leam=m.get("use_leam", False),
        mtp_n=m.get("mtp_n", 1),
    )

    model = CodeGPTLMHeadModel(model_cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    from transformers import PreTrainedTokenizerFast
    _tp = Path(args.tokenizer)
    if not _tp.is_absolute():
        _tp = Path.cwd() / _tp
    _j = _tp if _tp.suffix == ".json" else _tp / "tokenizer.json"
    if not _j.exists():
        raise FileNotFoundError(f"Tokenizer not found: {_j}. Train tokenizer (e.g. python data/tokenizer_train.py --output data/tokenizer).")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(_j)) if _j.suffix == ".json" else PreTrainedTokenizerFast.from_pretrained(str(_tp))
    encode = lambda s: tokenizer.encode(s)
    decode = lambda t: tokenizer.decode(t)
    # LEAM++ (Plan §2): bei use_leam Logits vor Sampling grammatikalisch beschränken.
    leam_constrainer = LEAMGrammarConstrainer(tokenizer) if getattr(model_cfg, "use_leam", False) else None

    input_ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    print("Generating...")
    with torch.no_grad():
        for _ in range(args.max_tokens):
            if input_ids.size(1) > model_cfg.max_seq_len:
                input_ids = input_ids[:, -model_cfg.max_seq_len:]
                
            logits = model(input_ids)["logits"]
            next_token_logits = logits[0, -1, :].clone()
            if leam_constrainer is not None:
                # LEAM: constrain_logits(sequence_ids, logits) vor Sampling.
                next_token_logits = leam_constrainer.constrain_logits(
                    input_ids[0].tolist(), next_token_logits.unsqueeze(0)
                )[0]

            if args.temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = decode(input_ids[0].tolist())
    print("\n--- Output ---")
    print(generated_text)
    print("--------------")

if __name__ == "__main__":
    main()
