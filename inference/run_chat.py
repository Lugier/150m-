"""
Chat-Inferenz für das Instruction-Tuned Modell (SFT/RL). ChatML-Format.
Bei config.use_leam wird LEAMGrammarConstrainer vor dem Sampling angewendet (Plan §2).
"""

import argparse
import sys
import torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.gpt import CodeGPTLMHeadModel
from model.config import ModelConfig
from data.chat_format import format_chat_history, IM_START, IM_END, ASSISTANT_PROMPT
from model.leam import LEAMGrammarConstrainer

def load_generator(checkpoint_path: str, device: torch.device):
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        # Fallbacks for sft/rl or standard checkpoints
        for name in ["final_rl.pt", "final_sft.pt", "final.pt"]:
            if (ckpt_path / name).exists():
                ckpt_path = ckpt_path / name
                break
                
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_cfg = ckpt.get("config", {})

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
    
    return model, model_cfg

def chat_loop(model, model_cfg, tokenizer, device, temp=0.2, max_tokens=256):
    """Run interactive chat. tokenizer must have .encode(s, add_special_tokens=False) and .decode(ids)."""
    messages = [
        {"role": "system", "content": "You are Giant-Killer, an expert code LLM."}
    ]
    # LEAM++ (Plan §2): Grammar-Guard maskiert Logits, die zu SyntaxError führen.
    leam_constrainer = LEAMGrammarConstrainer(tokenizer) if getattr(model_cfg, "use_leam", False) else None

    def encode(s: str):
        out = tokenizer.encode(s, add_special_tokens=False)
        return out if isinstance(out, list) else list(out)

    def decode(ids):
        return tokenizer.decode(ids)

    print("="*50)
    print("Giant-Killer 150M Instruction-Tuned Model loaded.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)

    while True:
        try:
            user_input = input("\nUser: ")
        except EOFError:
            break
            
        if user_input.lower() in ("quit", "exit"):
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # We need the history PLUS the start of the assistant's turn
        history_str = format_chat_history(messages)
        prompt = history_str + f"{IM_START}{ASSISTANT_PROMPT}\n"
        
        input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated_ids = input_ids.clone()
        
        print(f"\nGiant-Killer: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if generated_ids.size(1) > model_cfg.max_seq_len:
                    generated_ids = generated_ids[:, -model_cfg.max_seq_len:]
                    
                logits = model(generated_ids)["logits"]
                next_token_logits = logits[0, -1, :].clone()
                if leam_constrainer is not None:
                    # Syntax-sichere Logits vor Temperatur/Sampling.
                    next_token_logits = leam_constrainer.constrain_logits(
                        generated_ids[0].tolist(), next_token_logits.unsqueeze(0)
                    )[0]

                if temp == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                new_word = decode([next_token.item()])
                print(new_word, end="", flush=True)
                
                # Basic stop condition
                if IM_END in new_word or next_token.item() == 2: # EOS
                    break
                    
        print()
        
        # Extract just the newly generated content and append to history
        full_output = decode(generated_ids[0].tolist())
        assistant_reply = full_output[len(prompt):].replace(IM_END, "").strip()
        messages.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT or RL checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    model, config = load_generator(args.checkpoint, device)

    tokenizer_path = _ROOT / args.tokenizer if not (args.tokenizer.startswith("/") or (len(args.tokenizer) > 1 and args.tokenizer[1] == ":")) else Path(args.tokenizer)
    _j = tokenizer_path if tokenizer_path.suffix == ".json" else tokenizer_path / "tokenizer.json"
    if not _j.exists():
        raise FileNotFoundError(f"Tokenizer not found: {_j}. Train tokenizer (e.g. python data/tokenizer_train.py --output data/tokenizer) and pass --tokenizer.")
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(_j))
    chat_loop(model, config, tokenizer, device, temp=args.temperature, max_tokens=args.max_tokens)
