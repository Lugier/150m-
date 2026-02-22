"""
GRPO RL Training (Instruction + Plan §2.7): Execution-Reward + optional CodeRL+ Semantics.
Lädt SFT-Checkpoint, bewertet Kandidaten per Sandbox; bei reference_solution/target_code
im RL-JSONL wird Reward aus Execution und Semantics-Match gemischt (CodeRL+).
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path
from math import log
from typing import List, Dict, Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.gpt import CodeGPTLMHeadModel
from model.config import ModelConfig
from post_training.execution_reward import compute_execution_reward
from post_training.coderl_plus import compute_semantics_reward
from data.chat_format import format_chat_history, IM_START, ASSISTANT_PROMPT
from data.dataloader import load_tokenizer_for_training
from training.device_utils import resolve_device

def generate_candidates(model, tokenizer, prompt: str, num_candidates: int, max_new_tokens: int = 128, device: str = "cuda"):
    """
    Generates N candidates for a given prompt using multinomial sampling to ensure diversity 
    for the GRPO advantage calculation.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids] * num_candidates, dtype=torch.long, device=device)
    
    generated = input_tensor.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.size(1) > model.config.max_seq_len:
                break
                
            logits = model(generated)["logits"]
            next_token_logits = logits[:, -1, :]
            
            # Apply a high temperature to force diverse candidates
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Very basic early stopping heuristic if all candidates produced EOS
            # (assuming EOS is 2)
            if (next_token == 2).all():
                break
                
    # Return just the generated portion
    prompt_len = input_tensor.size(1)
    return generated[:, prompt_len:]

def get_candidate_logprobs(model, input_ids, prompt_len):
    """
    Computes log probabilities for the *generated* portion of the combined sequences.
    """
    logits = model(input_ids)["logits"]
    # Shift logits and labels by 1 for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Calculate log_softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probs of the actual tokens chosen
    # Shape: [Batch, SeqLen]
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out the prompt tokens (prompt length - 1 because we shifted)
    mask = torch.zeros_like(target_log_probs)
    mask[:, prompt_len - 1:] = 1.0
    
    # Return matched log probs across the generated span
    return target_log_probs * mask, mask

def grpo_train_step(
    policy_model, ref_model, optimizer, tokenizer, prompt: str, tests: list,
    num_candidates: int = 4, beta: float = 0.04, device: str | torch.device = "cuda",
    reference_code: str | None = None, coderl_plus_weight: float = 0.5,
):
    """
    Ein GRPO-Schritt: K Kandidaten generieren, mit Execution-Reward (+ optional CodeRL+ Semantics) bewerten.
    reference_code aus RL-JSONL (reference_solution/target_code) → Reward = (1-w)*exec + w*semantics_match.
    """
    policy_model.train()

    # 1. Generation Phase
    gen_tokens = generate_candidates(policy_model, tokenizer, prompt, num_candidates, device=device)

    # Try decode for reward evaluation
    candidates_text = []
    for row in gen_tokens:
        candidates_text.append(tokenizer.decode(row.tolist()))

    # 2. Reward: Execution (Sandbox) + bei Vorhandensein CodeRL+ Semantics-Match (Plan §2.7).
    rewards = []
    for text in candidates_text:
        exec_score = compute_execution_reward(text, tests)
        if reference_code is not None and reference_code.strip():
            semantics_score = min(1.0, compute_semantics_reward(text, reference_code))
            score = (1.0 - coderl_plus_weight) * exec_score + coderl_plus_weight * semantics_score
        else:
            score = exec_score
        rewards.append(score)
        
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Compute Advantage (Reward - Mean Base)
    if num_candidates > 1:
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
    else:
        advantages = torch.zeros_like(rewards_tensor)
        
    # Re-encode input combined for full forward passes
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    full_input = torch.cat([torch.tensor([prompt_ids]*num_candidates, device=device), gen_tokens], dim=1)
    
    # 3. Probabilities (ref in no_grad; policy in autocast on CUDA for efficiency, Plan §6)
    with torch.no_grad():
        ref_log_probs, mask = get_candidate_logprobs(ref_model, full_input, prompt_len)

    use_amp = device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")
    with torch.autocast(device_type="cuda", enabled=use_amp):
        pol_log_probs, _ = get_candidate_logprobs(policy_model, full_input, prompt_len)
        ratio = torch.exp(pol_log_probs - ref_log_probs)
        eps = 0.2
        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2)
        approx_kl = ref_log_probs - pol_log_probs
        loss = (policy_loss + beta * approx_kl) * mask
        loss = loss.sum() / mask.sum()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), rewards_tensor.mean().item()


def load_rl_prompts(rl_data_path: str) -> List[Dict[str, Any]]:
    """
    Lädt RL-Prompts aus JSONL (prompt, tests). Optional: reference_solution oder target_code
    für CodeRL+ (Semantics-Match-Reward); wird an grpo_train_step übergeben.
    """
    path = Path(rl_data_path)
    if not path.is_absolute():
        path = _ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"RL data path does not exist: {path}. Provide --rl-data with path to instruction_sft.jsonl or RL JSONL.")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt" in row:
                # reference_solution/target_code → CodeRL+ blended reward in grpo_train_step.
                out.append({
                    "prompt": row["prompt"],
                    "tests": row.get("tests", []),
                    "reference_solution": row.get("reference_solution") or row.get("target_code") or "",
                })
            elif "messages" in row:
                messages = row["messages"]
                if not messages:
                    continue
                prefix = messages[:-1]
                prompt_str = format_chat_history(prefix) + f"{IM_START}{ASSISTANT_PROMPT}\n"
                out.append({"prompt": prompt_str, "tests": row.get("tests", [])})
    if not out:
        raise ValueError(f"RL data file is empty or has no valid lines: {path}")
    return out


def run_rl_training(
    sft_checkpoint: str = "sft_checkpoints/final_sft.pt",
    output_dir: str = "rl_checkpoints",
    epochs: int = 2,
    rl_data_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device_override: Optional[str] = None,
):
    if not rl_data_path:
        raise ValueError("RL training requires --rl-data with path to instruction_sft.jsonl or RL JSONL.")
    device = resolve_device(device_override)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # RTX 3090
    print(f"Loading SFT model from {sft_checkpoint} for GRPO onto {device}...")

    ckpt = torch.load(_ROOT / sft_checkpoint, map_location="cpu")
    train_cfg = ckpt["config"]
    
    m = train_cfg["model"]
    model_cfg = ModelConfig(
        d_model=m["d_model"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        vocab_size=m["vocab_size"],
        max_seq_len=m["max_seq_len"],
        use_bitnet=m.get("use_bitnet", False),
        use_mamba_hybrid=m.get("use_mamba_hybrid", False),
        use_blt=m.get("use_blt", False),
        use_leam=m.get("use_leam", False),
        mtp_n=m.get("mtp_n", 1),
    )
    
    # Active Policy to train
    policy_model = CodeGPTLMHeadModel(model_cfg)
    policy_model.load_state_dict(ckpt["model"])
    policy_model.to(device)
    
    # Frozen Reference Model
    ref_model = CodeGPTLMHeadModel(model_cfg)
    ref_model.load_state_dict(ckpt["model"])
    ref_model.to(device)
    ref_model.eval()
    
    # Simple Opt for RL (smaller LR)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer: required at data/tokenizer or --tokenizer
    _tok_path = tokenizer_path or "data/tokenizer"
    _tok_path_resolved = _ROOT / _tok_path if not os.path.isabs(_tok_path) else Path(_tok_path)
    _raw = load_tokenizer_for_training(tokenizer_path=_tok_path_resolved)
    if not (hasattr(_raw, "encode") and hasattr(_raw, "decode")):
        raise AttributeError("Tokenizer must expose encode and decode. Check tokenizer at " + str(_tok_path_resolved))
    class _TokenizerAdapter:
        def __init__(self, t: Any):
            self._t = t
        def encode(self, s: str, add_special_tokens: bool = False, **kwargs) -> list:
            try:
                out = self._t.encode(s, add_special_tokens=add_special_tokens, **kwargs)
            except TypeError:
                out = self._t.encode(s, **kwargs)
            return out if isinstance(out, list) else list(out)
        def decode(self, ids: list, **kwargs) -> str:
            return self._t.decode(ids, **kwargs)
    tokenizer = _TokenizerAdapter(_raw)

    rl_prompts = load_rl_prompts(rl_data_path)
    print(f"RL data: {len(rl_prompts)} prompts loaded; first prompt length {len(rl_prompts[0]['prompt'])} chars, tests: {len(rl_prompts[0].get('tests', []))}")

    global_step = 0
    for epoch in range(epochs):
        for data in rl_prompts:
            prompt = data["prompt"]
            tests = data["tests"]
            reference_code = data.get("reference_solution", "") or ""
            # CodeRL+: bei reference_code wird Semantics-Match mit 0.5 Gewicht zugemischt.
            loss, reward = grpo_train_step(
                policy_model, ref_model, optimizer, tokenizer, prompt, tests,
                num_candidates=4, device=device,
                reference_code=reference_code if reference_code else None,
                coderl_plus_weight=0.5,
            )
            global_step += 1
            
            print(f"RL Step {global_step} | Loss {loss:.4f} | Avg Reward {reward:.2f}")
            
    # Save
    final_path = Path(output_dir) / "final_rl.pt"
    torch.save({
        "model": policy_model.state_dict(),
        "config": train_cfg
    }, final_path)
    print(f"RL Training Done. Saved {final_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GRPO RL training from SFT checkpoint")
    p.add_argument("--sft-checkpoint", default="sft_checkpoints/final_sft.pt", help="Path to SFT checkpoint")
    p.add_argument("--output-dir", default="rl_checkpoints", help="Output directory for RL checkpoint")
    p.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    p.add_argument("--rl-data", type=str, required=True, help="Path to RL prompts JSONL (e.g. data/processed/instruction_sft.jsonl)")
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (default: data/tokenizer)")
    p.add_argument("--device", type=str, default=None, help="Override device (cuda, mps, cpu); default auto")
    args = p.parse_args()
    run_rl_training(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        rl_data_path=args.rl_data,
        tokenizer_path=args.tokenizer,
        device_override=args.device,
    )
