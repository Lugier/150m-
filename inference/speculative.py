"""
Speculative Decoding (Plan §2): small draft model proposes K tokens; main model verifies in one forward.
1.5–6.5× faster decode, lossless. Draft ~15–30M params for 300M main.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import torch


def speculative_decode_step(
    main_model: Any,
    input_ids: torch.Tensor,
    draft_tokens: List[int],
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """
    One verify step: main model forward on input_ids + draft_tokens; return accepted tokens and count.
    Accept draft token at position i iff argmax(main_logits[i]) == draft_tokens[i].
    """
    if not draft_tokens:
        return input_ids, 0
    device = input_ids.device
    draft = torch.tensor([draft_tokens], dtype=torch.long, device=device)
    full = torch.cat([input_ids, draft], dim=1)
    with torch.no_grad():
        out = main_model(full)["logits"]
    # logits at positions that predict each draft token: out[:, -(len(draft_tokens)+1):-1]
    # we need logits at positions [len(input_ids), len(input_ids)+1, ...] i.e. after prompt
    start = input_ids.size(1) - 1
    accepted = 0
    for i, tok in enumerate(draft_tokens):
        logits_i = out[0, start + i, :]
        pred = logits_i.argmax().item()
        if pred == tok:
            accepted += 1
        else:
            break
    if accepted == 0:
        return input_ids, 0
    new_ids = torch.cat([
        input_ids,
        torch.tensor([draft_tokens[:accepted]], dtype=torch.long, device=device),
    ], dim=1)
    return new_ids, accepted


def draft_generate(
    draft_model: Any,
    input_ids: torch.Tensor,
    k: int,
    temperature: float = 0.0,
) -> List[int]:
    """Draft model generates K tokens autoregressively."""
    tokens: List[int] = []
    current = input_ids
    for _ in range(k):
        with torch.no_grad():
            logits = draft_model(current)["logits"]
        next_logits = logits[0, -1, :]
        if temperature <= 0:
            next_tok = next_logits.argmax().item()
        else:
            probs = torch.softmax(next_logits.float() / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
        tokens.append(next_tok)
        current = torch.cat([
            current,
            torch.tensor([[next_tok]], dtype=current.dtype, device=current.device),
        ], dim=1)
        if current.size(1) >= draft_model.config.max_seq_len:
            break
    return tokens


def speculative_decode(
    main_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    k: int = 4,
    temperature: float = 0.0,
    draft_model: Optional[Any] = None,
    decode_fn: Optional[Callable[[List[int]], str]] = None,
) -> Tuple[torch.Tensor, str]:
    """
    Generate with optional speculative decoding. If draft_model is None, standard autoregressive.
    Returns (output_ids, decoded_string).
    """
    device = input_ids.device
    seq = input_ids.clone()
    decode_fn = decode_fn or (lambda ids: tokenizer.decode(ids))

    if draft_model is None:
        # Standard decode
        for _ in range(max_new_tokens):
            if seq.size(1) >= main_model.config.max_seq_len:
                break
            with torch.no_grad():
                logits = main_model(seq)["logits"]
            next_logits = logits[0, -1, :]
            if temperature <= 0:
                next_tok = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            seq = torch.cat([
                seq,
                torch.tensor([[next_tok]], dtype=seq.dtype, device=device),
            ], dim=1)
        return seq, decode_fn(seq[0].tolist())

    # Speculative loop
    generated = 0
    while generated < max_new_tokens:
        if seq.size(1) >= main_model.config.max_seq_len:
            break
        draft_tokens = draft_generate(draft_model, seq, k, temperature)
        if not draft_tokens:
            break
        seq, accepted = speculative_decode_step(main_model, seq, draft_tokens, temperature)
        generated += accepted
        if accepted < len(draft_tokens):
            # Rejected at position accepted; sample one from main at that position
            with torch.no_grad():
                logits = main_model(seq)["logits"]
            next_logits = logits[0, -1, :]
            if temperature <= 0:
                next_tok = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            seq = torch.cat([
                seq,
                torch.tensor([[next_tok]], dtype=seq.dtype, device=device),
            ], dim=1)
            generated += 1
    return seq, decode_fn(seq[0].tolist())
