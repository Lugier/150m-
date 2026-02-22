"""
LEAM++ (Syntax-Awareness): AST metadata stream alongside byte/token stream.
Limits prediction to valid grammar rules; reduces invalid mutations.
"""
from __future__ import annotations
import ast
import torch
import torch.nn as nn
from typing import List

class ASTMetadataEmbedding(nn.Module):
    """Embed AST node type and depth for syntax-aware conditioning."""
    def __init__(self, num_node_types: int = 128, depth_bins: int = 32, d_model: int = 384):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_types, d_model)
        self.depth_embed = nn.Embedding(depth_bins, d_model)

    def forward(self, node_type_ids: torch.Tensor, depth_ids: torch.Tensor) -> torch.Tensor:
        return self.node_embed(node_type_ids) + self.depth_embed(depth_ids)

class LEAMGrammarConstrainer:
    """
    LEAM++ (Plan §2): Grammar-Guard in der Inferenz.
    Wird in run_chat.py / run_torch.py verwendet, wenn config.use_leam True.
    Maskiert Next-Token-Logits, die zu sofortigem Python-SyntaxError führen.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def constrain_logits(self, current_sequence_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Logits shape (1, vocab_size); top-k Kandidaten werden per ast.parse geprüft, Verstöße auf -1e9 gesetzt."""
        if not self.tokenizer:
            return logits

        # We decode the top K predictions to explicitly parse for syntax fatalities
        top_k = 5
        top_indices = torch.topk(logits, top_k, dim=-1).indices
        
        base_code = self.tokenizer.decode(current_sequence_ids)
        
        for idx in top_indices[0]: # Assuming batch size 1 for inference constraints
            token_str = self.tokenizer.decode([idx.item()])
            speculative_code = base_code + token_str
            
            # Simple AST verification barrier
            # If the token creates an immediate fatal syntax closure, penalize it
            # We ignore unclosed strings/indentations which are expected mid-generation,
            # focusing on illegal keyword sequences.
            try:
                ast.parse(speculative_code + "\npass")
            except SyntaxError as e:
                # If the error is early in the file, or involves structural violations,
                # we mask the logit.
                if "invalid syntax" in str(e) and "EOF" not in str(e):
                    logits[0, idx.item()] = -1e9

        return logits
