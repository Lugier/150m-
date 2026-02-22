"""
ChatML Formatting and Tokenizer Span Logic.
Handles the conversion of conversation histories into ChatML strings,
and determines the token-level spans of the assistant's responses for loss masking.
"""

from typing import List, Dict, Tuple, Any

# ChatML Tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYS_PROMPT = "system"
USER_PROMPT = "user"
ASSISTANT_PROMPT = "assistant"

# We assume that the tokenizer has/will have these special tokens
CHATML_SPECIAL_TOKENS = [IM_START, IM_END]

def format_chat_message(role: str, content: str) -> str:
    """Formats a single message into a ChatML string block."""
    return f"{IM_START}{role}\n{content}{IM_END}\n"

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Given a list of messages [{'role': 'user', 'content': '...'}],
    returns the complete ChatML formatted string.
    """
    formatted = ""
    for msg in messages:
        formatted += format_chat_message(msg["role"], msg["content"])
    return formatted

def parse_chat_to_message_spans(tokenizer, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
    """
    Given a list of messages, formats them and outputs:
    1. The fully tokenized input_ids
    2. A labels array where every token NOT belonging to the ASSISTANT's actual 
       response payload is set to -100 (for loss masking).
       
    This ensures that the model only learns to predict the assistant's content,
    not the user's prompt or the ChatML boilerplate.
    """
    input_ids = []
    labels = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # We split the message into its 'header' (which shouldn't be learned)
        # and the 'content'. 
        # Note: In strict ChatML, the assistant header is <|im_start|>assistant\n
        header = f"{IM_START}{role}\n"
        footer = f"{IM_END}\n"
        
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = tokenizer.encode(footer, add_special_tokens=False)
        
        # Extend the main sequence
        input_ids.extend(header_ids)
        input_ids.extend(content_ids)
        input_ids.extend(footer_ids)
        
        # Extend the labels sequence
        # System and User turns are entirely masked
        if role != ASSISTANT_PROMPT:
            labels.extend([-100] * len(header_ids))
            labels.extend([-100] * len(content_ids))
            labels.extend([-100] * len(footer_ids))
        else:
            # For the assistant:
            # We DONT predict the header (<|im_start|>assistant\n)
            labels.extend([-100] * len(header_ids))
            
            # We DO predict the content
            labels.extend(content_ids)
            
            # We DO predict the footer (<|im_end|>\n) so it learns to stop
            labels.extend(footer_ids)
            
    return input_ids, labels

def inject_chatml_special_tokens(tokenizer):
    """
    Injects ChatML special tokens into a HuggingFace tokenizer.
    Returns the tokenizer and the possibly updated vocab size.
    """
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': CHATML_SPECIAL_TOKENS})
    return tokenizer, len(tokenizer)
