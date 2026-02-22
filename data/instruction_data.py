"""
Evol-Instruct logic and Teacher API wrappers for generating synthetic 
instruction tuning data for the Code-LLM.
"""

import json
import os
import requests
import random
from typing import List, Dict, Optional

# Example endpoint for a local Ollama instance running Qwen2.5-Coder or DeepSeek
# Change to OpenAI or other providers if needed.
DEFAULT_TEACHER_API = os.getenv("TEACHER_API_URL", "http://localhost:11434/api/chat")
DEFAULT_TEACHER_MODEL = os.getenv("TEACHER_API_MODEL", "qwen2.5-coder")

EVOL_INSTRUCT_SYSTEM_PROMPT = """You are an expert software engineer and AI teacher.
Your goal is to take a simple coding instruction and EVOLVE it into a more 
complex, challenging, and realistic programming task.
Rules for evolving:
1. Add specific constraints (e.g. time/space complexity).
2. Require error handling or input validation.
3. Make it part of a larger system context.
4. Keep the output as a single, clear instruction. Do not answer it.

Original Instruction: {instruction}
Evolved Instruction:"""

def call_teacher_api(messages: List[Dict[str, str]], 
                     api_url: str = DEFAULT_TEACHER_API, 
                     model: str = DEFAULT_TEACHER_MODEL) -> Optional[str]:
    """
    Calls a Chat Completions API (OpenAI compatible format by default, 
    adapted here for Ollama as an accessible local option for large code models).
    """
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        
        # Adjust parsing based on the exact API (OpenAI vs Ollama)
        data = response.json()
        if "message" in data:  # Ollama format
            return data["message"]["content"]
        elif "choices" in data: # OpenAI format
            return data["choices"][0]["message"]["content"]
        else:
            return None
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None

def evolve_instruction(seed_instruction: str) -> str:
    """Evolves a seed instruction to make it more complex using the Teacher model."""
    prompt = EVOL_INSTRUCT_SYSTEM_PROMPT.format(instruction=seed_instruction)
    messages = [{"role": "user", "content": prompt}]
    
    evolved = call_teacher_api(messages)
    return evolved.strip() if evolved else seed_instruction

def generate_teacher_response(instruction: str, system_prompt: str = "You are a helpful coding assistant.") -> str:
    """Generates the grounded coding response from the Teacher model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    response = call_teacher_api(messages)
    return response.strip() if response else ""
