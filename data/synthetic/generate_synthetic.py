import os
import json

def generate_synthetic_stage3(output_path: str):
    """
    Simulates generating phi-1 style Teacher datasets.
    Format: Docstring -> Code -> Tests (with CoT comments).
    Ensures high-quality data for Stage 3 instruction tuning.
    """
    print("Generating Stage 3 Synthetic Data (phi-1 style)...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    synthetic_samples = [
        {
            "instruction": "Write a python function to compute the gcd.",
            "docstring": "def gcd(a: int, b: int) -> int:\n    '''Returns the greatest common divisor of a and b.'''",
            "code": "    while b:\n        a, b = b, a % b\n    return a",
            "tests": "assert gcd(48, 18) == 6\nassert gcd(10, 5) == 5"
        }
    ]
    
    with open(output_path, "w") as f:
        for s in synthetic_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Generated {len(synthetic_samples)} synthetic examples to {output_path}")

if __name__ == "__main__":
    generate_synthetic_stage3("data/processed/stage_3/synthetic.jsonl")
