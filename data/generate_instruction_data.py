import argparse
import json
import os
import tqdm
from pathlib import Path

from data.instruction_data import evolve_instruction, generate_teacher_response
from data.chat_format import format_chat_history

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT instruction data via Evol-Instruct & Teacher Models.")
    parser.add_argument("--seed_file", type=str, help="Path to a txt file with one seed instruction per line.", default=None)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of synthetic samples to generate if missing seed_file.")
    parser.add_argument("--evolve", action="store_true", help="Apply Evol-Instruct to make inputs more complex.")
    parser.add_argument("--output_file", type=str, default="data/processed/instruction_sft.jsonl")
    args = parser.parse_args()

    # Pre-defined simple seeds if no file
    seed_instructions = [
        "Write a Python function to sort a list.",
        "Create a simple HTTP server in Go.",
        "Write a React component for a login form.",
        "How do I reverse a string in C++?",
        "Write a SQL query to get the top 5 users by sales.",
        "Create a Python script to download an image from a URL.",
        "Write a Bash script to backup a directory.",
        "Implement binary search in Java.",
        "Write a Dockerfile for a Node.js app.",
        "Create a simple CLI calculator in Rust."
    ]

    if args.seed_file and os.path.exists(args.seed_file):
        with open(args.seed_file, "r") as f:
            seed_instructions = [line.strip() for line in f if line.strip()]

    # Limit to num_samples roughly
    seed_instructions = seed_instructions[:args.num_samples]

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(seed_instructions)} instruction SFT examples...")
    
    with open(args.output_file, "w") as out:
        for seed in tqdm.tqdm(seed_instructions):
            
            instruction = seed
            if args.evolve:
                instruction = evolve_instruction(seed)
                
            response = generate_teacher_response(instruction)
            
            # If the API call failed, skip writing
            if not response:
                continue

            # Store as rich raw JSON for later
            example = {
                "seed_instruction": seed,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"SFT Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
