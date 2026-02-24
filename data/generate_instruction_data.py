import argparse
import json
import os
import tqdm
from pathlib import Path

from data.instruction_data import (
    evolve_instruction,
    generate_teacher_response,
    generate_teacher_response_multi,
    extract_concepts_from_code,
    filter_selfcodealign_green,
)
from data.chat_format import format_chat_history


def _run_tests_sandbox(code: str, tests: list) -> tuple:
    try:
        from evaluation.eval_repair import run_tests_in_sandbox
        return run_tests_in_sandbox(code, tests)
    except Exception:
        return False, ""


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT instruction data via Evol-Instruct & Teacher Models.")
    parser.add_argument("--seed_file", type=str, help="Path to a txt file with one seed instruction per line.", default=None)
    parser.add_argument("--seed_code_file", type=str, default=None, help="Path to file with code snippets (one per line or JSONL with 'code' key) for SelfCodeAlign concept extraction.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of synthetic samples to generate if missing seed_file.")
    parser.add_argument("--evolve", action="store_true", help="Apply Evol-Instruct to make inputs more complex.")
    parser.add_argument("--selfcodealign", action="store_true", help="SelfCodeAlign-style: concepts from code, multi-sample, only test-green pairs (requires tests in seed or --tests_file).")
    parser.add_argument("--selfcodealign_multi", type=int, default=3, help="Number of teacher responses per instruction when using --selfcodealign.")
    parser.add_argument("--tests_file", type=str, default=None, help="Path to JSONL with 'tests' list per line (optional, for sandbox filter).")
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

    # SelfCodeAlign: extract instructions from seed code file
    if args.selfcodealign and args.seed_code_file and os.path.exists(args.seed_code_file):
        seed_instructions = []
        with open(args.seed_code_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    code = row.get("code", row.get("content", line))
                except json.JSONDecodeError:
                    code = line
                for item in extract_concepts_from_code(code):
                    seed_instructions.append(item["instruction"])
        seed_instructions = seed_instructions[: max(args.num_samples, len(seed_instructions))]

    # Limit to num_samples roughly
    seed_instructions = seed_instructions[:args.num_samples]

    default_tests: list = []
    if args.tests_file and os.path.exists(args.tests_file):
        with open(args.tests_file, "r") as f:
            first = f.readline()
            if first:
                try:
                    default_tests = json.loads(first).get("tests", [])
                except json.JSONDecodeError:
                    pass

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(seed_instructions)} instruction SFT examples... (selfcodealign={args.selfcodealign})")
    
    with open(args.output_file, "w") as out:
        for seed in tqdm.tqdm(seed_instructions):
            instruction = seed
            if args.evolve:
                instruction = evolve_instruction(seed)
            
            if args.selfcodealign:
                candidates = generate_teacher_response_multi(instruction, num_samples=args.selfcodealign_multi)
                if default_tests and candidates:
                    green = filter_selfcodealign_green(candidates, default_tests, _run_tests_sandbox)
                    response = green[0] if green else (candidates[0] if candidates else "")
                else:
                    response = candidates[0] if candidates else ""
            else:
                response = generate_teacher_response(instruction)
            
            if not response:
                continue

            example = {
                "seed_instruction": seed,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
            if default_tests:
                example["tests"] = default_tests
            out.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"SFT Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
