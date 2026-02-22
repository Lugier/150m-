"""
SFT auf Trajektorien (Plan §7): Prompt→Versuch→Fix; nur Tests grün.
"""
import json
from pathlib import Path
from typing import Iterator

def load_trajectories(path: str | Path) -> Iterator[dict]:
    """Plan §7: Load trajectory JSONL lines."""
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def filter_tests_green(trajectories: Iterator[dict]) -> Iterator[dict]:
    """Plan §7: Only trajectories with all_tests_passed."""
    for t in trajectories:
        if t.get("all_tests_passed", t.get("final_status") == "green"):
            yield t

def load_trajectories_for_sft(dataset_path: str):
    """
    SFT on Trajectories (Prompt->Attempt->Fix->Success).
    Prepares the dataset for standard HuggingFace trainers formatting.
    """
    dataset = []
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Flatten trajectory into a conversational format
                conversation = []
                conversation.append({"role": "user", "content": data["problem"]})
                for step in data.get("trajectory", []):
                    conversation.append({"role": "assistant", "content": step})
                dataset.append(conversation)
    except FileNotFoundError:
        return []
    return dataset

if __name__ == "__main__":
    trials = load_trajectories_for_sft("data/processed/sft_trajectories.jsonl")
    print(f"Loaded {len(trials)} SFT conversational trajectories.")
