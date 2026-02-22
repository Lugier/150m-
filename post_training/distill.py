"""
Trajectory Distillation (Plan §7): Teacher Problem→Lösung→Tests→Fix→…;
nur Episoden mit final alle Tests grün.
"""

import json
import os
from pathlib import Path
from typing import List, Any

def load_teacher_trajectories(path: str | Path) -> List[dict]:
    """Plan §7: Load teacher trajectory JSONL."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def filter_reward_green(trajectories: List[dict], key: str = "final_status") -> List[dict]:
    """Plan §7: Only trajectories with all tests green."""
    return [t for t in trajectories if t.get(key) == "green" or t.get("all_tests_passed", False)]

def distill_trajectories(teacher_file: str, student_dataset: str):
    """
    Trajectory Distillation:
    Converts teacher traces and extracts ONLY completely solved episodes 
    (with green tests) for the student to distill.
    """
    if not os.path.exists(teacher_file):
        raise FileNotFoundError(f"Teacher file not found: {teacher_file}. Provide teacher trajectory JSONL (e.g. from CodeRL+ rollouts).")

    clean_trajectories = []
    with open(teacher_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("final_status") == "green":
                clean_trajectories.append(data)
                
    os.makedirs(os.path.dirname(student_dataset), exist_ok=True)
    with open(student_dataset, 'w') as f:
        for t in clean_trajectories:
            f.write(json.dumps(t) + "\n")
            
    print(f"Distilled {len(clean_trajectories)} clean trajectories into {student_dataset}.")

if __name__ == "__main__":
    distill_trajectories("data/teacher_rollouts.jsonl", "data/processed/sft_trajectories.jsonl")
