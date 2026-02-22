"""
EvalPlus HumanEval+/MBPP+: pass@1, pass@10 (Plan Abschn. 9).
Sicht 1: Funktionale Korrektheit Docstring→Code; Repeated Sampling.
evalplus.evaluate --dataset humaneval --samples samples.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_evalplus(
    samples_path: str | Path,
    dataset: str = "humaneval",
    base_only: bool = False,
) -> dict[str, float]:
    """
    EvalPlus (Plan §9): pass@1, pass@10; greedy pass@1; Kontamination beachten.
    samples_path: JSONL mit task_id, completion (oder solution).
    """
    try:
        from evalplus.eval import evaluate
        result = evaluate(
            dataset=dataset,
            samples=str(samples_path),
            base_only=base_only,
        )
        metrics = result.get("metrics", {})
        return {
            "pass@1": float(metrics.get("pass@1", 0.0)),
            "pass@10": float(metrics.get("pass@10", 0.0)),
            **{k: v for k, v in metrics.items() if k not in ("pass@1", "pass@10")},
        }
    except ImportError:
        return {"pass@1": 0.0, "pass@10": 0.0, "note": "evalplus not installed"}
    except Exception as e:
        return {"pass@1": 0.0, "pass@10": 0.0, "error": str(e)}


def generate_humaneval_samples(
    model: Any,
    output_file: str,
    tokenizer: Any = None,
    num_tasks: int = 164,
    max_new_tokens: int = 256,
) -> None:
    """
    Generiert k Samples für HumanEval (164 Aufgaben). Ausgabe: JSONL für EvalPlus.
    """
    samples = []
    task_ids = [f"HumanEval/{i}" for i in range(num_tasks)]
    for task_id in task_ids:
        if model is None:
            samples.append({"task_id": task_id, "completion": "    return True\n"})
        else:
            # Platzhalter: echtes Modell würde hier generieren
            samples.append({"task_id": task_id, "completion": "    return True\n"})
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Samples written to {output_file}")
    print("Evaluate: evalplus.evaluate --dataset humaneval --samples", output_file)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="EvalPlus HumanEval+/MBPP+, pass@1/pass@10 (Plan §9)")
    p.add_argument("--samples", required=True, help="JSONL samples (task_id, completion)")
    p.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    p.add_argument("--base_only", action="store_true")
    args = p.parse_args()
    metrics = run_evalplus(args.samples, args.dataset, args.base_only)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
