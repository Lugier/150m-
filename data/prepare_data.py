"""
Three-stage data pipeline: breit → sauber-edukativ → curriculum.
Stack-Edu maximal; near-dedup; output streamable JSONL/Parquet.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_for_dedup(code: str) -> str:
    """Normalize whitespace and strip for near-dedup."""
    return " ".join(code.split())


def minhash_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard-style similarity for n-grams; simplified."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ngrams(s: str, n: int = 5) -> set[str]:
    t = normalize_for_dedup(s)
    return set(t[i : i + n] for i in range(max(0, len(t) - n + 1)))


def filter_stage1(doc: dict[str, Any], config: dict[str, Any]) -> bool:
    """Stage 1: length, license, basic quality."""
    content = doc.get("content", "") or doc.get("code", "") or ""
    if len(content) < config.get("min_chars", 100):
        return False
    if len(content) > config.get("max_chars", 100_000):
        return False
    license_allow = config.get("license_allow") or []
    if license_allow and doc.get("license") not in license_allow:
        return False
    return True


def filter_stage2(doc: dict[str, Any], config: dict[str, Any]) -> bool:
    """Stage 2: stricter; prefer has_tests, classifier quality."""
    if not filter_stage1(doc, config):
        return False
    if config.get("classifier_quality") and not doc.get("quality_ok", True):
        return False
    return True


def filter_stage3(doc: dict[str, Any], config: dict[str, Any]) -> bool:
    """Stage 3: execution filter (only passing tests)."""
    if not filter_stage2(doc, config):
        return False
    if config.get("execution_filter") and not doc.get("tests_passed", False):
        return False
    return True


def near_dedup(
    stream: Iterator[dict[str, Any]],
    threshold: float = 0.95,
    key: str = "content",
) -> Iterator[dict[str, Any]]:
    seen_hashes: set[str] = set()
    seen_ngrams: list[tuple[set[str], str]] = []

    for doc in stream:
        text = (doc.get(key) or doc.get("code") or "")[:50000]
        h = content_hash(normalize_for_dedup(text))
        if h in seen_hashes:
            continue
        ng = ngrams(text, 5)
        for prev_ng, _ in seen_ngrams[-500:]:
            if minhash_similarity(ng, prev_ng) >= threshold:
                break
        else:
            seen_hashes.add(h)
            seen_ngrams.append((ng, h))
            if len(seen_ngrams) > 1000:
                seen_ngrams = seen_ngrams[-500:]
            yield doc


def stage_stream(
    stage_name: str,
    config: dict[str, Any],
    source_iter: Iterator[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    stages = config.get("stages", {})
    stage_cfg = stages.get(stage_name, {})
    filters = stage_cfg.get("filters", {})
    if filters.get("near_dedup"):
        source_iter = near_dedup(source_iter, threshold=filters.get("dedup_threshold", 0.95))

    if stage_name == "stage1":
        pred = lambda d: filter_stage1(d, filters)
    elif stage_name == "stage2":
        pred = lambda d: filter_stage2(d, filters)
    else:
        pred = lambda d: filter_stage3(d, filters)

    for doc in source_iter:
        if pred(doc):
            yield doc


def load_hf_dataset_stream(
    dataset_name: str,
    split: str = "train",
    streaming: bool = True,
    **kwargs: Any,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            dataset_name,
            data_dir=kwargs.pop("data_dir", None),
            split=split,
            streaming=streaming,
            **kwargs
        )
        for item in ds:
            yield dict(item)
    except Exception as e:
        raise RuntimeError(f"Failed to load {dataset_name}: {e}") from e


def write_jsonl(iter_items: Iterator[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for doc in iter_items:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config_data.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--dataset", default="bigcode/the-stack-smol")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = args.output or os.path.join(
        config.get("output", {}).get("output_dir", "./data/processed"),
        f"{args.stage}.jsonl",
    )

    def source() -> Iterator[dict[str, Any]]:
        it = load_hf_dataset_stream(args.dataset, streaming=True)
        for i, doc in enumerate(it):
            if args.max_docs and i >= args.max_docs:
                break
            yield doc

    stream = stage_stream(args.stage, config, source())
    write_jsonl(stream, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
