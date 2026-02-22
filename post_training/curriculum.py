"""
StepCoder-style curriculum: CCCS (Completion Curriculum), FGO (Focused Gradient on executed segments).
Stages: short_functions -> medium_functions -> complex_tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import yaml


@dataclass
class CurriculumStage:
    name: str
    docstring_lines_max: int
    code_lines_max: int
    weight: float


def load_curriculum_config(config_path: str = "post_training/config_post.yaml") -> list[CurriculumStage]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    stages_cfg = cfg.get("curriculum", {}).get("stages", [])
    return [
        CurriculumStage(
            name=s["name"],
            docstring_lines_max=s.get("docstring_lines_max", 999),
            code_lines_max=s["code_lines_max"],
            weight=s["weight"],
        )
        for s in stages_cfg
    ]


def assign_stage(
    docstring_lines: int,
    code_lines: int,
    stages: list[CurriculumStage],
) -> CurriculumStage:
    """Assign sample to curriculum stage by line counts."""
    for s in stages:
        if docstring_lines <= s.docstring_lines_max and code_lines <= s.code_lines_max:
            return s
    return stages[-1]


def iter_curriculum_weighted(
    samples: list[dict],
    stages: list[CurriculumStage],
    text_key: str = "content",
) -> Iterator[dict]:
    """Yield samples with stage assigned; can be used to weight or order dataloader."""
    for sample in samples:
        text = sample.get(text_key, "") or sample.get("code", "")
        lines = text.split("\n")
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        docstring_lines = len([l for l in lines if '"""' in l or "'''" in l])
        stage = assign_stage(docstring_lines, code_lines, stages)
        sample["curriculum_stage"] = stage.name
        sample["curriculum_weight"] = stage.weight
        yield sample
