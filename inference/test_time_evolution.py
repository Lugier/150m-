"""
Test-Time Evolution (Plan §2.8): S*, AB-MCTS, DaJ, PoT.
S*: parallele Kandidaten + Sandbox-Auswahl (+ optional differenzierende Tests).
AB-MCTS/DaJ/PoT: Stubs/APIs für Adaptive MCTS, Judge, transiente LoRA/GRPO-Updates.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple


def generate_differentiating_tests(
    candidates: List[str],
    prompt: str,
    tests: List[str],
    max_extra: int = 5,
    run_tests_fn: Optional[Callable[[str, List[str]], Tuple[bool, str]]] = None,
) -> List[str]:
    """
    Generate test inputs that can differentiate between candidates (S* distinguishing inputs).
    If run_tests_fn is provided, runs each candidate and returns failing tests as extra
    discriminators. Otherwise returns [] (plug in LLM or heuristic for full implementation).
    """
    extra: List[str] = []
    if run_tests_fn is not None and candidates and tests:
        try:
            results = [run_tests_fn(c, tests) for c in candidates]
            passed = [r[0] for r in results]
            if sum(passed) != len(candidates) and sum(passed) != 0:
                for i, t in enumerate(tests):
                    if len(extra) >= max_extra:
                        break
                    if "assert" in t:
                        extra.append(t)
        except Exception:
            pass
    return extra[:max_extra]


def _run_tests(code: str, tests: List[str]) -> Tuple[bool, str]:
    """Lazy import um zirkuläre Imports zu vermeiden."""
    from evaluation.eval_repair import run_tests_in_sandbox
    return run_tests_in_sandbox(code, tests)


def s_star_select(
    candidates: List[str],
    tests: List[str],
    differentiating_tests: Optional[List[str]] = None,
) -> Tuple[int, List[Tuple[int, bool]]]:
    """
    S* Selection (Plan §2.8): Alle Kandidaten in Sandbox gegen tests (+ differentiating_tests)
    laufen lassen; Index des besten (erster der passt) und pro-Kandidat (idx, passed).
    """
    all_tests = list(tests) + (differentiating_tests or [])
    results: List[Tuple[int, bool]] = []
    for i, code in enumerate(candidates):
        passed, _ = _run_tests(code, all_tests)
        results.append((i, passed))
    # Best = first that passed; else first candidate
    for i, passed in results:
        if passed:
            return i, results
    return 0, results


def s_star_generate(
    generate_fn: Callable[[str, int], List[str]],
    prompt: str,
    tests: List[str],
    num_candidates: int = 16,
    differentiating_tests: Optional[List[str]] = None,
    differentiating_generator: Optional[Callable[[List[str], str, List[str]], List[str]]] = None,
) -> str:
    """
    S* Generation: generate_fn(prompt, n) liefert n Code-Strings; bester wird per
    s_star_select (Sandbox) gewählt. Wenn differentiating_generator gesetzt, werden
    zusätzliche Tests erzeugt, die Kandidaten unterscheiden (distinguishing inputs).
    """
    candidates = generate_fn(prompt, num_candidates)
    if not candidates:
        return ""
    if differentiating_tests is None and differentiating_generator is not None:
        differentiating_tests = differentiating_generator(candidates, prompt, tests)
    best_idx, _ = s_star_select(candidates, tests, differentiating_tests)
    return candidates[best_idx]


def ab_mcts_score(
    code: str,
    tests: List[str],
    num_rollouts: int = 4,
) -> float:
    """AB-MCTS (Plan §2.8): Stub – Bewertung hier nur über Sandbox Pass/Fail."""
    passed, _ = _run_tests(code, tests)
    return 1.0 if passed else 0.0


def daj_judge(
    solution_a: str,
    solution_b: str,
    problem: str,
    tests: List[str],
) -> int:
    """DaJ (Plan §2.8): Vergleich zweier Lösungen. Stub: Sieg per Sandbox (Pass/Fail); -1/0/1."""
    pa, _ = _run_tests(solution_a, tests)
    pb, _ = _run_tests(solution_b, tests)
    if pa and not pb:
        return -1
    if pb and not pa:
        return 1
    return 0


def pot_update_hook(
    model: Any,
    feedback: List[float],
    candidates: List[str],
    lr: float = 1e-6,
) -> None:
    """
    PoT (Plan §2.8): Transiente LoRA/GRPO-Updates zur Laufzeit. Stub: kein echtes Update;
    Produktion würde Execution-/Konfidenz-Feedback für on-the-fly Adapter nutzen.
    """
    pass
