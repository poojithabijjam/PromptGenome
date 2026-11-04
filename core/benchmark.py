"""Held-out evaluation: test examples never influence candidate selection."""
from __future__ import annotations

from typing import Callable, Iterable

from core.optimizer import TestCase, evaluate, optimize


def benchmark(
    task: str,
    training_cases: Iterable[TestCase],
    test_cases: Iterable[TestCase],
    execute: Callable[[str, str], str],
    generations: int = 3,
    mutator: Callable[[str], list[str]] | None = None,
    max_candidates: int = 5,
) -> dict:
    train, test = list(training_cases), list(test_cases)
    if not train or not test:
        raise ValueError("Training and held-out test cases must both be nonempty")
    train_inputs = {case.input for case in train}
    if any(case.input in train_inputs for case in test):
        raise ValueError("Training and test inputs must not overlap")
    result = optimize(task, train, execute, generations, mutator, max_candidates)
    baseline_score = evaluate(result["baseline"]["prompt"], test, execute)
    best_score = evaluate(result["best"]["prompt"], test, execute)
    return {
        "optimization": result,
        "held_out": {
            "case_count": len(test),
            "baseline_accuracy": baseline_score,
            "optimized_accuracy": best_score,
            "improvement_percentage_points": round((best_score - baseline_score) * 100, 2),
        },
    }
