"""Scoring agent with exact-match and structured JSON validation."""
import json
from difflib import SequenceMatcher
from core.optimizer import TestCase


def score_prediction(prediction: str, expected: str, metric: str = "exact") -> float:
    if metric == "exact":
        return float(prediction.strip() == expected.strip())
    if metric == "normalized":
        return float(prediction.strip().casefold() == expected.strip().casefold())
    if metric == "similarity":
        # Character-level similarity, not an embedding or semantic correctness metric.
        return SequenceMatcher(None, prediction.strip().casefold(), expected.strip().casefold()).ratio()
    if metric == "json":
        try:
            return float(json.loads(prediction) == json.loads(expected))
        except (ValueError, TypeError):
            return 0.0
    raise ValueError("metric must be exact, normalized, similarity, or json")


class EvaluationAgent:
    def __init__(self, metric: str = "exact"):
        score_prediction("", "", metric)
        self.metric = metric

    def run(self, prompt: str, cases: list[TestCase], execution_agent) -> float:
        if not cases:
            raise ValueError("At least one evaluation case is required")
        return sum(
            score_prediction(execution_agent.run(prompt, case.input), case.expected, self.metric)
            for case in cases
        ) / len(cases)
