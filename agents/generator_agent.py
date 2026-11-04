"""Baseline prompt generation agent."""
from core.optimizer import generate


class GeneratorAgent:
    def run(self, task: str) -> str:
        return generate(task)
