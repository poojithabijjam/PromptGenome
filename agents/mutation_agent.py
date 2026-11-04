"""Prompt mutation agent; supports an injected LLM mutator."""
from collections.abc import Callable
from core.optimizer import mutate


class MutationAgent:
    def __init__(self, mutator: Callable[[str], list[str]] | None = None):
        self.mutator = mutator or mutate

    def run(self, prompt: str, limit: int = 5) -> list[str]:
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")
        variants = self.mutator(prompt)
        if not isinstance(variants, list):
            raise ValueError("mutator must return a list")
        unique = []
        for variant in variants:
            if isinstance(variant, str) and variant.strip() and variant.strip() != prompt.strip() and variant.strip() not in unique:
                unique.append(variant.strip())
        return unique[:limit]
