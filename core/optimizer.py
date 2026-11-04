"""Provider-independent generation, mutation, execution, evaluation and selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable
import uuid


@dataclass(frozen=True)
class TestCase:
    input: str
    expected: str


@dataclass
class Candidate:
    id: str
    prompt: str
    generation: int
    parent_id: str | None = None
    score: float = 0.0


def generate(task: str) -> str:
    if not task.strip():
        raise ValueError("Task description cannot be empty")
    return f"{task.strip()}\nRespond accurately and concisely."


def mutate(prompt: str) -> list[str]:
    suffixes = [
        "Think through the task carefully before responding. Return only the final answer.",
        "Follow the requested output format exactly. Do not add unrelated information.",
        "Check the answer against the task requirements before responding.",
    ]
    return [prompt.rstrip() + "\n" + suffix for suffix in suffixes if suffix not in prompt]


def evaluate(prompt: str, cases: Iterable[TestCase], execute: Callable[[str, str], str]) -> float:
    """Case-sensitive exact match after trimming surrounding whitespace."""
    cases = list(cases)
    if not cases:
        raise ValueError("At least one evaluation case is required")
    return sum(execute(prompt, case.input).strip() == case.expected.strip() for case in cases) / len(cases)


def optimize(
    task: str,
    cases: Iterable[TestCase],
    execute: Callable[[str, str], str],
    generations: int = 3,
    mutator: Callable[[str], list[str]] | None = None,
    max_candidates: int = 5,
) -> dict:
    if not 0 <= generations <= 20:
        raise ValueError("generations must be between 0 and 20")
    if not 1 <= max_candidates <= 10:
        raise ValueError("max_candidates must be between 1 and 10")
    cases = list(cases)
    if not cases:
        raise ValueError("At least one evaluation case is required")
    baseline = Candidate(id=str(uuid.uuid4()), prompt=generate(task), generation=0)
    baseline.score = evaluate(baseline.prompt, cases, execute)
    history = [baseline]
    best = baseline
    seen = {baseline.prompt}
    mutation = mutator or mutate
    for generation in range(1, generations + 1):
        prompts = mutation(best.prompt)
        variants = []
        for prompt in prompts[:max_candidates]:
            if not isinstance(prompt, str) or not prompt.strip() or prompt in seen:
                continue
            seen.add(prompt)
            candidate = Candidate(
                id=str(uuid.uuid4()), prompt=prompt, generation=generation, parent_id=best.id
            )
            candidate.score = evaluate(candidate.prompt, cases, execute)
            variants.append(candidate)
        history.extend(variants)
        winner = max(variants, key=lambda c: c.score, default=best)
        if winner.score > best.score:
            best = winner
        else:
            break
    return {
        "baseline": asdict(baseline),
        "best": asdict(best),
        "history": [asdict(candidate) for candidate in history],
    }
