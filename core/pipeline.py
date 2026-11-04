"""Five-agent orchestration with isolated held-out benchmarking."""
from dataclasses import asdict
import uuid

from agents import GeneratorAgent, MutationAgent, ExecutionAgent, EvaluationAgent, SelectionAgent
from core.optimizer import Candidate, TestCase


class PromptGenomePipeline:
    def __init__(self, executor, mutator=None, metric="exact"):
        self.generator = GeneratorAgent()
        self.mutation = MutationAgent(mutator)
        self.execution = ExecutionAgent(executor)
        self.evaluation = EvaluationAgent(metric)
        self.selection = SelectionAgent()
        self.metric = metric

    def run(self, task: str, training_cases: list[TestCase], test_cases: list[TestCase],
            generations: int = 2, variants: int = 3) -> dict:
        if not 0 <= generations <= 20:
            raise ValueError("generations must be between 0 and 20")
        if not 1 <= variants <= 10:
            raise ValueError("variants must be between 1 and 10")
        if not training_cases or not test_cases:
            raise ValueError("Training and held-out cases must both be nonempty")
        if {c.input for c in training_cases} & {c.input for c in test_cases}:
            raise ValueError("Training and held-out inputs must not overlap")
        baseline = Candidate(id=str(uuid.uuid4()), prompt=self.generator.run(task), generation=0)
        baseline.score = self.evaluation.run(baseline.prompt, training_cases, self.execution)
        best, history, seen = baseline, [baseline], {baseline.prompt}
        for generation in range(1, generations + 1):
            candidates = []
            for prompt in self.mutation.run(best.prompt, variants):
                if prompt in seen:
                    continue
                seen.add(prompt)
                candidate = Candidate(id=str(uuid.uuid4()), prompt=prompt, generation=generation,
                                      parent_id=best.id)
                candidate.score = self.evaluation.run(prompt, training_cases, self.execution)
                candidates.append(candidate)
            history.extend(candidates)
            winner = self.selection.run(best, candidates)
            if winner is best:
                break
            best = winner
        baseline_test = self.evaluation.run(baseline.prompt, test_cases, self.execution)
        best_test = self.evaluation.run(best.prompt, test_cases, self.execution)
        return {
            "optimization": {
                "baseline": asdict(baseline),
                "best": asdict(best),
                "history": [asdict(c) for c in history],
            },
            "held_out": {
                "case_count": len(test_cases),
                "metric": self.metric,
                "baseline_accuracy": baseline_test,
                "optimized_accuracy": best_test,
                "improvement_percentage_points": round((best_test - baseline_test) * 100, 2),
            },
        }
