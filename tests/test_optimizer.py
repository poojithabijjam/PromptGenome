import unittest

from core.optimizer import TestCase, evaluate, generate, mutate, optimize
from core.benchmark import benchmark


def fake_executor(prompt: str, input_text: str) -> str:
    if "Follow the requested output format exactly" in prompt:
        return input_text.upper()
    return input_text.lower()


class OptimizerTests(unittest.TestCase):
    def test_empty_task_rejected(self):
        with self.assertRaises(ValueError):
            generate(" ")

    def test_mutations_are_distinct(self):
        variants = mutate(generate("Uppercase the input"))
        self.assertEqual(len(variants), len(set(variants)))

    def test_empty_cases_rejected(self):
        with self.assertRaises(ValueError):
            evaluate("prompt", [], fake_executor)

    def test_case_sensitive_exact_match(self):
        self.assertEqual(evaluate("prompt", [TestCase("hello", "HELLO")], fake_executor), 0)

    def test_selects_improved_variant(self):
        cases = [TestCase("hello", "HELLO"), TestCase("world", "WORLD")]
        result = optimize("Uppercase the input", cases, fake_executor, generations=3)
        self.assertEqual(result["baseline"]["score"], 0)
        self.assertEqual(result["best"]["score"], 1)
        self.assertEqual(result["best"]["generation"], 1)
        self.assertEqual(result["best"]["parent_id"], result["baseline"]["id"])

    def test_zero_generations_keeps_baseline(self):
        result = optimize("Echo", [TestCase("x", "x")], fake_executor, generations=0)
        self.assertEqual(len(result["history"]), 1)

    def test_benchmark_held_out(self):
        result = benchmark(
            "Uppercase the input",
            [TestCase("hello", "HELLO")],
            [TestCase("world", "WORLD")],
            fake_executor,
        )
        self.assertEqual(result["held_out"]["baseline_accuracy"], 0)
        self.assertEqual(result["held_out"]["optimized_accuracy"], 1)
        self.assertEqual(result["held_out"]["improvement_percentage_points"], 100)

    def test_benchmark_rejects_overlap(self):
        with self.assertRaises(ValueError):
            benchmark("Task", [TestCase("x", "X")], [TestCase("x", "X")], fake_executor)

    def test_benchmark_rejects_empty_test(self):
        with self.assertRaises(ValueError):
            benchmark("Task", [TestCase("x", "X")], [], fake_executor)


if __name__ == "__main__":
    unittest.main()
