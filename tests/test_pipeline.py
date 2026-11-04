import unittest

from agents.evaluation_agent import score_prediction
from core.optimizer import TestCase
from core.pipeline import PromptGenomePipeline


def execute(prompt, text):
    return text.upper() if "Follow the requested output format exactly" in prompt else text.lower()


class PipelineTests(unittest.TestCase):
    def test_full_five_agent_pipeline(self):
        result = PromptGenomePipeline(execute).run(
            "Uppercase text", [TestCase("hello", "HELLO")],
            [TestCase("world", "WORLD")], generations=2
        )
        self.assertEqual(result["held_out"]["baseline_accuracy"], 0)
        self.assertEqual(result["held_out"]["optimized_accuracy"], 1)
        self.assertEqual(result["optimization"]["best"]["generation"], 1)

    def test_json_equivalence(self):
        self.assertEqual(score_prediction('{"a": 1, "b": 2}', '{"b":2,"a":1}', "json"), 1)
        self.assertEqual(score_prediction('not json', '{"a":1}', "json"), 0)

    def test_similarity(self):
        self.assertEqual(score_prediction("Hello", "hello", "similarity"), 1)
        self.assertLess(score_prediction("a", "xyz", "similarity"), 1)

    def test_overlapping_inputs_rejected(self):
        with self.assertRaises(ValueError):
            PromptGenomePipeline(execute).run("Task", [TestCase("a", "a")], [TestCase("a", "a")])

    def test_no_generations(self):
        result = PromptGenomePipeline(execute).run(
            "Task", [TestCase("a", "a")], [TestCase("b", "b")], generations=0
        )
        self.assertEqual(len(result["optimization"]["history"]), 1)

    def test_llm_mutator_injected(self):
        pipeline = PromptGenomePipeline(execute, mutator=lambda p: [p + "\nFollow the requested output format exactly. Do not add unrelated information."])
        result = pipeline.run("Uppercase", [TestCase("x", "X")], [TestCase("y", "Y")])
        self.assertEqual(result["held_out"]["optimized_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
