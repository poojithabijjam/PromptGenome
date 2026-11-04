"""Offline checks for the bundled benchmark; no API key needed."""
import unittest
from benchmarks.run_support_tickets import TASK, load_cases
from core.pipeline import PromptGenomePipeline

class SupportTicketBenchmarkTests(unittest.TestCase):
    def test_split_and_labels(self):
        train = load_cases("support_tickets_train.json")
        test = load_cases("support_tickets_test.json")
        self.assertEqual((len(train), len(test)), (36, 12))
        self.assertFalse({x.input for x in train} & {x.input for x in test})
        self.assertEqual({x.expected for x in train + test},
                         {"billing", "technical", "account", "other"})

    def test_offline_pipeline_and_heldout_isolation(self):
        train = load_cases("support_tickets_train.json")
        test = load_cases("support_tickets_test.json")
        def fake_execute(prompt, message):
            if "billed" in message.lower() or "invoice" in message.lower() or "refund" in message.lower():
                return "billing"
            if "crashes" in message.lower() or "500 error" in message.lower() or "freezes" in message.lower():
                return "technical"
            if "password" in message.lower() or "email on" in message.lower() or "login" in message.lower():
                return "account"
            return "other"
        result = PromptGenomePipeline(fake_execute, metric="normalized").run(
            TASK, train, test, generations=1, variants=1)
        self.assertEqual(result["held_out"]["case_count"], 12)
        self.assertEqual(result["held_out"]["baseline_accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
