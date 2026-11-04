"""Offline checks for the bundled Streamlit preset data."""
import json
import unittest
from pathlib import Path
from benchmarks.run_support_tickets import load_cases

class PresetDataTests(unittest.TestCase):
    def test_support_ticket_json_loads(self):
        for filename, size in [("support_tickets_train.json", 36),
                               ("support_tickets_test.json", 12)]:
            cases = load_cases(filename)
            self.assertEqual(len(cases), size)
            self.assertTrue(all(isinstance(x.input, str) and isinstance(x.expected, str) for x in cases))

    def test_no_overlap(self):
        train = load_cases("support_tickets_train.json")
        test = load_cases("support_tickets_test.json")
        self.assertFalse({x.input for x in train} & {x.input for x in test})

if __name__ == "__main__":
    unittest.main()
