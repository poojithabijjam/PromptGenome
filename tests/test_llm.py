import json
import os
import unittest
from unittest.mock import MagicMock, patch

from core.llm import OpenAIAdapter, Usage


class AdapterTests(unittest.TestCase):
    def test_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIAdapter()

    @patch("core.llm.OpenAI")
    def test_execution_and_usage(self, mock_openai):
        response = MagicMock()
        response.choices[0].message.content = "answer"
        response.usage.prompt_tokens = 12
        response.usage.completion_tokens = 4
        mock_openai.return_value.chat.completions.create.return_value = response
        adapter = OpenAIAdapter(api_key="test-key")
        self.assertEqual(adapter.execute("instruction", "question"), "answer")
        self.assertEqual(adapter.usage.to_dict(), {"calls": 1, "input_tokens": 12, "output_tokens": 4})

    @patch("core.llm.OpenAI")
    def test_mutations_parse_json(self, mock_openai):
        response = MagicMock()
        response.choices[0].message.content = json.dumps(["variant 1", "variant 2", "variant 1"])
        response.usage = None
        mock_openai.return_value.chat.completions.create.return_value = response
        adapter = OpenAIAdapter(api_key="test-key")
        self.assertEqual(adapter.mutate("original"), ["variant 1", "variant 2"])


if __name__ == "__main__":
    unittest.main()
