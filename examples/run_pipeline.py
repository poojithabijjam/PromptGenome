"""Offline end-to-end five-agent demo; no OpenAI key required."""
from core.optimizer import TestCase
from core.pipeline import PromptGenomePipeline


def execute(prompt, text):
    return text.upper() if "Follow the requested output format exactly" in prompt else text.lower()


if __name__ == "__main__":
    result = PromptGenomePipeline(execute).run(
        "Convert the input to uppercase.",
        [TestCase("hello", "HELLO"), TestCase("world", "WORLD")],
        [TestCase("unseen", "UNSEEN"), TestCase("example", "EXAMPLE")],
    )
    print(result["held_out"])
