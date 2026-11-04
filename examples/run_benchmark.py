"""Offline benchmark with separate optimization and held-out examples.

The fake executor is a demonstration, NOT an LLM quality benchmark.
"""
from core.benchmark import benchmark
from core.optimizer import TestCase


def fake_executor(prompt: str, text: str) -> str:
    if "Follow the requested output format exactly" in prompt:
        return text.upper()
    return text.lower()


if __name__ == "__main__":
    result = benchmark(
        "Convert input to uppercase.",
        [TestCase("hello", "HELLO"), TestCase("world", "WORLD")],
        [TestCase("example", "EXAMPLE"), TestCase("test", "TEST")],
        fake_executor,
    )
    print(result["held_out"])
