"""Offline deterministic demo; no API key or external services needed."""
from core.optimizer import TestCase, optimize


def executor(prompt: str, text: str) -> str:
    if "Follow the requested output format exactly" in prompt:
        return text.upper()
    return text.lower()


if __name__ == "__main__":
    result = optimize(
        "Convert input text to uppercase.",
        [TestCase("hello", "HELLO"), TestCase("world", "WORLD")],
        executor,
    )
    print("Baseline score:", result["baseline"]["score"])
    print("Best score:", result["best"]["score"])
    print("Candidates evaluated:", len(result["history"]))
