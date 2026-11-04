"""Execute candidate instructions against labeled examples."""
from collections.abc import Callable


class ExecutionAgent:
    def __init__(self, executor: Callable[[str, str], str]):
        self.executor = executor

    def run(self, prompt: str, input_text: str) -> str:
        output = self.executor(prompt, input_text)
        if not isinstance(output, str):
            raise TypeError("executor must return a string")
        return output
