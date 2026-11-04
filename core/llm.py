"""OpenAI model adapter, usage accounting, and LLM-powered prompt mutations."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, response):
        self.calls += 1
        if response.usage:
            self.input_tokens += response.usage.prompt_tokens or 0
            self.output_tokens += response.usage.completion_tokens or 0

    def to_dict(self):
        return vars(self).copy()


class OpenAIAdapter:
    def __init__(self, model="gpt-4o-mini", api_key=None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY to run live optimization.")
        self.client = OpenAI(api_key=key, timeout=30.0, max_retries=2)
        self.model = model
        self.usage = Usage()

    def complete(self, system, user):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self.usage.record(response)
        return (response.choices[0].message.content or "").strip()

    def execute(self, prompt, test_input):
        return self.complete(prompt, test_input)

    def mutate(self, prompt, count=3):
        if count < 1 or count > 5:
            raise ValueError("count must be between 1 and 5")
        system = (
            "You are a prompt optimization agent. Rewrite an instruction into distinct, "
            "clear variants while preserving the original task, safety constraints, and "
            "output requirements. Do not include answers to example inputs. "
            "Return ONLY a JSON array of strings."
        )
        raw = self.complete(system, f"Generate {count} alternative prompts:\n{prompt}")
        try:
            # Accept JSON fenced output, but not arbitrary prose.
            clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            variants = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError("Mutation model returned invalid JSON") from exc
        if not isinstance(variants, list) or any(not isinstance(p, str) for p in variants):
            raise ValueError("Mutation model must return a JSON array of strings")
        unique = []
        for variant in variants:
            candidate = variant.strip()
            if candidate and candidate != prompt.strip() and candidate not in unique:
                unique.append(candidate)
        if not unique:
            raise ValueError("Mutation model returned no distinct candidates")
        return unique[:count]


def make_executor(model="gpt-4o-mini", temperature=0.0):
    """Backward-compatible executor factory (temperature retained for API compatibility)."""
    return OpenAIAdapter(model=model).execute
