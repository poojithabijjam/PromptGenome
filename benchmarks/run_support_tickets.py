"""Run the labeled support-ticket benchmark with an OpenAI API key.

Usage: python -m benchmarks.run_support_tickets --generations 2 --variants 3
No network requests occur until this command is explicitly invoked.
"""
import argparse
import json
from pathlib import Path
from core.llm import OpenAIAdapter
from core.optimizer import TestCase
from core.pipeline import PromptGenomePipeline

HERE = Path(__file__).resolve().parent
TASK = "Classify the customer message into exactly one lowercase label: billing, technical, account, or other. Return the label only, with no punctuation or explanation."

def load_cases(filename):
    data = json.loads((HERE / filename).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Expected nonempty JSON array")
    cases = [TestCase(item["input"], item["expected"]) for item in data]
    if any(case.expected not in {"billing", "technical", "account", "other"} for case in cases):
        raise ValueError("Invalid label in benchmark")
    return cases

def run(model="gpt-4o-mini", generations=2, variants=3, mutation="llm"):
    train = load_cases("support_tickets_train.json")
    test = load_cases("support_tickets_test.json")
    adapter = OpenAIAdapter(model=model)
    mutator = (lambda prompt: adapter.mutate(prompt, min(variants, 5))) if mutation == "llm" else None
    result = PromptGenomePipeline(adapter.execute, mutator=mutator, metric="normalized").run(
        TASK, train, test, generations=generations, variants=variants
    )
    result["usage"] = adapter.usage.to_dict()
    result["configuration"] = {"model": model, "generations": generations,
        "variants": variants, "mutation": mutation, "metric": "normalized"}
    return result

def main():
    parser = argparse.ArgumentParser(description="Run a real, billable support-ticket benchmark")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--mutation", choices=["llm", "deterministic"], default="llm")
    parser.add_argument("--output", default="support_tickets_result.json")
    args = parser.parse_args()
    result = run(args.model, args.generations, args.variants, args.mutation)
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    scores = result["held_out"]
    print(f"Baseline: {scores['baseline_accuracy']:.1%} | Optimized: {scores['optimized_accuracy']:.1%}")
    print(f"Delta: {scores['improvement_percentage_points']:+.1f} pp | API calls: {result['usage']['calls']}")
    print(f"Saved experiment to {output}")

if __name__ == "__main__":
    main()
