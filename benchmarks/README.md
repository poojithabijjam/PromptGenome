# Support ticket classification benchmark

This is a **synthetic, labeled** 48-case dataset: 36 optimization cases and 12 held-out cases. The four classes are `billing`, `technical`, `account`, and `other`. Messages use distinct subjects and templates; the held-out set contains different subjects, but **shares templates** with training, so it is not an independent distribution-shift test.

The benchmark deliberately uses a simple initial task. **No improvement is guaranteed.** Selection uses training scores only; the held-out scores are computed after selecting the winner. This dataset is a demonstration, not a statistically representative real-world benchmark.

To run live (billable OpenAI calls):

```bash
export OPENAI_API_KEY="your-key"
python -m benchmarks.run_support_tickets --generations 2 --variants 3 --mutation llm
```

The JSON result is written locally. Do not commit sensitive outputs or API credentials. Report model, parameters, data split, score, calls and tokens. For credible comparisons, repeat runs and test on independently authored examples. Do not tune prompts based on held-out results.
