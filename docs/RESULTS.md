# Benchmark results — fill only after a real run

| Field | Measured value |
|---|---|
| Date (UTC) | Not yet measured |
| Model | Not yet measured |
| Dataset | Synthetic support-ticket classification (36 train / 12 held-out) |
| Generations / variants | Not yet measured |
| Mutation strategy | Not yet measured |
| Metric | normalized |
| Baseline held-out score | Not yet measured |
| Selected held-out score | Not yet measured |
| Delta (percentage points) | Not yet measured |
| API calls | Not yet measured |
| Input / output tokens | Not yet measured |
| Number of repeated trials | Not yet measured |

## Interpretation

Do not claim an accuracy improvement without recorded experiment output. The earlier uppercase smoke test achieved 100% baseline and 100% optimized accuracy on two held-out examples (0 percentage-point improvement). That verifies basic end-to-end behavior, not search effectiveness.

For the ticket dataset, training and held-out messages use related templates, so even a positive score delta is limited evidence. Repeat trials and use a separately authored, template-disjoint test set for a stronger claim.
