# PromptGenome demo walkthrough

Use the [restricted-access Streamlit app](https://promptgenome.streamlit.app/) with an authorized account.

## Five-minute demonstration

1. Expand **How PromptGenome works · Architecture** to explain the five roles and training-versus-held-out separation.
2. Choose **Uppercase conversion (smoke test)**, 1 generation, 1 variant, deterministic mutation, exact metric. This is a small billable integration check, not evidence of optimization improvement.
3. Choose **Support ticket classification (48 cases)**. The UI fills the task and 36 training / 12 held-out cases automatically. Select normalized scoring.
4. Review the **maximum API calls** estimate before running. For 1 generation / 1 variant, the upper bound is 96 calls; an LLM mutation adds up to one mutation call. More variants/generations can substantially increase usage.
5. Run the experiment. Show the selected prompt, training-score chart, candidate lineage, and held-out accuracy comparison.
6. Download the experiment JSON. Use its actual scores and usage in your portfolio. A 0 pp or negative improvement is a legitimate outcome.

## Explaining the results

**Baseline held-out score:** Original prompt accuracy on cases not used for selection.

**Optimized held-out score:** Selected prompt accuracy on those same held-out cases.

**Delta:** Optimized minus baseline, in percentage points. A positive result on one small synthetic split is not proof of generalization.

**Candidate lineage:** Parent IDs and generations show the search path.

**Usage:** Counts OpenAI calls and tokens; these are not dollar-cost estimates.

## Evidence checklist before publishing

- [ ] GitHub Actions is green for the current feature-branch commit.
- [ ] Streamlit app loads and displays the architecture expander.
- [ ] Both benchmark presets populate task and JSON fields.
- [ ] Small smoke test completes without errors.
- [ ] Support-ticket benchmark completes and exports JSON.
- [ ] Capture screenshots of the architecture, preset selector, result cards, lineage, and usage **from the actual app**.
- [ ] Record model, date, parameters, train/test sizes, measured scores, API calls, and tokens.
- [ ] Repeat the experiment and test with independently authored holdout messages before claiming reliable improvement.

Do not publish API keys, private emails, or sensitive ticket text in screenshots or experiment exports.
