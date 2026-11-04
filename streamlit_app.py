"""PromptGenome end-to-end Streamlit interface."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.pipeline import PromptGenomePipeline
from core.llm import OpenAIAdapter
from core.optimizer import TestCase
from benchmarks.run_support_tickets import TASK as SUPPORT_TASK

load_dotenv()
st.set_page_config(page_title="PromptGenome", page_icon="🧬", layout="wide")

ROOT = Path(__file__).resolve().parent
PRESETS = {
    "Support ticket classification (48 cases)": {
        "task": SUPPORT_TASK,
        "train": ROOT / "benchmarks" / "support_tickets_train.json",
        "test": ROOT / "benchmarks" / "support_tickets_test.json",
        "metric": "normalized",
    },
    "Uppercase conversion (smoke test)": {
        "task": "Convert the input text to uppercase.",
        "train": [{"input": "hello", "expected": "HELLO"}, {"input": "world", "expected": "WORLD"}],
        "test": [{"input": "example", "expected": "EXAMPLE"}, {"input": "testing", "expected": "TESTING"}],
        "metric": "exact",
    },
}

def preset_examples(source):
    if isinstance(source, Path):
        return json.loads(source.read_text(encoding="utf-8"))
    return source

def apply_preset(name):
    preset = PRESETS[name]
    st.session_state["task"] = preset["task"]
    st.session_state["train_json"] = json.dumps(preset_examples(preset["train"]), indent=2)
    st.session_state["test_json"] = json.dumps(preset_examples(preset["test"]), indent=2)
    st.session_state["metric"] = preset["metric"]
    st.session_state.pop("result", None)

if "task" not in st.session_state:
    apply_preset("Support ticket classification (48 cases)")

st.title("🧬 PromptGenome")
st.caption("Five composable agents: generate → mutate → execute → evaluate → select; benchmark on held-out examples.")

with st.expander("How PromptGenome works · Architecture", expanded=False):
    st.image(str(ROOT / "assets" / "architecture.svg"), caption="Training data drives candidate selection; held-out cases are evaluated afterward.")
    st.markdown("**Generate** baseline → **Mutate** candidates → **Execute** with OpenAI → **Evaluate** on training labels → **Select** only if training score improves. The held-out set never influences selection.")
    st.caption("The five roles are testable Python components, not autonomous framework agents. OpenAI is used for candidate execution and optionally for mutation.")

with st.sidebar:
    st.header("Experiment configuration")
    benchmark = st.selectbox("Benchmark preset", list(PRESETS), key="benchmark",
                             on_change=lambda: apply_preset(st.session_state["benchmark"]))
    st.caption("Changing the preset loads its labeled examples. You can edit them before running.")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    generations = st.slider("Maximum generations", 0, 5, 1)
    variants = st.slider("Variants per generation", 1, 5, 1)
    strategy = st.selectbox("Mutation strategy", ["Deterministic", "LLM-generated"])
    metric = st.selectbox("Evaluation metric", ["exact", "normalized", "json", "similarity"], key="metric")
    st.warning("Live model calls are billable. The call estimate below is an upper bound; the pipeline can stop early.")

task = st.text_area("Task instruction", key="task", height=100)
train_json = st.text_area("Optimization examples (JSON)", key="train_json", height=180)
test_json = st.text_area("Held-out examples (JSON; different inputs)", key="test_json", height=180)

def parse_cases(raw):
    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("Cases must be a nonempty JSON array.")
    if any(not isinstance(x, dict) or not isinstance(x.get("input"), str)
           or not isinstance(x.get("expected"), str) for x in data):
        raise ValueError("Every case needs string input and expected fields.")
    return [TestCase(x["input"], x["expected"]) for x in data]

try:
    training, testing = parse_cases(train_json), parse_cases(test_json)
    if {x.input for x in training} & {x.input for x in testing}:
        raise ValueError("Training and held-out inputs must not overlap.")
    # One baseline + up to generations × variants candidates, and two held-out evaluations.
    max_calls = (1 + generations * variants) * len(training) + 2 * len(testing)
    if strategy == "LLM-generated":
        max_calls += generations
    st.info(f"Dataset: {len(training)} training / {len(testing)} held-out cases. "
            f"Estimated maximum: **{max_calls} API calls** (including mutation calls). "
            "Actual usage may be lower.")
    valid = True
except (ValueError, json.JSONDecodeError) as exc:
    st.error(f"Dataset validation: {exc}")
    valid = False

if st.button("Run optimization and held-out benchmark", type="primary", disabled=not valid):
    try:
        adapter = OpenAIAdapter(model=model)
        mutator = (lambda p: adapter.mutate(p, variants)) if strategy == "LLM-generated" else None
        with st.spinner("Running model evaluations..."):
            result = PromptGenomePipeline(adapter.execute, mutator=mutator, metric=metric).run(
                task, training, testing, generations, variants
            )
        result["usage"] = adapter.usage.to_dict()
        result["configuration"] = {"benchmark": benchmark, "model": model, "generations": generations,
                                    "variants": variants, "mutation": strategy, "metric": metric,
                                    "training_count": len(training), "held_out_count": len(testing)}
        st.session_state["result"] = result
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Model request failed ({type(exc).__name__}). Check API access and retry.")

result = st.session_state.get("result")
if result:
    run = result["optimization"]
    scores = result["held_out"]
    usage = result["usage"]
    a, b, c, d = st.columns(4)
    a.metric("Baseline held-out score", f"{scores['baseline_accuracy']:.1%}")
    b.metric("Optimized held-out score", f"{scores['optimized_accuracy']:.1%}",
             delta=f"{scores['improvement_percentage_points']:+.1f} pp")
    c.metric("Candidates evaluated", len(run["history"]))
    d.metric("API calls", usage["calls"])
    st.caption(f"Token usage: {usage['input_tokens']} input / {usage['output_tokens']} output. Pricing varies by model; dollar cost is not estimated.")
    if scores["improvement_percentage_points"] > 0:
        st.success("The selected prompt improved held-out performance in this run. Repeat with independent cases before claiming generalization.")
    elif scores["improvement_percentage_points"] < 0:
        st.warning("The selected prompt performed worse on held-out cases. Training improvement does not guarantee generalization.")
    else:
        st.info("No held-out improvement in this run. This can happen when the baseline is already strong or mutations do not help.")
    st.subheader("Selected prompt")
    st.code(run["best"]["prompt"], language="text")
    frame = pd.DataFrame(run["history"])
    st.subheader("Training accuracy by generation")
    st.line_chart(frame.groupby("generation")["score"].max())
    st.subheader("Candidate lineage")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button("Export experiment JSON", json.dumps(result, indent=2),
                       file_name="promptgenome-experiment.json", mime="application/json")
