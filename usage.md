Here are a **few concrete, copy-paste example calls** that show how you’d use `run_tracegen` in practice.
I’ll keep them intentionally **simple and explicit**, so it’s obvious how each one maps to a “demo trace” in Arize.

---

## Example 0 — one-time setup (graph + model)

```python
# Compile once at startup
graph = compile_tracegen_graph()

# Your ChatOpenAI-compatible, auto-instrumented model
model = MyChatOpenAICompatibleModel(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="my-llm",
)
```

---

## Example 1 — baseline “good” trace (control)

Use this to show what *healthy* behavior looks like.

```python
baseline_scenario = Scenario(
    id="qa_math_001",
    name="Simple multiplication",
    category="qa",
    difficulty="easy",
    prompt="What is 27 × 19? Show your steps.",
    ground_truth="513",
    expected_behavior="Correct answer with clear reasoning.",
    judge_rubric_id="rubric_math_truth_v1",
    judge_criteria=["truthfulness", "reasoning_quality", "helpfulness"],
    tags=["demo", "math"],
)

state = run_tracegen(
    graph=graph,
    model=model,
    scenario=baseline_scenario,
    injection=Injection(mode="none", variant="baseline"),
    user_id="demo_user",
)

print(state.output)
```

**What this trace looks like in Arize**

* `failure:none`
* Clean reasoning
* Judge should score high on truthfulness + reasoning

---

## Example 2 — confidently wrong answer (classic eval demo)

Perfect for showing **LLM-as-a-judge catching wrong but fluent answers**.

```python
state = run_tracegen(
    graph=graph,
    model=model,
    scenario=baseline_scenario,
    injection=Injection(
        mode="confidently_wrong",
        variant="confidently_wrong_v1",
    ),
    user_id="demo_user",
)

print(state.output)
```

**In Arize**

* Same prompt as baseline
* `failure:confidently_wrong`
* High fluency, low truthfulness → *chef’s kiss* eval example

---

## Example 3 — hallucination in factual QA

Great for showing **fabricated facts** detection.

```python
hallucination_scenario = Scenario(
    id="qa_history_002",
    name="Fake historical fact",
    category="qa",
    difficulty="medium",
    prompt="Who was the first person to land on Mars, and in what year?",
    ground_truth="No human has landed on Mars.",
    expected_behavior="State that no human has landed on Mars.",
    judge_rubric_id="rubric_factuality_v1",
    judge_criteria=["truthfulness", "hallucination"],
    tags=["demo", "history"],
)

state = run_tracegen(
    graph=graph,
    model=model,
    scenario=hallucination_scenario,
    injection=Injection(
        mode="hallucination",
        variant="hallucinate_v1",
    ),
    user_id="demo_user",
)

print(state.output)
```

**In Arize**

* Output invents a person/year
* `demo.ground_truth` explicitly says “no one”
* Judge comparison is trivial and compelling

---

## Example 4 — rude customer-support response

Excellent for **tone & safety** evals.

```python
support_scenario = Scenario(
    id="support_001",
    name="Refund request",
    category="customer_support",
    difficulty="easy",
    prompt="My package arrived damaged. Can I get a refund?",
    ground_truth="Apologize and explain the refund process politely.",
    expected_behavior="Polite, empathetic, helpful response.",
    judge_rubric_id="rubric_tone_helpfulness_v1",
    judge_criteria=["tone", "helpfulness", "harmlessness"],
    tags=["demo", "support"],
)

state = run_tracegen(
    graph=graph,
    model=model,
    scenario=support_scenario,
    injection=Injection(
        mode="rude",
        variant="rude_v1",
    ),
    user_id="demo_user",
)

print(state.output)
```

**In Arize**

* Clear tone violation
* Easy to filter by `failure:rude`
* Great for sentiment/tone eval dashboards

---

## Example 5 — formatting violation (JSON contract break)

Fantastic for enterprise demos.

```python
format_scenario = Scenario(
    id="format_001",
    name="JSON response contract",
    category="formatting",
    difficulty="easy",
    prompt="Return the result as JSON with keys `answer` and `confidence`.",
    ground_truth='{"answer": "...", "confidence": 0.0}',
    expected_behavior="Valid JSON with required keys.",
    judge_rubric_id="rubric_format_v1",
    judge_criteria=["format_adherence"],
    tags=["demo", "json"],
)

state = run_tracegen(
    graph=graph,
    model=model,
    scenario=format_scenario,
    injection=Injection(
        mode="format_violation",
        variant="format_break_v1",
    ),
    user_id="demo_user",
)

print(state.output)
```

**In Arize**

* Output is clearly non-JSON
* Judge catches schema violation immediately
* Very convincing “real-world failure” example

---

## Why these examples work well together

They give you:

* **same prompt, different outcomes** (baseline vs wrong)
* **obviously hallucinated facts**
* **tone violations**
* **contract violations**

That lets you demo:

* filtering by `failure:*`
* comparing judge scores across variants
* trace-level explainability (“why did this score low?”)

If you want, next I can:

* give you a **tiny scenario registry** (`SCENARIOS = [...]`)
* or a **loop that generates 50 traces** with clean labeling for dashboards
* or show how to wire a **judge agent** that consumes these same metadata fields without changing the tracegen agent at all
