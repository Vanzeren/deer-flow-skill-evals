# Skill-based Agent Evaluation Framework Design

**Goal:** Build a trace-level evaluation harness for skill-based agents on top of Inspect AI, so evals can judge whether a loaded skill changed agent behavior in the expected direction instead of only grading the final answer.

**Status:** Design spec for implementation planning.

**Primary integration point:** `backend/skill_eval/` as a backend-local evaluation package, with Inspect tasks under `backend/evals/` and data cases under `backend/cases/`.

---

## Problem

Inspect AI provides useful eval primitives: datasets, tasks, solvers, scorers, sandboxes, and eval logs. It does not know DeerFlow skills, skill activation, tool-call policy, LangGraph traces, or whether a skill was merely loaded versus actually used.

The framework must add a stable skill-eval layer that can answer these questions:

1. Was the expected skill loaded into the agent runtime?
2. Did the agent behavior show that the skill was used?
3. Were required tools called?
4. Were forbidden tools avoided?
5. Were tool arguments and tool order consistent with the skill?
6. Was the trace complete enough to trust the result?
7. Was the final answer correct?
8. Did the with-skill run improve over a baseline run without skills?

The core value is trace-level behavior evaluation, not simple final-answer grading.

---

## Design Principles

- Inspect is the eval runner, not the owner of skill semantics.
- The solver executes the agent and writes standardized trace metadata.
- Scorers evaluate behavior from `AgentTrace`, not raw LangChain, LangGraph, or DeerFlow messages.
- `Sample.target` remains the final-answer reference.
- `Sample.metadata["case"]` carries behavior expectations.
- `state.output.completion` stores the final answer.
- `state.metadata["agent_trace"]` stores the standardized runtime trace.
- The assertion engine is pure Python and testable without Inspect.
- Deterministic rule scorers and LLM-as-judge scorers are separate.
- Single-run scoring and baseline comparison are separate.
- `skill.loaded` and `skill.used` are distinct states.
- The trace schema is the stable contract. Runtime adapters may change; scorers should not.

---

## Architecture

```text
SkillEvalCase JSONL
        ↓
dataset_loader.py
        ↓
Inspect Sample
  target = final-answer reference
  metadata["case"] = behavior expectations
        ↓
Inspect Task
        ↓
skill_agent_solver()
        ↓
AgentRunner
  - MockAgentRunner
  - DeerFlowAgentRunner
  - LangChainAgentRunner
        ↓
AgentRunResult
        ↓
AgentTrace
        ↓
Scorers
  - trace_integrity_scorer
  - skill_assertion_scorer
  - tool_call_scorer
  - output_rule_scorer
  - performance_scorer
  - optional model-graded scorers
        ↓
Inspect Eval Log
        ↓
report.py / comparison.py
```

### Layer Responsibilities

| Layer | Responsibility |
|---|---|
| `SkillEvalCase` | Data-driven user input, target answer, expected skills, assertions, tags, difficulty. |
| Inspect dataset | Converts JSONL cases into Inspect `Sample` objects. |
| Inspect task | Wires dataset, solver, scorers, and sandbox. |
| Custom solver | Calls the configured agent runner and stores final answer plus standardized trace. |
| Agent runner | Executes a concrete runtime: mock, DeerFlow, LangChain, or future agents. |
| Runtime adapter | Converts runtime-native messages/events into `AgentTrace`. |
| `AgentTrace` | Stable representation of messages, tool calls, skill invocation, steps, errors, latency, tokens. |
| Assertion engine | Converts declarative assertions into pass/fail results. |
| Inspect scorer | Converts assertion results into Inspect `Score`. |
| Report module | Aggregates scores, failures, assertion pass rates, and skill pass rates. |
| Comparison module | Aligns baseline and with-skill runs by case id and reports impact. |

---

## Proposed File Layout

```text
backend/skill_eval/
  __init__.py
  case_schema.py
  trace_schema.py
  agent_runner.py
  assertion_engine.py
  dataset_loader.py
  inspect_solver.py
  inspect_scorer.py
  report.py
  comparison.py
  adapters/
    __init__.py
    mock.py
    deerflow.py
    langchain.py

backend/evals/
  skills_eval.py
  baseline_eval.py
  with_skill_eval.py

backend/cases/
  no_write_todos.jsonl
  gcp_skills.jsonl

backend/tests/skill_eval/
  test_assertion_engine.py
  test_dataset_loader.py
  test_trace_integrity_scorer.py
  test_skill_assertion_scorer.py
  test_mock_eval.py
```

`backend/skill_eval/` is intentionally separate from `packages/harness/deerflow/`. The eval harness can import DeerFlow for adapters, but DeerFlow core should not depend on the eval harness.

---

## Core Data Model

### `SkillAssertionSpec`

`SkillAssertionSpec` is the declarative assertion language for behavior checks.

```python
from typing import Literal

from pydantic import BaseModel, Field


AssertionName = Literal[
    "skill_loaded",
    "skill_used",
    "skill_not_used",
    "tool_called",
    "tool_not_called",
    "tool_args_contains",
    "tool_args_match",
    "tool_call_order",
    "tool_error_absent",
    "output_contains",
    "output_not_contains",
    "regex_match",
    "json_valid",
    "success_is_true",
    "trace_complete",
    "latency_under",
    "tokens_under",
    "tool_count_under",
    "max_steps_under",
    "no_unexpected_clarification",
]


class SkillAssertionSpec(BaseModel):
    name: AssertionName
    target: str | None = None
    threshold: int | float | None = None
    message: str | None = None
```

### `SkillEvalCase`

```python
class SkillEvalCase(BaseModel):
    id: str
    input: str
    target: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    candidate_skills: list[str] = Field(default_factory=list)
    assertions: list[SkillAssertionSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["smoke", "normal", "hard"] = "normal"
```

Field semantics:

- `input`: User task sent to the agent.
- `target`: Final-answer reference used by final-answer scorers.
- `required_skills`: Skills expected to be used for this case.
- `candidate_skills`: Skills the solver may load for this case.
- `assertions`: Trace-level and output-level behavior checks.
- `tags`: Filtering and reporting dimensions such as `tool-use`, `negative`, `skill-compliance`.
- `difficulty`: Eval tier for smoke, normal, and hard suites.

### `AgentTrace`

```python
from typing import Any

from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None


class SkillInvocation(BaseModel):
    name: str
    path: str | None = None
    loaded: bool = False
    used: bool = False
    trigger_reason: str | None = None


class AgentTrace(BaseModel):
    input: str
    final_answer: str
    success: bool
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    skill_invocations: list[SkillInvocation] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
```

`SkillInvocation.loaded` means the skill was loaded into the agent context or runtime. `SkillInvocation.used` means the agent behavior showed evidence of that skill's workflow, policy, or constraints. A skill can be loaded but unused.

---

## Agent Runner Interface

`agent_runner.py` owns the runtime-independent interface.

```python
from typing import Protocol

from pydantic import BaseModel

from skill_eval.trace_schema import AgentTrace


class AgentRunResult(BaseModel):
    final_answer: str
    success: bool
    trace: AgentTrace


class AgentRunner(Protocol):
    async def run(
        self,
        user_input: str,
        skills: list[str],
        sandbox: str | None = None,
    ) -> AgentRunResult:
        ...


async def run_agent(
    user_input: str,
    skills: list[str],
    sandbox: str | None = None,
    runner: AgentRunner | None = None,
) -> AgentRunResult:
    if runner is None:
        from skill_eval.adapters.mock import MockAgentRunner

        runner = MockAgentRunner()

    return await runner.run(
        user_input=user_input,
        skills=skills,
        sandbox=sandbox,
    )
```

### Mock Runner

The MVP uses `MockAgentRunner` to prove the Inspect → solver → trace → scorer chain without a real agent dependency.

```python
from skill_eval.agent_runner import AgentRunResult
from skill_eval.trace_schema import AgentTrace, AgentToolCall, SkillInvocation


class MockAgentRunner:
    async def run(
        self,
        user_input: str,
        skills: list[str],
        sandbox: str | None = None,
    ) -> AgentRunResult:
        tool_calls = []
        final_answer = "Mock answer."

        if "cloud run" in user_input.lower():
            final_answer = "Use gcloud run deploy to deploy a Cloud Run service."

        if "write todos" in user_input.lower() and "do not" not in user_input.lower():
            tool_calls.append(
                AgentToolCall(
                    name="write_todos",
                    args={"items": ["mock plan"]},
                )
            )

        trace = AgentTrace(
            input=user_input,
            final_answer=final_answer,
            success=True,
            tool_calls=tool_calls,
            skill_invocations=[
                SkillInvocation(
                    name=skill,
                    path=skill,
                    loaded=True,
                    used=bool(skills),
                    trigger_reason="mock runner loaded candidate skill",
                )
                for skill in skills
            ],
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": final_answer},
            ],
            steps=[
                {"type": "mock_start"},
                {"type": "mock_final_answer"},
            ],
        )

        return AgentRunResult(
            final_answer=final_answer,
            success=True,
            trace=trace,
        )
```

### DeerFlow Runner

The DeerFlow adapter should be phase 2, after the MVP is green.

Potential DeerFlow sources:

| `AgentTrace` field | DeerFlow source |
|---|---|
| `messages` | `DeerFlowClient.stream()` events or Gateway run messages. |
| `tool_calls` | AIMessage tool calls plus ToolMessage outputs. |
| `skill_invocations.loaded` | Skill activation middleware state, slash activation, loaded skill context, or solver-selected candidate skills. |
| `skill_invocations.used` | Explicit skill activation plus behavior evidence from assertions or adapter-specific markers. |
| `steps` | Subagent events, run event store events, or streaming custom events. |
| `latency_ms` | Runner wall-clock timing. |
| `input_tokens`, `output_tokens` | Token usage metadata or thread token usage API. |
| `errors` | Tool errors, model errors, run failures. |

The scorer must not read these DeerFlow-native structures directly.

---

## Inspect Solver

`inspect_solver.py` is the adapter between Inspect `TaskState` and the agent runner.

```python
from inspect_ai.solver import Generate, TaskState, solver

from skill_eval.agent_runner import AgentRunner, run_agent


@solver
def skill_agent_solver(
    agent_runner: AgentRunner | None = None,
    skills: list[str] | None = None,
    sandbox: str | None = "docker",
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case = state.metadata.get("case", {})
        selected_skills = skills if skills is not None else case.get("candidate_skills", [])

        result = await run_agent(
            user_input=state.input_text,
            skills=selected_skills,
            sandbox=sandbox,
            runner=agent_runner,
        )

        state.output.completion = result.final_answer
        state.metadata["agent_trace"] = result.trace.model_dump()
        state.metadata["success"] = result.success

        return state

    return solve
```

The solver usually does not call `await generate(state)` because the real model call happens inside the custom agent runtime. `generate(state)` is only appropriate when Inspect itself owns the model invocation.

---

## Assertion Engine

`assertion_engine.py` is pure Python. It accepts `SkillAssertionSpec`, `AgentTrace`, and final answer text, then returns an `AssertionResult`.

```python
from typing import Any

from pydantic import BaseModel, Field

from skill_eval.case_schema import SkillAssertionSpec
from skill_eval.trace_schema import AgentTrace


class AssertionResult(BaseModel):
    name: str
    passed: bool
    explanation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def evaluate_assertion(
    assertion: SkillAssertionSpec,
    trace: AgentTrace,
    final_answer: str,
) -> AssertionResult:
    if assertion.name == "tool_called":
        return evaluate_tool_called(assertion, trace)
    if assertion.name == "tool_not_called":
        return evaluate_tool_not_called(assertion, trace)
    if assertion.name == "output_contains":
        return evaluate_output_contains(assertion, final_answer)
    if assertion.name == "success_is_true":
        return evaluate_success_is_true(assertion, trace)
    if assertion.name == "trace_complete":
        return evaluate_trace_complete(assertion, trace)
    return AssertionResult(
        name=assertion.name,
        passed=False,
        explanation=f"Unsupported assertion in MVP: {assertion.name}",
    )
```

### MVP Assertions

The first implementation must support only these assertions:

| Assertion | Rule |
|---|---|
| `tool_called` | Any `trace.tool_calls[*].name == target`. |
| `tool_not_called` | No `trace.tool_calls[*].name == target`. |
| `output_contains` | `target in final_answer`. |
| `success_is_true` | `trace.success is True`. |
| `trace_complete` | Required trace fields are present and no fatal errors. |

`trace_complete` passes when:

- `trace.input` is non-empty.
- `trace.final_answer` is non-empty.
- `trace.success` is not missing.
- At least one of `trace.messages`, `trace.tool_calls`, or `trace.steps` is non-empty.
- `trace.errors` contains no fatal error.

### Later Assertions

Phase 2 and 3 can add:

```text
skill_loaded
skill_used
skill_not_used
tool_args_contains
tool_args_match
tool_call_order
tool_error_absent
output_not_contains
regex_match
json_valid
latency_under
tokens_under
tool_count_under
max_steps_under
no_unexpected_clarification
```

---

## Inspect Scorers

The scorer layer should be broader than the MVP implementation. The design has three scorer classes:

1. Rule-based scorers: deterministic checks over `AgentTrace` and final answer text.
2. Model-graded scorers: LLM-as-judge checks for semantic quality and complex skill compliance.
3. Comparison scorers: offline comparison across baseline and with-skill eval logs.

MVP implements only the two required rule-based scorers because they prove the end-to-end harness. The framework API must still reserve clear slots for the full scorer set below so the first implementation does not collapse everything into one giant scorer.

### Scorer Catalog

| Scorer | Class | Input | Output | Implementation phase |
|---|---|---|---|---|
| `trace_integrity_scorer()` | Rule-based | `state.metadata["agent_trace"]` | `Score` with trace validity explanation | MVP |
| `skill_assertion_scorer()` | Rule-based | `state.metadata["case"]`, `state.metadata["agent_trace"]` | `Score` with per-assertion results | MVP |
| `tool_call_scorer()` | Rule-based | `AgentTrace.tool_calls` plus case assertions | Tool behavior score | Phase 2 |
| `skill_usage_scorer()` | Rule-based | `AgentTrace.skill_invocations`, required/candidate skills | Skill loading/usage score | Phase 2 |
| `output_rule_scorer()` | Rule-based | final answer plus output assertions | Output format/content score | Phase 2 |
| `performance_scorer()` | Rule-based | latency, tokens, steps, tool count | Cost/loop score | Phase 2 |
| `answer_quality_scorer()` | Model-graded | final answer and `Sample.target` | Semantic answer score | Phase 3 |
| `skill_compliance_judge_scorer()` | Model-graded | task, skill summary, final answer, trace | Skill compliance score | Phase 3 |
| `clarification_judge_scorer()` | Model-graded | task, final answer, trace | Clarification necessity score | Phase 3 |
| `baseline_comparison_scorer()` | Comparison | paired baseline/with-skill records | Improvement/regression classification | Phase 4 |
| `skill_impact_scorer()` | Comparison | paired traces and score metadata | Behavior-impact score | Phase 4 |

### Rule-based Scorers

#### `trace_integrity_scorer()`

This scorer validates that eval data is trustworthy before behavior scoring. It should be included in every task.

```python
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from skill_eval.trace_schema import AgentTrace


@scorer
def trace_integrity_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if "agent_trace" not in state.metadata:
            return Score(value=0.0, explanation="Missing agent_trace in state.metadata.")

        try:
            trace = AgentTrace.model_validate(state.metadata["agent_trace"])
        except Exception as exc:
            return Score(value=0.0, explanation=f"Invalid AgentTrace: {exc}")

        failures = []

        if not trace.input:
            failures.append("Trace input is empty.")
        if not trace.final_answer:
            failures.append("Trace final_answer is empty.")
        if trace.success is None:
            failures.append("Trace success is missing.")
        if not trace.messages and not trace.tool_calls and not trace.steps:
            failures.append("Trace has no messages, tool calls, or steps.")
        if any("fatal" in error.lower() for error in trace.errors):
            failures.append(f"Trace contains fatal errors: {trace.errors}")

        return Score(
            value=0.0 if failures else 1.0,
            explanation="\n".join(failures) if failures else "Trace is complete.",
        )

    return score
```

#### `skill_assertion_scorer()`

This is the main business scorer. It interprets `SkillEvalCase.assertions` by delegating to `assertion_engine.evaluate_assertion()`.

```python
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from skill_eval.assertion_engine import evaluate_assertion
from skill_eval.case_schema import SkillEvalCase
from skill_eval.trace_schema import AgentTrace


@scorer
def skill_assertion_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if "case" not in state.metadata:
            return Score(value=0.0, explanation="Missing case metadata.")
        if "agent_trace" not in state.metadata:
            return Score(value=0.0, explanation="Missing agent_trace metadata.")

        case = SkillEvalCase.model_validate(state.metadata["case"])
        trace = AgentTrace.model_validate(state.metadata["agent_trace"])

        results = [
            evaluate_assertion(assertion, trace, trace.final_answer)
            for assertion in case.assertions
        ]

        failures = [result for result in results if not result.passed]

        return Score(
            value=0.0 if failures else 1.0,
            explanation="\n".join(result.explanation for result in failures)
            if failures
            else "All assertions passed.",
            metadata={
                "case_id": case.id,
                "assertion_results": [result.model_dump() for result in results],
            },
        )

    return score
```

#### `tool_call_scorer()`

Purpose: isolate tool behavior from general skill assertions, so tool-call diagnostics can be reported even when final-answer quality passes.

Supported assertions:

```text
tool_called
tool_not_called
tool_count_under
tool_args_contains
tool_args_match
tool_call_order
tool_error_absent
```

MVP-equivalent minimum for this scorer in Phase 2:

```python
TOOL_ASSERTIONS = {
    "tool_called",
    "tool_not_called",
    "tool_count_under",
    "tool_error_absent",
}
```

Behavior:

- Filter `case.assertions` to tool-related assertions.
- Evaluate them with the assertion engine.
- Return `Score(1.0)` only if every tool assertion passes.
- Include `tool_calls`, failing assertion names, and tool count in metadata.

#### `skill_usage_scorer()`

Purpose: make skill activation and usage visible as first-class metrics instead of burying them inside generic assertions.

Supported checks:

```text
skill_loaded
skill_used
skill_not_used
required_skill_used
unexpected_skill_not_used
skill_trigger_reason_present
```

Behavior:

- For every `case.required_skills`, require a matching `SkillInvocation` with `loaded=True`.
- For behavior-sensitive cases, require `used=True` either via explicit assertion or scorer option.
- Flag skills that are used but not in `required_skills` or `candidate_skills`.
- Report loaded-only skills separately from used skills.

Initial `used` evidence policy:

```text
used=True if the adapter records explicit runtime evidence, such as slash activation, skill middleware activation, or a runtime marker.
used=False if the skill was only passed in `candidate_skills` and no behavior evidence exists.
```

Later policy can add assertion-derived evidence, such as a skill-specific forbidden-tool assertion passing only in the with-skill run.

#### `output_rule_scorer()`

Purpose: deterministic final-answer checks that should not require a judge model.

Supported assertions:

```text
output_contains
output_not_contains
regex_match
json_valid
markdown_contains_section
command_contains
```

Behavior:

- Evaluate only output assertions.
- Preserve final-answer semantic quality for `answer_quality_scorer()`.
- Use this scorer for CI gates where exact content or structure matters.

#### `performance_scorer()`

Purpose: detect skill-induced loops, excessive tool usage, or excessive context expansion.

Supported assertions:

```text
latency_under
tokens_under
tool_count_under
max_steps_under
```

Behavior:

- Fail missing metrics for threshold assertions.
- Report observed latency, token total, tool count, and step count in metadata.
- Keep performance failure separate from behavior failure.

### Model-graded Scorers

Model-graded scorers should be optional and never the only source of truth for CI-critical behavior constraints.

#### `answer_quality_scorer()`

Use Inspect's `model_graded_qa()` when `Sample.target` is a semantic answer reference.

Purpose:

- Grade final-answer correctness.
- Ignore skill usage and tool behavior.

#### `skill_compliance_judge_scorer()`

Purpose: judge complex skill compliance when the skill document contains nuanced process rules that are hard to encode as deterministic assertions.

Inputs:

```text
user task
skill summary or selected skill document sections
final answer
tool calls
step summaries
skill invocation records
```

Required judge output:

```text
GRADE: PASS or FAIL
REASON: concise explanation
VIOLATIONS: bullet list, empty when PASS
```

Judgment dimensions:

- Did the agent follow required workflow steps?
- Did the agent avoid prohibited behavior?
- Did the agent use the skill only within its intended scope?
- Did the agent skip a required safety or verification step?
- Did tool behavior match the skill's instructions?

#### `clarification_judge_scorer()`

Purpose: determine whether a clarification request was necessary.

Rule-based prefilter:

```text
Can you clarify
Could you provide more details
I need more information
请澄清
能否提供更多信息
```

Judge inputs:

```text
user task
final answer
trace messages
skill instruction about clarification behavior
```

Pass condition:

- The agent did not ask for clarification; or
- The task was genuinely underspecified and clarification was required.

### Comparison Scorers

Comparison scorers operate after two eval runs. They are not normal single-sample Inspect scorers unless Inspect log-pair integration is added later.

#### `baseline_comparison_scorer()`

Purpose: classify each case as improved, regressed, unchanged pass, or unchanged fail.

Inputs:

```text
baseline_score
with_skill_score
baseline_trace
with_skill_trace
baseline_assertion_results
with_skill_assertion_results
```

Outputs:

```text
improved
regressed
unchanged_pass
unchanged_fail
behavior_changed
```

#### `skill_impact_scorer()`

Purpose: judge whether the skill changed behavior, independent of whether the final result passed.

Signals:

```text
tool-call sequence changed
forbidden tool count changed
required tool appeared
final answer changed
skill-specific output pattern appeared
latency/token/tool-count changed
assertion failure type changed
```

Output:

```text
impact_score: 0.0 to 1.0
impact_type: positive | negative | neutral | inconclusive
behavior_changed: bool
behavior_improved: bool
```

### Scorer Groups

Recommended groupings:

```python
# General skill eval
scorer=[
    trace_integrity_scorer(),
    skill_assertion_scorer(),
    model_graded_qa(),
]

# Tool-use eval
scorer=[
    trace_integrity_scorer(),
    tool_call_scorer(),
    output_rule_scorer(),
]

# Skill activation eval
scorer=[
    trace_integrity_scorer(),
    skill_usage_scorer(),
    skill_assertion_scorer(),
]

# Complex compliance eval
scorer=[
    trace_integrity_scorer(),
    skill_assertion_scorer(),
    skill_compliance_judge_scorer(),
]

# Clarification behavior eval
scorer=[
    trace_integrity_scorer(),
    skill_assertion_scorer(),
    clarification_judge_scorer(),
]

# Performance eval
scorer=[
    trace_integrity_scorer(),
    performance_scorer(),
]
```

MVP still implements only `trace_integrity_scorer()` and `skill_assertion_scorer()`. That is an implementation-scope decision, not the full scorer design.

---

## Dataset Loader

`dataset_loader.py` loads JSONL and converts each row into an Inspect `Sample`.

```python
from pathlib import Path

from inspect_ai.dataset import Sample

from skill_eval.case_schema import SkillEvalCase


def load_skill_cases(
    path: str,
    tags: list[str] | None = None,
    difficulty: str | None = None,
    required_skill: str | None = None,
) -> list[Sample]:
    samples: list[Sample] = []

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            case = SkillEvalCase.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Invalid skill eval case at {path}:{line_number}: {exc}") from exc

        if tags and not set(tags).issubset(set(case.tags)):
            continue
        if difficulty and case.difficulty != difficulty:
            continue
        if required_skill and required_skill not in case.required_skills:
            continue

        samples.append(
            Sample(
                id=case.id,
                input=case.input,
                target=case.target or "",
                metadata={"case": case.model_dump()},
            )
        )

    return samples
```

---

## Demo Cases

`backend/cases/no_write_todos.jsonl`:

```json
{"id":"no-write-todos-001","input":"Create a plan for this task, but do not write todos.","target":"The agent should answer without calling write_todos.","required_skills":["no-write-todos-in-pro"],"candidate_skills":["skills/no-write-todos-in-pro"],"assertions":[{"name":"tool_not_called","target":"write_todos"},{"name":"success_is_true"},{"name":"trace_complete"}],"tags":["negative-tool","skill-compliance"],"difficulty":"smoke"}
```

`backend/cases/gcp_skills.jsonl`:

```json
{"id":"cloud-run-001","input":"How do I deploy a Cloud Run service?","target":"The answer should mention gcloud run deploy.","required_skills":["gcp-cloud-run"],"candidate_skills":["skills/gcp-cloud-run"],"assertions":[{"name":"output_contains","target":"gcloud run deploy"},{"name":"success_is_true"},{"name":"trace_complete"}],"tags":["gcp","cloud-run","answer-quality"],"difficulty":"smoke"}
```

---

## Inspect Tasks

`backend/evals/skills_eval.py`:

```python
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.scorer import model_graded_qa

from skill_eval.dataset_loader import load_skill_cases
from skill_eval.inspect_scorer import skill_assertion_scorer, trace_integrity_scorer
from skill_eval.inspect_solver import skill_agent_solver


@task
def skills_eval(
    case_file: str = "cases/gcp_skills.jsonl",
    skills_folder: str = "skills",
    use_model_graded_qa: bool = False,
):
    samples = load_skill_cases(case_file)

    skill_files = (Path.cwd() / skills_folder).rglob("SKILL.md")
    all_skills = [str(skill_file.parent) for skill_file in skill_files]

    scorers = [
        trace_integrity_scorer(),
        skill_assertion_scorer(),
    ]

    if use_model_graded_qa:
        scorers.append(model_graded_qa())

    return Task(
        dataset=samples,
        solver=skill_agent_solver(skills=all_skills, sandbox="docker"),
        scorer=scorers,
        sandbox="docker",
    )
```

`baseline_eval.py` uses `skills=[]`.

`with_skill_eval.py` uses `skills=None`, allowing the solver to read `case.candidate_skills`.

---

## Baseline vs With-skill Comparison

Single with-skill evals prove whether a skilled agent passed. They do not prove the skill caused the improvement. The framework must support paired runs:

```text
Run A: baseline agent, skills=[]
Run B: with-skill agent, skills=case.candidate_skills
```

The comparison module aligns records by case id.

### Comparison Outcomes

| Baseline | With skill | Outcome |
|---|---|---|
| fail | pass | `improved` |
| pass | fail | `regressed` |
| pass | pass | `unchanged_pass` |
| fail | fail | `unchanged_fail` |

### Behavior Change Signals

`behavior_changed` is true if any of these changed:

- final answer text
- tool-call sequence
- forbidden tool count
- assertion failure types
- skill invocation records
- step count
- token count or latency beyond configured tolerance

`behavior_changed` is not the same as `behavior_improved`. A skill can change behavior negatively.

### Comparison Data Model

```python
from pydantic import BaseModel, Field

from skill_eval.trace_schema import AgentTrace


class CaseComparison(BaseModel):
    case_id: str
    baseline_passed: bool
    with_skill_passed: bool
    outcome: str
    behavior_changed: bool
    explanation: str
    metadata: dict = Field(default_factory=dict)
```

### Comparison Metrics

Report-level metrics:

```text
baseline_pass_rate
with_skill_pass_rate
delta_pass_rate
improved_cases
regressed_cases
unchanged_pass_cases
unchanged_fail_cases
behavior_changed_cases
forbidden_tool_reduction
average_tool_count_delta
average_latency_delta
average_token_delta
most_common_baseline_failures
most_common_with_skill_failures
```

---

## Report Design

### Single Eval Report

```text
Skill Eval Summary

Total cases: 20
Passed: 16
Failed: 4
Pass rate: 80.0%

By assertion:
- tool_not_called: 19 / 20
- skill_used: 14 / 20
- trace_complete: 20 / 20
- success_is_true: 17 / 20

Failed cases:
- no-write-todos-003
  Reason: Forbidden tool `write_todos` was called.
- gcp-cloud-run-002
  Reason: Expected output to contain `gcloud run deploy`.
```

### Baseline Comparison Report

```text
Baseline vs With-skill Summary

Total cases: 20

Baseline pass rate: 55%
With-skill pass rate: 80%
Delta: +25%

Improved:
- no-write-todos-001
- no-write-todos-002

Regressed:
- cloud-run-003

Behavior changes:
- forbidden tool calls reduced from 6 to 1
- average tool count changed from 3.2 to 2.1
- average latency changed from 4200ms to 5100ms
```

---

## MVP Scope

MVP includes:

1. `SkillEvalCase` schema.
2. `AgentTrace` schema.
3. JSONL loader.
4. Mock agent runner.
5. Inspect solver.
6. Assertion engine with:
   - `tool_called`
   - `tool_not_called`
   - `output_contains`
   - `success_is_true`
   - `trace_complete`
7. Inspect scorers:
   - `skill_assertion_scorer()`
   - `trace_integrity_scorer()`
8. Demo JSONL case files.
9. Demo Inspect task.
10. Unit tests for schema, loader, assertion engine, scorers, and mock eval flow.

MVP excludes:

- LLM-as-judge scorers.
- Automatic case generation.
- Full DeerFlow adapter.
- Langfuse or Phoenix integration.
- Advanced baseline report.
- Full skill impact scoring.
- JSONPath matching for tool args.
- Full sandbox fixture design.

---

## Implementation Phases

### Phase 1: MVP Harness

Build the minimal inspect-compatible harness with mock agent execution.

Deliverables:

- `backend/skill_eval/case_schema.py`
- `backend/skill_eval/trace_schema.py`
- `backend/skill_eval/agent_runner.py`
- `backend/skill_eval/adapters/mock.py`
- `backend/skill_eval/assertion_engine.py`
- `backend/skill_eval/dataset_loader.py`
- `backend/skill_eval/inspect_solver.py`
- `backend/skill_eval/inspect_scorer.py`
- demo cases
- demo task
- unit tests

Validation:

- Unit tests pass.
- Mock eval produces a trace.
- `skill_assertion_scorer()` returns pass/fail from case assertions.

### Phase 2: Full Deterministic Assertions

Add all rule assertions from `AssertionName`.

Validation:

- Unit tests for each assertion pass and fail paths.
- Assertion result metadata includes useful debug details.

### Phase 3: DeerFlow Adapter

Add `DeerFlowAgentRunner` and `DeerFlowTraceAdapter`.

Validation:

- One smoke eval runs against DeerFlow embedded client or Gateway.
- Tool calls and messages appear in `AgentTrace`.
- Skill activation produces `SkillInvocation` records.

### Phase 4: Reports and Baseline Comparison

Add `report.py` and `comparison.py`.

Validation:

- Single-run report aggregates assertion failures.
- Paired baseline/with-skill report identifies improved and regressed cases.

### Phase 5: Model-graded Scorers

Add optional judge-based scorers.

Validation:

- Judge prompts return structured grades.
- Rule scorers remain usable without judge model configuration.

---

## Unit Test Plan

### `test_assertion_engine.py`

Required tests:

- `tool_called` passes when expected tool appears.
- `tool_called` fails when expected tool is absent.
- `tool_not_called` passes when forbidden tool is absent.
- `tool_not_called` fails when forbidden tool appears.
- `output_contains` passes and fails.
- `success_is_true` passes and fails.
- `trace_complete` passes for a valid trace.
- `trace_complete` fails for empty input.
- `trace_complete` fails for empty final answer.
- `trace_complete` fails with no messages, tool calls, or steps.
- `trace_complete` fails on fatal errors.

### `test_dataset_loader.py`

Required tests:

- Valid JSONL loads into `Sample` objects.
- Blank lines are ignored.
- Invalid JSONL raises a `ValueError` with file and line information.
- `Sample.target` equals final-answer target.
- `Sample.metadata["case"]` preserves behavior expectations.
- Tag filtering works.
- Difficulty filtering works.
- Required-skill filtering works.

### `test_trace_integrity_scorer.py`

Required tests:

- Missing `agent_trace` fails.
- Invalid `AgentTrace` payload fails.
- Empty trace fails.
- Valid trace passes.
- Fatal error fails.

### `test_skill_assertion_scorer.py`

Required tests:

- All assertions pass returns `Score(value=1.0)`.
- Any failed assertion returns `Score(value=0.0)`.
- Score metadata includes `assertion_results`.
- Missing case metadata fails.
- Missing trace metadata fails.

### `test_mock_eval.py`

Required tests:

- `MockAgentRunner` returns `AgentRunResult` with `AgentTrace`.
- Solver writes `state.output.completion`.
- Solver writes `state.metadata["agent_trace"]`.
- Skill assertion scorer can score a mock trace.

---

## Risks and Decisions

### Inspect dependency placement

Decision: add Inspect as a backend dev dependency when implementation starts.

Reason: this is an eval/test harness, not production runtime. It should not increase the core DeerFlow package runtime surface unless later promoted.

### Skill `used` semantics

Decision: `used` starts as adapter-provided evidence, then later can be strengthened by assertion-derived evidence or model-graded compliance.

Reason: `loaded` is usually observable directly; `used` is behavioral and may require runtime markers, trace evidence, and case-specific rules.

### DeerFlow coupling

Decision: keep DeerFlow coupling inside `skill_eval/adapters/deerflow.py`.

Reason: scorers must remain reusable for LangChain or other agents.

### Baseline comparison timing

Decision: implement comparison after single-run scoring is reliable.

Reason: comparison depends on stable per-case score metadata and trace records.

---

## Acceptance Criteria

The MVP is complete when:

1. JSONL cases load into Inspect samples.
2. A mock runner can execute cases through the Inspect solver.
3. The solver writes final answer and standardized trace metadata.
4. `trace_integrity_scorer()` validates trace completeness.
5. `skill_assertion_scorer()` evaluates the MVP assertions.
6. Unit tests cover pass and fail paths for loader, assertion engine, scorers, and mock flow.
7. Demo cases show both output and tool-behavior checks.

The full framework is complete when:

1. DeerFlow runs can be converted into `AgentTrace`.
2. Rule scorers cover all listed assertion types.
3. Optional model-graded scorers can evaluate semantic skill compliance.
4. Baseline and with-skill eval logs can be compared by case id.
5. Reports show pass rates, assertion failure types, skill-level aggregates, improved cases, and regressions.
