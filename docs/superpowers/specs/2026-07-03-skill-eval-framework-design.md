# Skill-based Agent Evaluation Framework Design

**Goal:** Build a trace-level evaluation harness for skill-based agents on top of Inspect AI, so evals can judge whether a loaded skill changed agent behavior in the expected direction instead of only grading the final answer.

**Status:** Design spec for implementation planning.

**Primary integration point:** `backend/skill_eval/` as a backend-local evaluation package, with Inspect tasks under `backend/evals/` and data cases under `backend/cases/`.

---

## Problem

Inspect AI provides useful eval primitives: datasets, tasks, solvers, scorers, sandboxes, and eval logs. It does not know DeerFlow skills, skill activation, tool-call policy, LangGraph traces, or whether a skill was merely loaded versus actually used.

The framework must add a stable skill-eval layer that can answer these questions:

1. Was the expected skill loaded into the agent runtime?
2. Did the agent select the skill by slash activation or by reading `SKILL.md`, and did later behavior apply the skill's rules?
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
- Generic scorers evaluate behavior from `AgentTrace`, not raw DeerFlow or LangGraph messages; raw runtime data remains adapter input and debug evidence.
- `Sample.target` remains the final-answer reference.
- `Sample.metadata["case"]` carries behavior expectations.
- `state.output.completion` stores the final answer.
- `state.metadata["agent_trace"]` stores the standardized runtime trace.
- The assertion engine is pure Python and testable without Inspect.
- Deterministic rule scorers and LLM-as-judge scorers are separate.
- Single-run scoring and baseline comparison are separate.
- `skill.loaded`, `skill.used`, and `skill.applied` are distinct states: loaded means available in context, used means selected/activated, and applied means behavior complied with the skill.
- The trace schema is the stable evaluation contract. Runtime adapters may change; generic scorers should not. Adapter-specific diagnostic scorers may inspect raw runtime data, but they must be optional and isolated under adapter modules.

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
        ↓
AgentRunResult
        ↓
AgentTrace
        ↓
Scorers
  - trace_integrity_scorer
  - skill_assertion_scorer
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
| Agent runner | Executes the concrete runtime for this project: mock for MVP tests, then DeerFlow for real evals. |
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
    "skill_applied",
    "skill_not_applied",
    "tool_called",
    "tool_not_called",
    "tool_args_contains",
    "tool_args_match",
    "tool_call_order",
    "tool_error_absent",
    "tool_result_contains",
    "tool_result_match",
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
- `required_skills`: Skills expected to be selected and applied for this case.
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
    applied: bool | None = None
    trigger_reason: str | None = None
    evidence: list[str] = Field(default_factory=list)


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
    runtime: str | None = None
    raw_trace_ref: str | None = None
```

`SkillInvocation.loaded` means the skill content entered the agent context. `SkillInvocation.used` means the agent selected or activated the skill, either through slash activation or a successful `read_file` of that skill's `SKILL.md`. `SkillInvocation.applied` means later behavior complied with the skill's rules; it is `None` when the adapter cannot decide and is usually derived from skill-specific assertions or baseline comparison. A skill can be loaded and used but not applied. `AgentTrace.messages` and `AgentTrace.steps` are normalized evaluation evidence, not raw DeerFlow/LangGraph objects. `raw_trace_ref` points to the original Inspect log, DeerFlow run id, artifact, or adapter-specific trace file when deeper debugging needs the raw runtime payload.

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
        raise NotImplementedError


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
                    applied=None,
                    trigger_reason="mock runner loaded candidate skill",
                    evidence=["mock runner selected candidate skill"],
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
| `runtime` | Constant such as `deerflow-embedded` or `deerflow-gateway`. |
| `raw_trace_ref` | DeerFlow run id, Inspect log location, or persisted raw trace artifact. |
| `messages` | Normalized message records derived from `DeerFlowClient.stream()` events or Gateway run messages. |
| `tool_calls` | AIMessage tool calls plus ToolMessage outputs. |
| `skill_invocations.loaded` | Skill content entered context through slash activation, successful skill `read_file`, loaded skill context, or solver-selected mock skill. |
| `skill_invocations.used` | Explicit slash activation or successful `read_file` of `/mnt/skills/<skill>/SKILL.md`; in DeerFlow, reading `SKILL.md` counts as selecting/using the skill. |
| `skill_invocations.applied` | Behavior-level conclusion from skill-specific assertions or comparison; default `None` unless there is explicit evidence. |
| `steps` | Normalized subagent events, run event store events, or streaming custom events. |
| `latency_ms` | Runner wall-clock timing. |
| `input_tokens`, `output_tokens` | Token usage metadata or thread token usage API. |
| `errors` | Tool errors, model errors, run failures. |

Generic scorers must not parse DeerFlow-native structures directly. If a DeerFlow-only diagnostic check needs raw payloads, it belongs in an optional adapter-specific module such as `skill_eval/adapters/deerflow_scorer.py`, not in the generic scorer set.

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

Phase 2 can add the remaining deterministic assertions. These stay inside the assertion engine and are executed by `skill_assertion_scorer()` rather than by adding one scorer per concern:

```text
skill_loaded
skill_used
skill_not_used
skill_applied
skill_not_applied
tool_args_contains
tool_args_match
tool_call_order
tool_error_absent
tool_result_contains
tool_result_match
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

The scorer layer is intentionally narrow. The framework has two core Inspect scorers:

1. `trace_integrity_scorer()` — validates the standardized trace is present and evaluable.
2. `skill_assertion_scorer()` — executes every declarative `SkillAssertionSpec` through the pure assertion engine.

Tool behavior, output rules, performance limits, skill loaded/used/applied checks, and clarification checks are assertion types, not separate scorers. This keeps the evaluation semantics in one tested place: `assertion_engine.py`.

Baseline and with-skill impact analysis is also not an Inspect scorer in the MVP design. It is an offline comparison/report module that reads two eval logs or result sets and compares aligned case ids.

### Scorer Catalog

| Scorer | Class | Input | Output | Implementation phase |
|---|---|---|---|---|
| `trace_integrity_scorer()` | Rule-based Inspect scorer | `state.metadata["agent_trace"]` | `Score` with trace validity explanation | MVP |
| `skill_assertion_scorer()` | Rule-based Inspect scorer | `state.metadata["case"]`, `state.metadata["agent_trace"]` | `Score` with per-assertion results | MVP |

Optional future judge scoring should be added only when deterministic assertions cannot express the behavior being evaluated. Final-answer semantic grading can use Inspect's `model_graded_qa()` directly in task definitions; it does not need a custom wrapper in this harness.

### Assertion Coverage by Concern

The assertion engine, not separate scorers, owns these checks:

| Concern | Assertion names |
|---|---|
| Skill activation and usage | `skill_loaded`, `skill_used`, `skill_not_used`, `skill_applied`, `skill_not_applied` |
| Tool invocation | `tool_called`, `tool_not_called`, `tool_call_order`, `tool_count_under` |
| Tool input | `tool_args_contains`, `tool_args_match` |
| Tool output and failure | `tool_result_contains`, `tool_result_match`, `tool_error_absent` |
| Final answer rules | `output_contains`, `output_not_contains`, `regex_match`, `json_valid` |
| Runtime cost and loop guard | `latency_under`, `tokens_under`, `max_steps_under` |
| Run status and trace validity | `success_is_true`, `trace_complete` |
| Clarification behavior | `no_unexpected_clarification` |

This means a tool-use case and a performance case can both use the same scorer:

```python
scorer=[
    trace_integrity_scorer(),
    skill_assertion_scorer(),
]
```

The difference lives in the case data:

```json
{"name": "tool_args_contains", "target": "gcloud run deploy"}
{"name": "tool_result_contains", "target": "Deployment successful"}
{"name": "tokens_under", "threshold": 4000}
```

### `trace_integrity_scorer()`

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

### `skill_assertion_scorer()`

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

### Model-graded scoring

Use Inspect's built-in `model_graded_qa()` in task definitions when `Sample.target` is a semantic final-answer reference.

Do not add custom LLM-as-judge scorers in the MVP. Add a custom judge later only when deterministic assertions cannot express a skill compliance requirement. That future judge should read the same `AgentTrace` and case metadata, not raw DeerFlow internals.

### Baseline comparison

Baseline comparison is handled by `comparison.py` and `report.py`, not by an Inspect scorer. It compares baseline and with-skill outputs after both eval runs finish.

Comparison outputs should include:

```text
improved
regressed
unchanged_pass
unchanged_fail
behavior_changed
behavior_improved
impact_type: positive | negative | neutral | inconclusive
```

Comparison signals include:

```text
tool-call sequence changed
forbidden tool count changed
required tool appeared
tool arguments changed
tool results improved or errors reduced
final answer changed
skill-specific output pattern appeared
latency/token/tool-count changed
assertion failure type changed
```

Recommended single-run scorer setup:

```python
scorer=[
    trace_integrity_scorer(),
    skill_assertion_scorer(),
]
```

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
- skill_applied: 14 / 20
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

Add the remaining deterministic assertion types to `assertion_engine.py`, including skill usage, tool arguments, tool results, output rules, cost limits, and clarification behavior. Do not add new Inspect scorers for these concerns.

Validation:

- Unit tests cover pass and fail paths for every assertion type.
- Assertion result metadata includes useful debug details, especially matched tool calls, tool result snippets, observed thresholds, and failed skill invocation records.

### Phase 3: DeerFlow Adapter

Add `DeerFlowAgentRunner` and `DeerFlowTraceAdapter`.

Validation:

- One smoke eval runs against DeerFlow embedded client or Gateway.
- Tool calls and messages appear in `AgentTrace`.
- Skill activation produces `SkillInvocation` records.

### Phase 4: Reports and Baseline Comparison

Add `report.py` and `comparison.py`. These modules are offline result processors, not Inspect scorers.

Validation:

- Single-run report aggregates assertion failures by case, skill, tag, and assertion type.
- Paired baseline/with-skill report identifies improved, regressed, unchanged-pass, and unchanged-fail cases.
- Comparison output distinguishes `behavior_changed` from `behavior_improved`.
- Tool impact signals include required tool added, forbidden tool reduced, tool arguments changed, tool results improved, and tool errors reduced.

### Phase 5: Optional Model-Graded Evaluation

Use Inspect's `model_graded_qa()` for final-answer semantic grading when a case has a meaningful `target`. Add a custom skill-compliance judge only after deterministic assertions are insufficient for a real case family.

Validation:

- Rule scorers remain usable with no judge model configured.
- Judge-based checks read `AgentTrace` and case metadata, not raw DeerFlow internals.

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

### Skill `used` and `applied` semantics

Decision: `used` means the agent selected the skill, either by slash activation or by successfully reading `/mnt/skills/<skill>/SKILL.md`. `applied` means later behavior complied with the skill's rules and may be derived from skill-specific assertions or baseline comparison.

Reason: in DeerFlow, reading `SKILL.md` is the concrete action that selects/uses a skill. Keeping `applied` separate prevents false confidence when the agent reads a skill but then violates its instructions.

### DeerFlow coupling

Decision: keep DeerFlow coupling inside `skill_eval/adapters/deerflow.py`.

Reason: scorers must remain reusable across the mock runner and the real DeerFlow adapter, and future runtimes can be added only if needed.

### AgentTrace versus raw runtime messages

Decision: generic scorers read `AgentTrace`; raw DeerFlow and LangGraph payloads are preserved by reference through `raw_trace_ref` and may be used only by optional adapter-specific diagnostics.

Reason: `AgentTrace` keeps rule scorers portable between the mock runner and DeerFlow, and makes baseline comparison possible without coupling scoring semantics to DeerFlow event internals. Raw runtime payloads are still needed for debugging and adapter validation, but making them the primary scorer interface would couple evaluation semantics to one runtime.

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
2. `skill_assertion_scorer()` covers all listed deterministic assertion types.
3. Optional model-graded evaluation can assess semantic skill compliance when deterministic assertions are insufficient.
4. Baseline and with-skill eval logs can be compared by case id.
5. Reports show pass rates, assertion failure types, skill-level aggregates, improved cases, and regressions.
