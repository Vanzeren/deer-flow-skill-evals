# Skill Eval Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MVP Inspect AI skill-evaluation harness for trace-level behavior evaluation with mock agent execution.

**Architecture:** Add a backend-local `skill_eval` package that owns stable schemas, a pure assertion engine, JSONL case loading, an agent runner protocol, Inspect solver/scorer adapters, and demo Inspect tasks. MVP runs through a mock runner first; DeerFlow runtime adaptation stays outside this implementation plan except for stable trace fields that support it.

**Tech Stack:** Python 3.12, Pydantic v2, Inspect AI (`inspect-ai` package), pytest, pytest-asyncio, uv, ruff.

## Global Constraints

- Generic scorers MUST evaluate `AgentTrace`, not raw DeerFlow or LangGraph messages.
- Raw runtime data MUST remain adapter input or debug evidence through `AgentTrace.raw_trace_ref`.
- `Sample.target` MUST hold final-answer reference text.
- `Sample.metadata["case"]` MUST hold behavior expectations.
- `state.output.completion` MUST hold the agent final answer.
- `state.metadata["agent_trace"]` MUST hold `AgentTrace.model_dump()`.
- Assertion engine MUST be pure Python and testable without Inspect.
- MVP assertions are exactly `tool_called`, `tool_not_called`, `output_contains`, `success_is_true`, and `trace_complete`.
- MVP scorers are exactly `skill_assertion_scorer()` and `trace_integrity_scorer()`.
- Do not add DeerFlow runtime adapter in this MVP.
- Do not add LLM-as-judge scorers in this MVP.
- The only planned real runtime adapter is DeerFlow; the mock runner exists only as the MVP test seam.
- Keep all implementation under `backend/skill_eval/`, `backend/evals/`, `backend/cases/`, and `backend/tests/skill_eval/`.

---

## File Structure

Create:

- `backend/skill_eval/__init__.py` — package marker and public module docstring.
- `backend/skill_eval/case_schema.py` — `SkillAssertionSpec`, `SkillEvalCase`, and `AssertionName`.
- `backend/skill_eval/trace_schema.py` — `AgentToolCall`, `SkillInvocation`, and `AgentTrace`.
- `backend/skill_eval/agent_runner.py` — `AgentRunResult`, `AgentRunner` protocol, and `run_agent()` dispatcher.
- `backend/skill_eval/adapters/__init__.py` — adapter package marker.
- `backend/skill_eval/adapters/mock.py` — deterministic mock runner for MVP evals.
- `backend/skill_eval/assertion_engine.py` — pure assertion evaluation functions and `AssertionResult`.
- `backend/skill_eval/dataset_loader.py` — JSONL case loader returning Inspect `Sample` objects.
- `backend/skill_eval/inspect_solver.py` — `skill_agent_solver()`.
- `backend/skill_eval/inspect_scorer.py` — `trace_integrity_scorer()` and `skill_assertion_scorer()`.
- `backend/evals/__init__.py` — eval package marker.
- `backend/evals/skills_eval.py` — demo Inspect task with optional model-graded QA.
- `backend/evals/baseline_eval.py` — baseline task using `skills=[]`.
- `backend/evals/with_skill_eval.py` — with-skill task using case `candidate_skills`.
- `backend/cases/no_write_todos.jsonl` — demo negative tool case.
- `backend/cases/gcp_skills.jsonl` — demo output-content case.
- `backend/tests/skill_eval/test_assertion_engine.py` — assertion engine unit tests.
- `backend/tests/skill_eval/test_dataset_loader.py` — JSONL loader tests.
- `backend/tests/skill_eval/test_trace_integrity_scorer.py` — scorer trace-integrity tests.
- `backend/tests/skill_eval/test_skill_assertion_scorer.py` — assertion scorer tests.
- `backend/tests/skill_eval/test_mock_eval.py` — mock runner and solver integration tests.

Modify:

- `backend/pyproject.toml` — add `inspect-ai` to the `dev` dependency group.
- `backend/AGENTS.md` — add a short development note for the backend-local skill eval harness after the implementation is verified.

---

### Task 1: Add Inspect AI development dependency

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`

**Interfaces:**
- Consumes: existing uv dependency group layout.
- Produces: importable `inspect_ai` package for tests and eval tasks.

- [ ] **Step 1: Add the dependency**

Run from `backend/`:

```bash
uv add --dev inspect-ai
```

Expected: `pyproject.toml` dev dependency group includes `inspect-ai`, and `uv.lock` changes.

- [ ] **Step 2: Verify import works**

Run:

```bash
uv run python -c "import inspect_ai; print(inspect_ai.__name__)"
```

Expected output:

```text
inspect_ai
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "test: add inspect ai dev dependency"
```

---

### Task 2: Add core case and trace schemas

**Files:**
- Create: `backend/skill_eval/__init__.py`
- Create: `backend/skill_eval/case_schema.py`
- Create: `backend/skill_eval/trace_schema.py`
- Test: `backend/tests/skill_eval/test_assertion_engine.py` receives schema smoke tests in this task.

**Interfaces:**
- Produces: `SkillAssertionSpec`, `SkillEvalCase`, `AgentToolCall`, `SkillInvocation`, `AgentTrace`.
- Later tasks import these exact classes.

- [ ] **Step 1: Write failing schema tests**

Create `backend/tests/skill_eval/test_assertion_engine.py` with schema smoke coverage first:

```python
import pytest
from pydantic import ValidationError

from skill_eval.case_schema import SkillAssertionSpec, SkillEvalCase
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


def test_skill_eval_case_defaults():
    case = SkillEvalCase(id="case-1", input="Do the task")

    assert case.target is None
    assert case.required_skills == []
    assert case.candidate_skills == []
    assert case.assertions == []
    assert case.tags == []
    assert case.difficulty == "normal"


def test_skill_assertion_rejects_unknown_name():
    with pytest.raises(ValidationError):
        SkillAssertionSpec(name="unknown_assertion")


def test_agent_trace_captures_normalized_evidence_and_raw_ref():
    trace = AgentTrace(
        input="Use the skill",
        final_answer="Done",
        success=True,
        tool_calls=[AgentToolCall(name="bash", args={"cmd": "pwd"})],
        skill_invocations=[SkillInvocation(name="demo", loaded=True, used=False)],
        messages=[{"role": "assistant", "content": "Done"}],
        steps=[{"type": "final"}],
        runtime="mock",
        raw_trace_ref="artifact://trace",
    )

    assert trace.runtime == "mock"
    assert trace.raw_trace_ref == "artifact://trace"
    assert trace.tool_calls[0].name == "bash"
    assert trace.skill_invocations[0].loaded is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'skill_eval'`.

- [ ] **Step 3: Create package marker**

Create `backend/skill_eval/__init__.py`:

```python
"""Trace-level skill evaluation harness for Inspect AI."""
```

- [ ] **Step 4: Implement case schema**

Create `backend/skill_eval/case_schema.py`:

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

- [ ] **Step 5: Implement trace schema**

Create `backend/skill_eval/trace_schema.py`:

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
    runtime: str | None = None
    raw_trace_ref: str | None = None
```

- [ ] **Step 6: Run schema tests**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -v
```

Expected: PASS for the three schema smoke tests.

- [ ] **Step 7: Commit**

```bash
git add skill_eval tests/skill_eval/test_assertion_engine.py
git commit -m "feat: add skill eval schemas"
```

---

### Task 3: Implement MVP assertion engine

**Files:**
- Create: `backend/skill_eval/assertion_engine.py`
- Modify: `backend/tests/skill_eval/test_assertion_engine.py`

**Interfaces:**
- Consumes: `SkillAssertionSpec`, `AgentTrace`.
- Produces: `AssertionResult` and `evaluate_assertion(assertion, trace, final_answer)`.

- [ ] **Step 1: Add failing assertion tests**

Append these tests to `backend/tests/skill_eval/test_assertion_engine.py`:

```python
from skill_eval.assertion_engine import evaluate_assertion


def _valid_trace(**overrides):
    data = {
        "input": "Do it",
        "final_answer": "Done with gcloud run deploy",
        "success": True,
        "messages": [{"role": "assistant", "content": "Done"}],
    }
    data.update(overrides)
    return AgentTrace(**data)


def test_tool_called_passes_when_expected_tool_appears():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_called", target="bash"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True
    assert "was called" in result.explanation


def test_tool_called_fails_when_expected_tool_is_absent():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="read_file")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_called", target="bash"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False
    assert "Expected tool `bash`" in result.explanation


def test_tool_not_called_passes_when_forbidden_tool_is_absent():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="read_file")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_not_called", target="write_todos"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_tool_not_called_fails_when_forbidden_tool_is_present():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="write_todos")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_not_called", target="write_todos"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False
    assert "Forbidden tool `write_todos` was called" in result.explanation


def test_output_contains_passes_and_fails():
    trace = _valid_trace(final_answer="Use gcloud run deploy")

    passing = evaluate_assertion(
        SkillAssertionSpec(name="output_contains", target="gcloud run deploy"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="output_contains", target="kubectl apply"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert failing.passed is False


def test_success_is_true_passes_and_fails():
    passing_trace = _valid_trace(success=True)
    failing_trace = _valid_trace(success=False)

    assert evaluate_assertion(SkillAssertionSpec(name="success_is_true"), passing_trace, passing_trace.final_answer).passed is True
    assert evaluate_assertion(SkillAssertionSpec(name="success_is_true"), failing_trace, failing_trace.final_answer).passed is False


def test_trace_complete_passes_for_valid_trace():
    trace = _valid_trace()

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is True


def test_trace_complete_fails_for_empty_input():
    trace = _valid_trace(input="")

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "trace.input is empty" in result.explanation


def test_trace_complete_fails_for_empty_final_answer():
    trace = _valid_trace(final_answer="")

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "trace.final_answer is empty" in result.explanation


def test_trace_complete_fails_without_evidence():
    trace = _valid_trace(messages=[], tool_calls=[], steps=[])

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "trace has no messages, tool_calls, or steps" in result.explanation


def test_trace_complete_fails_on_fatal_error():
    trace = _valid_trace(errors=["fatal: run crashed"])

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "fatal" in result.explanation
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'skill_eval.assertion_engine'`.

- [ ] **Step 3: Implement assertion engine**

Create `backend/skill_eval/assertion_engine.py`:

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


def evaluate_assertion(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
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
    return _fail(assertion, f"Unsupported assertion in MVP: {assertion.name}")


def evaluate_tool_called(assertion: SkillAssertionSpec, trace: AgentTrace) -> AssertionResult:
    called = any(call.name == assertion.target for call in trace.tool_calls)
    if called:
        return _pass(assertion, f"Tool `{assertion.target}` was called.")
    return _fail(assertion, f"Expected tool `{assertion.target}` to be called.")


def evaluate_tool_not_called(assertion: SkillAssertionSpec, trace: AgentTrace) -> AssertionResult:
    called = any(call.name == assertion.target for call in trace.tool_calls)
    if not called:
        return _pass(assertion, f"Tool `{assertion.target}` was not called.")
    return _fail(assertion, f"Forbidden tool `{assertion.target}` was called.")


def evaluate_output_contains(assertion: SkillAssertionSpec, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    if target in final_answer:
        return _pass(assertion, f"Output contained `{target}`.")
    return _fail(assertion, f"Expected output to contain `{target}`.")


def evaluate_success_is_true(assertion: SkillAssertionSpec, trace: AgentTrace) -> AssertionResult:
    if trace.success is True:
        return _pass(assertion, "Trace success is true.")
    return _fail(assertion, "Trace success is not true.")


def evaluate_trace_complete(assertion: SkillAssertionSpec, trace: AgentTrace) -> AssertionResult:
    failures: list[str] = []
    if not trace.input:
        failures.append("trace.input is empty")
    if not trace.final_answer:
        failures.append("trace.final_answer is empty")
    if trace.success is None:
        failures.append("trace.success is missing")
    if not trace.messages and not trace.tool_calls and not trace.steps:
        failures.append("trace has no messages, tool_calls, or steps")
    if any("fatal" in error.lower() for error in trace.errors):
        failures.append(f"trace contains fatal errors: {trace.errors}")

    if failures:
        return _fail(assertion, "; ".join(failures), failures=failures)
    return _pass(assertion, "Trace is complete.")


def _pass(assertion: SkillAssertionSpec, explanation: str, **metadata: Any) -> AssertionResult:
    return AssertionResult(name=assertion.name, passed=True, explanation=assertion.message or explanation, metadata=metadata)


def _fail(assertion: SkillAssertionSpec, explanation: str, **metadata: Any) -> AssertionResult:
    return AssertionResult(name=assertion.name, passed=False, explanation=assertion.message or explanation, metadata=metadata)
```

- [ ] **Step 4: Run assertion engine tests**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skill_eval/assertion_engine.py tests/skill_eval/test_assertion_engine.py
git commit -m "feat: add skill assertion engine"
```

---

### Task 4: Add JSONL dataset loader

**Files:**
- Create: `backend/skill_eval/dataset_loader.py`
- Create: `backend/tests/skill_eval/test_dataset_loader.py`

**Interfaces:**
- Consumes: `SkillEvalCase`.
- Produces: `load_skill_cases(path, tags=None, difficulty=None, required_skill=None) -> list[Sample]`.

- [ ] **Step 1: Write failing loader tests**

Create `backend/tests/skill_eval/test_dataset_loader.py`:

```python
import pytest

from skill_eval.dataset_loader import load_skill_cases


def test_load_skill_cases_preserves_target_and_case_metadata(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text(
        '{"id":"case-1","input":"Say hi","target":"hi","required_skills":["demo"],"candidate_skills":["skills/demo"],"assertions":[{"name":"output_contains","target":"hi"}],"tags":["smoke"],"difficulty":"smoke"}\n',
        encoding="utf-8",
    )

    samples = load_skill_cases(str(case_file))

    assert len(samples) == 1
    assert samples[0].id == "case-1"
    assert samples[0].input == "Say hi"
    assert samples[0].target == "hi"
    assert samples[0].metadata["case"]["assertions"][0]["name"] == "output_contains"


def test_load_skill_cases_ignores_blank_lines(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text('\n{"id":"case-1","input":"Say hi"}\n\n', encoding="utf-8")

    samples = load_skill_cases(str(case_file))

    assert len(samples) == 1


def test_load_skill_cases_reports_invalid_line(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text('{"id":"case-1","input":"ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_skill_cases(str(case_file))

    assert "cases.jsonl:2" in str(exc_info.value)


def test_load_skill_cases_filters_tags_difficulty_and_required_skill(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text(
        '\n'.join(
            [
                '{"id":"keep","input":"A","required_skills":["demo"],"tags":["tool-use","smoke"],"difficulty":"smoke"}',
                '{"id":"drop-tag","input":"B","required_skills":["demo"],"tags":["other"],"difficulty":"smoke"}',
                '{"id":"drop-difficulty","input":"C","required_skills":["demo"],"tags":["tool-use","smoke"],"difficulty":"hard"}',
                '{"id":"drop-skill","input":"D","required_skills":["other"],"tags":["tool-use","smoke"],"difficulty":"smoke"}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )

    samples = load_skill_cases(str(case_file), tags=["tool-use"], difficulty="smoke", required_skill="demo")

    assert [sample.id for sample in samples] == ["keep"]
```

- [ ] **Step 2: Run loader tests to verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_dataset_loader.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'skill_eval.dataset_loader'`.

- [ ] **Step 3: Implement dataset loader**

Create `backend/skill_eval/dataset_loader.py`:

```python
from pathlib import Path

from inspect_ai.dataset import Sample

from skill_eval.case_schema import SkillEvalCase


def load_skill_cases(path: str, tags: list[str] | None = None, difficulty: str | None = None, required_skill: str | None = None) -> list[Sample]:
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

- [ ] **Step 4: Run loader tests**

Run:

```bash
uv run pytest tests/skill_eval/test_dataset_loader.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skill_eval/dataset_loader.py tests/skill_eval/test_dataset_loader.py
git commit -m "feat: add skill case loader"
```

---

### Task 5: Add mock runner and runner protocol

**Files:**
- Create: `backend/skill_eval/agent_runner.py`
- Create: `backend/skill_eval/adapters/__init__.py`
- Create: `backend/skill_eval/adapters/mock.py`
- Create: `backend/tests/skill_eval/test_mock_eval.py`

**Interfaces:**
- Produces: `AgentRunResult`, `AgentRunner`, `run_agent()`, `MockAgentRunner`.
- Later solver task depends on `run_agent()`.

- [ ] **Step 1: Write failing mock runner tests**

Create `backend/tests/skill_eval/test_mock_eval.py`:

```python
import pytest

from skill_eval.adapters.mock import MockAgentRunner
from skill_eval.agent_runner import run_agent


@pytest.mark.asyncio
async def test_mock_runner_returns_agent_trace_for_cloud_run():
    runner = MockAgentRunner()

    result = await runner.run("How do I deploy a Cloud Run service?", skills=["skills/gcp-cloud-run"])

    assert result.success is True
    assert "gcloud run deploy" in result.final_answer
    assert result.trace.runtime == "mock"
    assert result.trace.skill_invocations[0].loaded is True
    assert result.trace.messages


@pytest.mark.asyncio
async def test_mock_runner_records_write_todos_when_not_forbidden():
    result = await run_agent("Please write todos for this task", skills=[], runner=MockAgentRunner())

    assert [call.name for call in result.trace.tool_calls] == ["write_todos"]


@pytest.mark.asyncio
async def test_mock_runner_avoids_write_todos_when_user_forbids_it():
    result = await run_agent("Create a plan, but do not write todos.", skills=["skills/no-write-todos-in-pro"], runner=MockAgentRunner())

    assert result.trace.tool_calls == []
    assert result.trace.skill_invocations[0].used is True
```

- [ ] **Step 2: Run mock tests to verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_mock_eval.py -v
```

Expected: FAIL with `ModuleNotFoundError` for the runner modules.

- [ ] **Step 3: Implement runner protocol**

Create `backend/skill_eval/agent_runner.py`:

```python
from typing import Protocol

from pydantic import BaseModel

from skill_eval.trace_schema import AgentTrace


class AgentRunResult(BaseModel):
    final_answer: str
    success: bool
    trace: AgentTrace


class AgentRunner(Protocol):
    async def run(self, user_input: str, skills: list[str], sandbox: str | None = None) -> AgentRunResult:
        raise NotImplementedError


async def run_agent(user_input: str, skills: list[str], sandbox: str | None = None, runner: AgentRunner | None = None) -> AgentRunResult:
    if runner is None:
        from skill_eval.adapters.mock import MockAgentRunner

        runner = MockAgentRunner()

    return await runner.run(user_input=user_input, skills=skills, sandbox=sandbox)
```

- [ ] **Step 4: Create adapter package marker**

Create `backend/skill_eval/adapters/__init__.py`:

```python
"""Agent runtime adapters for the skill evaluation harness."""
```

- [ ] **Step 5: Implement mock runner**

Create `backend/skill_eval/adapters/mock.py`:

```python
from skill_eval.agent_runner import AgentRunResult
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


class MockAgentRunner:
    async def run(self, user_input: str, skills: list[str], sandbox: str | None = None) -> AgentRunResult:
        tool_calls: list[AgentToolCall] = []
        final_answer = "Mock answer."
        lowered = user_input.lower()

        if "cloud run" in lowered:
            final_answer = "Use gcloud run deploy to deploy a Cloud Run service."

        if "write todos" in lowered and "do not" not in lowered:
            tool_calls.append(AgentToolCall(name="write_todos", args={"items": ["mock plan"]}))

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
            steps=[{"type": "mock_start"}, {"type": "mock_final_answer"}],
            runtime="mock",
        )

        return AgentRunResult(final_answer=final_answer, success=True, trace=trace)
```

- [ ] **Step 6: Run mock runner tests**

Run:

```bash
uv run pytest tests/skill_eval/test_mock_eval.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add skill_eval/agent_runner.py skill_eval/adapters tests/skill_eval/test_mock_eval.py
git commit -m "feat: add mock skill eval runner"
```

---

### Task 6: Add Inspect solver

**Files:**
- Create: `backend/skill_eval/inspect_solver.py`
- Modify: `backend/tests/skill_eval/test_mock_eval.py`

**Interfaces:**
- Consumes: `run_agent()`, optional injected `AgentRunner`, `TaskState`.
- Produces: `skill_agent_solver(agent_runner=None, skills=None, sandbox="docker")`.

- [ ] **Step 1: Add failing solver integration test**

Append to `backend/tests/skill_eval/test_mock_eval.py`:

```python
from inspect_ai.model import ModelOutput
from inspect_ai.solver import TaskState

from skill_eval.inspect_solver import skill_agent_solver


@pytest.mark.asyncio
async def test_skill_agent_solver_writes_completion_and_trace_metadata():
    state = TaskState(
        model="mock-model",
        sample_id="cloud-run-001",
        epoch=1,
        input="How do I deploy a Cloud Run service?",
        target="The answer should mention gcloud run deploy.",
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=""),
        metadata={"case": {"candidate_skills": ["skills/gcp-cloud-run"]}},
    )

    async def unused_generate(inner_state):
        return inner_state

    solve = skill_agent_solver(agent_runner=MockAgentRunner(), skills=None, sandbox="docker")
    result_state = await solve(state, unused_generate)

    assert "gcloud run deploy" in result_state.output.completion
    assert result_state.metadata["success"] is True
    assert result_state.metadata["agent_trace"]["runtime"] == "mock"
    assert result_state.metadata["agent_trace"]["skill_invocations"][0]["name"] == "skills/gcp-cloud-run"
```

- [ ] **Step 2: Run solver test to verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_mock_eval.py::test_skill_agent_solver_writes_completion_and_trace_metadata -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'skill_eval.inspect_solver'`.

- [ ] **Step 3: Implement Inspect solver**

Create `backend/skill_eval/inspect_solver.py`:

```python
from inspect_ai.solver import Generate, TaskState, solver

from skill_eval.agent_runner import AgentRunner, run_agent


@solver
def skill_agent_solver(agent_runner: AgentRunner | None = None, skills: list[str] | None = None, sandbox: str | None = "docker"):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case = state.metadata.get("case", {})
        selected_skills = skills if skills is not None else case.get("candidate_skills", [])

        result = await run_agent(user_input=state.input_text, skills=selected_skills, sandbox=sandbox, runner=agent_runner)

        state.output.completion = result.final_answer
        state.metadata["agent_trace"] = result.trace.model_dump()
        state.metadata["success"] = result.success

        return state

    return solve
```

- [ ] **Step 4: Run solver integration test**

Run:

```bash
uv run pytest tests/skill_eval/test_mock_eval.py::test_skill_agent_solver_writes_completion_and_trace_metadata -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skill_eval/inspect_solver.py tests/skill_eval/test_mock_eval.py
git commit -m "feat: add inspect skill agent solver"
```

---

### Task 7: Add Inspect scorers

**Files:**
- Create: `backend/skill_eval/inspect_scorer.py`
- Create: `backend/tests/skill_eval/test_trace_integrity_scorer.py`
- Create: `backend/tests/skill_eval/test_skill_assertion_scorer.py`

**Interfaces:**
- Consumes: `AgentTrace`, `SkillEvalCase`, `evaluate_assertion()`.
- Produces: `trace_integrity_scorer()` and `skill_assertion_scorer()`.

- [ ] **Step 1: Write trace integrity scorer tests**

Create `backend/tests/skill_eval/test_trace_integrity_scorer.py`:

```python
import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from skill_eval.inspect_scorer import trace_integrity_scorer
from skill_eval.trace_schema import AgentTrace


def _state(metadata):
    return TaskState(
        model="mock-model",
        sample_id="case-1",
        epoch=1,
        input="input",
        target="target",
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=""),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_trace_integrity_scorer_fails_when_trace_missing():
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({}), Target(""))

    assert score.value == 0.0
    assert "Missing agent_trace" in score.explanation


@pytest.mark.asyncio
async def test_trace_integrity_scorer_fails_when_trace_invalid():
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({"agent_trace": {"input": "x"}}), Target(""))

    assert score.value == 0.0
    assert "Invalid AgentTrace" in score.explanation


@pytest.mark.asyncio
async def test_trace_integrity_scorer_passes_for_valid_trace():
    trace = AgentTrace(
        input="input",
        final_answer="answer",
        success=True,
        messages=[{"role": "assistant", "content": "answer"}],
    )
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({"agent_trace": trace.model_dump()}), Target(""))

    assert score.value == 1.0
    assert score.explanation == "Trace is complete."


@pytest.mark.asyncio
async def test_trace_integrity_scorer_fails_on_fatal_error():
    trace = AgentTrace(
        input="input",
        final_answer="answer",
        success=True,
        messages=[{"role": "assistant", "content": "answer"}],
        errors=["fatal: crashed"],
    )
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({"agent_trace": trace.model_dump()}), Target(""))

    assert score.value == 0.0
    assert "fatal" in score.explanation
```

- [ ] **Step 2: Write skill assertion scorer tests**

Create `backend/tests/skill_eval/test_skill_assertion_scorer.py`:

```python
import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from skill_eval.inspect_scorer import skill_assertion_scorer
from skill_eval.trace_schema import AgentTrace, AgentToolCall


def _state(metadata):
    return TaskState(
        model="mock-model",
        sample_id="case-1",
        epoch=1,
        input="input",
        target="target",
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=""),
        metadata=metadata,
    )


def _trace():
    return AgentTrace(
        input="input",
        final_answer="Use gcloud run deploy",
        success=True,
        tool_calls=[AgentToolCall(name="bash")],
        messages=[{"role": "assistant", "content": "Use gcloud run deploy"}],
    )


@pytest.mark.asyncio
async def test_skill_assertion_scorer_passes_all_assertions():
    metadata = {
        "case": {
            "id": "case-1",
            "input": "input",
            "assertions": [
                {"name": "tool_called", "target": "bash"},
                {"name": "output_contains", "target": "gcloud run deploy"},
                {"name": "success_is_true"},
                {"name": "trace_complete"},
            ],
        },
        "agent_trace": _trace().model_dump(),
    }
    score_fn = skill_assertion_scorer()

    score = await score_fn(_state(metadata), Target(""))

    assert score.value == 1.0
    assert score.metadata["case_id"] == "case-1"
    assert len(score.metadata["assertion_results"]) == 4


@pytest.mark.asyncio
async def test_skill_assertion_scorer_fails_any_failed_assertion():
    metadata = {
        "case": {
            "id": "case-1",
            "input": "input",
            "assertions": [{"name": "tool_not_called", "target": "bash"}],
        },
        "agent_trace": _trace().model_dump(),
    }
    score_fn = skill_assertion_scorer()

    score = await score_fn(_state(metadata), Target(""))

    assert score.value == 0.0
    assert "Forbidden tool `bash` was called" in score.explanation


@pytest.mark.asyncio
async def test_skill_assertion_scorer_fails_missing_case_metadata():
    score_fn = skill_assertion_scorer()

    score = await score_fn(_state({"agent_trace": _trace().model_dump()}), Target(""))

    assert score.value == 0.0
    assert "Missing case metadata" in score.explanation


@pytest.mark.asyncio
async def test_skill_assertion_scorer_fails_missing_trace_metadata():
    score_fn = skill_assertion_scorer()

    score = await score_fn(_state({"case": {"id": "case-1", "input": "input"}}), Target(""))

    assert score.value == 0.0
    assert "Missing agent_trace metadata" in score.explanation
```

- [ ] **Step 3: Run scorer tests to verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_trace_integrity_scorer.py tests/skill_eval/test_skill_assertion_scorer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'skill_eval.inspect_scorer'`.

- [ ] **Step 4: Implement Inspect scorers**

Create `backend/skill_eval/inspect_scorer.py`:

```python
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from skill_eval.assertion_engine import evaluate_assertion
from skill_eval.case_schema import SkillEvalCase
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

        failures: list[str] = []
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

        return Score(value=0.0 if failures else 1.0, explanation="\n".join(failures) if failures else "Trace is complete.")

    return score


@scorer
def skill_assertion_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if "case" not in state.metadata:
            return Score(value=0.0, explanation="Missing case metadata.")
        if "agent_trace" not in state.metadata:
            return Score(value=0.0, explanation="Missing agent_trace metadata.")

        case = SkillEvalCase.model_validate(state.metadata["case"])
        trace = AgentTrace.model_validate(state.metadata["agent_trace"])
        results = [evaluate_assertion(assertion, trace, trace.final_answer) for assertion in case.assertions]
        failures = [result for result in results if not result.passed]

        return Score(
            value=0.0 if failures else 1.0,
            explanation="\n".join(result.explanation for result in failures) if failures else "All assertions passed.",
            metadata={"case_id": case.id, "assertion_results": [result.model_dump() for result in results]},
        )

    return score
```

- [ ] **Step 5: Run scorer tests**

Run:

```bash
uv run pytest tests/skill_eval/test_trace_integrity_scorer.py tests/skill_eval/test_skill_assertion_scorer.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add skill_eval/inspect_scorer.py tests/skill_eval/test_trace_integrity_scorer.py tests/skill_eval/test_skill_assertion_scorer.py
git commit -m "feat: add inspect skill eval scorers"
```

---

### Task 8: Add demo cases and Inspect tasks

**Files:**
- Create: `backend/evals/__init__.py`
- Create: `backend/evals/skills_eval.py`
- Create: `backend/evals/baseline_eval.py`
- Create: `backend/evals/with_skill_eval.py`
- Create: `backend/cases/no_write_todos.jsonl`
- Create: `backend/cases/gcp_skills.jsonl`
- Modify: `backend/tests/skill_eval/test_mock_eval.py`

**Interfaces:**
- Consumes: loader, solver, scorers.
- Produces: runnable Inspect task definitions and demo JSONL cases.

- [ ] **Step 1: Add eval package marker**

Create `backend/evals/__init__.py`:

```python
"""Inspect task entrypoints for DeerFlow evaluation harnesses."""
```

- [ ] **Step 2: Add demo cases**

Create `backend/cases/no_write_todos.jsonl`:

```json
{"id":"no-write-todos-001","input":"Create a plan for this task, but do not write todos.","target":"The agent should answer without calling write_todos.","required_skills":["no-write-todos-in-pro"],"candidate_skills":["skills/no-write-todos-in-pro"],"assertions":[{"name":"tool_not_called","target":"write_todos"},{"name":"success_is_true"},{"name":"trace_complete"}],"tags":["negative-tool","skill-compliance"],"difficulty":"smoke"}
```

Create `backend/cases/gcp_skills.jsonl`:

```json
{"id":"cloud-run-001","input":"How do I deploy a Cloud Run service?","target":"The answer should mention gcloud run deploy.","required_skills":["gcp-cloud-run"],"candidate_skills":["skills/gcp-cloud-run"],"assertions":[{"name":"output_contains","target":"gcloud run deploy"},{"name":"success_is_true"},{"name":"trace_complete"}],"tags":["gcp","cloud-run","answer-quality"],"difficulty":"smoke"}
```

- [ ] **Step 3: Implement main skills eval task**

Create `backend/evals/skills_eval.py`:

```python
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.scorer import model_graded_qa

from skill_eval.dataset_loader import load_skill_cases
from skill_eval.inspect_scorer import skill_assertion_scorer, trace_integrity_scorer
from skill_eval.inspect_solver import skill_agent_solver


@task
def skills_eval(case_file: str = "cases/gcp_skills.jsonl", skills_folder: str = "skills", use_model_graded_qa: bool = False):
    samples = load_skill_cases(case_file)
    skill_files = (Path.cwd() / skills_folder).rglob("SKILL.md")
    all_skills = [str(skill_file.parent) for skill_file in skill_files]

    scorers = [trace_integrity_scorer(), skill_assertion_scorer()]
    if use_model_graded_qa:
        scorers.append(model_graded_qa())

    return Task(dataset=samples, solver=skill_agent_solver(skills=all_skills, sandbox="docker"), scorer=scorers, sandbox="docker")
```

- [ ] **Step 4: Implement baseline eval task**

Create `backend/evals/baseline_eval.py`:

```python
from inspect_ai import Task, task

from skill_eval.dataset_loader import load_skill_cases
from skill_eval.inspect_scorer import skill_assertion_scorer, trace_integrity_scorer
from skill_eval.inspect_solver import skill_agent_solver


@task
def baseline_eval(case_file: str = "cases/gcp_skills.jsonl"):
    return Task(
        dataset=load_skill_cases(case_file),
        solver=skill_agent_solver(skills=[], sandbox="docker"),
        scorer=[trace_integrity_scorer(), skill_assertion_scorer()],
        sandbox="docker",
    )
```

- [ ] **Step 5: Implement with-skill eval task**

Create `backend/evals/with_skill_eval.py`:

```python
from inspect_ai import Task, task

from skill_eval.dataset_loader import load_skill_cases
from skill_eval.inspect_scorer import skill_assertion_scorer, trace_integrity_scorer
from skill_eval.inspect_solver import skill_agent_solver


@task
def with_skill_eval(case_file: str = "cases/gcp_skills.jsonl"):
    return Task(
        dataset=load_skill_cases(case_file),
        solver=skill_agent_solver(skills=None, sandbox="docker"),
        scorer=[trace_integrity_scorer(), skill_assertion_scorer()],
        sandbox="docker",
    )
```

- [ ] **Step 6: Add task construction tests**

Append to `backend/tests/skill_eval/test_mock_eval.py`:

```python
from evals.baseline_eval import baseline_eval
from evals.skills_eval import skills_eval
from evals.with_skill_eval import with_skill_eval


def test_demo_inspect_tasks_construct():
    assert skills_eval(case_file="cases/gcp_skills.jsonl").dataset
    assert baseline_eval(case_file="cases/gcp_skills.jsonl").dataset
    assert with_skill_eval(case_file="cases/gcp_skills.jsonl").dataset
```

- [ ] **Step 7: Run demo task construction test**

Run:

```bash
uv run pytest tests/skill_eval/test_mock_eval.py::test_demo_inspect_tasks_construct -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evals cases tests/skill_eval/test_mock_eval.py
git commit -m "feat: add demo skill eval tasks"
```

---

### Task 9: Run focused verification and update backend docs

**Files:**
- Modify: `backend/AGENTS.md`

**Interfaces:**
- Consumes: completed MVP harness.
- Produces: verified tests and backend development documentation note.

- [ ] **Step 1: Run focused skill eval test suite**

Run:

```bash
uv run pytest tests/skill_eval -v
```

Expected: PASS for all tests in `tests/skill_eval/`.

- [ ] **Step 2: Run formatting check for changed package**

Run:

```bash
uv run ruff format --check skill_eval evals tests/skill_eval
```

Expected: PASS.

- [ ] **Step 3: Run lint for changed package**

Run:

```bash
uv run ruff check skill_eval evals tests/skill_eval
```

Expected: PASS.

- [ ] **Step 4: Update backend AGENTS documentation**

Add this section to `backend/AGENTS.md` near the backend test/development documentation area:

```markdown
### Skill Evaluation Harness

`skill_eval/` is a backend-local Inspect AI evaluation harness for skill-based agent behavior. It is not part of the publishable `deerflow-harness` package. The MVP path is JSONL cases → Inspect `Sample` → `skill_agent_solver()` → `AgentRunResult` → normalized `AgentTrace` → rule scorers.

Key boundaries:
- `Sample.target` stores final-answer references.
- `Sample.metadata["case"]` stores `SkillEvalCase` behavior expectations.
- `state.output.completion` stores the agent final answer.
- `state.metadata["agent_trace"]` stores `AgentTrace.model_dump()`.
- Generic scorers read `AgentTrace`, not raw DeerFlow or LangGraph messages.
- `MockAgentRunner` is the MVP runner for testing the Inspect integration before adding the DeerFlow adapter.

Focused tests live in `tests/skill_eval/` and can be run with `uv run pytest tests/skill_eval -v`.
```

- [ ] **Step 5: Run final focused tests after docs update**

Run:

```bash
uv run pytest tests/skill_eval -v
```

Expected: PASS.

- [ ] **Step 6: Commit docs and final verification state**

```bash
git add AGENTS.md
git commit -m "docs: document skill eval harness"
```

---

## Plan Self-Review

### Spec Coverage

- SkillEvalCase schema: Task 2.
- AgentTrace schema with `runtime` and `raw_trace_ref`: Task 2.
- JSONL case loader: Task 4.
- Custom Inspect solver: Task 6.
- Custom Inspect scorers: Task 7.
- Assertion engine MVP assertions: Task 3.
- Demo case files: Task 8.
- Demo Inspect tasks: Task 8.
- Mock agent runner: Task 5.
- Unit tests: Tasks 2 through 9.
- Baseline/with-skill task entrypoints: Task 8.
- DeerFlow adapter excluded from MVP: Global Constraints and task scope.

### Type Consistency

- `SkillEvalCase.assertions` uses `list[SkillAssertionSpec]` in schema and scorer tests.
- `AgentTrace.tool_calls` uses `list[AgentToolCall]` in schema and assertion tests.
- `AgentTrace.skill_invocations` uses `list[SkillInvocation]` in schema and mock runner.
- `run_agent()` accepts `runner: AgentRunner | None` and is used by `skill_agent_solver()`.
- `skill_agent_solver()` writes `state.output.completion`, `state.metadata["agent_trace"]`, and `state.metadata["success"]`.
- `skill_assertion_scorer()` returns per-assertion metadata under `assertion_results`.

### Placeholder Scan

The plan intentionally contains no unresolved implementation placeholders. Deferred features are explicitly excluded from MVP and named in Global Constraints.
