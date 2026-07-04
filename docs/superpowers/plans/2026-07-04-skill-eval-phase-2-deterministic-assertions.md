# Skill Eval Phase 2 Deterministic Assertions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the skill-eval MVP assertion engine from 10 assertions to the full Phase 2 deterministic assertion set without adding new Inspect scorers.

**Architecture:** Keep all evaluation semantics inside `skill_eval/assertion_engine.py`; `skill_assertion_scorer()` continues to dispatch every declarative case assertion through the pure engine. Extend `SkillAssertionSpec.name` to accept the Phase 2 names, add small pure helpers for string matching, JSON serialization, threshold checks, and selected tool-call extraction, and cover each assertion with pass/fail unit tests.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, pytest-asyncio, Inspect AI, ruff.

## Global Constraints

- Generic scorers MUST evaluate `AgentTrace`, not raw DeerFlow or LangGraph messages.
- Raw runtime data MUST remain adapter input or debug evidence through `AgentTrace.raw_trace_ref`.
- Do not add new Inspect scorers in Phase 2.
- Every new assertion MUST route through `skill_assertion_scorer()` by adding a handler in `assertion_engine.py`.
- Assertion engine MUST remain pure Python and testable without Inspect.
- Unit tests MUST cover pass and fail paths for every new assertion type.
- `AssertionResult.metadata` SHOULD include useful debug details: matched tool calls, observed threshold values, snippets, or observed sequences.
- Do not add the DeerFlow adapter in this phase; it is Phase 3.
- Keep implementation under `backend/skill_eval/` and `backend/tests/skill_eval/`.

---

## File Structure

Modify:

- `skill_eval/case_schema.py` — extend `AssertionName` with Phase 2 names.
- `skill_eval/assertion_engine.py` — add deterministic assertion handlers and focused helper functions.
- `tests/skill_eval/test_assertion_engine.py` — add pass/fail coverage for every new handler.

No new files are required.

---

### Task 1: Add output-rule assertions

**Files:**
- Modify: `skill_eval/case_schema.py`
- Modify: `skill_eval/assertion_engine.py`
- Test: `tests/skill_eval/test_assertion_engine.py`

**Interfaces:**
- Consumes: `SkillAssertionSpec(name, target, threshold, message)`, `AgentTrace.final_answer`.
- Produces assertion handlers: `output_not_contains`, `regex_match`, `json_valid`.

- [ ] **Step 1: Extend assertion names for output rules**

In `skill_eval/case_schema.py`, add these names to `AssertionName`:

```python
    "output_not_contains",
    "regex_match",
    "json_valid",
```

Expected location: the existing `AssertionName = Literal[...]` list.

- [ ] **Step 2: Add failing output-rule tests**

Append these tests to `tests/skill_eval/test_assertion_engine.py`:

```python
def test_output_not_contains_passes_and_fails():
    trace = _valid_trace(final_answer="Safe deployment complete")

    passing = evaluate_assertion(SkillAssertionSpec(name="output_not_contains", target="password"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="output_not_contains", target="deployment"), trace, trace.final_answer)

    assert passing.passed is True
    assert failing.passed is False
    assert failing.metadata["target"] == "deployment"


def test_regex_match_passes_and_fails():
    trace = _valid_trace(final_answer="Run id: abc-123")

    passing = evaluate_assertion(SkillAssertionSpec(name="regex_match", target=r"[a-z]+-\d+"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="regex_match", target=r"^done$"), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["pattern"] == r"[a-z]+-\d+"
    assert failing.passed is False


def test_regex_match_fails_invalid_pattern():
    trace = _valid_trace(final_answer="answer")

    result = evaluate_assertion(SkillAssertionSpec(name="regex_match", target="["), trace, trace.final_answer)

    assert result.passed is False
    assert "Invalid regex" in result.explanation
    assert "error" in result.metadata


def test_json_valid_passes_and_fails():
    valid_trace = _valid_trace(final_answer='{"status":"ok"}')
    invalid_trace = _valid_trace(final_answer="not json")

    passing = evaluate_assertion(SkillAssertionSpec(name="json_valid"), valid_trace, valid_trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="json_valid"), invalid_trace, invalid_trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["json_type"] == "dict"
    assert failing.passed is False
    assert "error" in failing.metadata
```

- [ ] **Step 3: Run output-rule tests and verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -k "output_not_contains or regex_match or json_valid" -v
```

Expected: FAIL because the new assertion names are not registered yet.

- [ ] **Step 4: Implement output-rule handlers**

In `skill_eval/assertion_engine.py`, add imports:

```python
import json
import re
```

Append these handlers near the existing output assertion:

```python
@register_assertion("output_not_contains")
def evaluate_output_not_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    if target not in final_answer:
        return _pass(assertion, f"Output did not contain `{target}`.", target=target)
    return _fail(assertion, f"Forbidden output `{target}` was present.", target=target)


@register_assertion("regex_match")
def evaluate_regex_match(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    pattern = assertion.target or ""
    try:
        match = re.search(pattern, final_answer)
    except re.error as exc:
        return _fail(assertion, f"Invalid regex `{pattern}`: {exc}", pattern=pattern, error=str(exc))

    if match:
        return _pass(assertion, f"Output matched regex `{pattern}`.", pattern=pattern, match=match.group(0))
    return _fail(assertion, f"Expected output to match regex `{pattern}`.", pattern=pattern)


@register_assertion("json_valid")
def evaluate_json_valid(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    try:
        parsed = json.loads(final_answer)
    except json.JSONDecodeError as exc:
        return _fail(assertion, f"Expected output to be valid JSON: {exc}", error=str(exc))

    return _pass(assertion, "Output is valid JSON.", json_type=type(parsed).__name__)
```

- [ ] **Step 5: Verify output-rule tests pass**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -k "output_not_contains or regex_match or json_valid" -v
```

Expected: PASS.

- [ ] **Step 6: Commit output-rule assertions**

```bash
git add skill_eval/case_schema.py skill_eval/assertion_engine.py tests/skill_eval/test_assertion_engine.py
git commit -m "feat: add skill eval output assertions"
```

---

### Task 2: Add tool argument, result, error, count, and order assertions

**Files:**
- Modify: `skill_eval/case_schema.py`
- Modify: `skill_eval/assertion_engine.py`
- Test: `tests/skill_eval/test_assertion_engine.py`

**Interfaces:**
- Consumes: `AgentTrace.tool_calls: list[AgentToolCall]` where each call has `name`, `args`, `result`, and `error`.
- Produces assertion handlers: `tool_args_contains`, `tool_args_match`, `tool_call_order`, `tool_error_absent`, `tool_result_contains`, `tool_result_match`, `tool_count_under`.

- [ ] **Step 1: Extend assertion names for tool behavior**

In `skill_eval/case_schema.py`, add these names to `AssertionName`:

```python
    "tool_args_contains",
    "tool_args_match",
    "tool_call_order",
    "tool_error_absent",
    "tool_result_contains",
    "tool_result_match",
    "tool_count_under",
```

- [ ] **Step 2: Add failing tool assertion tests**

Append these tests to `tests/skill_eval/test_assertion_engine.py`:

```python
def test_tool_args_contains_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", args={"cmd": "gcloud run deploy app"})])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_args_contains", target="gcloud run deploy"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_args_contains", target="kubectl apply"), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["matched_tool_call"]["name"] == "bash"
    assert failing.passed is False


def test_tool_args_match_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", args={"cmd": "gcloud run deploy app"})])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_args_match", target=r"gcloud\s+run\s+deploy"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_args_match", target=r"kubectl\s+apply"), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["pattern"] == r"gcloud\s+run\s+deploy"
    assert failing.passed is False


def test_tool_call_order_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="read_file"), AgentToolCall(name="bash"), AgentToolCall(name="present_files")])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_call_order", target="read_file,bash"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_call_order", target="bash,read_file"), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["observed_order"] == ["read_file", "bash", "present_files"]
    assert failing.passed is False


def test_tool_error_absent_passes_and_fails():
    clean_trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", error=None)])
    failing_trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", error="permission denied")])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_error_absent", target="bash"), clean_trace, clean_trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_error_absent", target="bash"), failing_trace, failing_trace.final_answer)

    assert passing.passed is True
    assert failing.passed is False
    assert failing.metadata["errored_tool_calls"][0]["error"] == "permission denied"


def test_tool_result_contains_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", result="Deployment successful")])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_result_contains", target="successful"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_result_contains", target="failed"), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["matched_tool_call"]["name"] == "bash"
    assert failing.passed is False


def test_tool_result_match_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", result="revision rev-42 ready")])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_result_match", target=r"rev-\d+"), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_result_match", target=r"error:\s+"), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["match"] == "rev-42"
    assert failing.passed is False


def test_tool_count_under_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash"), AgentToolCall(name="read_file")])

    passing = evaluate_assertion(SkillAssertionSpec(name="tool_count_under", threshold=3), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tool_count_under", threshold=2), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["observed"] == 2
    assert failing.passed is False
    assert failing.metadata["threshold"] == 2
```

- [ ] **Step 3: Run tool assertion tests and verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -k "tool_args or tool_call_order or tool_error_absent or tool_result or tool_count_under" -v
```

Expected: FAIL because the new assertion handlers are not registered yet.

- [ ] **Step 4: Add serialization and matching helpers**

In `skill_eval/assertion_engine.py`, add these helpers before `_pass()`:

```python
def _tool_call_dump(call: AgentToolCall) -> dict[str, Any]:
    return call.model_dump()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _target_tool_calls(assertion: SkillAssertionSpec, trace: AgentTrace) -> list[AgentToolCall]:
    if not assertion.target or assertion.target not in {call.name for call in trace.tool_calls}:
        return trace.tool_calls
    return [call for call in trace.tool_calls if call.name == assertion.target]


def _compile_pattern(assertion: SkillAssertionSpec) -> tuple[re.Pattern[str] | None, AssertionResult | None]:
    pattern = assertion.target or ""
    try:
        return re.compile(pattern), None
    except re.error as exc:
        return None, _fail(assertion, f"Invalid regex `{pattern}`: {exc}", pattern=pattern, error=str(exc))
```

- [ ] **Step 5: Implement tool assertion handlers**

Append these handlers in `skill_eval/assertion_engine.py`:

```python
@register_assertion("tool_args_contains")
def evaluate_tool_args_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    for call in trace.tool_calls:
        args_text = _stringify(call.args)
        if target in args_text:
            return _pass(assertion, f"Tool arguments contained `{target}`.", target=target, matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool arguments to contain `{target}`.", target=target, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_args_match")
def evaluate_tool_args_match(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    pattern, error = _compile_pattern(assertion)
    if error is not None:
        return error
    assert pattern is not None
    for call in trace.tool_calls:
        match = pattern.search(_stringify(call.args))
        if match:
            return _pass(assertion, f"Tool arguments matched regex `{pattern.pattern}`.", pattern=pattern.pattern, match=match.group(0), matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool arguments to match regex `{pattern.pattern}`.", pattern=pattern.pattern, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_call_order")
def evaluate_tool_call_order(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    expected = [name.strip() for name in (assertion.target or "").split(",") if name.strip()]
    observed = [call.name for call in trace.tool_calls]
    cursor = 0
    for name in observed:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    if cursor == len(expected):
        return _pass(assertion, f"Tool call order contained `{expected}`.", expected_order=expected, observed_order=observed)
    return _fail(assertion, f"Expected tool call order `{expected}` within observed order `{observed}`.", expected_order=expected, observed_order=observed)


@register_assertion("tool_error_absent")
def evaluate_tool_error_absent(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    calls = _target_tool_calls(assertion, trace)
    errored = [call for call in calls if call.error]
    if not errored:
        return _pass(assertion, "No matching tool errors were present.", checked_tool_calls=[_tool_call_dump(call) for call in calls])
    return _fail(assertion, "Expected no matching tool errors.", errored_tool_calls=[_tool_call_dump(call) for call in errored])


@register_assertion("tool_result_contains")
def evaluate_tool_result_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    for call in trace.tool_calls:
        result_text = _stringify(call.result)
        if target in result_text:
            return _pass(assertion, f"Tool result contained `{target}`.", target=target, matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool result to contain `{target}`.", target=target, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_result_match")
def evaluate_tool_result_match(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    pattern, error = _compile_pattern(assertion)
    if error is not None:
        return error
    assert pattern is not None
    for call in trace.tool_calls:
        match = pattern.search(_stringify(call.result))
        if match:
            return _pass(assertion, f"Tool result matched regex `{pattern.pattern}`.", pattern=pattern.pattern, match=match.group(0), matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool result to match regex `{pattern.pattern}`.", pattern=pattern.pattern, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_count_under")
def evaluate_tool_count_under(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    threshold = assertion.threshold
    observed = len(trace.tool_calls)
    if threshold is None:
        return _fail(assertion, "tool_count_under requires a threshold.", observed=observed, threshold=None)
    if observed < threshold:
        return _pass(assertion, f"Tool count {observed} was under {threshold}.", observed=observed, threshold=threshold)
    return _fail(assertion, f"Tool count {observed} was not under {threshold}.", observed=observed, threshold=threshold)
```

- [ ] **Step 6: Verify tool assertion tests pass**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -k "tool_args or tool_call_order or tool_error_absent or tool_result or tool_count_under" -v
```

Expected: PASS.

- [ ] **Step 7: Commit tool assertions**

```bash
git add skill_eval/case_schema.py skill_eval/assertion_engine.py tests/skill_eval/test_assertion_engine.py
git commit -m "feat: add skill eval tool assertions"
```

---

### Task 3: Add runtime limit and clarification assertions

**Files:**
- Modify: `skill_eval/case_schema.py`
- Modify: `skill_eval/assertion_engine.py`
- Test: `tests/skill_eval/test_assertion_engine.py`

**Interfaces:**
- Consumes: `AgentTrace.latency_ms`, `AgentTrace.input_tokens`, `AgentTrace.output_tokens`, `AgentTrace.steps`, and `AgentTrace.tool_calls`.
- Produces assertion handlers: `latency_under`, `tokens_under`, `max_steps_under`, `no_unexpected_clarification`.

- [ ] **Step 1: Extend assertion names for limits and clarification**

In `skill_eval/case_schema.py`, add these names to `AssertionName`:

```python
    "latency_under",
    "tokens_under",
    "max_steps_under",
    "no_unexpected_clarification",
```

- [ ] **Step 2: Add failing runtime and clarification tests**

Append these tests to `tests/skill_eval/test_assertion_engine.py`:

```python
def test_latency_under_passes_and_fails():
    trace = _valid_trace(latency_ms=120)

    passing = evaluate_assertion(SkillAssertionSpec(name="latency_under", threshold=200), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="latency_under", threshold=100), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata == {"observed": 120, "threshold": 200}
    assert failing.passed is False


def test_tokens_under_uses_input_plus_output_tokens():
    trace = _valid_trace(input_tokens=30, output_tokens=20)

    passing = evaluate_assertion(SkillAssertionSpec(name="tokens_under", threshold=60), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="tokens_under", threshold=50), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["observed"] == 50
    assert failing.passed is False


def test_max_steps_under_passes_and_fails():
    trace = _valid_trace(steps=[{"type": "start"}, {"type": "finish"}])

    passing = evaluate_assertion(SkillAssertionSpec(name="max_steps_under", threshold=3), trace, trace.final_answer)
    failing = evaluate_assertion(SkillAssertionSpec(name="max_steps_under", threshold=2), trace, trace.final_answer)

    assert passing.passed is True
    assert passing.metadata["observed"] == 2
    assert failing.passed is False


def test_no_unexpected_clarification_passes_when_absent():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash")])

    result = evaluate_assertion(SkillAssertionSpec(name="no_unexpected_clarification"), trace, trace.final_answer)

    assert result.passed is True


def test_no_unexpected_clarification_fails_for_ask_clarification_tool():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="ask_clarification", args={"question": "Which project?"})])

    result = evaluate_assertion(SkillAssertionSpec(name="no_unexpected_clarification"), trace, trace.final_answer)

    assert result.passed is False
    assert result.metadata["clarification_tool_calls"][0]["name"] == "ask_clarification"
```

- [ ] **Step 3: Run runtime tests and verify failure**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -k "latency_under or tokens_under or max_steps_under or no_unexpected_clarification" -v
```

Expected: FAIL because the new assertion handlers are not registered yet.

- [ ] **Step 4: Add numeric threshold helper**

In `skill_eval/assertion_engine.py`, add this helper before `_pass()`:

```python
def _threshold(assertion: SkillAssertionSpec, observed: int | float | None, label: str) -> AssertionResult | None:
    if assertion.threshold is None:
        return _fail(assertion, f"{assertion.name} requires a threshold.", observed=observed, threshold=None)
    if observed is None:
        return _fail(assertion, f"Trace has no observed {label} value.", observed=None, threshold=assertion.threshold)
    return None
```

- [ ] **Step 5: Implement runtime and clarification handlers**

Append these handlers in `skill_eval/assertion_engine.py`:

```python
@register_assertion("latency_under")
def evaluate_latency_under(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    error = _threshold(assertion, trace.latency_ms, "latency_ms")
    if error is not None:
        return error
    assert trace.latency_ms is not None and assertion.threshold is not None
    if trace.latency_ms < assertion.threshold:
        return _pass(assertion, f"Latency {trace.latency_ms}ms was under {assertion.threshold}ms.", observed=trace.latency_ms, threshold=assertion.threshold)
    return _fail(assertion, f"Latency {trace.latency_ms}ms was not under {assertion.threshold}ms.", observed=trace.latency_ms, threshold=assertion.threshold)


@register_assertion("tokens_under")
def evaluate_tokens_under(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    observed = None if trace.input_tokens is None and trace.output_tokens is None else (trace.input_tokens or 0) + (trace.output_tokens or 0)
    error = _threshold(assertion, observed, "token count")
    if error is not None:
        return error
    assert observed is not None and assertion.threshold is not None
    if observed < assertion.threshold:
        return _pass(assertion, f"Token count {observed} was under {assertion.threshold}.", observed=observed, threshold=assertion.threshold)
    return _fail(assertion, f"Token count {observed} was not under {assertion.threshold}.", observed=observed, threshold=assertion.threshold)


@register_assertion("max_steps_under")
def evaluate_max_steps_under(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    observed = len(trace.steps)
    error = _threshold(assertion, observed, "step count")
    if error is not None:
        return error
    assert assertion.threshold is not None
    if observed < assertion.threshold:
        return _pass(assertion, f"Step count {observed} was under {assertion.threshold}.", observed=observed, threshold=assertion.threshold)
    return _fail(assertion, f"Step count {observed} was not under {assertion.threshold}.", observed=observed, threshold=assertion.threshold)


@register_assertion("no_unexpected_clarification")
def evaluate_no_unexpected_clarification(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    clarification_calls = [call for call in trace.tool_calls if call.name == "ask_clarification"]
    if not clarification_calls:
        return _pass(assertion, "No unexpected clarification was requested.")
    return _fail(assertion, "Unexpected clarification was requested.", clarification_tool_calls=[_tool_call_dump(call) for call in clarification_calls])
```

- [ ] **Step 6: Verify runtime tests pass**

Run:

```bash
uv run pytest tests/skill_eval/test_assertion_engine.py -k "latency_under or tokens_under or max_steps_under or no_unexpected_clarification" -v
```

Expected: PASS.

- [ ] **Step 7: Commit runtime assertions**

```bash
git add skill_eval/case_schema.py skill_eval/assertion_engine.py tests/skill_eval/test_assertion_engine.py
git commit -m "feat: add skill eval runtime assertions"
```

---

### Task 4: Integrate, verify, and update docs

**Files:**
- Modify: `tests/skill_eval/test_assertion_engine.py`
- Modify: `backend/AGENTS.md` if the assertion catalog note needs to mention Phase 2.

**Interfaces:**
- Consumes: all Phase 2 assertion handlers registered in `ASSERTION_REGISTRY`.
- Produces: focused verification evidence for the complete Phase 2 assertion set.

- [ ] **Step 1: Update registry coverage test**

In `tests/skill_eval/test_assertion_engine.py`, update `test_mvp_assertions_are_registered` or rename it to `test_deterministic_assertions_are_registered`. The expected set must be:

```python
{
    "tool_called",
    "tool_not_called",
    "tool_args_contains",
    "tool_args_match",
    "tool_call_order",
    "tool_error_absent",
    "tool_result_contains",
    "tool_result_match",
    "tool_count_under",
    "output_contains",
    "output_not_contains",
    "regex_match",
    "json_valid",
    "latency_under",
    "tokens_under",
    "max_steps_under",
    "no_unexpected_clarification",
    "success_is_true",
    "trace_complete",
    "skill_loaded",
    "skill_used",
    "skill_not_used",
    "skill_applied",
    "skill_not_applied",
}
```

- [ ] **Step 2: Update schema validation test**

In `tests/skill_eval/test_assertion_engine.py`, update the test that checks allowed assertion names so `SkillAssertionSpec(name="regex_match")` now validates successfully and only an actually unknown name raises `ValidationError`.

Use this structure:

```python
def test_skill_assertion_accepts_all_deterministic_assertion_names():
    for name in get_args(AssertionName):
        assert SkillAssertionSpec(name=name).name == name

    with pytest.raises(ValidationError):
        SkillAssertionSpec(name="unknown_assertion")
```

- [ ] **Step 3: Run full skill-eval tests**

Run:

```bash
uv run pytest tests/skill_eval -v
```

Expected: PASS.

- [ ] **Step 4: Run formatting check**

Run:

```bash
uv run ruff format --check skill_eval tests/skill_eval
```

Expected: PASS.

- [ ] **Step 5: Run lint**

Run:

```bash
uv run ruff check skill_eval tests/skill_eval
```

Expected: PASS.

- [ ] **Step 6: Update backend docs only if needed**

If `backend/AGENTS.md` still says the harness only has MVP assertions, update the Skill Evaluation Harness note with this sentence:

```markdown
Phase 2 deterministic assertions extend the same pure assertion engine for tool arguments/results/errors/order/count, output rules, runtime limits, and unexpected clarification checks; no additional Inspect scorers are introduced for these checks.
```

Do not add a separate scorer catalog section unless the code now exposes a new scorer.

- [ ] **Step 7: Commit final verification/doc state**

```bash
git add skill_eval tests/skill_eval AGENTS.md
git commit -m "test: cover skill eval deterministic assertions"
```

---

## Plan Self-Review

### Spec Coverage

- Phase 2 assertion catalog is covered: Task 1 covers output rules; Task 2 covers tool behavior; Task 3 covers runtime and clarification; Task 4 verifies registry coverage.
- No new Inspect scorer is planned; every assertion remains routed through `ASSERTION_REGISTRY` and `skill_assertion_scorer()`.
- `AgentTrace` remains the only generic scorer input; no DeerFlow/LangGraph raw structures are introduced.
- DeerFlow adapter remains excluded and deferred to Phase 3.

### Type Consistency

- New names are added to `AssertionName`, then registered with the same exact string names in `assertion_engine.py`.
- Handlers keep the existing signature: `(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult`.
- Metadata keys used by tests are emitted by the proposed handlers: `matched_tool_call`, `observed`, `threshold`, `pattern`, `match`, `target`, `errored_tool_calls`, and `clarification_tool_calls`.

### Placeholder Scan

- No placeholder implementation steps remain.
- Every code-changing step includes the concrete code to add or the exact assertion-name list to update.
- Every task has focused verification commands and expected outcomes.
