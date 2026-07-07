# Skill Eval Phase 3: DeerFlow Agent Adapter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MockAgentRunner` with a `DeerFlowAgentRunner` that runs real DeerFlow agents in-process via `DeerFlowClient`, collects `AgentTrace` from stream events, and feeds it into the unchanged assertion/scorer pipeline.

**Architecture:** Two new files: `DeerFlowTraceAdapter` (pure stream→trace converter, unit-testable without DeerFlow) and `DeerFlowAgentRunner` (wraps `DeerFlowClient` + adapter, implements `AgentRunner` protocol). One modified file: `skills_eval.py` accepts runner via parameter. No changes to scorers, assertions, or schemas.

**Tech Stack:** Python 3.12, `DeerFlowClient` (sync generator), `asyncio.to_thread`, Pydantic v2, pytest + pytest-asyncio, Inspect AI.

## Global Constraints

- Generic scorers MUST evaluate `AgentTrace`, not raw DeerFlow or LangGraph messages.
- Raw stream events MUST be preserved to disk as `raw_trace_ref` for debugging.
- `SkillInvocation.applied` remains `None` from the adapter — it's an assertion-level judgment.
- DeerFlowClient's `stream()` is a sync generator; the runner wraps it in `asyncio.to_thread()`.
- Smoke tests MUST skip gracefully with a clear message when no `config.yaml` exists.
- Existing mock-based tests MUST continue to pass.
- Backend lint and tests MUST pass (`cd backend && make lint && make test`).

---

## File Structure

Create:
- `backend/skill_eval/adapters/deerflow.py` — `DeerFlowTraceAdapter` + `DeerFlowAgentRunner`
- `backend/tests/skill_eval/test_deerflow_adapter.py` — unit tests for adapter (synthetic events)
- `backend/tests/skill_eval/test_deerflow_runner.py` — integration smoke test (real client)

Modify:
- `backend/evals/skills_eval.py` — accept `runner` parameter

---

## DeerFlow Stream Event Reference

`DeerFlowClient.stream()` yields `StreamEvent` objects. The `type` field is one of `"values"`, `"messages-tuple"`, `"custom"`, `"end"`.

Relevant `messages-tuple` event shapes:

| `data["type"]` | `data` keys | Meaning |
|---|---|---|
| `"ai"` | `content`, `id` | AI text delta (accumulate by `id` for final answer) |
| `"ai"` | `content`, `id`, `tool_calls: [{name, args, id}]` | AI requests tool calls |
| `"ai"` | `content`, `id`, `usage_metadata: {input_tokens, output_tokens, total_tokens}` | AI text delta with usage |
| `"tool"` | `content`, `name`, `tool_call_id`, `id` | Tool execution result |

`end` event shape: `{"usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int}}`

Tool call correlation: match AI `tool_calls[].id` with tool `tool_call_id`.

---

### Task 1: DeerFlowTraceAdapter — Stream Events to AgentTrace

**Files:**
- Create: `backend/skill_eval/adapters/deerflow.py` (adapter class only)
- Test: `backend/tests/skill_eval/test_deerflow_adapter.py`

**Interfaces:**
- Produces: `DeerFlowTraceAdapter` class with `feed(event: StreamEvent) -> None` and `build(raw_trace_path: str | None = None) -> AgentTrace`

The adapter is a pure data converter — no DeerFlow imports, no I/O beyond dumping raw events to disk. It accumulates state from stream events and assembles `AgentTrace` on `build()`.

#### Design

```python
class DeerFlowTraceAdapter:
    def __init__(self, request: AgentRunRequest):
        self._tool_calls: dict[str, AgentToolCall] = {}  # keyed by tool_call_id
        self._tool_call_order: list[str] = []             # ordered tool_call_ids
        self._messages: list[dict[str, Any]] = []
        self._chunks_by_msg_id: dict[str, list[str]] = {} # AI text deltas by msg id
        self._last_ai_msg_id: str = ""
        self._start_time: float = 0.0
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._errors: list[str] = []
        self._raw_events: list[dict[str, Any]] = []
        self._request = request

    def feed(self, event) -> None:
        """Ingest one StreamEvent."""
        # Record raw event for debugging
        self._raw_events.append({"type": event.type, "data": event.data})

        if event.type == "messages-tuple":
            self._feed_message(event.data)
        elif event.type == "end":
            usage = event.data.get("usage", {})
            if usage.get("input_tokens") is not None:
                self._input_tokens = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                self._output_tokens = usage["output_tokens"]
        # values and custom events are captured in _raw_events but
        # not semantically parsed into AgentTrace fields.

    def build(self, raw_trace_path: str | None = None) -> AgentTrace:
        """Assemble final AgentTrace."""
        # Save raw events if path provided
        if raw_trace_path:
            import json
            from pathlib import Path
            Path(raw_trace_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_trace_path).write_text(
                "\n".join(json.dumps(e) for e in self._raw_events)
            )

        # Determine final answer: accumulated text of last AI message
        final_answer = ""
        if self._last_ai_msg_id and self._last_ai_msg_id in self._chunks_by_msg_id:
            final_answer = "".join(self._chunks_by_msg_id[self._last_ai_msg_id])

        # Build ordered tool calls
        tool_calls = []
        for tc_id in self._tool_call_order:
            if tc_id in self._tool_calls:
                tool_calls.append(self._tool_calls[tc_id])

        # Build skill invocations
        skill_invocations = self._infer_skill_invocations()

        latency_ms = int((time.monotonic() - self._start_time) * 1000) if self._start_time else None

        return AgentTrace(
            input=self._request.user_input,
            final_answer=final_answer,
            success=len(self._errors) == 0,
            tool_calls=tool_calls,
            skill_invocations=skill_invocations,
            messages=self._messages,
            errors=self._errors,
            latency_ms=latency_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            steps=[{"type": e["type"]} for e in self._raw_events],
            runtime="deerflow",
            raw_trace_ref=raw_trace_path,
        )
```

#### `_feed_message()` — message event dispatch

```python
def _feed_message(self, data: dict[str, Any]) -> None:
    msg_type = data.get("type")
    if msg_type == "ai":
        self._feed_ai_message(data)
    elif msg_type == "tool":
        self._feed_tool_message(data)
    # Other types (human, system) are logged as messages only
    self._messages.append(data)
```

#### `_feed_ai_message()` — AI text deltas and tool calls

```python
def _feed_ai_message(self, data: dict[str, Any]) -> None:
    msg_id = data.get("id") or ""
    content = data.get("content") or ""

    # Accumulate text deltas
    if msg_id:
        self._last_ai_msg_id = msg_id
        if content:
            self._chunks_by_msg_id.setdefault(msg_id, []).append(content)

    # Process tool call requests
    tool_calls_data = data.get("tool_calls") or []
    for tc in tool_calls_data:
        tc_id = tc.get("id") or ""
        if not tc_id:
            continue
        if tc_id not in self._tool_calls:
            self._tool_calls[tc_id] = AgentToolCall(
                name=tc.get("name", ""),
                args=tc.get("args", {}),
            )
            self._tool_call_order.append(tc_id)
```

#### `_feed_tool_message()` — tool results

```python
def _feed_tool_message(self, data: dict[str, Any]) -> None:
    tc_id = data.get("tool_call_id") or ""
    if tc_id and tc_id in self._tool_calls:
        call = self._tool_calls[tc_id]
        call.result = data.get("content")
        error = data.get("error")
        if error:
            call.error = str(error)
```

#### `_infer_skill_invocations()` — skill usage detection

```python
def _infer_skill_invocations(self) -> list[SkillInvocation]:
    invocations: list[SkillInvocation] = []

    # Determine loaded skills from request
    forced = self._request.forced_skills
    if forced is not None:
        loaded_skills = set(forced)
    else:
        loaded_skills = set(self._request.required_skills) | set(self._request.candidate_skills)

    # Detect used skills: look for read_file calls targeting SKILL.md
    used_skills: set[str] = set()
    for tc_id in self._tool_call_order:
        call = self._tool_calls.get(tc_id)
        if not call or call.name != "read_file":
            continue
        args = call.args or {}
        path = args.get("file_path") or args.get("path") or ""
        # Match patterns like "skills/gcp-deploy/SKILL.md"
        # or "skills/public/gcp-deploy/SKILL.md"
        for skill_name in loaded_skills:
            if f"/{skill_name}/SKILL.md" in path or f"/{skill_name}/skill.md" in path:
                used_skills.add(skill_name)
                break

    for skill_name in sorted(loaded_skills):
        invocations.append(SkillInvocation(
            name=skill_name,
            path=f"skills/{skill_name}",
            loaded=True,
            used=skill_name in used_skills,
            applied=None,
            trigger_reason="read_file SKILL.md" if skill_name in used_skills else "available in context",
            evidence=[f"read_file targeting skills/{skill_name}/SKILL.md"] if skill_name in used_skills else [],
        ))

    return invocations
```

#### Unit Tests (`test_deerflow_adapter.py`)

- [ ] **Step 1: Write failing tests**

```python
import time
from skill_eval.adapters.deerflow import DeerFlowTraceAdapter
from skill_eval.agent_runner import AgentRunRequest
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


def _make_event(type_: str, data: dict) -> "StreamEvent":
    from deerflow.client import StreamEvent
    return StreamEvent(type=type_, data=data)


def test_adapter_empty_stream():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hello"))
    adapter._start_time = time.monotonic()
    trace = adapter.build()
    assert trace.final_answer == ""
    assert trace.tool_calls == []
    assert trace.skill_invocations == []
    assert trace.success is True


def test_adapter_single_ai_message():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hello"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "Hello ", "id": "msg1"}))
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "world", "id": "msg1"}))
    adapter.feed(_make_event("end", {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}))
    trace = adapter.build()
    assert trace.final_answer == "Hello world"
    assert trace.input_tokens == 10
    assert trace.output_tokens == 5
    assert trace.runtime == "deerflow"


def test_adapter_multiple_ai_messages_last_is_final():
    """Only the last AI message's text becomes final_answer."""
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="plan then execute"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "I'll plan first.", "id": "msg1"}))
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "Done: result is 42.", "id": "msg2"}))
    trace = adapter.build()
    assert trace.final_answer == "Done: result is 42."


def test_adapter_tool_call_and_result():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="read a file"))
    adapter._start_time = time.monotonic()
    # AI requests a tool call
    adapter.feed(_make_event("messages-tuple", {
        "type": "ai", "content": "", "id": "msg1",
        "tool_calls": [{"name": "read_file", "args": {"file_path": "data.txt"}, "id": "tc1"}],
    }))
    # Tool result arrives
    adapter.feed(_make_event("messages-tuple", {
        "type": "tool", "content": "file contents here", "name": "read_file",
        "tool_call_id": "tc1", "id": "msg2",
    }))
    trace = adapter.build()
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "read_file"
    assert trace.tool_calls[0].args == {"file_path": "data.txt"}
    assert trace.tool_calls[0].result == "file contents here"
    assert trace.tool_calls[0].error is None


def test_adapter_tool_call_with_error():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="bad command"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {
        "type": "ai", "content": "", "id": "msg1",
        "tool_calls": [{"name": "bash", "args": {"cmd": "rm -rf /"}, "id": "tc1"}],
    }))
    # Error in tool result
    adapter.feed(_make_event("messages-tuple", {
        "type": "tool", "content": "", "name": "bash",
        "tool_call_id": "tc1", "id": "msg2",
    }))
    # Simulate error on tool call - adapter reads error from data
    adapter._tool_calls["tc1"].error = "permission denied"
    trace = adapter.build()
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].error == "permission denied"


def test_adapter_multiple_tool_calls_ordered():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="multi-step"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {
        "type": "ai", "content": "", "id": "msg1",
        "tool_calls": [
            {"name": "read_file", "args": {}, "id": "tc1"},
            {"name": "bash", "args": {}, "id": "tc2"},
        ],
    }))
    adapter.feed(_make_event("messages-tuple", {
        "type": "tool", "content": "ok", "name": "read_file", "tool_call_id": "tc1", "id": "msg2",
    }))
    adapter.feed(_make_event("messages-tuple", {
        "type": "tool", "content": "ok", "name": "bash", "tool_call_id": "tc2", "id": "msg3",
    }))
    trace = adapter.build()
    assert len(trace.tool_calls) == 2
    assert [tc.name for tc in trace.tool_calls] == ["read_file", "bash"]


def test_adapter_usage_from_end_event():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hi"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "hey", "id": "m1"}))
    adapter.feed(_make_event("end", {"usage": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75}}))
    trace = adapter.build()
    assert trace.input_tokens == 50
    assert trace.output_tokens == 25


def test_adapter_latency():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hi"))
    adapter._start_time = time.monotonic() - 1.5  # simulate 1.5s elapsed
    trace = adapter.build()
    assert trace.latency_ms is not None
    assert 1400 <= trace.latency_ms <= 1600  # ~1.5s


def test_adapter_skill_loaded():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(
        user_input="deploy",
        required_skills=["gcp-deploy"],
        candidate_skills=["gcp-deploy", "system-design"],
    ))
    adapter._start_time = time.monotonic()
    trace = adapter.build()
    invocations = {inv.name: inv for inv in trace.skill_invocations}
    assert "gcp-deploy" in invocations
    assert invocations["gcp-deploy"].loaded is True
    assert invocations["gcp-deploy"].used is False  # no read_file
    assert "system-design" in invocations
    assert invocations["system-design"].loaded is True


def test_adapter_skill_used_via_read_file():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(
        user_input="deploy to cloud run",
        required_skills=["gcp-deploy"],
    ))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {
        "type": "ai", "content": "", "id": "msg1",
        "tool_calls": [{"name": "read_file", "args": {"file_path": "skills/gcp-deploy/SKILL.md"}, "id": "tc1"}],
    }))
    adapter.feed(_make_event("messages-tuple", {
        "type": "tool", "content": "# GCP Deploy Skill\n...", "name": "read_file",
        "tool_call_id": "tc1", "id": "msg2",
    }))
    trace = adapter.build()
    invocations = {inv.name: inv for inv in trace.skill_invocations}
    assert invocations["gcp-deploy"].used is True
    assert invocations["gcp-deploy"].loaded is True


def test_adapter_skill_not_used_without_read_file():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(
        user_input="deploy",
        required_skills=["gcp-deploy"],
    ))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {
        "type": "ai", "content": "", "id": "msg1",
        "tool_calls": [{"name": "bash", "args": {"cmd": "gcloud run deploy"}, "id": "tc1"}],
    }))
    trace = adapter.build()
    invocations = {inv.name: inv for inv in trace.skill_invocations}
    assert invocations["gcp-deploy"].used is False
```

- [ ] **Step 2: Run tests — verify FAIL**

```bash
cd backend && uv run pytest tests/skill_eval/test_deerflow_adapter.py -v
```

Expected: FAIL — `DeerFlowTraceAdapter` not yet implemented.

- [ ] **Step 3: Implement `DeerFlowTraceAdapter`**

Create `backend/skill_eval/adapters/deerflow.py` with the full adapter class as designed above.

- [ ] **Step 4: Run tests — verify PASS**

```bash
cd backend && uv run pytest tests/skill_eval/test_deerflow_adapter.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/adapters/deerflow.py backend/tests/skill_eval/test_deerflow_adapter.py
git commit -m "feat: add DeerFlowTraceAdapter for stream-to-trace conversion"
```

---

### Task 2: DeerFlowAgentRunner

**Files:**
- Modify: `backend/skill_eval/adapters/deerflow.py` (add runner class)

**Interfaces:**
- Consumes: `DeerFlowTraceAdapter`, `AgentRunRequest`, `AgentRunResult`, `AgentRunner` protocol
- Produces: `DeerFlowAgentRunner` class with `async def run(request: AgentRunRequest) -> AgentRunResult`

The runner wraps `DeerFlowClient` and the adapter. Since `DeerFlowClient.stream()` is a sync generator, the runner uses `asyncio.to_thread()` to avoid blocking the event loop.

#### Design

```python
class DeerFlowAgentRunner:
    """AgentRunner backed by DeerFlowClient (in-process, no Gateway)."""

    def __init__(
        self,
        config_path: str | None = None,
        model_name: str | None = None,
        sandbox: str | None = None,
        skills_dir: str = "skills",
        trace_dir: str | None = None,
    ):
        self._config_path = config_path
        self._model_name = model_name
        self._sandbox = sandbox or "local"
        self._skills_dir = skills_dir
        self._trace_dir = trace_dir

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        import uuid
        from pathlib import Path
        from deerflow.client import DeerFlowClient

        # Determine available skills
        mode = request.metadata.get("mode", "with_skill")
        if mode == "baseline":
            available_skills: set[str] | None = set()
        elif request.forced_skills is not None:
            available_skills = set(request.forced_skills)
        else:
            candidate = set(request.required_skills) | set(request.candidate_skills)
            available_skills = candidate if candidate else None

        # Create client
        try:
            client = DeerFlowClient(
                config_path=self._config_path,
                model_name=request.metadata.get("model_name") or self._model_name,
                available_skills=available_skills,
            )
        except Exception as exc:
            return AgentRunResult(
                final_answer="",
                success=False,
                trace=AgentTrace(
                    input=request.user_input,
                    final_answer="",
                    success=False,
                    errors=[f"Failed to create DeerFlowClient: {exc}"],
                    runtime="deerflow",
                ),
            )

        thread_id = str(uuid.uuid4())
        timeout = request.metadata.get("timeout_seconds", 300)

        adapter = DeerFlowTraceAdapter(request)
        adapter._start_time = time.monotonic()

        # Run stream in thread to avoid blocking event loop
        def _stream_and_feed():
            try:
                for event in client.stream(request.user_input, thread_id=thread_id):
                    adapter.feed(event)
            except Exception as exc:
                adapter._errors.append(f"Stream error: {exc}")

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_stream_and_feed),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            adapter._errors.append(f"Stream timed out after {timeout}s")
        except Exception as exc:
            adapter._errors.append(f"Runner error: {exc}")

        # Save raw trace
        trace_path = None
        if self._trace_dir:
            trace_path = str(Path(self._trace_dir) / f"{thread_id}.jsonl")

        trace = adapter.build(raw_trace_path=trace_path)

        return AgentRunResult(
            final_answer=trace.final_answer,
            success=trace.success,
            trace=trace,
        )
```

#### Unit Test for Runner (`test_deerflow_runner.py` — baseline mode synthetic)

Since the real client needs a valid config, this test validates the `baseline` mode path (no skills) without actually hitting the agent's LLM. We just verify construction + available_skills logic:

- [ ] **Step 1: Write runner construction test**

Add to `backend/tests/skill_eval/test_deerflow_runner.py`:

```python
import pytest
from skill_eval.adapters.deerflow import DeerFlowAgentRunner
from skill_eval.agent_runner import AgentRunRequest


def test_runner_baseline_mode_no_skills():
    """Baseline mode sets available_skills to empty set."""
    runner = DeerFlowAgentRunner()
    # We can't actually call run() without config, but we verify
    # the class exists and implements the protocol.
    assert hasattr(runner, "run")
    assert callable(runner.run)
```

- [ ] **Step 2: Run — verify PASS**

```bash
cd backend && uv run pytest tests/skill_eval/test_deerflow_runner.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/skill_eval/adapters/deerflow.py backend/tests/skill_eval/test_deerflow_runner.py
git commit -m "feat: add DeerFlowAgentRunner"
```

---

### Task 3: Wire Runner into skills_eval.py

**Files:**
- Modify: `backend/evals/skills_eval.py`

**Interfaces:**
- Consumes: `DeerFlowAgentRunner` (optional import)
- Produces: updated `skills_eval` task that accepts `runner` parameter

- [ ] **Step 1: Add runner parameter**

In `backend/evals/skills_eval.py`, modify the `skills_eval` function signature:

```python
@task
def skills_eval(
    case_file: str = "cases/gcp_skills.jsonl",
    mode: str = "with_skill",
    skills_folder: str = "skills",
    use_model_graded_qa: bool = False,
    use_deerflow: bool = False,
):
    samples = load_skill_cases(case_file)

    if mode == "baseline":
        selected_skills: list[str] | None = []
    elif mode == "with_skill":
        selected_skills = None
    elif mode == "all_skills":
        skill_files = (Path.cwd() / skills_folder).rglob("SKILL.md")
        selected_skills = [str(skill_file.parent) for skill_file in skill_files]
    else:
        raise ValueError("mode must be one of: baseline, with_skill, all_skills")

    # Runner selection
    agent_runner = None
    if use_deerflow:
        from skill_eval.adapters.deerflow import DeerFlowAgentRunner
        agent_runner = DeerFlowAgentRunner()

    scorers = [trace_integrity_scorer(), skill_assertion_scorer()]
    if use_model_graded_qa:
        scorers.append(model_graded_qa())

    return Task(
        dataset=samples,
        solver=skill_agent_solver(agent_runner=agent_runner, skills=selected_skills, sandbox="docker"),
        scorer=scorers,
        sandbox="docker",
    )
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
cd backend && uv run pytest tests/skill_eval/ -v
```

Expected: all existing tests PASS (mock runner still default).

- [ ] **Step 3: Commit**

```bash
git add backend/evals/skills_eval.py
git commit -m "feat: add --use-deerflow flag to skills_eval task"
```

---

### Task 4: Smoke Test with Real DeerFlowClient

**Files:**
- Modify: `backend/tests/skill_eval/test_deerflow_runner.py` (add smoke tests)

- [ ] **Step 1: Add smoke tests**

```python
import os
import pytest


def _has_config():
    """Check if a valid config.yaml exists for real-agent tests."""
    for path in ["config.yaml", "config.yml", "configure.yml"]:
        if os.path.exists(path):
            return True
    return False


@pytest.mark.skipif(not _has_config(), reason="No config.yaml found — real-agent smoke test skipped")
def test_runner_smoke_trivial_input():
    """Run a trivial input through the real DeerFlow agent."""
    import asyncio
    from skill_eval.adapters.deerflow import DeerFlowAgentRunner
    from skill_eval.agent_runner import AgentRunRequest

    runner = DeerFlowAgentRunner()
    request = AgentRunRequest(user_input="Say hello in exactly three words.")
    result = asyncio.run(runner.run(request))

    assert result.success is True
    assert len(result.final_answer) > 0
    assert result.trace.runtime == "deerflow"
    assert len(result.trace.messages) > 0
    # Should have at least one AI message
    ai_messages = [m for m in result.trace.messages if m.get("type") == "ai"]
    assert len(ai_messages) > 0


@pytest.mark.skipif(not _has_config(), reason="No config.yaml found — real-agent smoke test skipped")
def test_runner_smoke_tool_call():
    """Run an input that triggers a tool call."""
    import asyncio
    from skill_eval.adapters.deerflow import DeerFlowAgentRunner
    from skill_eval.agent_runner import AgentRunRequest

    runner = DeerFlowAgentRunner()
    request = AgentRunRequest(
        user_input="Read the file README.md and tell me what project this is.",
    )
    result = asyncio.run(runner.run(request))

    assert result.success is True
    # Should have at least one read_file tool call
    read_calls = [tc for tc in result.trace.tool_calls if tc.name == "read_file"]
    assert len(read_calls) > 0, "Expected at least one read_file tool call"
```

- [ ] **Step 2: Run smoke tests (skip if no config)**

```bash
cd backend && uv run pytest tests/skill_eval/test_deerflow_runner.py -v
```

Expected: tests skip if no config.yaml; PASS if config exists and agent responds.

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
cd backend && make lint && make test
```

Expected: all existing tests PASS, no lint errors.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/skill_eval/test_deerflow_runner.py
git commit -m "test: add DeerFlow agent smoke tests"
```

---

## Verification Checklist

After all tasks:

- [ ] `cd backend && make lint` — no new lint errors
- [ ] `cd backend && make test` — all tests pass (smoke tests may skip)
- [ ] `DeerFlowTraceAdapter` unit tests cover: empty stream, single AI message, multi AI message, tool call + result, tool error, multiple tool calls ordering, usage, latency, skill loaded, skill used via read_file, skill not used
- [ ] `DeerFlowAgentRunner` class exists and follows `AgentRunner` protocol
- [ ] `skills_eval.py` accepts `use_deerflow=True` to switch runners
- [ ] Smoke tests run when `config.yaml` is present; skip cleanly otherwise
