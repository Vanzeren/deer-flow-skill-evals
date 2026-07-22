# Skill Eval Phase 3: DeerFlow Agent Adapter

**Goal:** Replace `MockAgentRunner` with a real `DeerFlowAgentRunner` that drives `DeerFlowClient` in-process, collects `AgentTrace` from the live agent run, and feeds it into the existing assertion/scorer pipeline — without changing the eval framework's scorer/assertion contract.

**Status:** Historical design record — implemented architecture has evolved.

> **Do not use this file as the current integration guide.** The production
> implementation now uses spawned child processes rather than an in-process
> runner; DeerFlow tool calls/results arrive through `messages-tuple` events;
> and quick mode accepts text and/or tool calls. See
> [`docs/skill-eval-architecture.md`](../../skill-eval-architecture.md#四如何接入新的-agent-runtime)
> for the current contract, stop conditions, validation ladder, and lessons
> learned. This document remains unchanged below as the original design record.

**Depends on:** Phase 1 (MVP mock runner) and Phase 2 (full deterministic assertions) being green.

---

## Problem

The skill eval framework is built and validated against `MockAgentRunner`. To evaluate real skill behavior, we need to run actual DeerFlow agents with real skills loaded and extract the same `AgentTrace` contract the assertion engine and scorers consume.

Key unknowns:
1. Can we run `DeerFlowClient` in-process within an Inspect eval without Gateway?
2. How do we detect skill loading, selection, and compliance from DeerFlow's stream events?
3. How do we capture tool calls, errors, latency, and token usage from the live run?
4. What config surface does `DeerFlowClient` need (model, sandbox, skills, checkpointer)?

---

## Design Principles

- **Same contract, different runtime.** `DeerFlowAgentRunner` implements `AgentRunner` → produces `AgentRunResult` → `AgentTrace`. Scorers and assertions don't change.
- **In-process first.** Use `DeerFlowClient` (embedded Python client) directly, no Gateway/HTTP. This avoids service orchestration in evals.
- **Skill policy lives in the runner.** The runner decides which skills to force/allow per mode (`baseline`, `with_skill`, `all_skills`), mirroring the mock runner's `forced_skills` logic.
- **Trace extraction is best-effort and documented.** DeerFlow streaming events are not a 1:1 match with `AgentTrace`. The adapter converts what it can; gaps are explicit.
- **Raw trace preserved.** `AgentTrace.raw_trace_ref` stores a path to the full stream event log for post-hoc debugging.
- **Config is injectable.** Model, sandbox, checkpointer, and skills path are passed via `AgentRunRequest.metadata`, not hardcoded.

---

## Architecture

```text
Inspect Task (skills_eval.py)
        ↓
skill_agent_solver(agent_runner=DeerFlowAgentRunner(...))
        ↓
DeerFlowAgentRunner.run(AgentRunRequest)
        ↓
DeerFlowClient.stream(message, thread_id=uuid)
        ↓  (consumes stream events)
DeerFlowTraceAdapter  ←  NEW: converts stream → AgentTrace
        ↓
AgentTrace
  .tool_calls        ← extracted from ToolMessage / AIMessage.tool_calls
  .skill_invocations  ← inferred from skill_activation events + tool calls
  .messages           ← full message history
  .latency_ms         ← wall-clock timing
  .input_tokens       ← from usage metadata
  .output_tokens      ← from usage metadata
  .steps              ← stream event summary
  .raw_trace_ref      ← path to full event log
        ↓
Scorers (unchanged)
  trace_integrity_scorer
  skill_assertion_scorer
```

### New files

```text
backend/skill_eval/adapters/
  deerflow.py          # DeerFlowAgentRunner + DeerFlowTraceAdapter

backend/tests/skill_eval/
  test_deerflow_runner.py     # Unit tests for adapter conversion
  test_deerflow_smoke.py      # Smoke test with real DeerFlowClient (integration)
```

### Modified files

```text
backend/evals/skills_eval.py   # Wire DeerFlowAgentRunner as default (or opt-in)
```

---

## Core Data Flow: Stream Events → AgentTrace

`DeerFlowClient.stream()` yields `StreamEvent` objects with these types:

| `StreamEvent.type` | `data` shape | Maps to `AgentTrace` field |
|---|---|---|
| `messages-tuple` | `(role, content, id, ...)` | `messages[]`, `final_answer` (last AI delta accumulated) |
| `tool_call` | `{name, args, id}` | `tool_calls[].name`, `tool_calls[].args` |
| `tool_result` | `{content, tool_call_id, name}` | `tool_calls[].result` |
| `tool_error` | `{error, tool_call_id, name}` | `tool_calls[].error` |
| `end` | `{usage: {input_tokens, output_tokens}}` | `input_tokens`, `output_tokens` |
| `custom` | varies | `steps[]` (logged but not semantically parsed) |

### Skill invocation detection

DeerFlow does not emit dedicated "skill activated" events. The adapter infers `SkillInvocation` from:

1. **`loaded`**: The runner knows which skills were made available (`forced_skills` / `available_skills`). Every skill in the candidate set is marked `loaded=True`.
2. **`used`**: Detected by scanning tool calls for `read_file` targeting a `SKILL.md` path in the skills directory, or slash-command-style invocations (e.g. `/gcp-deploy` appearing in user/assistant messages). Heuristic — not 100% but sufficient for eval assertions.
3. **`applied`**: Always `None` from the adapter. This is an assertion-level judgment, not a runtime signal. The `skill_applied` / `skill_not_applied` assertions already evaluate this from trace evidence.

### Tool call correlation

DeerFlow emits tool calls and results as separate stream events. The adapter must correlate them by `tool_call_id`:
- Receive `tool_call` event → create `AgentToolCall(name, args)`, stash by id.
- Receive `tool_result` event → attach `result` to the stashed call.
- Receive `tool_error` event → attach `error` to the stashed call.
- At stream end, all stashed calls become `trace.tool_calls`.

---

## DeerFlowTraceAdapter

```python
class DeerFlowTraceAdapter:
    """Converts DeerFlowClient stream events into AgentTrace."""

    def __init__(self, request: AgentRunRequest):
        self.request = request
        self._tool_calls: dict[str, AgentToolCall] = {}
        self._messages: list[dict[str, Any]] = []
        self._steps: list[dict[str, Any]] = []
        self._final_chunks: list[str] = []
        self._last_ai_id: str = ""
        self._start_time: float | None = None
        self._latency_ms: int | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._success: bool = True
        self._errors: list[str] = []
        self._raw_events: list[dict[str, Any]] = []

    def feed(self, event: StreamEvent) -> None:
        """Ingest one stream event."""

    def build(self, raw_trace_path: str | None = None) -> AgentTrace:
        """Assemble final AgentTrace from accumulated events."""
```

Key decisions:
- `feed()` is called per stream event; it accumulates state.
- `build()` constructs the final `AgentTrace` after the stream ends.
- `_raw_events` is serialized to disk as `raw_trace_ref` for debugging.

---

## DeerFlowAgentRunner

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
        checkpointer: Any | None = None,
    ):
        ...

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        ...
```

### Config resolution

| Parameter | Source (priority) |
|---|---|
| `config_path` | Constructor → `DEER_FLOW_CONFIG` env → `config.yaml` (cwd) |
| `model_name` | Constructor → `request.metadata["model_name"]` → app config default |
| `sandbox` | Constructor → `request.sandbox` → `"local"` |
| `skills_dir` | Constructor → `request.metadata["skills_dir"]` → `"skills"` |
| `trace_dir` | Constructor → `request.metadata["trace_dir"]` → `tempfile.mkdtemp()` |

### Run flow

1. Create `DeerFlowClient(config_path=..., model_name=..., checkpointer=...)`.
2. Set `available_skills` based on eval mode:
   - `baseline` → `available_skills=set()` (no skills)
   - `with_skill` → `available_skills=set(request.required_skills + request.candidate_skills)`
   - `all_skills` → `available_skills=None` (all scanned)
3. Call `client.stream(request.user_input, thread_id=uuid4())`.
4. Feed each `StreamEvent` into `DeerFlowTraceAdapter.feed()`.
5. After stream ends, call `adapter.build()` → `AgentTrace`.
6. Save raw event log to `trace_dir / {thread_id}.jsonl`.
7. Return `AgentRunResult(final_answer=..., success=..., trace=...)`.

### Error handling

- If `DeerFlowClient` construction fails (no config, no model) → raise clear error with fix instructions.
- If the stream raises → capture error in `AgentTrace.errors`, set `success=False`, still return a partial trace.
- If the stream times out → configurable timeout via `request.metadata["timeout_seconds"]` (default 300).

---

## Skill selection policy

The runner respects the same `forced_skills` semantics as the mock runner, but maps them to DeerFlow's `available_skills`:

| Eval mode | `forced_skills` | `available_skills` | Effect |
|---|---|---|---|
| `baseline` | `[]` | `set()` | No skills available → agent runs without skill guidance |
| `with_skill` | `None` (not forced) | `{required + candidate}` | Agent sees only relevant skills; auto-selects via skill middleware |
| `with_skill` | `["gcp-deploy"]` | `{"gcp-deploy"}` | Agent forced to use specific skill |
| `all_skills` | `None` | `None` (all scanned) | Agent sees all installed skills |

---

## Validation Plan

### Unit tests (`test_deerflow_runner.py`)

1. **Adapter: empty stream** → `AgentTrace` with no tool calls, no skills, empty messages.
2. **Adapter: single AI message** → `final_answer` equals accumulated text.
3. **Adapter: tool call + result** → `AgentToolCall` has name, args, result, no error.
4. **Adapter: tool call + error** → `AgentToolCall` has name, args, error set.
5. **Adapter: multiple tool calls** → all appear in `trace.tool_calls` in order.
6. **Adapter: usage metadata** → `input_tokens` and `output_tokens` extracted from `end` event.
7. **Adapter: latency** → `latency_ms` > 0 and ≤ timeout.
8. **Adapter: skill invocation — loaded** → skills in `request.required_skills` appear as `loaded=True`.
9. **Adapter: skill invocation — used** → `read_file("skills/gcp-deploy/SKILL.md")` tool call marks `gcp-deploy` as `used=True`.
10. **Adapter: skill invocation — not used** → skill in candidate set but no read → `used=False`.
11. **Runner: construction with no config** → clear error message.
12. **Runner: baseline mode** → `available_skills` is empty set.

### Smoke test (`test_deerflow_smoke.py`, integration)

1. **Real agent runs without crash** — `DeerFlowAgentRunner.run()` with a trivial input ("say hello") returns `AgentRunResult(success=True)`.
2. **Real skill loading** — run with `request.required_skills=["gcp-deploy"]` and verify `trace.skill_invocations` includes `gcp-deploy` with `loaded=True`.
3. **Tool calls captured** — run with a task that triggers a tool (e.g. `read_file`) and verify `trace.tool_calls` is non-empty.
4. **End-to-end eval task** — run `skills_eval(case_file="cases/gcp_skills.jsonl")` with `DeerFlowAgentRunner` and verify it completes without framework errors (assertion failures are fine — those are case quality issues).

Smoke tests require a valid `config.yaml` with at least one model configured. They SHOULD be skipped with a clear message when no config exists.

---

## Configuration

### Required for real-agent runs

```yaml
# config.yaml (minimal)
models:
  - name: "default"
    model: "claude-sonnet-4-20250514"  # or any configured model
    provider: "anthropic"               # or "openai", "deepseek", etc.
```

### Optional eval-specific config

```yaml
# Can be passed via metadata or env
skill_eval:
  timeout_seconds: 300
  trace_dir: "./eval_traces"
  sandbox: "local"
```

---

## Non-goals (explicitly deferred)

- **No Gateway/HTTP mode.** Phase 3 is in-process only. A Gateway-backed runner (`DeerFlowGatewayRunner`) is a future option if the in-process path proves insufficient.
- **No sandbox integration.** The runner sets `sandbox="local"` (no Docker). Sandboxed eval runs are Phase 4+.
- **No baseline comparison.** `comparison.py` is Phase 4.
- **No model-graded scoring.** `model_graded_qa` is Phase 5.
- **No concurrency.** One eval run = one agent invocation. Concurrent eval runs are a CI concern, not an adapter concern.
- **No skill compliance auto-detection.** `SkillInvocation.applied` remains `None` from the adapter.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `DeerFlowClient` requires a full `config.yaml` with model credentials | Smoke tests skip gracefully when no config; docs state prerequisites clearly |
| Stream event schema is not a public API and may change | Adapter pins to current `StreamEvent` shape; raw event log preserved for debugging; unit tests freeze the expected shapes |
| Skill activation detection is heuristic | `skill_used` assertion metadata includes the evidence (which tool call triggered it); false negatives are debugable from `raw_trace_ref` |
| Real agent runs are slow (10-60s per case) | Unit tests use synthetic events (fast); smoke test uses trivial inputs; full evals are expected to be slow |
| `DeerFlowClient` is synchronous but inspect solver is async | Wrap `client.stream()` in `asyncio.to_thread()` or use a thread-pool executor |

---

## Acceptance Criteria

1. `DeerFlowAgentRunner` implements `AgentRunner` protocol.
2. Unit tests pass with synthetic stream events — no real agent needed.
3. Smoke test runs with real `DeerFlowClient` when `config.yaml` is present; skips cleanly otherwise.
4. `skill_assertion_scorer` and `trace_integrity_scorer` work unchanged against traces from the DeerFlow adapter.
5. `skills_eval.py` task can be configured to use `DeerFlowAgentRunner` via parameter or env var.
6. Existing mock-based tests continue to pass.
7. Backend lint and tests pass (`make lint && make test`).
