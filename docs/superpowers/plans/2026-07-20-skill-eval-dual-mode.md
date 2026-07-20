# Skill Eval Dual-Mode (Quick Turn / Full Output) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quick evaluation mode that judges the first assistant text turn after skill load, keep full mode judging final output, drop message history from judge evidence, and record the batch-structured tool-call chain in traces.

**Architecture:** Single-run quick capture: the DeerFlow adapter keeps consuming the stream after the routing decision until the first non-empty AI text turn completes, stores it as `AgentTrace.quick_turn`, and derives `tool_call_chain` (outer = time-ordered batches, inner = concurrent tool-call ids from one AI message) from data the adapter already collects. Judge evidence replaces `message[N]`/`tool_call[N]`/`tool_result[N]` items with `tool_chain[B]` batch items; a new `QuickJudgment` schema and `quick_turn_scorer` evaluate the captured turn. The POC gains `--quality-mode quick|full|both`.

**Tech Stack:** Python 3.12, pydantic v2, Inspect AI (`inspect_ai`), pytest + pytest-asyncio, ruff.

**Spec:** `docs/superpowers/specs/2026-07-20-skill-eval-dual-mode-design.md`

## Global Constraints

- Python >= 3.12; pydantic v2 models with `Field(default_factory=...)` for list fields.
- All backend commands run from `backend/`: lint `make lint`, focused tests `uv run pytest tests/skill_eval/<file> -v`.
- Async tests MUST use `@pytest.mark.asyncio` (project uses pytest-asyncio in marker mode).
- `JudgeEvidenceBundle` MUST NOT contain `expected_route` or the case rationale (existing leak guard, enforced by tests). Do not add them.
- `AgentTrace.messages` keeps being collected; it only leaves the judge evidence path. Reports never inline messages.
- `routing.py` observation LOGIC is frozen; only a read-only accessor may be added.
- No compatibility shim for removed `message`/`tool_call`/`tool_result` evidence kinds — clean cutover, update the tests that reference them.
- Quality-tagged cases in `cases/literature_skill_routing.jsonl`: 4 total, of which 3 expect a skill and 1 (`none-precision-recall-001`) expects `none`. Smoke ids: `slr-attention-variants-001`, `paper-review-arxiv-001`, `none-precision-recall-001`.

---

### Task 1: Tool-call chain in the collected trace

**Files:**
- Modify: `backend/skill_eval/trace_schema.py`
- Modify: `backend/skill_eval/adapters/deerflow.py` (`DeerFlowTraceAdapter.build`, plus two accessors)
- Test: `backend/tests/skill_eval/test_deerflow_adapter.py`

**Interfaces:**
- Produces: `QuickTurnCapture(message_id: str, skill: str, content: str)`; `AgentTrace.tool_call_chain: list[list[str]]`; `AgentTrace.quick_turn: QuickTurnCapture | None`; `DeerFlowTraceAdapter.ai_message_ids() -> tuple[str, ...]`; `DeerFlowTraceAdapter.ai_message_content(message_id: str) -> str`. Tasks 2–4 consume all of these.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/skill_eval/test_deerflow_adapter.py`:

```python
def test_adapter_builds_tool_call_chain_grouped_by_ai_message():
    adapter = DeerFlowTraceAdapter(request())
    adapter.start()
    adapter.feed(
        event(
            "messages-tuple",
            {
                "type": "ai",
                "id": "m1",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "name": "read_file", "args": {"path": "a"}},
                    {"id": "t2", "name": "bash", "args": {"cmd": "ls"}},
                ],
            },
        )
    )
    adapter.feed(event("messages-tuple", {"type": "tool", "id": "r1", "tool_call_id": "t1", "name": "read_file", "content": "x"}))
    adapter.feed(event("messages-tuple", {"type": "ai", "id": "m2", "content": "thinking"}))
    adapter.feed(
        event(
            "messages-tuple",
            {
                "type": "ai",
                "id": "m3",
                "content": "",
                "tool_calls": [{"id": "t3", "name": "write_file", "args": {"path": "b"}}],
            },
        )
    )

    trace = adapter.build(thread_id="thread-1")

    assert trace.tool_call_chain == [["t1", "t2"], ["t3"]]
    chained = [call_id for batch in trace.tool_call_chain for call_id in batch]
    assert sorted(chained) == sorted(call.id for call in trace.tool_calls)
    assert trace.quick_turn is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/skill_eval/test_deerflow_adapter.py::test_adapter_builds_tool_call_chain_grouped_by_ai_message -v`
Expected: FAIL with `AttributeError: 'AgentTrace' object has no attribute 'tool_call_chain'` (pydantic `ValidationError` on unexpected keyword is also acceptable — the point is it fails).

- [ ] **Step 3: Implement schema fields and chain derivation**

In `backend/skill_eval/trace_schema.py`, add before `AgentTrace`:

```python
class QuickTurnCapture(BaseModel):
    message_id: str
    skill: str
    content: str
```

Add two fields to `AgentTrace` (after `tool_calls`):

```python
    tool_call_chain: list[list[str]] = Field(default_factory=list)
    quick_turn: QuickTurnCapture | None = None
```

In `backend/skill_eval/adapters/deerflow.py`:
1. Import the new model: change the trace_schema import to
   `from skill_eval.trace_schema import AgentArtifact, AgentToolCall, AgentTrace, QuickTurnCapture`
2. Add accessors to `DeerFlowTraceAdapter` (next to `artifact_paths` property):

```python
    def ai_message_ids(self) -> tuple[str, ...]:
        return tuple(self._ai_messages)

    def ai_message_content(self, message_id: str) -> str:
        message = self._ai_messages.get(message_id)
        return str(message["content"]) if message is not None else ""
```

3. In `DeerFlowTraceAdapter.build`, derive the chain and pass it to `AgentTrace`. Immediately before `final_answer = ""`:

```python
        tool_call_chain = [
            [call["id"] for call in message["tool_calls"]]
            for message in self._messages
            if message["type"] == "ai" and message["tool_calls"]
        ]
```

and add `tool_call_chain=tool_call_chain,` to the `AgentTrace(...)` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_deerflow_adapter.py -v`
Expected: PASS (new test + all existing adapter tests; existing traces default to empty chain and `quick_turn=None`).

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/trace_schema.py backend/skill_eval/adapters/deerflow.py backend/tests/skill_eval/test_deerflow_adapter.py
git commit -m "feat(skill-eval): record batch-structured tool_call_chain in traces"
```

---

### Task 2: Quick run mode with quick-turn capture

**Files:**
- Modify: `backend/skill_eval/agent_runner.py` (RunMode literal)
- Modify: `backend/skill_eval/routing.py` (read-only accessor only)
- Modify: `backend/skill_eval/adapters/deerflow.py` (`_QuickTurnWatcher`, event loop, `_build_result`)
- Test: `backend/tests/skill_eval/test_deerflow_adapter.py`

**Interfaces:**
- Consumes: Task 1's `QuickTurnCapture`, `ai_message_ids`, `ai_message_content`.
- Produces: `RunMode = Literal["routing_probe", "quick", "full"]`; `RoutingObserver.decided_route: RouteLabel | Literal["ambiguous"] | None` (property, `None` until completed); `_QuickTurnWatcher` with `.skill`, `.target_id`, `.content`, `.complete`, `.start(skill=, existing_message_ids=)`, `.feed(event, adapter)`; quick mode fills `AgentTrace.quick_turn`. Task 5's scorer and Task 7's eval task consume these.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/skill_eval/test_deerflow_adapter.py`:

```python
def ai_text(message_id: str, content: str) -> StreamEvent:
    return event("messages-tuple", {"type": "ai", "id": message_id, "content": content, "tool_calls": []})


def test_quick_turn_watcher_waits_for_accumulated_content():
    adapter = DeerFlowTraceAdapter(request(mode="quick"))
    adapter.start()
    watcher = deerflow_module._QuickTurnWatcher()
    watcher.start(skill="systematic-literature-review", existing_message_ids=("m1",))

    e1 = event("messages-tuple", {"type": "ai", "id": "m2", "content": ""})
    adapter.feed(e1)
    watcher.feed(e1, adapter)
    assert watcher.target_id is None

    e2 = event("messages-tuple", {"type": "ai", "id": "m2", "content": "Now "})
    adapter.feed(e2)
    watcher.feed(e2, adapter)
    assert watcher.target_id == "m2"
    assert watcher.complete is False

    e3 = event("messages-tuple", {"type": "tool", "id": "r9", "tool_call_id": "t9", "name": "bash", "content": "x"})
    adapter.feed(e3)
    watcher.feed(e3, adapter)
    assert watcher.complete is True
    assert watcher.content == "Now "


def test_quick_mode_captures_first_text_turn_after_skill_load():
    holder = {}

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[
                ai_read_skill("systematic-literature-review"),
                tool_result(),
                ai_text("m2", "I will search for papers on the topic."),
                ai_text("m3", "this turn must not be captured"),
                AssertionError,
            ],
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(mode="quick"), client_factory=client_factory)

    assert result.success is True
    assert result.route_observation.observed_route == "systematic-literature-review"
    assert result.trace.quick_turn is not None
    assert result.trace.quick_turn.message_id == "m2"
    assert result.trace.quick_turn.skill == "systematic-literature-review"
    assert result.trace.quick_turn.content == "I will search for papers on the topic."
    assert holder["client"].stream_closed is True
    assert holder["client"].options["subagent_enabled"] is False


def test_quick_mode_none_route_runs_to_end_without_quick_turn():
    holder = {}

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[ai_text("m1", "Direct answer"), event("end", {"usage": {}})],
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(mode="quick"), client_factory=client_factory)

    assert result.success is True
    assert result.route_observation.observed_route == "none"
    assert result.trace.quick_turn is None


def test_quick_mode_breaks_immediately_on_ambiguous_route():
    holder = {}

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[
                event(
                    "messages-tuple",
                    {
                        "type": "ai",
                        "id": "m1",
                        "content": "",
                        "tool_calls": [
                            {"id": "t1", "name": "read_file", "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"}},
                            {"id": "t2", "name": "read_file", "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"}},
                        ],
                    },
                ),
                tool_result("t1"),
                tool_result("t2"),
                AssertionError,
            ],
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(mode="quick"), client_factory=client_factory)

    assert result.route_observation.observed_route == "ambiguous"
    assert result.trace.quick_turn is None


def test_quick_mode_stream_end_without_text_turn_leaves_quick_turn_none():
    holder = {}

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[
                ai_read_skill("systematic-literature-review"),
                tool_result(),
                event(
                    "messages-tuple",
                    {
                        "type": "ai",
                        "id": "m2",
                        "content": "",
                        "tool_calls": [{"id": "t2", "name": "bash", "args": {"cmd": "ls"}}],
                    },
                ),
                tool_result("t2", "files"),
                event("end", {"usage": {}}),
            ],
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(mode="quick"), client_factory=client_factory)

    assert result.success is True
    assert result.route_observation.observed_route == "systematic-literature-review"
    assert result.trace.quick_turn is None
```

(The watcher test goes through the `deerflow_module` alias the file already imports, so no new import is needed.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/skill_eval/test_deerflow_adapter.py -k quick -v`
Expected: FAIL — `AgentRunRequest` rejects `mode="quick"` (literal validation) and `_QuickTurnWatcher` does not exist.

- [ ] **Step 3: Implement quick mode**

3a. `backend/skill_eval/agent_runner.py` — change the RunMode line:

```python
type RunMode = Literal["routing_probe", "quick", "full"]
```

3b. `backend/skill_eval/routing.py` — add a read-only property to `RoutingObserver` (after `__init__`):

```python
    @property
    def decided_route(self) -> RouteLabel | Literal["ambiguous"] | None:
        return self._observed if self._completed else None
```

3c. `backend/skill_eval/adapters/deerflow.py` — add the watcher class after `DeerFlowTraceAdapter` (before `snapshot_artifact`):

```python
class _QuickTurnWatcher:
    """Track the first non-empty AI text turn after a skill-load routing decision."""

    def __init__(self) -> None:
        self.skill: str | None = None
        self.target_id: str | None = None
        self.content: str = ""
        self.complete: bool = False
        self._excluded_ids: set[str] = set()

    def start(self, *, skill: str, existing_message_ids: tuple[str, ...]) -> None:
        self.skill = skill
        self._excluded_ids = set(existing_message_ids)

    def feed(self, event: StreamEvent, adapter: DeerFlowTraceAdapter) -> None:
        if self.skill is None or self.complete:
            return
        if event.type == "end":
            if self.target_id is not None:
                self.content = adapter.ai_message_content(self.target_id)
                self.complete = True
            return
        if event.type != "messages-tuple":
            return
        message_id = str(event.data.get("id") or "")
        if self.target_id is None:
            if event.data.get("type") != "ai" or message_id in self._excluded_ids:
                return
            if adapter.ai_message_content(message_id).strip():
                self.target_id = message_id
            return
        if message_id != self.target_id:
            self.content = adapter.ai_message_content(self.target_id)
            self.complete = True
```

3d. In `_execute_deerflow`, rename `stopped_on_route` to `stopped_early` (all occurrences in that function), create the watcher, and extend the loop. Replace the block from `stream = None` through the end of the `try:` loop with:

```python
    stream = None
    saw_end = False
    stopped_early = False
    stream_failed = False
    watcher = _QuickTurnWatcher() if request.mode == "quick" else None
    try:
        with _sandbox_context(request.sandbox):
            stream = client.stream(request.user_input, thread_id=request.thread_id)
            for stream_event in stream:
                adapter.feed(stream_event)
                route_ready = observer.feed(stream_event)
                if stream_event.type == "end":
                    saw_end = True
                if request.mode == "routing_probe" and route_ready:
                    stopped_early = True
                    break
                if watcher is not None:
                    if route_ready and watcher.skill is None:
                        decided = observer.decided_route
                        if decided == "ambiguous":
                            stopped_early = True
                            break
                        if decided is not None:
                            watcher.start(skill=decided, existing_message_ids=adapter.ai_message_ids())
                    watcher.feed(stream_event, adapter)
                    if watcher.complete:
                        stopped_early = True
                        break
```

Update the guard after the loop:

```python
    if not saw_end and not stopped_early and not stream_failed:
```

After the artifacts block, before `observation = observer.finalize(...)`, build the capture:

```python
    quick_turn = None
    if watcher is not None and watcher.complete and watcher.skill is not None and watcher.target_id is not None:
        quick_turn = QuickTurnCapture(
            message_id=watcher.target_id,
            skill=watcher.skill,
            content=watcher.content,
        )
```

and change the final line of `_execute_deerflow` to pass it:

```python
    return _build_result(request, adapter, observation, artifacts=artifacts, quick_turn=quick_turn)
```

3e. `_build_result` — new keyword and trace update:

```python
def _build_result(
    request: AgentRunRequest,
    adapter: DeerFlowTraceAdapter,
    observation: RouteObservation,
    *,
    artifacts: list[AgentArtifact] | None = None,
    quick_turn: QuickTurnCapture | None = None,
) -> AgentRunResult:
```

Immediately after the `try/except OSError` block that builds `trace`, add:

```python
    if quick_turn is not None:
        trace = trace.model_copy(update={"quick_turn": quick_turn})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_deerflow_adapter.py tests/skill_eval/test_routing.py tests/skill_eval/test_deerflow_runner.py -v`
Expected: PASS (probe/full paths unchanged; `subagent_enabled` is `False` for quick because it keys off `mode == "full"`).

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/agent_runner.py backend/skill_eval/routing.py backend/skill_eval/adapters/deerflow.py backend/tests/skill_eval/test_deerflow_adapter.py
git commit -m "feat(skill-eval): add quick run mode capturing first post-load text turn"
```

---

### Task 3: Judge evidence without message trace

**Files:**
- Modify: `backend/skill_eval/judge.py`
- Test: `backend/tests/skill_eval/test_judge.py`

**Interfaces:**
- Consumes: `AgentTrace.tool_call_chain`, `AgentTrace.quick_turn` (Tasks 1–2).
- Produces: `EvidenceKind = Literal["tool_chain", "error", "artifact", "final_answer", "quick_turn"]`; `_PROCESS_EVIDENCE_KINDS = {"tool_chain", "error"}`; `_OUTPUT_EVIDENCE_KINDS = {"artifact", "final_answer", "quick_turn"}`; `build_judge_evidence(..., target: Literal["final_output", "quick_turn"] = "final_output")`; `JudgeEvidenceBundle.evaluation_target`; `_validate_evidence_references(bundle, references: list[str])`; module-level `_strip_fences(text) -> str`; `_build_repair_prompt(output, error, schema_model)`. Tasks 4–5 consume these.

- [ ] **Step 1: Update existing tests to the new contract (failing)**

In `backend/tests/skill_eval/test_judge.py`:

1. `full_trace` fixture: add `tool_call_chain=[["t1"]],` to the `AgentTrace(...)` call.
2. `valid_judgment_json`: change default evidence to `evidence or ["tool_chain[0]", "final_answer"]`.
3. `test_judge_bundle_omits_expected_label_and_rationale`: replace the five id assertions with:

```python
    assert "message[" not in payload
    assert "tool_call[" not in payload
    assert "tool_result[" not in payload
    assert "tool_chain[0]" in payload
    assert "artifact[report.md]" in payload
    assert "final_answer" in payload
```

4. `test_large_evidence_is_head_tail_truncated_with_hash` and `test_exhausted_evidence_budget_retains_omission_marker`: change the `bounded_evidence(...)` kind argument from `"tool_result"` to `"tool_chain"` (both call sites; the id string can stay or become `"tool_chain[0]"`).
5. `test_judge_rejects_unknown_evidence_without_repair`: change evidence to `["tool_chain[999]", "final_answer"]`.
6. Replace `test_judge_requires_trace_evidence` with:

```python
@pytest.mark.asyncio
async def test_judge_requires_process_evidence(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["final_answer"])])

    with pytest.raises(JudgeFailure, match="tool chain or error evidence"):
        await judge_quality(valid_bundle, model)
```

7. `test_judge_auto_adds_output_evidence_when_missing`: change evidence to `["tool_chain[0]"]` and the `output_kinds` set to `{"final_answer", "artifact"}`.

Add the new tests:

```python
def test_tool_chain_evidence_expands_concurrent_batch(routing_case, route_observation):
    trace = AgentTrace(
        input="Synthesize three papers.",
        final_answer="answer",
        success=True,
        thread_id="thread-1",
        tool_calls=[
            AgentToolCall(id="t1", message_id="m1", name="read_file", args={"path": "SKILL.md"}, result="body"),
            AgentToolCall(id="t2", message_id="m1", name="bash", args={"cmd": "ls"}, result="files"),
            AgentToolCall(id="t3", message_id="m2", name="write_file", args={"path": "out.md"}, result="ok"),
        ],
        tool_call_chain=[["t1", "t2"], ["t3"]],
    )

    bundle = build_judge_evidence(
        case=routing_case,
        trace=trace,
        observation=route_observation,
        skill_descriptions={
            "systematic-literature-review": "multi-paper",
            "academic-paper-review": "one-paper",
        },
    )

    chain_items = [item for item in bundle.evidence if item.kind == "tool_chain"]
    assert [item.id for item in chain_items] == ["tool_chain[0]", "tool_chain[1]"]
    first = json.loads(chain_items[0].content)
    assert [call["name"] for call in first] == ["read_file", "bash"]
    assert first[0]["result"] == "body"
    assert bundle.evaluation_target == "final_output"
    assert all(item.kind != "message" for item in bundle.evidence)


def test_quick_target_excludes_captured_turn_batch_and_final_answer(routing_case, route_observation):
    trace = AgentTrace(
        input="Review the paper.",
        final_answer="",
        success=True,
        thread_id="thread-1",
        tool_calls=[
            AgentToolCall(id="t1", message_id="m1", name="read_file", args={"path": "SKILL.md"}, result="skill"),
            AgentToolCall(id="t2", message_id="m2", name="bash", args={"cmd": "ls"}, result="x"),
        ],
        tool_call_chain=[["t1"], ["t2"]],
        quick_turn=QuickTurnCapture(message_id="m2", skill="systematic-literature-review", content="Plan: ..."),
    )

    bundle = build_judge_evidence(
        case=routing_case,
        trace=trace,
        observation=route_observation,
        skill_descriptions={
            "systematic-literature-review": "multi-paper",
            "academic-paper-review": "one-paper",
        },
        target="quick_turn",
    )

    ids = [item.id for item in bundle.evidence]
    assert ids == ["tool_chain[0]", "quick_turn"]
    assert bundle.evaluation_target == "quick_turn"
    quick_item = bundle.evidence[-1]
    assert quick_item.kind == "quick_turn"
    assert quick_item.content == "Plan: ..."


def test_quick_target_requires_captured_turn(routing_case, full_trace, route_observation):
    with pytest.raises(ValueError, match="quick turn"):
        build_judge_evidence(
            case=routing_case,
            trace=full_trace,
            observation=route_observation,
            skill_descriptions={
                "systematic-literature-review": "multi-paper",
                "academic-paper-review": "one-paper",
            },
            target="quick_turn",
        )


@pytest.mark.asyncio
async def test_judgment_may_cite_only_output_when_no_process_evidence(routing_case, route_observation):
    trace = AgentTrace(
        input="Say hi.",
        final_answer="hi",
        success=True,
        thread_id="thread-1",
    )
    bundle = build_judge_evidence(
        case=routing_case,
        trace=trace,
        observation=route_observation,
        skill_descriptions={
            "systematic-literature-review": "multi-paper",
            "academic-paper-review": "one-paper",
        },
    )
    model = FakeModel([valid_judgment_json(evidence=["final_answer"])])

    judgment = await judge_quality(bundle, model)

    assert judgment.overall_quality == 3
```

Update the import block of the test file to include `QuickTurnCapture`:

```python
from skill_eval.trace_schema import AgentArtifact, AgentToolCall, AgentTrace, QuickTurnCapture
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/skill_eval/test_judge.py -v`
Expected: FAIL — `bounded_evidence` rejects kind `"tool_chain"`, `build_judge_evidence` has no `target` parameter, evidence still contains `message[0]`.

- [ ] **Step 3: Implement evidence rework in judge.py**

3a. Replace the kinds block:

```python
_PROCESS_EVIDENCE_KINDS = {"tool_chain", "error"}
_OUTPUT_EVIDENCE_KINDS = {"artifact", "final_answer", "quick_turn"}
```

(delete `_TRACE_EVIDENCE_KINDS`) and:

```python
type EvidenceKind = Literal[
    "tool_chain",
    "error",
    "artifact",
    "final_answer",
    "quick_turn",
]
```

3b. `JudgeEvidenceBundle` gains a field (keep `expected_route` OUT of the bundle):

```python
class JudgeEvidenceBundle(BaseModel):
    user_input: str
    candidate_skills: dict[str, str]
    observed_route: str
    evaluation_target: Literal["quick_turn", "final_output"] = "final_output"
    evidence: list[EvidenceItem]
    expected_output: str | None = None
```

3c. `build_judge_evidence` — new signature and body for the item-collection portion:

```python
def build_judge_evidence(
    *,
    case: RoutingCase,
    trace: AgentTrace,
    observation: RouteObservation,
    skill_descriptions: dict[str, str],
    target: Literal["final_output", "quick_turn"] = "final_output",
) -> JudgeEvidenceBundle:
    expected_candidates = set(CANDIDATE_SKILLS)
    if set(skill_descriptions) != expected_candidates:
        raise ValueError("skill descriptions must contain exactly: " + ", ".join(CANDIDATE_SKILLS))
    quick_turn = trace.quick_turn
    if target == "quick_turn" and quick_turn is None:
        raise ValueError("quick_turn target requires a captured quick turn")

    calls_by_id = {call.id: call for call in trace.tool_calls}
    batches = trace.tool_call_chain
    if target == "quick_turn" and quick_turn is not None:
        batches = [
            batch
            for batch in batches
            if batch and calls_by_id[batch[0]].message_id != quick_turn.message_id
        ]

    raw_items: list[tuple[str, EvidenceKind, str]] = []
    for index, batch in enumerate(batches):
        if not batch:
            continue
        calls = []
        for call_id in batch:
            call = calls_by_id[call_id]
            calls.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.args,
                    "result": call.result,
                    "error": call.error,
                }
            )
        raw_items.append(
            (
                f"tool_chain[{index}]",
                "tool_chain",
                json.dumps(calls, ensure_ascii=False, sort_keys=True, default=str),
            )
        )
    for index, error in enumerate(trace.errors):
        raw_items.append((f"error[{index}]", "error", error))
```

Keep the artifact loop unchanged. Replace the trailing `final_answer` append with:

```python
    if target == "quick_turn" and quick_turn is not None:
        raw_items.append(("quick_turn", "quick_turn", quick_turn.content))
    else:
        raw_items.append(("final_answer", "final_answer", trace.final_answer))
```

Keep the budget loop unchanged. In the returned bundle add `evaluation_target=target,`.

3d. Move `_strip_fences` to module level (delete the nested copy inside `judge_quality`):

```python
def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text
```

3e. Parameterize the repair prompt:

```python
def _build_repair_prompt(output: str, error: Exception, schema_model: type[BaseModel]) -> str:
    schema = json.dumps(schema_model.model_json_schema(), ensure_ascii=False, sort_keys=True)
    return f"""format correction only; do not reconsider scores or reasons.
Return only corrected JSON matching this schema:
{schema}
Original output:
{output}
Parse or schema error:
{error}"""
```

Update the call in `judge_quality` to `_build_repair_prompt(output.completion, exc, QualityJudgment)`.

3f. New validation signature and rules:

```python
def _validate_evidence_references(
    bundle: JudgeEvidenceBundle,
    references: list[str],
) -> None:
    items = {item.id: item for item in bundle.evidence}
    unknown = [reference for reference in references if reference not in items]
    if unknown:
        raise JudgeFailure(f"unknown evidence reference(s): {', '.join(unknown)}")
    referenced_kinds = {items[reference].kind for reference in references}
    process_ids = {item.id for item in bundle.evidence if item.kind in _PROCESS_EVIDENCE_KINDS}
    if process_ids and not referenced_kinds.intersection(_PROCESS_EVIDENCE_KINDS):
        raise JudgeFailure("judgment must cite tool chain or error evidence")
    if not referenced_kinds.intersection(_OUTPUT_EVIDENCE_KINDS):
        raise JudgeFailure("judgment must cite output evidence")
```

3g. In `judge_quality`, replace the validation block with:

```python
    try:
        _validate_evidence_references(bundle, judgment.evidence)
    except JudgeFailure:
        output_ids = sorted(item.id for item in bundle.evidence if item.kind in _OUTPUT_EVIDENCE_KINDS)
        for reference in output_ids:
            if reference not in judgment.evidence:
                judgment.evidence.append(reference)
        _validate_evidence_references(bundle, judgment.evidence)

    return judgment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_judge.py -v`
Expected: PASS, including the new quick-target tests and the no-process-evidence allowance.

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/judge.py backend/tests/skill_eval/test_judge.py
git commit -m "feat(skill-eval): replace message trace with tool-chain evidence in judge bundles"
```

---

### Task 4: QuickJudgment schema and quick-turn judge

**Files:**
- Modify: `backend/skill_eval/judge.py`
- Test: `backend/tests/skill_eval/test_judge.py`

**Interfaces:**
- Consumes: Task 3's `build_judge_evidence(target="quick_turn")`, `_strip_fences`, `_build_repair_prompt(..., schema_model)`, `_validate_evidence_references`.
- Produces: `QuickJudgment(turn_quality: int, fatal_error: bool, rationale: str, evidence_references: list[str])`; `build_quick_judge_prompt(bundle) -> str`; `judge_quick_turn(bundle, model) -> QuickJudgment`. Task 5 consumes these.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/skill_eval/test_judge.py`:

```python
def valid_quick_judgment_json(evidence=None, **updates):
    payload = {
        "turn_quality": 3,
        "fatal_error": False,
        "rationale": "The turn follows the loaded skill workflow.",
        "evidence_references": evidence or ["tool_chain[0]", "quick_turn"],
    }
    payload.update(updates)
    return json.dumps(payload)


@pytest.fixture
def quick_bundle(routing_case, route_observation):
    trace = AgentTrace(
        input="Review the paper.",
        final_answer="",
        success=True,
        thread_id="thread-1",
        tool_calls=[
            AgentToolCall(id="t1", message_id="m1", name="read_file", args={"path": "SKILL.md"}, result="skill"),
        ],
        tool_call_chain=[["t1"]],
        quick_turn=QuickTurnCapture(message_id="m2", skill="systematic-literature-review", content="Plan: ..."),
    )
    return build_judge_evidence(
        case=routing_case,
        trace=trace,
        observation=route_observation,
        skill_descriptions={
            "systematic-literature-review": "multi-paper",
            "academic-paper-review": "one-paper",
        },
        target="quick_turn",
    )


@pytest.mark.asyncio
async def test_quick_judge_parses_structured_result(quick_bundle):
    model = FakeModel([valid_quick_judgment_json()])

    judgment = await judge_quick_turn(quick_bundle, model)

    assert judgment.turn_quality == 3
    assert judgment.fatal_error is False
    assert "first assistant text turn" in model.prompts[0]
    assert "expected_route" not in model.prompts[0]


@pytest.mark.asyncio
async def test_quick_judge_repairs_format_once(quick_bundle):
    model = FakeModel(["not json", valid_quick_judgment_json()])

    judgment = await judge_quick_turn(quick_bundle, model)

    assert judgment.turn_quality == 3
    assert len(model.prompts) == 2


@pytest.mark.asyncio
async def test_quick_judge_rejects_second_parse_failure(quick_bundle):
    model = FakeModel(["not json", "still not json"])

    with pytest.raises(JudgeFailure, match="after format repair"):
        await judge_quick_turn(quick_bundle, model)


@pytest.mark.asyncio
async def test_quick_judge_rejects_unknown_evidence(quick_bundle):
    model = FakeModel([valid_quick_judgment_json(evidence=["tool_chain[9]", "quick_turn"])])

    with pytest.raises(JudgeFailure, match="unknown evidence"):
        await judge_quick_turn(quick_bundle, model)


@pytest.mark.asyncio
async def test_quick_judge_requires_process_evidence(quick_bundle):
    model = FakeModel([valid_quick_judgment_json(evidence=["quick_turn"])])

    with pytest.raises(JudgeFailure, match="tool chain or error evidence"):
        await judge_quick_turn(quick_bundle, model)


@pytest.mark.asyncio
async def test_quick_judge_auto_adds_output_evidence_when_missing(quick_bundle):
    model = FakeModel([valid_quick_judgment_json(evidence=["tool_chain[0]"])])

    judgment = await judge_quick_turn(quick_bundle, model)

    assert "quick_turn" in judgment.evidence_references


def test_quick_judgment_rejects_blank_rationale():
    with pytest.raises(ValidationError):
        QuickJudgment.model_validate_json(valid_quick_judgment_json(rationale="  "))
```

Update the judge imports in the test file:

```python
from skill_eval.judge import (
    JudgeFailure,
    QuickJudgment,
    bounded_evidence,
    build_judge_evidence,
    judge_quality,
    judge_quick_turn,
    load_candidate_skill_descriptions,
)
```

(`ValidationError` from pydantic is already imported in the module under test; add `from pydantic import ValidationError` to the test file if absent.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/skill_eval/test_judge.py -k quick_judge -v`
Expected: FAIL — `judge_quick_turn` / `QuickJudgment` do not exist.

- [ ] **Step 3: Implement QuickJudgment and judge_quick_turn**

In `backend/skill_eval/judge.py`:

3a. Rubric constant (next to the other rubrics):

```python
_QUICK_TURN_RUBRIC = """Quick-turn rubric — you are scoring exactly one assistant turn:
the first text turn the agent produced after loading the named skill.
- the turn follows the loaded skill's workflow, format, and constraints;
- the turn responds to the user's actual request;
- the turn is coherent and self-sufficient given the observable evidence.
Do not score later steps; do not infer work that is not in the evidence."""
```

3b. Schema (after `QualityJudgment`):

```python
class QuickJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_quality: int = Field(ge=0, le=4)
    fatal_error: bool = False
    rationale: str = Field(min_length=1)
    evidence_references: list[str] = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def reject_blank_rationale(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("rationale must not be blank")
        return normalized

    @field_validator("evidence_references")
    @classmethod
    def reject_blank_entries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("entries must not be blank")
        return normalized
```

3c. Prompt builder and judge function (after `judge_quality`):

```python
def build_quick_judge_prompt(bundle: JudgeEvidenceBundle) -> str:
    schema = json.dumps(QuickJudgment.model_json_schema(), ensure_ascii=False, sort_keys=True)
    payload = bundle.model_dump_json()
    return f"""Evaluate only the observable behavior in the evidence bundle below.
The bundle captures the first assistant text turn after the skill named in observed_route was loaded.
Score that single turn only. Do not infer hidden reasoning or unobserved work.
Cite only stable evidence IDs present in the bundle.
Return JSON matching this schema and no prose outside JSON:
{schema}

{_QUICK_TURN_RUBRIC}

{_SCORE_ANCHORS}

Evidence bundle:
{payload}"""


async def judge_quick_turn(bundle: JudgeEvidenceBundle, model: Any) -> QuickJudgment:
    prompt = build_quick_judge_prompt(bundle)
    if bundle.expected_output:
        prompt += f"\n\n## Expected Output Reference\n\nThe following describes what a good first turn should cover. Compare the agent's actual turn against this reference when scoring turn_quality:\n\n{bundle.expected_output}\n"
    try:
        output = await model.generate(prompt)
    except Exception as exc:
        raise JudgeFailure(f"judge model call failed: {exc}") from exc

    try:
        judgment = QuickJudgment.model_validate_json(_strip_fences(output.completion))
    except (ValidationError, ValueError) as exc:
        repair_prompt = _build_repair_prompt(output.completion, exc, QuickJudgment)
        try:
            repaired_output = await model.generate(repair_prompt)
            judgment = QuickJudgment.model_validate_json(_strip_fences(repaired_output.completion))
        except Exception as repair_exc:
            raise JudgeFailure(f"judge output invalid after format repair: {repair_exc}") from repair_exc

    try:
        _validate_evidence_references(bundle, judgment.evidence_references)
    except JudgeFailure:
        output_ids = sorted(item.id for item in bundle.evidence if item.kind in _OUTPUT_EVIDENCE_KINDS)
        for reference in output_ids:
            if reference not in judgment.evidence_references:
                judgment.evidence_references.append(reference)
        _validate_evidence_references(bundle, judgment.evidence_references)

    return judgment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_judge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/judge.py backend/tests/skill_eval/test_judge.py
git commit -m "feat(skill-eval): add QuickJudgment schema and quick-turn judge"
```

---

### Task 5: quick_turn_scorer

**Files:**
- Modify: `backend/skill_eval/inspect_scorer.py`
- Test: `backend/tests/skill_eval/test_quick_scorer.py` (new)

**Interfaces:**
- Consumes: `build_judge_evidence(target="quick_turn")`, `judge_quick_turn`, `QuickJudgment` (Tasks 3–4).
- Produces: `quick_turn_scorer(judge_model, skill_descriptions)` — score metadata keys: `infrastructure_error` | `not_applicable_none_case` | `route_mismatch` | `quick_turn_missing` | `judge_failure` | `quick_judgment` + `quality_passed`. Task 7 wires it into the eval task; Task 6's report reads exactly these metadata keys.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/skill_eval/test_quick_scorer.py`:

```python
import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Target

from skill_eval.inspect_scorer import quick_turn_scorer


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelOutput.from_content("fake/judge", self.response)


def quick_judgment_json(**updates):
    payload = {
        "turn_quality": 3,
        "fatal_error": False,
        "rationale": "The turn follows the loaded skill workflow.",
        "evidence_references": ["tool_chain[0]", "quick_turn"],
    }
    payload.update(updates)
    return json.dumps(payload)


def descriptions():
    return {
        "systematic-literature-review": "multi-paper",
        "academic-paper-review": "one-paper",
    }


def quick_state():
    return SimpleNamespace(
        metadata={
            "case": {
                "id": "quick-1",
                "input": "Review the arXiv paper.",
                "expected_route": "academic-paper-review",
                "rationale": "PRIVATE HUMAN RATIONALE",
                "tags": ["quality"],
            },
            "agent_trace": {
                "input": "Review the arXiv paper.",
                "final_answer": "",
                "success": True,
                "thread_id": "t1",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "message_id": "m1",
                        "name": "read_file",
                        "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"},
                        "result": "skill body",
                        "error": None,
                    }
                ],
                "tool_call_chain": [["tc1"]],
                "quick_turn": {
                    "message_id": "m2",
                    "skill": "academic-paper-review",
                    "content": "I will review the paper along these axes.",
                },
                "messages": [],
                "artifacts": [],
                "errors": [],
            },
            "route_observation": {
                "observed_route": "academic-paper-review",
                "completed": True,
                "errors": [],
                "evidence": [],
            },
            "agent_success": True,
        }
    )


@pytest.mark.asyncio
async def test_quick_scorer_passes_threshold_and_hides_labels(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quick_turn_scorer("fake/judge", descriptions())(quick_state(), Target("academic-paper-review"))

    assert score.value == CORRECT
    assert score.metadata["quality_passed"] is True
    assert "expected_route" not in model.prompts[0]
    assert "PRIVATE HUMAN RATIONALE" not in model.prompts[0]


@pytest.mark.asyncio
async def test_quick_scorer_fails_below_threshold_or_fatal(monkeypatch):
    model = FakeModel(quick_judgment_json(turn_quality=2))
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quick_turn_scorer("fake/judge", descriptions())(quick_state(), Target("academic-paper-review"))

    assert score.value == INCORRECT
    assert score.metadata["quality_passed"] is False


@pytest.mark.asyncio
async def test_quick_scorer_rejects_failed_agent_run(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["agent_trace"]["success"] = False
    state.metadata["agent_trace"]["errors"] = ["agent failed"]

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert "infrastructure_error" in score.metadata
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_skips_none_expected_case(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["case"]["expected_route"] = "none"
    state.metadata["route_observation"]["observed_route"] = "none"

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("none"))

    assert score.value == NOANSWER
    assert score.metadata["not_applicable_none_case"] is True
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_skips_route_mismatch(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["route_observation"]["observed_route"] = "systematic-literature-review"

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert score.metadata["route_mismatch"] is True
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_reports_missing_quick_turn(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["agent_trace"]["quick_turn"] = None

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert score.metadata["quick_turn_missing"] is True
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_judge_failure_returns_noanswer(monkeypatch):
    model = FakeModel(quick_judgment_json(evidence_references=["tool_chain[9]", "quick_turn"]))
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quick_turn_scorer("fake/judge", descriptions())(quick_state(), Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert "unknown evidence" in score.metadata["judge_failure"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/skill_eval/test_quick_scorer.py -v`
Expected: FAIL — `quick_turn_scorer` does not exist.

- [ ] **Step 3: Implement the scorer**

In `backend/skill_eval/inspect_scorer.py`, extend the judge import:

```python
from skill_eval.judge import JudgeFailure, build_judge_evidence, judge_quality, judge_quick_turn
```

Append:

```python
@scorer(metrics=[])
def quick_turn_scorer(
    judge_model: str,
    skill_descriptions: dict[str, str],
):
    model = get_model(judge_model)

    async def score(state: TaskState, target: Target) -> Score:
        try:
            case = RoutingCase.model_validate(state.metadata["case"])
            trace = AgentTrace.model_validate(state.metadata["agent_trace"])
            observation = RouteObservation.model_validate(state.metadata["route_observation"])
        except Exception as exc:
            return Score(
                value=NOANSWER,
                explanation=f"Invalid quick-turn metadata: {exc}",
                metadata={"infrastructure_error": str(exc)},
            )

        infrastructure_errors = []
        if not trace.success:
            infrastructure_errors.extend(trace.errors or ["agent trace failed"])
        if not observation.completed:
            infrastructure_errors.extend(observation.errors or ["route observation incomplete"])
        if state.metadata.get("agent_success") is False:
            infrastructure_errors.append("agent run reported failure")
        if infrastructure_errors:
            message = "; ".join(dict.fromkeys(infrastructure_errors))
            return Score(
                value=NOANSWER,
                explanation=message,
                metadata={"infrastructure_error": message, "case_id": case.id},
            )

        if case.expected_route == "none":
            return Score(
                value=NOANSWER,
                explanation="quick turn not applicable to none-expected case",
                metadata={"not_applicable_none_case": True, "case_id": case.id},
            )
        if observation.observed_route != case.expected_route:
            return Score(
                value=NOANSWER,
                explanation=f"route mismatch: expected={case.expected_route} observed={observation.observed_route}",
                metadata={"route_mismatch": True, "case_id": case.id},
            )
        if trace.quick_turn is None:
            return Score(
                value=NOANSWER,
                explanation="quick turn not captured before the stream ended",
                metadata={"quick_turn_missing": True, "case_id": case.id},
            )

        try:
            bundle = build_judge_evidence(
                case=case,
                trace=trace,
                observation=observation,
                skill_descriptions=skill_descriptions,
                target="quick_turn",
            )
            judgment = await judge_quick_turn(bundle, model)
        except (JudgeFailure, KeyError, ValueError) as exc:
            return Score(
                value=NOANSWER,
                explanation=f"Quick turn judge failed: {exc}",
                metadata={"judge_failure": str(exc), "case_id": case.id},
            )

        quality_passed = not judgment.fatal_error and judgment.turn_quality >= 3
        return Score(
            value=CORRECT if quality_passed else INCORRECT,
            explanation=judgment.rationale,
            metadata={
                "quick_judgment": judgment.model_dump(),
                "quality_passed": quality_passed,
                "case_id": case.id,
            },
        )

    return score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_quick_scorer.py tests/skill_eval/test_quality_eval.py -v`
Expected: PASS (existing quality scorer untouched).

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/inspect_scorer.py backend/tests/skill_eval/test_quick_scorer.py
git commit -m "feat(skill-eval): add quick_turn_scorer with disjoint failure categories"
```

---

### Task 6: Report quick results

**Files:**
- Modify: `backend/skill_eval/report.py`
- Test: `backend/tests/skill_eval/test_report.py`

**Interfaces:**
- Consumes: `quick_turn_scorer` metadata keys (Task 5), `QuickJudgment` (Task 4).
- Produces: `QuickCaseResult(case_id, observed_route, judgment, category, detail, turn_quality, quality_passed, evidence_log)` with `category: Literal["infrastructure_error", "judge_failure", "quick_turn_missing", "route_mismatch", "not_applicable_none_case"] | None`; `extract_quick_results(log) -> list[QuickCaseResult]`; `PocSummary` fields `quality_mode`, `quick_results`, `quick_passed_cases`, `quick_turn_missing`, `quick_judge_failures`, `quick_infrastructure_failures`; markdown section `## Quick quality (first turn after skill load)`. Task 7 imports these.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/skill_eval/test_report.py`:

```python
def quick_score(case_id, *, judgment=None, category_metadata=None, value=CORRECT, passed=True):
    metadata = {"case_id": case_id}
    if judgment is not None:
        metadata["quick_judgment"] = judgment
        metadata["quality_passed"] = passed
    if category_metadata:
        metadata.update(category_metadata)
    return Score(value=value, explanation="detail text", metadata=metadata)


def quick_sample(case_id, score):
    return SimpleNamespace(
        id=case_id,
        epoch=1,
        metadata={},
        scores={
            "routing_scorer": Score(
                value=CORRECT,
                metadata={"case_id": case_id, "observed_route": "systematic-literature-review"},
            ),
            "quick_turn_scorer": score,
        },
    )


JUDGMENT = {
    "turn_quality": 3,
    "fatal_error": False,
    "rationale": "Turn follows the loaded skill.",
    "evidence_references": ["tool_chain[0]", "quick_turn"],
}


def test_extract_quick_results_reads_judgments_and_categories():
    log = SimpleNamespace(
        samples=[
            quick_sample("a", quick_score("a", judgment=JUDGMENT)),
            quick_sample("b", quick_score("b", category_metadata={"quick_turn_missing": True}, value=NOANSWER)),
            quick_sample("c", quick_score("c", category_metadata={"not_applicable_none_case": True}, value=NOANSWER)),
            quick_sample("d", quick_score("d", category_metadata={"route_mismatch": True}, value=NOANSWER)),
            quick_sample("e", quick_score("e", category_metadata={"judge_failure": "parse"}, value=NOANSWER)),
            quick_sample("f", quick_score("f", category_metadata={"infrastructure_error": "boom"}, value=NOANSWER)),
        ],
        location="logs/quick.eval",
    )

    results = extract_quick_results(log)

    assert len(results) == 6
    judged = results[0]
    assert judged.judgment is not None
    assert judged.turn_quality == 3
    assert judged.quality_passed is True
    assert judged.category is None
    assert [result.category for result in results[1:]] == [
        "quick_turn_missing",
        "not_applicable_none_case",
        "route_mismatch",
        "judge_failure",
        "infrastructure_error",
    ]
    assert all(result.evidence_log == "logs/quick.eval" for result in results)


def test_extract_quick_results_requires_scorer_output():
    log = SimpleNamespace(samples=[SimpleNamespace(id="x", epoch=1, metadata={}, scores={})], location="l")

    with pytest.raises(ValueError, match="quick_turn_scorer"):
        extract_quick_results(log)
```

Add a render test (there is no existing `PocSummary` helper in `test_report.py`, so define one — `make_balanced_summary` from the existing file provides the routing metrics):

```python
def make_poc_summary() -> PocSummary:
    return PocSummary(
        run_id="test-run",
        mode="full",
        identity=RunIdentity(
            agent_model="default",
            judge_model="mockllm/judge",
            inspect_ai_version="0.3.test",
            deerflow_version="test",
            case_file_sha256="a" * 64,
            skill_file_sha256={skill: "b" * 64 for skill in CANDIDATE_SKILLS},
            runtime_config={},
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            inspect_logs=["logs/routing.eval"],
        ),
        routing=make_balanced_summary(valid_run_rate=0.95, macro_precision=0.8, macro_recall=0.8),
        quality_results=[],
        quality_passed_cases=0,
        judge_failures=0,
        infrastructure_failures=0,
        acceptance=[],
    )


def test_markdown_includes_quick_section():
    summary = make_poc_summary()
    summary = summary.model_copy(
        update={
            "quality_mode": "both",
            "quick_results": [
                QuickCaseResult(
                    case_id="a",
                    observed_route="systematic-literature-review",
                    judgment=QuickJudgment(
                        turn_quality=3,
                        fatal_error=False,
                        rationale="solid turn",
                        evidence_references=["tool_chain[0]", "quick_turn"],
                    ),
                    category=None,
                    detail=None,
                    turn_quality=3,
                    quality_passed=True,
                    evidence_log="logs/quick.eval",
                ),
                QuickCaseResult(
                    case_id="b",
                    category="quick_turn_missing",
                    detail="quick turn not captured before the stream ended",
                    quality_passed=False,
                    evidence_log="logs/quick.eval",
                ),
            ],
            "quick_passed_cases": 1,
            "quick_turn_missing": 1,
        }
    )

    markdown = render_poc_markdown(summary)

    assert "## Quick quality (first turn after skill load)" in markdown
    assert "turn_quality=3" in markdown
    assert "quick_turn_missing" in markdown
```

(Add to the test file imports: `QuickJudgment` from `skill_eval.judge`; `QuickCaseResult`, `extract_quick_results` from `skill_eval.report`; `CANDIDATE_SKILLS` from `skill_eval.case_schema`. `RunIdentity`, `PocSummary`, `datetime`, `UTC` are already imported there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/skill_eval/test_report.py -v`
Expected: FAIL — `extract_quick_results` / `QuickCaseResult` do not exist.

- [ ] **Step 3: Implement report changes**

In `backend/skill_eval/report.py`:

3a. Imports: add `QuickJudgment` to the judge import; `NOANSWER` is already imported.

3b. `QuickCaseResult` MUST be defined before `PocSummary` (which references it). Add the model immediately after `QualityCaseResult`:

```python
class QuickCaseResult(BaseModel):
    case_id: str
    observed_route: ObservedRoute | None = None
    judgment: QuickJudgment | None = None
    category: Literal[
        "infrastructure_error",
        "judge_failure",
        "quick_turn_missing",
        "route_mismatch",
        "not_applicable_none_case",
    ] | None = None
    detail: str | None = None
    turn_quality: int | None = None
    quality_passed: bool = False
    evidence_log: str
```

Add `extract_quick_results` after `extract_quality_results`:

```python
def extract_quick_results(log: EvalLog) -> list[QuickCaseResult]:
    results: list[QuickCaseResult] = []
    log_location = str(getattr(log, "location", ""))
    for sample in log.samples or []:
        scores = sample.scores or {}
        routing_score = scores.get("routing_scorer")
        quick_score = scores.get("quick_turn_scorer")
        if routing_score is None:
            raise ValueError(f"Quick sample {sample.id} has no routing_scorer output")
        if quick_score is None:
            raise ValueError(f"Quick sample {sample.id} has no quick_turn_scorer output")
        routing_metadata = routing_score.metadata or {}
        quick_metadata = quick_score.metadata or {}
        observed_route = routing_metadata.get("observed_route")

        category = None
        detail = None
        judgment = None
        turn_quality = None
        if quick_metadata.get("infrastructure_error"):
            category, detail = "infrastructure_error", str(quick_metadata["infrastructure_error"])
        elif quick_metadata.get("judge_failure"):
            category, detail = "judge_failure", str(quick_metadata["judge_failure"])
        elif quick_metadata.get("quick_turn_missing"):
            category, detail = "quick_turn_missing", quick_score.explanation
        elif quick_metadata.get("route_mismatch"):
            category, detail = "route_mismatch", quick_score.explanation
        elif quick_metadata.get("not_applicable_none_case"):
            category, detail = "not_applicable_none_case", quick_score.explanation
        elif quick_score.value == NOANSWER:
            category, detail = "judge_failure", str(quick_score.explanation or "quick scorer returned NOANSWER")
        else:
            try:
                judgment = QuickJudgment.model_validate(quick_metadata["quick_judgment"])
            except Exception as exc:
                raise ValueError(f"Quick sample {sample.id} has invalid judgment: {exc}") from exc
            turn_quality = judgment.turn_quality

        results.append(
            QuickCaseResult(
                case_id=str(sample.id),
                observed_route=observed_route,
                judgment=judgment,
                category=category,
                detail=detail,
                turn_quality=turn_quality,
                quality_passed=bool(quick_metadata.get("quality_passed", False)),
                evidence_log=log_location,
            )
        )
    return results
```

3c. `PocSummary` — bump the schema literal to `"deerflow.agent-routing-poc.v2"` (both the `Literal[...]` and the default) and add fields:

```python
    quality_mode: Literal["quick", "full", "both"] = "full"
    quick_results: list[QuickCaseResult] = Field(default_factory=list)
    quick_passed_cases: int = 0
    quick_turn_missing: int = 0
    quick_judge_failures: int = 0
    quick_infrastructure_failures: int = 0
```

(`quality_mode` defaults to `"full"` so pre-change constructions and summaries keep legacy exit-code semantics.)

3d. `render_poc_markdown` — insert before the `"## Quality judgments"` block:

```python
    lines.extend(["", "## Quick quality (first turn after skill load)", ""])
    if summary.quick_results:
        judged = [result for result in summary.quick_results if result.judgment is not None]
        lines.append(f"- Passed: {summary.quick_passed_cases}/{len(summary.quick_results)}")
        lines.append(
            f"- quick_turn_missing: {summary.quick_turn_missing}; "
            f"infrastructure: {summary.quick_infrastructure_failures}; "
            f"judge failures: {summary.quick_judge_failures}"
        )
        if judged:
            mean_quality = sum(result.turn_quality or 0 for result in judged) / len(judged)
            lines.append(f"- Mean turn quality: {mean_quality:.2f}")
        for result in summary.quick_results:
            if result.judgment is not None:
                lines.append(
                    f"- `{result.case_id}`: turn_quality={result.turn_quality}, "
                    f"passed={result.quality_passed}, rationale: {result.judgment.rationale}"
                )
            else:
                lines.append(f"- `{result.case_id}`: {result.category} `{result.detail}`")
    else:
        lines.append("- Skipped.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_report.py -v`
Expected: PASS (existing constructions keep working via the new field defaults).

- [ ] **Step 5: Commit**

```bash
git add backend/skill_eval/report.py backend/tests/skill_eval/test_report.py
git commit -m "feat(skill-eval): report quick-turn results with disjoint failure buckets"
```

---

### Task 7: Quick eval task and POC `--quality-mode`

**Files:**
- Create: `backend/evals/skills_quick_eval.py`
- Modify: `backend/skill_eval/poc.py`
- Test: `backend/tests/skill_eval/test_quick_eval.py` (new)
- Test: `backend/tests/skill_eval/test_poc.py` (update + additions)

**Interfaces:**
- Consumes: `quick_turn_scorer` (Task 5), `extract_quick_results` and the `PocSummary` quick fields (Task 6).
- Produces: `skills_quick_eval(case_file, agent_model, judge_model, skills_root="../skills/public", sample_ids=None, trace_dir=None, config_path=None, sandbox="configured") -> Task`; `PocConfig.quality_mode: Literal["quick", "full", "both"] = "both"`; CLI flag `--quality-mode`; `PocSummary.quality_mode: Literal["quick", "full", "both"] = "full"` plus quick counters (field names defined in Task 6).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/skill_eval/test_quick_eval.py`:

```python
from types import SimpleNamespace

import pytest

import evals.skills_quick_eval as quick_module
from evals.skills_quick_eval import skills_quick_eval
from skill_eval.agent_runner import AgentRunResult
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace


class ScriptedRunner:
    def __init__(self):
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return AgentRunResult(
            final_answer="done",
            success=True,
            thread_id=request.thread_id,
            route_observation=RouteObservation(observed_route="none", completed=True),
            trace=AgentTrace(
                input=request.user_input,
                final_answer="done",
                success=True,
                thread_id=request.thread_id,
            ),
        )


def descriptions():
    return {
        "systematic-literature-review": "multi-paper",
        "academic-paper-review": "one-paper",
    }


def quick_state():
    return SimpleNamespace(
        input_text="Review the arXiv paper.",
        metadata={
            "case": {
                "id": "quick-1",
                "input": "Review the arXiv paper.",
                "expected_route": "academic-paper-review",
                "rationale": "single paper",
                "tags": ["quality"],
            }
        },
        output=SimpleNamespace(completion=""),
    )


@pytest.mark.asyncio
async def test_quick_task_selects_quality_cases_and_quick_runner(monkeypatch, tmp_path):
    runner = ScriptedRunner()
    runner_options = {}
    monkeypatch.setattr(
        quick_module,
        "DeerFlowAgentRunner",
        lambda **options: runner_options.update(options) or runner,
    )
    monkeypatch.setattr(
        quick_module,
        "load_candidate_skill_descriptions",
        lambda _: descriptions(),
    )
    task = skills_quick_eval(
        case_file="cases/literature_skill_routing.jsonl",
        agent_model="default",
        judge_model="mockllm/judge",
        skills_root="../skills/public",
        trace_dir=tmp_path,
        config_path="eval-config.yaml",
        sandbox="local",
    )
    state = quick_state()

    await task.solver(state, generate=None)

    assert len(task.dataset) == 4
    assert task.time_limit == 330
    assert len(task.scorer) == 2
    assert runner.requests[0].mode == "quick"
    assert runner.requests[0].timeout_seconds == 300
    assert runner_options["trace_dir"] == str(tmp_path)
    assert runner_options["config_path"] == "eval-config.yaml"
    assert runner_options["sandbox"] == "local"


@pytest.mark.asyncio
async def test_quick_task_sample_ids_bypass_quality_tag_filter(monkeypatch, tmp_path):
    smoke_ids = {
        "slr-attention-variants-001",
        "paper-review-arxiv-001",
        "none-precision-recall-001",
    }
    monkeypatch.setattr(quick_module, "DeerFlowAgentRunner", lambda **options: ScriptedRunner())
    monkeypatch.setattr(quick_module, "load_candidate_skill_descriptions", lambda _: descriptions())

    task = skills_quick_eval(
        case_file="cases/literature_skill_routing.jsonl",
        agent_model="default",
        judge_model="mockllm/judge",
        sample_ids=smoke_ids,
    )

    assert {str(sample.id) for sample in task.dataset} == smoke_ids


def test_quick_task_rejects_unknown_sample_id(monkeypatch):
    monkeypatch.setattr(quick_module, "DeerFlowAgentRunner", lambda **options: ScriptedRunner())
    monkeypatch.setattr(quick_module, "load_candidate_skill_descriptions", lambda _: descriptions())

    with pytest.raises(ValueError, match="Unknown routing sample id"):
        skills_quick_eval(
            case_file="cases/literature_skill_routing.jsonl",
            agent_model="default",
            judge_model="mockllm/judge",
            sample_ids={"does-not-exist"},
        )
```

Add to `backend/tests/skill_eval/test_poc.py` — first the quick log helper (after `quality_log`):

```python
def quick_log(cases):
    samples = []
    for case in cases:
        if case.expected_route == "none":
            quick_score = Score(
                value=NOANSWER,
                explanation="quick turn not applicable to none-expected case",
                metadata={"not_applicable_none_case": True, "case_id": case.id},
            )
        else:
            quick_score = Score(
                value=CORRECT,
                metadata={
                    "case_id": case.id,
                    "quick_judgment": {
                        "turn_quality": 3,
                        "fatal_error": False,
                        "rationale": "Turn follows the loaded skill.",
                        "evidence_references": ["tool_chain[0]", "quick_turn"],
                    },
                    "quality_passed": True,
                },
            )
        samples.append(
            SimpleNamespace(
                id=case.id,
                epoch=1,
                metadata={"case": case.model_dump()},
                scores={
                    "routing_scorer": routing_score(case),
                    "quick_turn_scorer": quick_score,
                },
            )
        )
    return SimpleNamespace(samples=samples, location="logs/quick.eval")
```

(`NOANSWER` must be added to the `inspect_ai.scorer` import.)

Update the four existing run-level tests:

1. `test_run_poc_calls_routing_three_epochs_and_quality_one`: change the logs line to

```python
    quality_cases = [case for case in cases if "quality" in case.tags]
    logs = [routing_log(cases), quick_log(quality_cases), quality_log(quality_cases)]
```

and add assertions:

```python
    assert len(calls) == 3
    assert summary.quick_passed_cases == 3
    assert len(summary.quick_results) == 4
```

2. `test_smoke_selects_fixed_three_cases_and_skips_quality`: smoke now runs routing + quick (full quality stays skipped). Replace `fake_eval` with:

```python
    logs = [routing_log(cases, epochs=1), quick_log(cases)]

    def fake_eval(task, **kwargs):
        calls.append((task, kwargs))
        return [logs.pop(0)]
```

and replace the tail assertions with:

```python
    assert len(calls) == 2
    assert calls[0][1]["epochs"] == 1
    assert {sample.id for sample in calls[0][0].dataset} == smoke_ids
    assert summary.mode == "smoke"
    assert summary.quality_results == []
    assert len(summary.quick_results) == 3
    assert summary.quick_passed_cases == 2
    assert exit_code == 0
```

3. `test_incomplete_eval_log_returns_invalid_exit`: change the logs line to

```python
    quality_cases = [case for case in cases if "quality" in case.tags]
    logs = [incomplete, quick_log(quality_cases), quality_log(quality_cases)]
```

4. `test_exit_codes_separate_quality_failure_and_invalid_evaluation`: change the logs line to

```python
    quality_cases = [case for case in cases if "quality" in case.tags]
    logs = [routing_log(cases), quick_log(quality_cases), quality]
```

Add the new tests:

```python
def test_quality_mode_quick_skips_full_quality(
    monkeypatch,
    valid_config,
    preflight_record,
):
    cases = read_routing_cases(valid_config.case_file)
    quality_cases = [case for case in cases if "quality" in case.tags]
    logs = [routing_log(cases), quick_log(quality_cases)]
    calls = []

    def fake_eval(*args, **kwargs):
        calls.append((args, kwargs))
        return [logs.pop(0)]

    monkeypatch.setattr("skill_eval.poc.inspect_eval", fake_eval)
    monkeypatch.setattr("skill_eval.poc.preflight", lambda config: preflight_record)

    summary, exit_code = run_poc(valid_config.model_copy(update={"quality_mode": "quick"}))

    assert len(calls) == 2
    assert summary.quality_results == []
    assert len(summary.quick_results) == 4
    assert exit_code == 0


def test_quick_turn_missing_invalidates_evaluation(
    monkeypatch,
    valid_config,
    preflight_record,
):
    cases = read_routing_cases(valid_config.case_file)
    quality_cases = [case for case in cases if "quality" in case.tags]
    quick = quick_log(quality_cases)
    quick.samples[0].scores["quick_turn_scorer"] = Score(
        value=NOANSWER,
        explanation="quick turn not captured before the stream ended",
        metadata={"quick_turn_missing": True, "case_id": quality_cases[0].id},
    )
    logs = [routing_log(cases), quick, quality_log(quality_cases)]
    monkeypatch.setattr(
        "skill_eval.poc.inspect_eval",
        lambda *args, **kwargs: [logs.pop(0)],
    )
    monkeypatch.setattr("skill_eval.poc.preflight", lambda config: preflight_record)

    summary, _ = run_poc(valid_config)

    assert exit_code_for(summary) == 2
    assert summary.quick_turn_missing == 1


def test_parser_defaults_and_accepts_quality_mode():
    parser = _build_parser()
    assert parser.parse_args([]).quality_mode == "both"
    assert parser.parse_args(["--quality-mode", "quick"]).quality_mode == "quick"
```

(add `_build_parser` to the `skill_eval.poc` import list.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/skill_eval/test_quick_eval.py tests/skill_eval/test_poc.py -v`
Expected: FAIL — `skills_quick_eval` module and `quality_mode` do not exist.

- [ ] **Step 3: Implement the eval task and POC wiring**

3a. Create `backend/evals/skills_quick_eval.py`:

```python
from pathlib import Path

from inspect_ai import Task, task

from skill_eval.adapters.deerflow import DeerFlowAgentRunner
from skill_eval.agent_runner import SandboxMode
from skill_eval.dataset_loader import load_routing_samples
from skill_eval.inspect_scorer import quick_turn_scorer, routing_scorer
from skill_eval.inspect_solver import deerflow_solver
from skill_eval.judge import load_candidate_skill_descriptions

_QUICK_TIMEOUT_SECONDS = 300
_QUICK_TASK_TIME_LIMIT_SECONDS = 330


@task
def skills_quick_eval(
    case_file: str,
    agent_model: str,
    judge_model: str,
    skills_root: str | Path = "../skills/public",
    sample_ids: set[str] | None = None,
    trace_dir: str | Path | None = None,
    config_path: str | None = None,
    sandbox: SandboxMode = "configured",
) -> Task:
    if sample_ids is None:
        samples = load_routing_samples(case_file, tags={"quality"})
    else:
        samples = load_routing_samples(case_file)
        samples = [sample for sample in samples if str(sample.id) in sample_ids]
        found = {str(sample.id) for sample in samples}
        if found != sample_ids:
            missing = ", ".join(sorted(sample_ids - found))
            raise ValueError(f"Unknown routing sample id(s): {missing}")
    skill_descriptions = load_candidate_skill_descriptions(Path(skills_root))
    runner = DeerFlowAgentRunner(
        config_path=config_path,
        trace_dir=str(trace_dir) if trace_dir is not None else None,
        sandbox=sandbox,
    )
    return Task(
        dataset=samples,
        solver=deerflow_solver(
            runner,
            mode="quick",
            model_name=agent_model,
            timeout_seconds=_QUICK_TIMEOUT_SECONDS,
        ),
        scorer=[
            routing_scorer(),
            quick_turn_scorer(judge_model, skill_descriptions),
        ],
        time_limit=_QUICK_TASK_TIME_LIMIT_SECONDS,
    )
```

3b. `backend/skill_eval/poc.py`:

- Imports: add `Literal` to the `typing` import; add `from evals.skills_quick_eval import skills_quick_eval`; add `extract_quick_results` to the `skill_eval.report` import list.
- `PocConfig`: add field

```python
    quality_mode: Literal["quick", "full", "both"] = "both"
```

- `PocConfig.from_env`: add keyword `quality_mode: str = "both"` and `values["quality_mode"] = quality_mode`.
- In `run_poc`, between the routing block and the quality block, insert:

```python
    quick_log = None
    if config.quality_mode in {"quick", "both"}:
        quick_task = skills_quick_eval(
            case_file=str(config.case_file),
            agent_model=config.agent_model,
            judge_model=config.judge_model,
            skills_root=config.skills_root,
            sample_ids=_SMOKE_CASE_IDS if config.smoke else None,
            trace_dir=trace_dir,
            config_path=config.config_path,
            sandbox="local",
        )
        try:
            quick_logs = inspect_eval(
                quick_task,
                model=None,
                epochs=1,
                max_samples=1,
                log_dir=str(config.log_dir),
                fail_on_error=False,
                score_on_error=True,
            )
            quick_log = _single_log(quick_logs, "quick quality")
        except Exception as exc:
            raise PocInvalidEvaluationError(f"Quick quality evaluation failed: {exc}") from exc
```

- Gate the existing full-quality block: change `if not config.smoke:` to `if not config.smoke and config.quality_mode in {"full", "both"}:`.
- After the existing quality extraction block, add:

```python
    quick_results = []
    if quick_log is not None:
        try:
            quick_results = extract_quick_results(quick_log)
        except Exception as exc:
            errors.append(f"Cannot extract quick quality results: {exc}")
        expected_quick = 3 if config.smoke else 4
        if len(quick_results) != expected_quick:
            errors.append(f"Expected {expected_quick} quick quality results, found {len(quick_results)}")
```

- Extend `inspect_logs`: after the quality_log append add

```python
    if quick_log is not None:
        inspect_logs.append(str(getattr(quick_log, "location", "")))
```

- Compute quick counters before `_acceptance_checks`:

```python
    quick_passed_cases = sum(result.quality_passed for result in quick_results)
    quick_turn_missing = sum(result.category == "quick_turn_missing" for result in quick_results)
    quick_judge_failures = sum(result.category == "judge_failure" for result in quick_results)
    quick_infrastructure_failures = sum(result.category == "infrastructure_error" for result in quick_results)
```

- `_acceptance_checks` gains keyword params `quick_passed_cases: int = 0, include_quick: bool = False` and appends when `include_quick`:

```python
    if include_quick:
        checks.append(
            AcceptanceCheck(
                name="quick quality cases passing turn threshold",
                actual=quick_passed_cases,
                threshold=">= 3 of 4",
                passed=quick_passed_cases >= 3,
            )
        )
```

Call it with `include_quick=not config.smoke and config.quality_mode in {"quick", "both"}` and `quick_passed_cases=quick_passed_cases`.
- `PocSummary(...)` gains `quality_mode=config.quality_mode, quick_results=quick_results, quick_passed_cases=quick_passed_cases, quick_turn_missing=quick_turn_missing, quick_judge_failures=quick_judge_failures, quick_infrastructure_failures=quick_infrastructure_failures,`.
- Replace `exit_code_for` with:

```python
def exit_code_for(summary: PocSummary) -> int:
    expected_routing_runs = 3 if summary.mode == "smoke" else 60
    full_quality_expected = 4 if (summary.mode == "full" and summary.quality_mode in {"full", "both"}) else 0
    quick_expected = (3 if summary.mode == "smoke" else 4) if summary.quality_mode in {"quick", "both"} else 0
    evaluation_invalid = (
        bool(summary.errors)
        or len(summary.routing.results) != expected_routing_runs
        or summary.judge_failures > 0
        or summary.infrastructure_failures > 0
        or len(summary.quality_results) != full_quality_expected
        or len(summary.quick_results) != quick_expected
        or summary.quick_judge_failures > 0
        or summary.quick_infrastructure_failures > 0
        or summary.quick_turn_missing > 0
    )
    if evaluation_invalid:
        return 2
    full_quality_passed = full_quality_expected == 0 or summary.quality_passed_cases >= 3
    quick_passed = quick_expected == 0 or summary.mode == "smoke" or summary.quick_passed_cases >= 3
    return 0 if routing_acceptance(summary.routing) and full_quality_passed and quick_passed else 1
```

- `_build_parser`: add

```python
    parser.add_argument("--quality-mode", choices=["quick", "full", "both"], default="both")
```

- `main`: pass `quality_mode=args.quality_mode` to `PocConfig.from_env`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/skill_eval/test_quick_eval.py tests/skill_eval/test_poc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/evals/skills_quick_eval.py backend/skill_eval/poc.py backend/tests/skill_eval/test_quick_eval.py backend/tests/skill_eval/test_poc.py
git commit -m "feat(skill-eval): add quick eval task and --quality-mode poc flag"
```

---

### Task 8: Full-suite verification and smoke

**Files:**
- None (verification only)

- [ ] **Step 1: Lint**

Run: `cd backend && make lint`
Expected: `ruff check .` passes with no errors.

- [ ] **Step 2: Full backend test suite**

Run: `cd backend && make test`
Expected: all tests pass, including the previously existing 277 plus the new ones.

- [ ] **Step 3: Live smoke (requires configured models)**

Run: `cd backend && AGENT_MODEL=<configured> JUDGE_MODEL=<configured> uv run python -m skill_eval.poc --smoke --quality-mode quick`
Expected: exit code 0; `eval-results/<run_id>/summary.md` contains `## Quick quality (first turn after skill load)`; `summary.json` has `schema_version: deerflow.agent-routing-poc.v2`, 3 `quick_results`, and judge prompts (Inspect log) contain no `message[` evidence ids. If live models are unavailable in the environment, record this as NOT RUN and rely on Step 2 plus the mockllm-based tests.
