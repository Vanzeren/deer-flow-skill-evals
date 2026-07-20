# Skill Eval Dual-Mode Design: Quick Turn vs Full Output

**Status:** Approved design for implementation planning
**Date:** 2026-07-20
**Primary goal:** Add a fast quality-evaluation mode that judges only the first assistant text turn after a skill loads, keep the full-run mode for final-output judgment, shrink judge evidence by dropping the message-history trace, and record the tool-call chain (grouped by concurrent batch) in collected traces.

## 1. Problem

The current POC (`docs/superpowers/specs/2026-07-13-agent-routing-eval-poc-design.md`) evaluates quality in exactly one way: run the agent to completion (`mode="full"`) and hand the whole `AgentTrace` — including the full `messages` history — to an LLM judge.

Three observed problems:

1. **Full runs are the only quality signal.** A skill's influence is concentrated in the single assistant turn produced right after `read_file(.../SKILL.md)` succeeds: that turn is where the skill's process, format, and constraints first shape behavior. Judging only the multi-minute final output conflates skill effect with everything that happened afterwards (search, sandbox, retries), and makes each quality data point cost a full run.
2. **Judge evidence is too large.** `build_judge_evidence()` embeds every AI/tool message as `message[N]` items. For real tool-using runs the bundle saturates its 80KB cap, most content is truncated, and judge calls are slow enough that larger eval batches do not finish in practice.
3. **The collected trace loses call structure.** `AgentTrace.tool_calls` is a flat time-ordered list. Which tools were issued concurrently (same AI message) versus sequentially is recoverable only by re-joining on `message_id`, and no report or judge can see that structure directly.

## 2. Goals

The implementation MUST:

1. Add a **quick evaluation mode** (`mode="quick"`) that, in a single agent run, produces the route observation AND captures the first assistant text turn after a candidate skill loads, then judges that one turn with an LLM judge.
2. Keep the **full evaluation mode** (`mode="full"`) judging the agent's final output, unchanged in scoring semantics.
3. Keep routing evaluation (`mode="routing_probe"`, `routing_scorer`) untouched; quick mode reuses its observation machinery.
4. Remove the message history (`message[N]` items) from judge evidence in BOTH modes; replace flat `tool_call[N]`/`tool_result[N]` items with batch-structured `tool_chain[B]` items.
5. Add `tool_call_chain: list[list[str]]` to the collected `AgentTrace` — outer list ordered by time, inner list = tool calls issued concurrently by one AI message, stored as ids referencing the flat `tool_calls` list.
6. Report quick-mode metrics separately from full-mode metrics, with `quick_turn_missing` as a distinct failure category beside `infrastructure_error` and `judge_failure`.
7. Expose mode selection as `--quality-mode quick|full|both` on the POC entrypoint.

## 3. Non-goals

The implementation MUST NOT:

- change routing observation logic (`routing.py`), the routing dataset, or routing metrics;
- change `case_schema.py` or the case files;
- change epoch counts, parallelism, or the subprocess isolation design;
- remove `AgentTrace.messages` from collection — it stays for debugging and raw-trace cross-reference; it only leaves the judge evidence path;
- introduce new candidate skills or new judge models;
- judge quick turns for cases whose expected route is `none`, or when the observed route did not match the expected one (the routing scorer owns that failure);
- keep any compatibility shim for the old `message[N]` evidence shape (clean cutover).

## 4. Design Principles

- **One run, two facts.** Quick mode reuses the routing-probe stream; the route decision and the quick-turn capture come from the same trajectory, so they can never disagree.
- **Skill effect is measured where it lands.** The first text turn after skill load is the unit of quick judgment — not a later summarization of it.
- **Evidence pays for itself.** Every evidence kind in the judge bundle must be something the judge is scored on. Message history fails that test; tool chain and outputs pass it.
- **Failure categories stay disjoint.** `infrastructure_error`, `judge_failure`, `quick_turn_missing`, `route_mismatch`, and real quality failures never share a bucket.
- **Clean cutover.** Old evidence kinds are deleted, not deprecated.

## 5. Run Modes and Quick-Turn Capture

### 5.1 `RunMode`

`backend/skill_eval/agent_runner.py`:

```python
type RunMode = Literal["routing_probe", "quick", "full"]
```

Existing modes keep their exact current behavior. `AgentRunRequest` / `AgentRunResult` shapes are otherwise unchanged; quick-turn data travels inside `AgentTrace`.

### 5.2 Quick-mode stream consumption (`adapters/deerflow.py`)

The event loop in `_execute_deerflow()` gains a quick watcher:

1. Feed every event to the `RoutingObserver` and the trace adapter exactly as today.
2. When `observer.feed(event)` returns `True` **and** the observation's decided route is a concrete candidate (not `ambiguous`), record `loaded_skill` and start watching. If the decision is `ambiguous`, break immediately (same as `routing_probe`) — there is no single skill whose turn to capture.
3. While watching, the trace adapter already accumulates per-message AI content. The watcher marks the first AI message whose accumulated `content.strip()` is non-empty as the **target turn**. Tool calls attached to that same message are ignored for capture purposes (text only).
4. The target turn is **complete** when the stream produces any event belonging to a different message id (AI or tool) or the stream ends. On completion: break, record the capture.
5. Degenerate paths:
   - Route decided `none` (never loads a candidate): identical to `routing_probe` — run to stream end or the existing `route_ready` break; `quick_turn` stays `None`.
   - Stream ends or times out while watching with no non-empty text turn: `quick_turn = None`, run is still `success=True` if no errors; the scorer classifies this as `quick_turn_missing`, NOT infrastructure failure.
6. `timeout_seconds` and the subprocess/anti-deadlock/exit-grace machinery are unchanged. `subagent_enabled` stays `False` for quick (same as `routing_probe`).

### 5.3 Capture contract (`trace_schema.py`)

```python
class QuickTurnCapture(BaseModel):
    message_id: str
    skill: str          # the loaded candidate skill whose turn this is
    content: str        # full text of the captured turn (not truncated at collection)

class AgentTrace(BaseModel):
    # ... existing fields unchanged ...
    tool_call_chain: list[list[str]] = Field(default_factory=list)
    quick_turn: QuickTurnCapture | None = None
```

`quick_turn` is set only in quick mode when capture completed. `final_answer` keeps its current meaning in all modes.

## 6. Tool-Call Chain

### 6.1 Structure

`tool_call_chain` is derived, not separately collected: `DeerFlowTraceAdapter` already records each AI message's tool calls under `_ai_messages[message_id]["tool_calls"]` in stream order. `build()` walks the AI messages in order and appends one inner list per message that issued at least one tool call:

```python
chain = [
    [call["id"] for call in message["tool_calls"]]
    for message in ai_messages_in_stream_order
    if message["tool_calls"]
]
```

Properties (unit-tested):

- inner list = one concurrent batch (same `message_id`), order within batch = stream order;
- outer list = batch time order;
- every id in the chain exists in the flat `tool_calls` list, and every id in `tool_calls` appears in exactly one batch;
- pure-text messages contribute no batch.

### 6.2 Consumption

The judge evidence builder is the only consumer that expands the chain; reports reference it structurally (batch counts) but never inline message history.

## 7. Judge Evidence Without Trace (`judge.py`)

### 7.1 Evidence kinds

```python
type EvidenceKind = Literal[
    "tool_chain",     # one concurrent batch, expanded
    "error",
    "artifact",
    "final_answer",
    "quick_turn",
]
```

`"message"`, `"tool_call"`, and `"tool_result"` are deleted.

A `tool_chain[B]` item's content is the JSON array of that batch's calls, each `{id, name, args, result, error}` expanded from the flat list. Budgets are unchanged: 12KB per item, 80KB per bundle, truncation flagged per item.

### 7.2 Reference validation

```python
_PROCESS_EVIDENCE_KINDS = {"tool_chain", "error"}
_OUTPUT_EVIDENCE_KINDS = {"artifact", "final_answer", "quick_turn"}
```

`_validate_evidence_references()` rules:

1. every cited id must exist in the bundle (unchanged);
2. if the bundle contains at least one process item, the judgment must cite at least one process item;
3. the judgment must always cite at least one output item.

Rule 2's conditional form covers the no-tool-call run, where no process evidence can exist.

### 7.3 Bundle

```python
class JudgeEvidenceBundle(BaseModel):
    user_input: str
    expected_route: str
    observed_route: str | None
    evaluation_target: Literal["quick_turn", "final_output"]
    route_rubric: str
    evidence: list[EvidenceItem]
    truncated: bool
    expected_output: str | None = None
```

`build_judge_evidence(trace, observation, skill_descriptions, target)` builds:

- `target="final_output"`: `tool_chain` batches over the whole run + `error` + `artifact` + `final_answer`.
- `target="quick_turn"`: `tool_chain` batches up to and including the skill-load batch + `error` + one `quick_turn` item (the captured text). No `final_answer` item.

## 8. Quick Judgment (`judge.py`)

### 8.1 Schema

```python
class QuickJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_quality: int          # 0-4, same anchors as QualityJudgment
    fatal_error: bool
    rationale: str
    evidence_references: list[str]
```

### 8.2 Rubric

A new `_QUICK_TURN_RUBRIC` tells the judge it is scoring exactly one assistant turn — the first after the named skill loaded — on:

- whether the turn follows the loaded skill's workflow, format, and constraints;
- whether it responds to the user's actual request;
- whether it is coherent and self-sufficient given what is observable.

The prompt states explicitly that later steps are out of scope and must not be inferred.

### 8.3 Execution

`judge_quick_turn(bundle, model) -> QuickJudgment` reuses the existing machinery: `build_judge_prompt` variant, `_strip_fences`, one repair retry on parse failure, `_validate_evidence_references`. Any parse/validation failure after repair raises `JudgeFailure`.

## 9. Scorers (`inspect_scorer.py`)

### 9.1 `quick_turn_scorer(judge_model, skill_descriptions)`

Order of checks, first match wins:

1. metadata parse failure → `NOANSWER` + `infrastructure_error`;
2. `not result.success` or `not observation.completed` → `NOANSWER` + `infrastructure_error`;
3. `case.expected_route == "none"` → `NOANSWER` + `not_applicable_none_case`;
4. `observation.observed_route != case.expected_route` → `NOANSWER` + `route_mismatch` (routing scorer owns the route failure; no double penalty);
5. `trace.quick_turn is None` → `NOANSWER` + `quick_turn_missing`;
6. judge → `CORRECT` iff `not fatal_error and turn_quality >= 3`, else `INCORRECT`; `JudgeFailure` → `NOANSWER` + `judge_failure`.

Explanations carry the category string so the report can bucket them.

### 9.2 `quality_judge_scorer`

Unchanged except it builds evidence with `target="final_output"` (no message items). Thresholds stay `>= 3` on all three dimensions.

## 10. Assembly, POC, and Report

### 10.1 Solver / task assembly

`deerflow_solver` already passes `mode` through; `AgentTrace.model_dump()` carries `tool_call_chain` and `quick_turn` into sample metadata with no solver change. A new Inspect task builder registers the quick eval: same runner, `mode="quick"`, scorers `[routing_scorer, quick_turn_scorer(...)]`. The full task keeps `[quality_judge_scorer(...)]`.

### 10.2 POC entrypoint (`poc.py`)

```
--quality-mode {quick,full,both}   (default: both)
```

- `routing eval` always runs (unchanged 60/3-run behavior).
- `quick` → quick task over the quality-case subset; `full` → existing full task; `both` → quick then full.
- Acceptance checks extend: quick pass rate and `quick_turn_missing` rate are reported; existing routing acceptance is untouched. Exit-code semantics (`0` pass / `1` metric miss / `2` invalid) are preserved, with quick metrics included in the quality side when enabled.

### 10.3 Report (`report.py`)

- `extract_quality_results` generalizes to extract per-scorer results by scorer name; a new `summarize_quick()` produces: run count, pass rate, `turn_quality` mean and 0-4 distribution, and disjoint failure buckets (`infrastructure_error`, `judge_failure`, `quick_turn_missing`, `route_mismatch`, `not_applicable_none_case`).
- `render_poc_markdown()` gains a **Quick quality** section beside the existing full-quality section.
- Reports MUST NOT inline `messages`; process structure is referenced via batch counts and evidence ids only.

## 11. Failure Taxonomy (final)

| Category | Meaning | Scored by |
|---|---|---|
| `infrastructure_error` | run/observation broken | both scorers, NOANSWER |
| `route_mismatch` | observed route != expected (quick mode: skip quality) | routing scorer INCORRECT; quick scorer NOANSWER |
| `quick_turn_missing` | route hit but no text turn captured | quick scorer NOANSWER |
| `judge_failure` | judge parse/validation failed after repair | either scorer NOANSWER |
| quality failure | judge ran, below threshold | INCORRECT |

## 12. Verification

Unit tests (`backend/tests/skill_eval/`):

1. `tool_call_chain`: concurrent batch grouping, sequential batches, empty-content messages skipped, chain ids are a partition of flat ids, order preserved.
2. Quick watcher: capture completes on next-message-id boundary and on stream end; none-route degenerates to probe behavior; ambiguous breaks immediately with `quick_turn=None`; timeout while watching yields `quick_turn=None` without failing the run.
3. `build_judge_evidence`: zero `message`/`tool_call`/`tool_result` items; `tool_chain` items ordered and expanded; quick target excludes `final_answer` and truncates chain at the load batch; truncation flags set.
4. `_validate_evidence_references`: new process/output rules including the empty-chain allowance.
5. `quick_turn_scorer`: every branch in §9.1.
6. `judge_quick_turn`: fence-stripped parse, one repair retry, reference validation failure → `JudgeFailure`.

Integration:

- Smoke mode (3 cases) with `--quality-mode quick` runs end to end and writes summaries containing the quick section.
- A retained smoke log demonstrates the bundle contains no `message[` evidence ids (asserted in test, not eyeballed).

## 13. File Change Map

| File | Change |
|---|---|
| `backend/skill_eval/agent_runner.py` | `RunMode` gains `"quick"` |
| `backend/skill_eval/trace_schema.py` | `QuickTurnCapture`, `tool_call_chain`, `quick_turn` |
| `backend/skill_eval/adapters/deerflow.py` | quick watcher, chain build, quick-turn capture |
| `backend/skill_eval/judge.py` | evidence kinds, `tool_chain` items, `QuickJudgment`, `judge_quick_turn`, quick rubric, bundle fields, validation rules |
| `backend/skill_eval/inspect_scorer.py` | `quick_turn_scorer`; full scorer evidence target |
| `backend/skill_eval/inspect_solver.py` | quick task registration |
| `backend/skill_eval/poc.py` | `--quality-mode`, quick acceptance |
| `backend/skill_eval/report.py` | quick extraction/summary/markdown section |
| `backend/tests/skill_eval/` | tests per §12 |
