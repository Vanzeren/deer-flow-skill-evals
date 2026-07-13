# Agent Routing Evaluation POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the assertion-heavy skill-eval prototype with a real DeerFlow routing benchmark (20 cases × 3 epochs) plus four bounded full executions evaluated by an evidence-grounded LLM judge.

**Architecture:** Both tracks use `DeerFlowAgentRunner`: `routing_probe` mode consumes the real `DeerFlowClient` stream until the current skill-load batch settles, while `full` mode runs through final output and artifacts. Inspect stores per-sample evidence; deterministic aggregation computes routing metrics, and a separate Inspect model judges semantic process/output quality for four tagged cases.

**Tech Stack:** Python 3.12, Pydantic v2, Inspect AI `>=0.3.244`, DeerFlow embedded Python client, pytest `>=9.0.3`, pytest-asyncio `>=1.3.0`, Ruff.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-13-agent-routing-eval-poc-design.md`.
- Both tracks MUST call the real `DeerFlowAgentRunner`; no mock agent or standalone classification prompt may produce benchmark results.
- Candidate skills are exactly `systematic-literature-review` and `academic-paper-review`.
- The routing dataset is exactly 20 cases: 8 SLR, 6 paper-review, 6 none; exactly four carry the `quality` tag.
- Routing runs use three raw epochs; do not majority-vote away individual results.
- Routing acceptance is valid-run rate `>= 0.95`, macro precision `>= 0.80`, and macro recall `>= 0.80`.
- The quality track runs exactly four cases for one epoch and requires at least three passing judgments.
- Cases contain no assertion lists, expected tool sequences, target substrings, or skill activation commands.
- The judge MUST NOT receive `expected_route`, case `rationale`, or hidden chain-of-thought.
- Judge failures use Inspect `NOANSWER` and remain distinct from agent quality failures.
- `AGENT_MODEL` is a DeerFlow config model name; `JUDGE_MODEL` is an Inspect model spec accepted by `inspect_ai.model.get_model()`.
- Run Inspect samples serially with `max_samples=1`; DeerFlow configuration is process-global and is not safe to mutate concurrently in this POC.
- Do not add a web UI. Retain Inspect logs and emit `summary.json` plus `summary.md`.
- Do not add dependencies. `inspect-ai` is already a dev dependency.
- `backend/uv.lock` has a pre-existing user modification. Do not modify, stage, or commit it.

## Locked File Structure

Create:

- `backend/skill_eval/routing.py` — route evidence models and stream observer.
- `backend/skill_eval/judge.py` — bounded evidence bundle, shared rubrics, structured judge call, format-only repair.
- `backend/skill_eval/report.py` — result extraction, metrics, acceptance, JSON/Markdown rendering.
- `backend/skill_eval/poc.py` — preflight and one-command orchestration.
- `backend/evals/skills_routing_eval.py` — Inspect routing task.
- `backend/evals/skills_quality_eval.py` — Inspect full-run + judge task.
- `backend/cases/literature_skill_routing.jsonl` — 20 reviewed cases.
- `backend/tests/skill_eval/test_routing.py` — observer behavior.
- `backend/tests/skill_eval/test_report.py` — aggregation and report contracts.
- `backend/tests/skill_eval/test_judge.py` — judge payload, parsing, repair, evidence validation.
- `backend/tests/skill_eval/test_poc.py` — orchestration, preflight, outputs, exit codes.

Modify:

- `backend/skill_eval/case_schema.py` — replace assertion types with `RoutingCase`.
- `backend/skill_eval/dataset_loader.py` — load and validate routing cases.
- `backend/skill_eval/trace_schema.py` — stable IDs, artifacts, compact observable trace.
- `backend/skill_eval/agent_runner.py` — request/result contracts and required real-runner boundary.
- `backend/skill_eval/adapters/deerflow.py` — compact trace adapter, observer integration, probe/full subprocess execution.
- `backend/skill_eval/inspect_solver.py` — route/full run metadata wiring.
- `backend/skill_eval/inspect_scorer.py` — deterministic route scorer and judge scorer adapter.
- `backend/tests/skill_eval/test_dataset_loader.py` — new schema and suite validation.
- `backend/tests/skill_eval/test_deerflow_adapter.py` — compact trace, artifacts, route evidence.
- `backend/tests/skill_eval/test_deerflow_runner.py` — real runner protocol and optional smoke.

Remove only after the new path passes a real three-case smoke:

- `backend/skill_eval/assertion_engine.py`
- `backend/skill_eval/adapters/mock.py`
- `backend/evals/skills_eval.py`
- `backend/cases/gcp_skills.jsonl`
- `backend/cases/no_write_todos.jsonl`
- `backend/tests/skill_eval/test_assertion_engine.py`
- `backend/tests/skill_eval/test_mock_eval.py`
- `backend/tests/skill_eval/test_skill_assertion_scorer.py`
- `backend/tests/skill_eval/test_trace_integrity_scorer.py`

---

### Task 1: Routing Case Contract and Balanced Dataset

**Files:**
- Modify: `backend/skill_eval/case_schema.py`
- Modify: `backend/skill_eval/dataset_loader.py`
- Create: `backend/cases/literature_skill_routing.jsonl`
- Modify: `backend/tests/skill_eval/test_dataset_loader.py`

**Interfaces:**
- Produces: `RouteLabel`, `CANDIDATE_SKILLS`, `RoutingCase`.
- Produces: `read_routing_cases(path: str | Path) -> list[RoutingCase]`.
- Produces: `load_routing_samples(path: str | Path, *, tags: set[str] | None = None) -> list[Sample]`.
- Produces: `validate_poc_suite(cases: Sequence[RoutingCase]) -> None`.
- Consumers: Inspect task factories and `skill_eval.poc` preflight.

- [ ] **Step 1: Replace loader tests with the routing-case contract**

Write these tests in `backend/tests/skill_eval/test_dataset_loader.py`:

```python
import json
from collections import Counter

import pytest

from skill_eval.case_schema import CANDIDATE_SKILLS, RoutingCase
from skill_eval.dataset_loader import load_routing_samples, read_routing_cases, validate_poc_suite


def test_routing_case_rejects_unknown_label():
    with pytest.raises(ValueError):
        RoutingCase(
            id="bad-route",
            input="Review several papers",
            expected_route="deep-research",
            rationale="Not a benchmark class",
        )


def test_load_routing_samples_preserves_label_and_private_rationale(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "slr-001",
                "input": "Compare methods across five papers",
                "expected_route": "systematic-literature-review",
                "rationale": "Multiple-paper synthesis",
                "tags": ["implicit"],
            }
        )
        + "\n"
    )

    samples = load_routing_samples(path)

    assert len(samples) == 1
    assert samples[0].id == "slr-001"
    assert str(samples[0].target) == "systematic-literature-review"
    assert samples[0].metadata["case"]["rationale"] == "Multiple-paper synthesis"


def test_quality_filter_selects_only_tagged_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {"id": "a", "input": "A", "expected_route": "none", "rationale": "A", "tags": []},
        {"id": "b", "input": "B", "expected_route": "none", "rationale": "B", "tags": ["quality"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert [sample.id for sample in load_routing_samples(path, tags={"quality"})] == ["b"]


def test_committed_poc_suite_is_balanced_and_non_leaking():
    cases = read_routing_cases("cases/literature_skill_routing.jsonl")
    validate_poc_suite(cases)

    assert len(cases) == 20
    assert Counter(case.expected_route for case in cases) == {
        "systematic-literature-review": 8,
        "academic-paper-review": 6,
        "none": 6,
    }
    assert sum("quality" in case.tags for case in cases) == 4
    assert len({case.id for case in cases}) == 20
    for case in cases:
        assert not case.input.lstrip().startswith("/")
        assert all(skill not in case.input for skill in CANDIDATE_SKILLS)
```

- [ ] **Step 2: Run the tests and verify the old schema fails**

Run:

```bash
cd backend
uv run pytest tests/skill_eval/test_dataset_loader.py -v
```

Expected: FAIL because `RoutingCase`, `CANDIDATE_SKILLS`, and routing loader functions do not exist.

- [ ] **Step 3: Replace `case_schema.py` with the minimal route schema**

Use this implementation:

```python
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

RouteLabel: TypeAlias = Literal[
    "systematic-literature-review",
    "academic-paper-review",
    "none",
]

CANDIDATE_SKILLS = (
    "systematic-literature-review",
    "academic-paper-review",
)


class RoutingCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    expected_route: RouteLabel
    rationale: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "input", "rationale")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value]
        if any(not tag for tag in normalized):
            raise ValueError("tags must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("tags must be unique")
        return normalized
```

Do not add assertion compatibility aliases.

- [ ] **Step 4: Implement the routing dataset loader and suite validator**

Replace `backend/skill_eval/dataset_loader.py` with:

```python
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from inspect_ai.dataset import Sample

from skill_eval.case_schema import CANDIDATE_SKILLS, RoutingCase

_EXPECTED_COUNTS = {
    "systematic-literature-review": 8,
    "academic-paper-review": 6,
    "none": 6,
}


def read_routing_cases(path: str | Path) -> list[RoutingCase]:
    source = Path(path)
    cases: list[RoutingCase] = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(RoutingCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid routing case at {source}:{line_number}: {exc}") from exc
    return cases


def load_routing_samples(path: str | Path, *, tags: set[str] | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for case in read_routing_cases(path):
        if tags and not tags.issubset(case.tags):
            continue
        samples.append(
            Sample(
                id=case.id,
                input=case.input,
                target=case.expected_route,
                metadata={"case": case.model_dump()},
            )
        )
    return samples


def validate_poc_suite(cases: Sequence[RoutingCase]) -> None:
    errors: list[str] = []
    if len(cases) != 20:
        errors.append(f"expected 20 cases, found {len(cases)}")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case ids must be unique")
    counts = Counter(case.expected_route for case in cases)
    if dict(counts) != _EXPECTED_COUNTS:
        errors.append(f"expected route counts {_EXPECTED_COUNTS}, found {dict(counts)}")
    quality_count = sum("quality" in case.tags for case in cases)
    if quality_count != 4:
        errors.append(f"expected 4 quality cases, found {quality_count}")
    for case in cases:
        if case.input.lstrip().startswith("/"):
            errors.append(f"{case.id}: slash activation is not an autonomous routing case")
        leaked = [skill for skill in CANDIDATE_SKILLS if skill in case.input]
        if leaked:
            errors.append(f"{case.id}: prompt leaks skill name(s): {', '.join(leaked)}")
    if errors:
        raise ValueError("Invalid POC suite:\n- " + "\n- ".join(errors))
```

- [ ] **Step 5: Create the exact 20-case JSONL dataset**

Write one JSON object per line to `backend/cases/literature_skill_routing.jsonl` using these exact records:

```json
{"id":"slr-explicit-gnn-001","input":"Survey recent papers on graph neural networks for drug discovery. Compare 5 papers and use BibTeX format.","expected_route":"systematic-literature-review","rationale":"The request explicitly asks for a bounded comparison across multiple papers.","tags":["explicit","multi-paper","quality"]}
{"id":"slr-implicit-rlhf-001","input":"What does the literature say about RLHF? Synthesize 3 representative papers and use APA format.","expected_route":"systematic-literature-review","rationale":"The literature plus a three-paper synthesis requires cross-paper review.","tags":["implicit","multi-paper","quality"]}
{"id":"slr-attention-variants-001","input":"Survey transformer attention variants published in the last two years on arXiv cs.CL.","expected_route":"systematic-literature-review","rationale":"A time-bounded survey across publications is a literature review.","tags":["explicit","time-window"]}
{"id":"slr-few-shot-bibtex-001","input":"What methods do recent papers use for few-shot learning in vision and language? Give me 15 papers in BibTeX.","expected_route":"systematic-literature-review","rationale":"The request asks for methods synthesized across a specified paper set.","tags":["implicit","citation-format"]}
{"id":"slr-rag-findings-001","input":"Review the literature on retrieval-augmented generation, focusing on key findings, limitations, and open questions.","expected_route":"systematic-literature-review","rationale":"The request explicitly requires thematic synthesis across a literature body.","tags":["explicit","synthesis"]}
{"id":"slr-hallucination-frameworks-001","input":"Compare evaluation frameworks used across LLM hallucination detection papers.","expected_route":"systematic-literature-review","rationale":"Cross-paper framework comparison requires multi-paper synthesis.","tags":["implicit","comparison"]}
{"id":"slr-mortgage-risk-001","input":"Summarize recent work on Monte Carlo methods for mortgage risk from the last three years.","expected_route":"systematic-literature-review","rationale":"Recent work over a time window denotes a research-literature survey.","tags":["implicit","time-window"]}
{"id":"slr-agentic-bibliography-001","input":"Create an annotated bibliography on agentic tool use covering 20 papers in IEEE format.","expected_route":"systematic-literature-review","rationale":"An annotated bibliography across twenty papers is a literature-review task.","tags":["explicit","citation-format"]}
{"id":"paper-review-arxiv-001","input":"Review this paper and assess its methodology, contribution, strengths, and weaknesses: https://arxiv.org/abs/2310.06825","expected_route":"academic-paper-review","rationale":"One specified arXiv paper requires depth-first paper review.","tags":["explicit","sibling-collision","quality"]}
{"id":"paper-review-upload-001","input":"I attached one research paper. Summarize its method and critique the evidence supporting its conclusions.","expected_route":"academic-paper-review","rationale":"A single attached paper requires paper review rather than a survey.","tags":["implicit","sibling-collision"]}
{"id":"paper-review-methodology-001","input":"Analyze the experimental methodology in this preprint: https://arxiv.org/abs/1706.03762","expected_route":"academic-paper-review","rationale":"The request targets methodology in one identified preprint.","tags":["explicit","sibling-collision"]}
{"id":"paper-review-peer-review-001","input":"Write a constructive peer review for the single manuscript at https://arxiv.org/abs/2203.02155","expected_route":"academic-paper-review","rationale":"A peer review of one manuscript belongs to single-paper review.","tags":["explicit","sibling-collision"]}
{"id":"paper-review-study-summary-001","input":"Summarize this study and tell me whether its conclusions follow from the reported results.","expected_route":"academic-paper-review","rationale":"The demonstrative reference to one study requests analysis of a single work.","tags":["implicit","sibling-collision"]}
{"id":"paper-review-attention-001","input":"Explain the main contribution and limitations of the paper Attention Is All You Need.","expected_route":"academic-paper-review","rationale":"A named single paper requires focused review.","tags":["implicit","sibling-collision"]}
{"id":"none-precision-recall-001","input":"Explain the difference between precision and recall with one simple example.","expected_route":"none","rationale":"A direct conceptual explanation requires neither academic skill.","tags":["near-domain","direct-answer","quality"]}
{"id":"none-attention-concept-001","input":"What is attention in transformers?","expected_route":"none","rationale":"A factual concept question does not request paper analysis or synthesis.","tags":["near-domain","direct-answer"]}
{"id":"none-ai-news-001","input":"Search for the latest news about AI regulation and summarize the top developments.","expected_route":"none","rationale":"General news research is not academic paper review or literature synthesis.","tags":["research","non-academic"]}
{"id":"none-bibtex-code-001","input":"Write a Python function that parses BibTeX files into dictionaries.","expected_route":"none","rationale":"A coding request involving BibTeX is not an academic review task.","tags":["keyword-collision","coding"]}
{"id":"none-translation-001","input":"Translate this paragraph to Chinese: Machine learning systems should be evaluated carefully.","expected_route":"none","rationale":"Translation requires neither candidate skill.","tags":["unrelated","translation"]}
{"id":"none-best-paper-001","input":"Find me one good introductory paper on reinforcement learning.","expected_route":"none","rationale":"Finding one recommendation is not reviewing a supplied paper or synthesizing a literature set.","tags":["near-boundary","single-result"]}
```

- [ ] **Step 6: Run the loader tests**

Run:

```bash
cd backend
uv run pytest tests/skill_eval/test_dataset_loader.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the case contract**

```bash
git add backend/skill_eval/case_schema.py backend/skill_eval/dataset_loader.py backend/cases/literature_skill_routing.jsonl backend/tests/skill_eval/test_dataset_loader.py
git commit -m "feat: add balanced skill routing cases"
```

---

### Task 2: Deterministic Routing Observer

**Files:**
- Create: `backend/skill_eval/routing.py`
- Create: `backend/tests/skill_eval/test_routing.py`

**Interfaces:**
- Consumes: `CANDIDATE_SKILLS` and DeerFlow `StreamEvent`.
- Produces: `RouteEvidence`, `RouteObservation`, `RoutingObserver`.
- Produces: `RoutingObserver.feed(event: StreamEvent) -> bool`, where `True` means probe mode may close the stream after the current candidate-read batch settled.
- Produces: `RoutingObserver.fail(message: str) -> None` and `RoutingObserver.finalize(*, stream_completed: bool) -> RouteObservation`.

- [ ] **Step 1: Write observer tests for discovery, selection, ambiguity, none, and errors**

Create `backend/tests/skill_eval/test_routing.py` with event helpers and these contracts:

```python
from deerflow.client import StreamEvent

from skill_eval.case_schema import CANDIDATE_SKILLS
from skill_eval.routing import RoutingObserver


def ai_tools(message_id: str, *calls: dict) -> StreamEvent:
    return StreamEvent(type="messages-tuple", data={"type": "ai", "id": message_id, "content": "", "tool_calls": list(calls)})


def tool_result(call_id: str, name: str, content: str) -> StreamEvent:
    return StreamEvent(type="messages-tuple", data={"type": "tool", "tool_call_id": call_id, "name": name, "content": content, "id": f"result-{call_id}"})


def end() -> StreamEvent:
    return StreamEvent(type="end", data={"usage": {}})


def test_describe_then_load_selects_skill_only_after_successful_read():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    assert observer.feed(ai_tools("m1", {"id": "d1", "name": "describe_skill", "args": {"name": "systematic-literature-review"}})) is False
    assert observer.feed(tool_result("d1", "describe_skill", "description")) is False
    assert observer.feed(ai_tools("m2", {"id": "r1", "name": "read_file", "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"}})) is False
    assert observer.feed(tool_result("r1", "read_file", "---\nname: systematic-literature-review\n---")) is True

    result = observer.finalize(stream_completed=False)
    assert result.completed is True
    assert result.observed_route == "systematic-literature-review"
    assert [e.kind for e in result.evidence] == ["described", "load_requested", "loaded"]


def test_describe_without_load_finishes_as_none():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(ai_tools("m1", {"id": "d1", "name": "describe_skill", "args": {"name": "academic-paper-review"}}))
    observer.feed(tool_result("d1", "describe_skill", "description"))
    observer.feed(end())
    assert observer.finalize(stream_completed=True).observed_route == "none"


def test_failed_skill_read_does_not_select_route():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(ai_tools("m1", {"id": "r1", "name": "read_file", "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"}}))
    assert observer.feed(tool_result("r1", "read_file", "Error: file not found")) is False
    observer.feed(end())
    result = observer.finalize(stream_completed=True)
    assert result.observed_route == "none"
    assert [e.kind for e in result.evidence] == ["load_requested", "load_failed"]


def test_two_successful_loads_in_same_message_are_ambiguous():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {"id": "r1", "name": "read_file", "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"}},
            {"id": "r2", "name": "read_file", "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"}},
        )
    )
    assert observer.feed(tool_result("r1", "read_file", "skill one")) is False
    assert observer.feed(tool_result("r2", "read_file", "skill two")) is True
    assert observer.finalize(stream_completed=False).observed_route == "ambiguous"


def test_later_batches_do_not_replace_first_route():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(ai_tools("m1", {"id": "r1", "name": "read_file", "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"}}))
    assert observer.feed(tool_result("r1", "read_file", "skill one")) is True
    assert observer.feed(ai_tools("m2", {"id": "r2", "name": "read_file", "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"}})) is True
    assert observer.finalize(stream_completed=True).observed_route == "systematic-literature-review"


def test_unrelated_read_is_not_route_evidence():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(ai_tools("m1", {"id": "r1", "name": "read_file", "args": {"path": "/mnt/user-data/workspace/paper.md"}}))
    observer.feed(tool_result("r1", "read_file", "paper"))
    observer.feed(end())
    result = observer.finalize(stream_completed=True)
    assert result.observed_route == "none"
    assert result.evidence == []


def test_shadow_skill_path_is_not_route_evidence():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(ai_tools("m1", {"id": "r1", "name": "read_file", "args": {"path": "/tmp/systematic-literature-review/SKILL.md"}}))
    observer.feed(tool_result("r1", "read_file", "not the mounted skill"))
    observer.feed(end())
    assert observer.finalize(stream_completed=True).observed_route == "none"


def test_stream_failure_is_not_none():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.fail("stream timed out")
    result = observer.finalize(stream_completed=False)
    assert result.completed is False
    assert result.observed_route is None
    assert result.errors == ["stream timed out"]
```

Also assert stable evidence IDs (`route_evidence[0]`, `route_evidence[1]`) and that a tool result with no matching call is ignored.

- [ ] **Step 2: Run the new observer tests and verify they fail**

```bash
cd backend
uv run pytest tests/skill_eval/test_routing.py -v
```

Expected: FAIL because `skill_eval.routing` does not exist.

- [ ] **Step 3: Implement route evidence and batch-aware observation**

Implement `backend/skill_eval/routing.py` with these models and state transitions:

```python
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, Field

from deerflow.client import StreamEvent
from skill_eval.case_schema import RouteLabel

EvidenceKind: TypeAlias = Literal["described", "load_requested", "loaded", "load_failed"]


class RouteEvidence(BaseModel):
    id: str
    kind: EvidenceKind
    skill: str
    tool_call_id: str
    detail: str | None = None


class RouteObservation(BaseModel):
    observed_route: RouteLabel | Literal["ambiguous"] | None = None
    evidence: list[RouteEvidence] = Field(default_factory=list)
    completed: bool = False
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None


class _PendingCall(BaseModel):
    batch_id: str
    kind: Literal["describe", "load"]
    skill: str


class RoutingObserver:
    def __init__(self, candidates: tuple[str, ...]):
        self._candidates = frozenset(candidates)
        self._pending: dict[str, _PendingCall] = {}
        self._batch_loads: dict[str, set[str]] = defaultdict(set)
        self._batch_pending_loads: dict[str, set[str]] = defaultdict(set)
        self._evidence: list[RouteEvidence] = []
        self._errors: list[str] = []
        self._observed: RouteLabel | Literal["ambiguous"] | None = None
        self._completed = False

    def feed(self, event: StreamEvent) -> bool:
        if self._completed:
            return True
        if event.type == "messages-tuple" and event.data.get("type") == "ai":
            self._feed_ai(event.data)
        elif event.type == "messages-tuple" and event.data.get("type") == "tool":
            self._feed_tool(event.data)
        elif event.type == "end" and self._observed is None and not self._errors:
            if self._pending_candidate_loads():
                self.fail("stream ended with unresolved candidate skill reads")
            else:
                self._observed = "none"
                self._completed = True
        return self._completed and self._observed != "none"

    def fail(self, message: str) -> None:
        self._errors.append(message)
        self._completed = False
        self._observed = None

    def finalize(self, *, stream_completed: bool, latency_ms: int | None = None) -> RouteObservation:
        if self._observed is None and not self._errors and stream_completed:
            self._observed = "none"
            self._completed = True
        if self._observed is None and not self._errors:
            self.fail("stream stopped before a routing decision")
        return RouteObservation(
            observed_route=self._observed,
            evidence=list(self._evidence),
            completed=self._completed,
            errors=list(self._errors),
            latency_ms=latency_ms,
        )

    def _feed_ai(self, data: dict) -> None:
        batch_id = str(data.get("id") or f"batch-{len(self._pending)}")
        for call in data.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            name = call.get("name")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if not call_id:
                continue
            if name == "describe_skill" and args.get("name") in self._candidates:
                self._pending[call_id] = _PendingCall(batch_id=batch_id, kind="describe", skill=args["name"])
                continue
            skill = self._skill_from_read(name, args)
            if skill:
                self._pending[call_id] = _PendingCall(batch_id=batch_id, kind="load", skill=skill)
                self._batch_pending_loads[batch_id].add(call_id)
                self._record("load_requested", skill, call_id)

    def _feed_tool(self, data: dict) -> None:
        call_id = str(data.get("tool_call_id") or "")
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        content = str(data.get("content") or "")
        failed = content.lstrip().startswith("Error:")
        if pending.kind == "describe":
            if not failed:
                self._record("described", pending.skill, call_id)
            return
        self._batch_pending_loads[pending.batch_id].discard(call_id)
        if failed:
            self._record("load_failed", pending.skill, call_id, content[:500])
        else:
            self._batch_loads[pending.batch_id].add(pending.skill)
            self._record("loaded", pending.skill, call_id)
        if self._batch_pending_loads[pending.batch_id]:
            return
        loaded = self._batch_loads[pending.batch_id]
        if len(loaded) == 1:
            self._observed = cast(RouteLabel, next(iter(loaded)))
            self._completed = True
        elif len(loaded) > 1:
            self._observed = "ambiguous"
            self._completed = True

    def _skill_from_read(self, name: object, args: dict) -> str | None:
        if name not in {"read_file", "read_file_tool"}:
            return None
        path = args.get("path") or args.get("file_path") or args.get("filepath")
        if not isinstance(path, str):
            return None
        pure = PurePosixPath(path)
        parts = pure.parts
        if (
            pure.name != "SKILL.md"
            or pure.parent.name not in self._candidates
            or len(parts) < 4
            or parts[-4:-2] != ("skills", "public")
        ):
            return None
        return pure.parent.name

    def _pending_candidate_loads(self) -> bool:
        return any(call.kind == "load" for call in self._pending.values())

    def _record(self, kind: EvidenceKind, skill: str, call_id: str, detail: str | None = None) -> None:
        self._evidence.append(
            RouteEvidence(
                id=f"route_evidence[{len(self._evidence)}]",
                kind=kind,
                skill=skill,
                tool_call_id=call_id,
                detail=detail,
            )
        )
```


- [ ] **Step 4: Run observer tests**

```bash
cd backend
uv run pytest tests/skill_eval/test_routing.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the observer**

```bash
git add backend/skill_eval/routing.py backend/tests/skill_eval/test_routing.py
git commit -m "feat: observe skill routing from DeerFlow events"
```

---

### Task 3: Real Runner Probe/Full Modes and Compact Trace

**Files:**
- Modify: `backend/skill_eval/trace_schema.py`
- Modify: `backend/skill_eval/agent_runner.py`
- Modify: `backend/skill_eval/adapters/deerflow.py`
- Modify: `backend/tests/skill_eval/test_deerflow_adapter.py`
- Modify: `backend/tests/skill_eval/test_deerflow_runner.py`

**Interfaces:**
- Consumes: `RoutingObserver`, `CANDIDATE_SKILLS`.
- Produces: `RunMode = Literal["routing_probe", "full"]`.
- Produces: `AgentRunRequest(case_id, user_input, mode, model_name, candidate_skills, timeout_seconds)`.
- Produces: `AgentRunResult(final_answer, success, trace, route_observation, thread_id)`.
- Produces: `DeerFlowAgentRunner.run(request) -> AgentRunResult` using a spawned child process with hard timeout cleanup.
- Full mode snapshots paths emitted by `values.artifacts` through `DeerFlowClient.get_artifact()`.

- [ ] **Step 1: Write failing adapter tests for stable tool IDs, merged message chunks, route observation, and artifacts**

Update `test_deerflow_adapter.py` so its core assertions include:

```python
def test_adapter_merges_ai_chunks_by_message_id():
    request = AgentRunRequest(case_id="c1", user_input="hello", mode="full", model_name="default")
    adapter = DeerFlowTraceAdapter(request)
    adapter.start()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "id": "m1", "content": "Hel"}))
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "id": "m1", "content": "lo"}))
    adapter.feed(_make_event("end", {"usage": {"input_tokens": 2, "output_tokens": 1}}))

    trace = adapter.build(thread_id="thread-1")

    assert trace.final_answer == "Hello"
    assert [m["content"] for m in trace.messages if m.get("id") == "m1"] == ["Hello"]


def test_adapter_correlates_tool_result_with_stable_ids():
    request = AgentRunRequest(case_id="c1", user_input="read", mode="full", model_name="default")
    adapter = DeerFlowTraceAdapter(request)
    adapter.start()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "id": "m1", "content": "", "tool_calls": [{"id": "t1", "name": "read_file", "args": {"path": "x"}}]}))
    adapter.feed(_make_event("messages-tuple", {"type": "tool", "id": "tm1", "tool_call_id": "t1", "name": "read_file", "content": "body"}))

    call = adapter.build(thread_id="thread-1").tool_calls[0]
    assert call.id == "t1"
    assert call.message_id == "m1"
    assert call.result == "body"
    assert call.error is None


def test_values_events_retain_unique_artifact_paths():
    request = AgentRunRequest(case_id="c1", user_input="write", mode="full", model_name="default")
    adapter = DeerFlowTraceAdapter(request)
    adapter.start()
    adapter.feed(_make_event("values", {"messages": [], "artifacts": ["/mnt/user-data/outputs/report.md"]}))
    adapter.feed(_make_event("values", {"messages": [], "artifacts": ["/mnt/user-data/outputs/report.md"]}))
    assert adapter.artifact_paths == ["/mnt/user-data/outputs/report.md"]
```

Add a scripted fake client test for `_execute_deerflow()`:

```python
import pytest

from deerflow.client import StreamEvent

from skill_eval.adapters.deerflow import _execute_deerflow
from skill_eval.agent_runner import AgentRunRequest


def _make_event(event_type, data):
    return StreamEvent(type=event_type, data=data)


class ScriptedStream:
    def __init__(self, client, events):
        self._client = client
        self._events = iter(events)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._events)
        if item is pytest.fail:
            pytest.fail("stream consumed past routing decision")
        return item

    def close(self):
        self._client.stream_closed = True


class ScriptedClient:
    def __init__(self, events):
        self.events = events
        self.stream_closed = False

    def stream(self, message, *, thread_id):
        return ScriptedStream(self, self.events)

    def get_artifact(self, thread_id, path):
        raise AssertionError("probe mode must not fetch artifacts")


def ai_read_skill_event(skill, call_id):
    return _make_event(
        "messages-tuple",
        {
            "type": "ai",
            "id": "m1",
            "content": "",
            "tool_calls": [{"id": call_id, "name": "read_file", "args": {"path": f"/mnt/skills/public/{skill}/SKILL.md"}}],
        },
    )


def tool_skill_result(call_id):
    return _make_event(
        "messages-tuple",
        {"type": "tool", "id": f"result-{call_id}", "tool_call_id": call_id, "name": "read_file", "content": "skill body"},
    )


def test_probe_mode_closes_real_stream_after_route_batch():
    client = ScriptedClient(events=[
        ai_read_skill_event("systematic-literature-review", "t1"),
        tool_skill_result("t1"),
        pytest.fail,
    ])
    request = AgentRunRequest(case_id="c1", user_input="survey papers", mode="routing_probe", model_name="default")

    result = _execute_deerflow(request, client_factory=lambda **_: client)

    assert result.route_observation.observed_route == "systematic-literature-review"
    assert client.stream_closed is True
    assert result.trace.runtime == "deerflow"
```

The sentinel `pytest.fail` MUST never be consumed.

- [ ] **Step 2: Run adapter/runner tests and verify they fail**

```bash
cd backend
uv run pytest tests/skill_eval/test_deerflow_adapter.py tests/skill_eval/test_deerflow_runner.py -v
```

Expected: FAIL on the new request/result schema and trace fields.

- [ ] **Step 3: Replace runner and trace contracts**

Implement these fields in `trace_schema.py`:

```python
from typing import Any

from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    id: str
    message_id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None


class AgentArtifact(BaseModel):
    path: str
    mime_type: str
    content: str
    original_bytes: int
    sha256: str
    truncated: bool


class AgentTrace(BaseModel):
    input: str
    final_answer: str
    success: bool
    thread_id: str
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[AgentArtifact] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    runtime: str = "deerflow"
    raw_trace_ref: str | None = None
```

Replace `agent_runner.py` contracts with:

```python
from typing import Literal, Protocol, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field

from skill_eval.case_schema import CANDIDATE_SKILLS
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace

RunMode: TypeAlias = Literal["routing_probe", "full"]


class AgentRunRequest(BaseModel):
    case_id: str
    user_input: str
    mode: RunMode
    model_name: str
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_skills: tuple[str, ...] = CANDIDATE_SKILLS
    timeout_seconds: int = 300
    trace_dir: str | None = None


class AgentRunResult(BaseModel):
    final_answer: str
    success: bool
    trace: AgentTrace
    route_observation: RouteObservation
    thread_id: str


class AgentRunner(Protocol):
    async def run(self, request: AgentRunRequest) -> AgentRunResult: ...
```

Remove `target`, `required_skills`, `forced_skills`, sandbox fields, mock default dispatch, and `SkillInvocation`.

- [ ] **Step 4: Refactor `DeerFlowTraceAdapter` to compact messages and expose artifacts**

Required behavior:

- Merge token deltas by AI message ID.
- Emit one normalized AI message per ID in first-seen order.
- Retain tool message records separately.
- Store tool call `id` and parent AI `message_id`.
- Infer a tool error when result text starts with `Error:`.
- Union `values.artifacts` in first-seen order.
- Preserve usage from `end`.
- Build an `AgentTrace` with the supplied thread ID.

Use `dict` insertion order; do not sort message IDs or tool calls.

- [ ] **Step 5: Implement artifact snapshots and synchronous real execution**

In `adapters/deerflow.py`, add:

```python
_ARTIFACT_HEAD_BYTES = 6_000
_ARTIFACT_TAIL_BYTES = 6_000


def _snapshot_artifact(client, thread_id: str, path: str) -> AgentArtifact:
    data, mime_type = client.get_artifact(thread_id, path)
    digest = hashlib.sha256(data).hexdigest()
    truncated = len(data) > _ARTIFACT_HEAD_BYTES + _ARTIFACT_TAIL_BYTES
    if truncated:
        preview = data[:_ARTIFACT_HEAD_BYTES] + b"\n...[truncated]...\n" + data[-_ARTIFACT_TAIL_BYTES:]
    else:
        preview = data
    return AgentArtifact(
        path=path,
        mime_type=mime_type,
        content=preview.decode("utf-8", errors="replace"),
        original_bytes=len(data),
        sha256=digest,
        truncated=truncated,
    )
```

Implement `_execute_deerflow(request, client_factory=DeerFlowClient)`. It MUST:

1. Use `request.thread_id` for the client stream, trace, artifacts, and every failure result; the parent creates it before spawning so timeout/crash evidence remains correlated.
2. Create the real client with `model_name=request.model_name`, `available_skills=set(request.candidate_skills)`, and `subagent_enabled=request.mode == "full"`.
3. Feed every event to both `DeerFlowTraceAdapter` and `RoutingObserver`.
4. In probe mode, break only when `observer.feed(event)` returns true.
5. Close the stream in `finally`.
6. Mark `stream_completed=True` only after an `end` event.
7. Snapshot artifacts only in full mode.
8. Convert construction, stream, and artifact errors into retained result errors rather than throwing away the partial trace.

The function returns `AgentRunResult` even on agent/runtime failure.

- [ ] **Step 6: Put each real run behind a spawned process timeout**

Use `multiprocessing.get_context("spawn")` and a one-way `Pipe`. The module-level child target receives only JSON-compatible request data and config values, calls `_execute_deerflow`, and sends either a result dump or a concise child error.

`DeerFlowAgentRunner.run()` MUST:

- start the process;
- close the child send handle in the parent;
- await `process.join(request.timeout_seconds)` via `asyncio.to_thread`;
- terminate and join a still-live process;
- return a failed `AgentRunResult` with `route_observation.completed=False` on timeout/crash;
- close process and pipe handles;
- never return while a child process is alive.

Do not use `asyncio.wait_for(asyncio.to_thread(client.stream))`: cancellation cannot stop a running Python thread and reproduces the current leaked-worker risk.

- [ ] **Step 7: Run adapter and runner tests**

```bash
cd backend
uv run pytest tests/skill_eval/test_deerflow_adapter.py tests/skill_eval/test_deerflow_runner.py tests/skill_eval/test_routing.py -v
```

Expected: PASS. Real smoke tests may skip only when config/model prerequisites are absent.

- [ ] **Step 8: Commit the real runner modes**

```bash
git add backend/skill_eval/trace_schema.py backend/skill_eval/agent_runner.py backend/skill_eval/adapters/deerflow.py backend/tests/skill_eval/test_deerflow_adapter.py backend/tests/skill_eval/test_deerflow_runner.py
git commit -m "feat: add real routing probe and full runner modes"
```

---

### Task 4: Inspect Routing Task and Deterministic Scorer

**Files:**
- Modify: `backend/skill_eval/inspect_solver.py`
- Modify: `backend/skill_eval/inspect_scorer.py`
- Create: `backend/evals/skills_routing_eval.py`
- Create: `backend/tests/skill_eval/test_routing_eval.py`

**Interfaces:**
- Produces: `deerflow_solver(runner: AgentRunner, *, mode: RunMode, model_name: str, timeout_seconds: int)`.
- Produces metadata keys `agent_trace`, `route_observation`, and `agent_success`.
- Produces: `routing_scorer()` returning `CORRECT`, `INCORRECT`, or `NOANSWER`.
- Produces: `skills_routing_eval(case_file, agent_model)` with a 180-second per-sample timeout.

- [ ] **Step 1: Write failing solver/scorer/task tests**

Create `test_routing_eval.py` with a scripted runner:

```python
import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Target
from inspect_ai.solver import TaskState

from evals.skills_routing_eval import skills_routing_eval
from skill_eval.agent_runner import AgentRunResult
from skill_eval.inspect_scorer import routing_scorer
from skill_eval.inspect_solver import deerflow_solver
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace


class ScriptedRunner:
    async def run(self, request):
        assert request.user_input == "survey papers"
        assert request.mode == "routing_probe"
        return AgentRunResult(
            final_answer="",
            success=True,
            thread_id=request.thread_id,
            route_observation=RouteObservation(observed_route="systematic-literature-review", completed=True),
            trace=AgentTrace(
                input=request.user_input,
                final_answer="",
                success=True,
                thread_id=request.thread_id,
                runtime="deerflow",
            ),
        )


@pytest.fixture
def scripted_runner():
    return ScriptedRunner()


@pytest.fixture
def routing_state():
    case = {
        "id": "route-1",
        "input": "survey papers",
        "expected_route": "systematic-literature-review",
        "rationale": "multi-paper request",
        "tags": [],
    }
    return TaskState(
        model="mock-model",
        sample_id=case["id"],
        epoch=1,
        input=case["input"],
        target=case["expected_route"],
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=""),
        metadata={"case": case},
    )


@pytest.mark.asyncio
async def test_solver_writes_route_and_trace_metadata(scripted_runner, routing_state):
    solver = deerflow_solver(scripted_runner, mode="routing_probe", model_name="default", timeout_seconds=180)
    result = await solver(routing_state, generate=None)
    assert result.metadata["route_observation"]["observed_route"] == "systematic-literature-review"
    assert result.metadata["agent_trace"]["runtime"] == "deerflow"


@pytest.mark.asyncio
async def test_routing_scorer_uses_exact_route_label(routing_state):
    routing_state.metadata["route_observation"] = {
        "observed_route": "systematic-literature-review",
        "evidence": [],
        "completed": True,
        "errors": [],
        "latency_ms": 10,
    }
    score = await routing_scorer()(routing_state, Target("systematic-literature-review"))
    assert score.value == CORRECT


@pytest.mark.asyncio
async def test_routing_scorer_marks_ambiguous_incorrect(routing_state):
    routing_state.metadata["route_observation"] = {
        "observed_route": "ambiguous",
        "evidence": [],
        "completed": True,
        "errors": [],
    }
    score = await routing_scorer()(routing_state, Target("systematic-literature-review"))
    assert score.value == INCORRECT


@pytest.mark.asyncio
async def test_routing_scorer_marks_infrastructure_failure_noanswer(routing_state):
    routing_state.metadata["route_observation"] = {
        "observed_route": None,
        "evidence": [],
        "completed": False,
        "errors": ["timeout"],
    }
    score = await routing_scorer()(routing_state, Target("none"))
    assert score.value == NOANSWER
    assert score.metadata["infrastructure_error"] == "timeout"


def test_routing_task_has_twenty_samples():
    task = skills_routing_eval(case_file="cases/literature_skill_routing.jsonl", agent_model="default")
    assert len(task.dataset) == 20
```


- [ ] **Step 2: Run tests and verify failure**

```bash
cd backend
uv run pytest tests/skill_eval/test_routing_eval.py -v
```

Expected: FAIL because new solver/scorer/task names do not exist.

- [ ] **Step 3: Implement the single DeerFlow solver**

`deerflow_solver()` validates `RoutingCase` from metadata, constructs `AgentRunRequest` with the requested mode/model, awaits the injected real runner, sets `state.output.completion`, and stores:

```python
state.metadata["agent_trace"] = result.trace.model_dump()
state.metadata["route_observation"] = result.route_observation.model_dump()
state.metadata["agent_success"] = result.success
state.metadata["thread_id"] = result.thread_id
```

The solver MUST NOT send `expected_route` or `rationale` to the runner.

- [ ] **Step 4: Implement the deterministic route scorer**

Use Inspect constants:

```python
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, Target, scorer


@scorer(metrics=[])
def routing_scorer():
    async def score(state, target: Target) -> Score:
        try:
            observation = RouteObservation.model_validate(state.metadata["route_observation"])
            case = RoutingCase.model_validate(state.metadata["case"])
        except Exception as exc:
            return Score(value=NOANSWER, explanation=f"Invalid routing metadata: {exc}", metadata={"infrastructure_error": str(exc)})
        if not observation.completed or observation.observed_route is None:
            message = "; ".join(observation.errors) or "incomplete routing observation"
            return Score(value=NOANSWER, explanation=message, metadata={"case_id": case.id, "infrastructure_error": message, "route_observation": observation.model_dump()})
        passed = observation.observed_route == case.expected_route
        return Score(
            value=CORRECT if passed else INCORRECT,
            explanation=f"expected={case.expected_route} observed={observation.observed_route}",
            metadata={"case_id": case.id, "expected_route": case.expected_route, "observed_route": observation.observed_route, "route_observation": observation.model_dump()},
        )
    return score
```

- [ ] **Step 5: Create the routing Inspect task**

`skills_routing_eval()` loads all cases, creates a `DeerFlowAgentRunner`, uses `deerflow_solver(..., mode="routing_probe", timeout_seconds=180)`, and installs only `routing_scorer()`. The task factory takes explicit `case_file` and `agent_model` parameters.

- [ ] **Step 6: Run routing task tests**

```bash
cd backend
uv run pytest tests/skill_eval/test_routing_eval.py tests/skill_eval/test_dataset_loader.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Inspect routing integration**

```bash
git add backend/skill_eval/inspect_solver.py backend/skill_eval/inspect_scorer.py backend/evals/skills_routing_eval.py backend/tests/skill_eval/test_routing_eval.py
git commit -m "feat: add Inspect skill routing benchmark"
```

---

### Task 5: Routing Metrics and Evidence Report

**Files:**
- Create: `backend/skill_eval/report.py`
- Create: `backend/tests/skill_eval/test_report.py`

**Interfaces:**
- Produces: `RoutingEpochResult`, `ClassMetrics`, `RoutingMetrics`.
- Produces: `extract_routing_results(log: EvalLog) -> list[RoutingEpochResult]`.
- Produces: `summarize_routing(results, *, planned_runs: int) -> RoutingMetrics`.
- Produces: `routing_acceptance(metrics) -> bool`.

- [ ] **Step 1: Write failing metric tests using synthetic epoch results**

Create `test_report.py` with these contracts:

```python
from skill_eval.report import RoutingEpochResult, routing_acceptance, summarize_routing


def result(case_id, epoch, expected, observed=None, error=None):
    return RoutingEpochResult(
        case_id=case_id,
        epoch=epoch,
        expected_route=expected,
        observed_route=observed,
        infrastructure_error=error,
        evidence=[],
        log_location=f"logs/{case_id}.eval",
    )


def test_summary_builds_confusion_metrics_and_valid_rate():
    results = [
        result("a", 1, "systematic-literature-review", "systematic-literature-review"),
        result("a", 2, "systematic-literature-review", "academic-paper-review"),
        result("b", 1, "academic-paper-review", "academic-paper-review"),
        result("b", 2, "academic-paper-review", "ambiguous"),
        result("c", 1, "none", "none"),
        result("c", 2, "none", error="timeout"),
    ]

    summary = summarize_routing(results, planned_runs=6)

    assert summary.valid_runs == 5
    assert summary.valid_run_rate == 5 / 6
    assert summary.confusion["systematic-literature-review"]["academic-paper-review"] == 1
    assert summary.confusion["academic-paper-review"]["ambiguous"] == 1
    assert summary.per_class["systematic-literature-review"].recall == 0.5
    assert summary.stable_cases == 0


def test_acceptance_requires_valid_rate_precision_and_recall():
    passing = make_balanced_summary(valid_run_rate=0.95, macro_precision=0.8, macro_recall=0.8)
    assert routing_acceptance(passing) is True
    assert routing_acceptance(passing.model_copy(update={"valid_run_rate": 0.94})) is False
    assert routing_acceptance(passing.model_copy(update={"macro_precision": 0.79})) is False
    assert routing_acceptance(passing.model_copy(update={"macro_recall": 0.79})) is False
```

Also test zero predicted-count precision returns `0.0`, raw epoch results are retained, and a case with any infrastructure result is unstable.

- [ ] **Step 2: Run tests and verify failure**

```bash
cd backend
uv run pytest tests/skill_eval/test_report.py -v
```

Expected: FAIL because `skill_eval.report` does not exist.

- [ ] **Step 3: Implement typed routing result extraction**

`extract_routing_results()` iterates `log.samples or []`, reads `sample.scores["routing_scorer"]`, and maps score metadata into `RoutingEpochResult`. For `NOANSWER`, populate `infrastructure_error`; for valid scores, require an observed label. Store `sample.epoch` and `log.location`.

Raise `ValueError` for missing scorer metadata rather than fabricating a route.

- [ ] **Step 4: Implement the confusion matrix and macro metrics**

Use fixed expected labels from `CANDIDATE_SKILLS + ("none",)` and observed columns plus `ambiguous`. Valid results include `ambiguous`; infrastructure results do not.

For each class:

```python
tp = confusion[label][label]
predicted = sum(confusion[expected][label] for expected in labels)
actual = sum(confusion[label].values())
precision = tp / predicted if predicted else 0.0
recall = tp / actual if actual else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

Stability requires exactly the expected number of epoch results for a case, no infrastructure error, and one unique observed label.

- [ ] **Step 5: Run report tests**

```bash
cd backend
uv run pytest tests/skill_eval/test_report.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit routing reporting**

```bash
git add backend/skill_eval/report.py backend/tests/skill_eval/test_report.py
git commit -m "feat: aggregate skill routing metrics"
```

---

### Task 6: Evidence-Grounded LLM Judge and Quality Task

**Files:**
- Create: `backend/skill_eval/judge.py`
- Modify: `backend/skill_eval/inspect_scorer.py`
- Create: `backend/evals/skills_quality_eval.py`
- Create: `backend/tests/skill_eval/test_judge.py`
- Create: `backend/tests/skill_eval/test_quality_eval.py`

**Interfaces:**
- Produces: `QualityJudgment`, `EvidenceItem`, `JudgeEvidenceBundle`.
- Produces: `build_judge_evidence(case, trace, observation, skill_descriptions) -> JudgeEvidenceBundle` without expected label/rationale.
- Produces: `load_candidate_skill_descriptions(skills_root: Path) -> dict[str, str]`.
- Produces: `judge_quality(bundle, model) -> QualityJudgment` with one format-only repair.
- Produces: `quality_judge_scorer(judge_model: str, skill_descriptions: dict[str, str])`.
- Produces: `skills_quality_eval(case_file, agent_model, judge_model)` with a 900-second per-sample timeout.

- [ ] **Step 1: Write failing evidence and judge tests**

Create `test_judge.py` with a fake Inspect-compatible model whose `generate()` returns `ModelOutput.from_content(...)`.

Required tests:

```python
import json

import pytest
from inspect_ai.model import ModelOutput

from skill_eval.case_schema import RoutingCase
from skill_eval.judge import JudgeFailure, bounded_evidence, build_judge_evidence, judge_quality
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentArtifact, AgentToolCall, AgentTrace


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelOutput.from_content("fake/judge", next(self.responses))


def valid_judgment_json(evidence=None):
    return json.dumps(
        {
            "recommended_route": "systematic-literature-review",
            "route_quality": 3,
            "process_quality": 3,
            "output_quality": 3,
            "overall_quality": 3,
            "fatal_error": False,
            "reasons": ["The observable run satisfies the bounded task."],
            "evidence": evidence or ["tool_call[0]", "final_answer"],
        }
    )


@pytest.fixture
def routing_case():
    return RoutingCase(
        id="quality-1",
        input="Synthesize three papers.",
        expected_route="systematic-literature-review",
        rationale="Multiple-paper synthesis",
        tags=["quality"],
    )


@pytest.fixture
def full_trace():
    return AgentTrace(
        input="Synthesize three papers.",
        final_answer="The papers converge on two findings.",
        success=True,
        thread_id="thread-1",
        tool_calls=[AgentToolCall(id="t1", message_id="m1", name="read_file", args={"path": "SKILL.md"}, result="body")],
        artifacts=[
            AgentArtifact(
                path="/mnt/user-data/outputs/report.md",
                mime_type="text/markdown",
                content="# Report",
                original_bytes=8,
                sha256="0" * 64,
                truncated=False,
            )
        ],
    )


@pytest.fixture
def route_observation():
    return RouteObservation(observed_route="systematic-literature-review", completed=True)


@pytest.fixture
def valid_bundle(routing_case, full_trace, route_observation):
    return build_judge_evidence(
        case=routing_case,
        trace=full_trace,
        observation=route_observation,
        skill_descriptions={"systematic-literature-review": "multi-paper", "academic-paper-review": "one-paper"},
    )

def test_judge_bundle_omits_expected_label_and_rationale(routing_case, full_trace, route_observation):
    bundle = build_judge_evidence(
        case=routing_case,
        trace=full_trace,
        observation=route_observation,
        skill_descriptions={"systematic-literature-review": "multi-paper", "academic-paper-review": "one-paper"},
    )
    payload = bundle.model_dump_json()
    assert "expected_route" not in payload
    assert routing_case.rationale not in payload
    assert "tool_call[0]" in payload
    assert "tool_result[0]" in payload
    assert "final_answer" in payload


def test_large_evidence_is_head_tail_truncated_with_hash():
    item = bounded_evidence("tool_result[0]", "tool_result", "x" * 100_000, remaining_bytes=20_000)
    assert item.truncated is True
    assert item.original_bytes == 100_000
    assert len(item.sha256) == 64
    assert "[truncated" in item.content


@pytest.mark.asyncio
async def test_judge_parses_structured_quality_result(valid_bundle):
    model = FakeModel([valid_judgment_json()])
    result = await judge_quality(valid_bundle, model)
    assert result.recommended_route == "systematic-literature-review"
    assert result.overall_quality == 3


@pytest.mark.asyncio
async def test_judge_repairs_format_once_without_rejudging(valid_bundle):
    model = FakeModel(["not json", valid_judgment_json()])
    result = await judge_quality(valid_bundle, model)
    assert result.overall_quality == 3
    assert len(model.prompts) == 2
    assert "format correction only" in model.prompts[1]


@pytest.mark.asyncio
async def test_judge_rejects_unknown_evidence_reference(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["tool_call[999]", "final_answer"])])
    with pytest.raises(JudgeFailure, match="unknown evidence"):
        await judge_quality(valid_bundle, model)
```

Also cover out-of-range scores, second parse failure, missing trace evidence, and missing final-answer/artifact evidence.

- [ ] **Step 2: Run judge tests and verify failure**

```bash
cd backend
uv run pytest tests/skill_eval/test_judge.py -v
```

Expected: FAIL because `skill_eval.judge` does not exist.

- [ ] **Step 3: Implement bounded, complete observable evidence**

Define:

```python
class EvidenceItem(BaseModel):
    id: str
    kind: Literal["message", "tool_call", "tool_result", "error", "artifact", "final_answer"]
    content: str
    original_bytes: int
    sha256: str
    truncated: bool


class JudgeEvidenceBundle(BaseModel):
    user_input: str
    candidate_skills: dict[str, str]
    observed_route: str
    evidence: list[EvidenceItem]
```

Use an 80,000-byte total payload budget and 12,000-byte per-item budget. Every message, tool call, result, error, artifact, and final answer receives an item ID. When the total content budget is exhausted, retain the item with digest, original size, and an explicit omitted marker; do not drop the item.

Implement `load_candidate_skill_descriptions()` using `deerflow.skills.frontmatter.split_skill_markdown()` against `skills_root / candidate / "SKILL.md"`. Reject a missing file, invalid frontmatter, a declared name that differs from the candidate directory, or a blank/non-string description. The quality task and POC preflight MUST use this same function so the Judge prompt and recorded hashes refer to the same skill files.

- [ ] **Step 4: Implement shared rubrics and strict prompt**

The prompt MUST state:


- evaluate observable behavior only;
- select `recommended_route` independently;
- do not infer hidden reasoning;
- use the 0–4 anchors from the spec;
- cite stable evidence IDs;
- return JSON matching `QualityJudgment.model_json_schema()` and no prose outside JSON.

Include all three shared rubrics verbatim in the module. Do not choose a rubric based on `expected_route` before sending the prompt.

- [ ] **Step 5: Implement parse, evidence validation, and one repair**

`judge_quality()` calls `await model.generate(prompt)`, parses `output.completion` with `QualityJudgment.model_validate_json`, and validates:

- all score fields are 0–4;
- every evidence reference names an existing bundle item;
- at least one reference is a `tool_call`, `tool_result`, `message`, or `error`;
- at least one reference is `final_answer` or an artifact.

On the first parse/schema error only, call the model again with the original output, schema, parse error, and the exact instruction “format correction only; do not reconsider scores or reasons.” Evidence-semantic failures do not receive a repair call.

- [ ] **Step 6: Add the quality judge Inspect scorer**

`quality_judge_scorer(judge_model, skill_descriptions)`:

1. Validates `RoutingCase`, `AgentTrace`, and `RouteObservation` from state metadata.
2. Uses the explicit description mapping supplied by the quality task; it never derives a rubric from the expected label.
3. Calls `get_model(judge_model)` explicitly.
4. Returns `Score(value=judgment.overall_quality, metadata={"quality_judgment": ..., "quality_passed": ...})`.
5. Returns `Score(value=NOANSWER, metadata={"judge_failure": ...})` on `JudgeFailure`.

The scorer never reads Inspect `Target` into the judge prompt.

- [ ] **Step 7: Create and test the quality task**

`skills_quality_eval()` loads only `tags={"quality"}`, reads both candidate descriptions from their `SKILL.md` frontmatter, uses `deerflow_solver(..., mode="full", timeout_seconds=900)` with a real `DeerFlowAgentRunner`, and installs both `routing_scorer()` and `quality_judge_scorer(judge_model, skill_descriptions)`.

Tests MUST assert:

- dataset size is four;
- runner request mode is `full`;
- the judge scorer receives no expected route/rationale;
- a judge failure yields `NOANSWER` and `judge_failure` metadata;
- a valid judgment computes `quality_passed` from all three dimension thresholds, not overall score alone.

Run:

```bash
cd backend
uv run pytest tests/skill_eval/test_judge.py tests/skill_eval/test_quality_eval.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit judge and quality task**

```bash
git add backend/skill_eval/judge.py backend/skill_eval/inspect_scorer.py backend/evals/skills_quality_eval.py backend/tests/skill_eval/test_judge.py backend/tests/skill_eval/test_quality_eval.py
git commit -m "feat: judge full skill execution quality"
```

---

### Task 7: One-Command POC Orchestration and Combined Report

**Files:**
- Modify: `backend/skill_eval/report.py`
- Create: `backend/skill_eval/poc.py`
- Create: `backend/tests/skill_eval/test_poc.py`
- Modify: `backend/tests/skill_eval/test_report.py`

**Interfaces:**
- Produces: `PocSummary` schema version `deerflow.agent-routing-poc.v1`.
- Produces: `run_poc(config: PocConfig) -> tuple[PocSummary, int]`.
- Produces CLI: `python -m skill_eval.poc [--smoke] [--case-file PATH] [--output-dir PATH]`.
- Normal execution reads required `AGENT_MODEL` and `JUDGE_MODEL` environment variables.

- [ ] **Step 1: Write failing preflight/orchestration tests**

Create `test_poc.py` with monkeypatched `inspect_ai.eval` and preflight boundaries. Cover:

```python
def test_preflight_requires_both_model_inputs(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    with pytest.raises(PocConfigurationError, match="AGENT_MODEL"):
        PocConfig.from_env()


def test_preflight_rejects_missing_candidate_skill(fake_client):
    fake_client.skills = [{"name": "systematic-literature-review", "enabled": True}]
    with pytest.raises(PocConfigurationError, match="academic-paper-review"):
        preflight(valid_config, client_factory=lambda **_: fake_client)


def test_run_poc_calls_routing_three_epochs_and_quality_one(monkeypatch, synthetic_logs, tmp_path):
    calls = []
    monkeypatch.setattr("skill_eval.poc.inspect_eval", lambda *args, **kwargs: calls.append(kwargs) or synthetic_logs.pop(0))
    summary, exit_code = run_poc(valid_config.model_copy(update={"output_dir": tmp_path}))
    assert calls[0]["epochs"] == 3
    assert calls[1]["epochs"] == 1
    assert all(call["max_samples"] == 1 for call in calls)
    assert exit_code == 0
    assert (tmp_path / summary.run_id / "summary.json").exists()
    assert (tmp_path / summary.run_id / "summary.md").exists()


def test_exit_codes_separate_quality_failure_and_invalid_evaluation():
    assert exit_code_for(passing_summary()) == 0
    assert exit_code_for(passing_summary(quality_passed_cases=2)) == 1
    assert exit_code_for(passing_summary(judge_failures=1)) == 2
```

Also test routing threshold failure returns `1`, eval/log incompleteness returns `2`, and smoke mode selects one fixed case per class with one epoch and skips full quality/Judge execution.

- [ ] **Step 2: Run CLI/report tests and verify failure**

```bash
cd backend
uv run pytest tests/skill_eval/test_poc.py tests/skill_eval/test_report.py -v
```

Expected: FAIL because orchestration and combined summary do not exist.

- [ ] **Step 3: Implement preflight**

`PocConfig.from_env()` resolves:

- `AGENT_MODEL` — required DeerFlow model name;
- `JUDGE_MODEL` — required Inspect model spec;
- case file default `cases/literature_skill_routing.jsonl`;
- output root default `eval-results`;
- log root default `logs`.

`preflight()` MUST:

1. Load and `validate_poc_suite()` the cases.
2. Instantiate `DeerFlowClient(model_name=agent_model, available_skills=set(CANDIDATE_SKILLS))` without running the agent.
3. Verify `agent_model` exists in `client.list_models()["models"]`.
4. Verify both skills exist and are enabled in `client.list_skills(enabled_only=True)["skills"]`.
5. Call `get_model(judge_model)` to validate the Inspect model spec.
6. Compute SHA-256 for the case file and both `../skills/public/<name>/SKILL.md` files.
7. Return an identity record without secrets.

- [ ] **Step 4: Run both Inspect tasks programmatically**

Use `inspect_ai.eval` (imported as `inspect_eval`) with `model=None`, `max_samples=1`, explicit log directory, `fail_on_error=False`, and `score_on_error=True`.

Normal mode:

```python
routing_logs = inspect_eval(
    skills_routing_eval(case_file=str(config.case_file), agent_model=config.agent_model),
    model=None,
    epochs=3,
    max_samples=1,
    log_dir=str(config.log_dir),
    fail_on_error=False,
    score_on_error=True,
)
quality_logs = inspect_eval(
    skills_quality_eval(case_file=str(config.case_file), agent_model=config.agent_model, judge_model=config.judge_model),
    model=None,
    epochs=1,
    max_samples=1,
    log_dir=str(config.log_dir),
    fail_on_error=False,
    score_on_error=True,
)
```

Smoke mode selects these three sample IDs for one routing epoch and skips the quality task:

```text
slr-attention-variants-001
paper-review-arxiv-001
none-precision-recall-001
```

- [ ] **Step 5: Extend report models for quality and run identities**

Add:

- `RunIdentity` with model names, versions, hashes, timestamps, and log locations.
- `QualityCaseResult` with case ID, route observation, judgment or judge failure, quality pass, and evidence log.
- `PocSummary` with schema version, run ID, identity, routing metrics, quality results, acceptance details, and errors.

A valid POC requires all four quality judgments to be schema/evidence valid. Agent quality passes when at least three cases meet all dimension thresholds. Judge failure makes the evaluation invalid (exit `2`), not merely low quality.

- [ ] **Step 6: Render deterministic JSON and Markdown**

Write JSON with `model_dump_json(indent=2)`. Markdown MUST include:

- identity/hashes;
- 3×4 confusion matrix;
- per-class P/R/F1 and macro values;
- valid-run and stable-case counts;
- failed/unstable cases and evidence IDs;
- four quality judgments with scores, reasons, evidence, and label-review flags;
- infrastructure and judge errors;
- each acceptance threshold with actual value and pass/fail.

Write into `eval-results/{run_id}/` via a temporary file followed by `Path.replace()` so partial summaries are not mistaken for completed runs.

- [ ] **Step 7: Implement CLI exit handling**

`main()` catches `PocConfigurationError` and invalid-evaluation errors, prints the concise reason to stderr, and returns `2`. A completed summary returns `0` or `1` based on thresholds. End the module with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run orchestration/report tests**

```bash
cd backend
uv run pytest tests/skill_eval/test_poc.py tests/skill_eval/test_report.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit the one-command POC**

```bash
git add backend/skill_eval/poc.py backend/skill_eval/report.py backend/tests/skill_eval/test_poc.py backend/tests/skill_eval/test_report.py
git commit -m "feat: add one-command routing eval poc"
```

---

### Task 8: Real Smoke, Clean Cutover, and Full POC Verification

**Files:**
- Remove the obsolete files listed in the locked file structure.
- Modify: `backend/skill_eval/__init__.py`
- Modify: any imports proven stale by focused diagnostics.

**Interfaces:**
- The only supported evaluation entrypoint after cutover is `python -m skill_eval.poc`.
- No compatibility aliases for assertion names, mock runner, or `skills_eval` remain.

- [ ] **Step 1: Run all new focused tests before touching old files**

```bash
cd backend
uv run pytest \
  tests/skill_eval/test_dataset_loader.py \
  tests/skill_eval/test_routing.py \
  tests/skill_eval/test_deerflow_adapter.py \
  tests/skill_eval/test_deerflow_runner.py \
  tests/skill_eval/test_routing_eval.py \
  tests/skill_eval/test_report.py \
  tests/skill_eval/test_judge.py \
  tests/skill_eval/test_quality_eval.py \
  tests/skill_eval/test_poc.py -v
```

Expected: PASS, with only explicitly prerequisite-gated real-agent tests skipped.

- [ ] **Step 2: Run the three-class real routing smoke**

From `backend/`, first export `AGENT_MODEL` as a configured DeerFlow model name and `JUDGE_MODEL` as an Inspect provider/model spec, then run:

```bash
test -n "$AGENT_MODEL" && test -n "$JUDGE_MODEL"
uv run python -m skill_eval.poc --smoke
```

Expected:

- command exits `0` when all three routes match;
- one real Inspect log exists;
- each sample metadata contains `route_observation` and real DeerFlow trace;
- positive cases stop after the candidate skill-load batch;
- the none case reaches normal stream completion;
- no child process remains alive after each sample.

If a route is wrong, treat it as an evaluation finding and inspect the retained trace; do not weaken labels or add string assertions.

- [ ] **Step 3: Remove the obsolete assertion/mock path only after smoke works**

Delete exactly the obsolete files listed above. Replace `backend/skill_eval/__init__.py` docstring with:

```python
"""Real-runtime DeerFlow skill routing and quality evaluation POC."""
```

Remove stale imports surfaced by Python import errors. Do not leave shims, aliases, re-exports, or deprecation comments.

- [ ] **Step 4: Run the complete focused skill-eval suite**

```bash
cd backend
uv run pytest tests/skill_eval -v
uv run ruff check skill_eval evals tests/skill_eval
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Verify module construction without model calls**

```bash
cd backend
uv run python -m skill_eval.poc --help
```

Expected: exit `0`; help documents required environment variables, `--smoke`, case path, output directory, and log directory.

- [ ] **Step 6: Run the complete approved POC**

```bash
cd backend
test -n "$AGENT_MODEL" && test -n "$JUDGE_MODEL"
uv run python -m skill_eval.poc
```

Expected observable completion:

- 60 routing sample epochs are present;
- four quality samples are present;
- four valid judge results are present;
- `summary.json` and `summary.md` exist under one run directory;
- summary counts reconcile with raw Inspect samples;
- exit is `0` when all approved thresholds pass, `1` when valid evidence misses a quality threshold, or `2` only when evaluation evidence is invalid/incomplete.

- [ ] **Step 7: Manually inspect all four quality judgments**

Open their Inspect traces and check:

- every cited evidence ID exists;
- reasons match observable trace/output/artifacts;
- no expected route or case rationale appears in the judge request;
- any judge/human route disagreement appears as `label_review_needed`;
- no hidden reasoning is stored in the judge evidence bundle.

Record discrepancies as test failures and fix the evidence/judge boundary; do not hand-edit generated summaries.

- [ ] **Step 8: Commit the clean cutover**

Stage only the evaluation files. Explicitly exclude `backend/uv.lock`.

```bash
git add backend/skill_eval backend/evals backend/cases backend/tests/skill_eval
git commit -m "refactor: cut over to routing eval poc"
```

- [ ] **Step 9: Request code review before completion**

Invoke `superpowers:requesting-code-review` with the approved spec, this plan, the focused test output, the real smoke output, and the generated POC summary. Address correctness findings before claiming completion.
