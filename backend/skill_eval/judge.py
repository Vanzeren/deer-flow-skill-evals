import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from deerflow.skills.frontmatter import split_skill_markdown
from skill_eval.case_schema import CANDIDATE_SKILLS, RouteLabel, RoutingCase
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace

_EVIDENCE_TOTAL_BYTES = 80_000
_EVIDENCE_ITEM_BYTES = 12_000
_PROCESS_EVIDENCE_KINDS = {"tool_chain", "error"}
_OUTPUT_EVIDENCE_KINDS = {"artifact", "final_answer", "quick_turn"}

_SYSTEMATIC_REVIEW_RUBRIC = """Systematic literature review:
- handle multi-paper scope and requested constraints coherently;
- use a relevant and bounded retrieval process;
- synthesize findings across papers instead of listing papers independently;
- produce internally consistent citations and requested artifacts;
- report limitations and avoid unsupported claims."""

_PAPER_REVIEW_RUBRIC = """Academic paper review:
- remain grounded in the specified single paper;
- identify the paper's method, contribution, evidence, strengths, weaknesses, and limitations;
- distinguish statements from the paper from the agent's critique;
- produce a useful and coherent review."""

_NO_SKILL_RUBRIC = """No skill:
- answer the request directly;
- avoid unnecessary skill loading and tool work;
- remain correct, relevant, and proportionate to the request."""

_COMMON_PROCESS_RUBRIC = """Common process rubric — assess observable execution only:
- tool choice and ordering are coherent;
- tool errors are handled rather than ignored;
- repeated or unused calls are penalized;
- final claims are supported by retrieved evidence;
- final output agrees with the retained trace and artifacts."""

_SCORE_ANCHORS = """All quality fields use these 0-4 anchors:
0: No evaluable result or completely wrong.
1: Severe omissions or largely unusable.
2: Partially satisfies the task with material problems.
3: Satisfies the task with sound evidence and no major defect.
4: Excellent, well-supported, efficient, and complete."""


type EvidenceKind = Literal[
    "tool_chain",
    "error",
    "artifact",
    "final_answer",
    "quick_turn",
]


class EvidenceItem(BaseModel):
    id: str
    kind: EvidenceKind
    content: str
    original_bytes: int
    sha256: str
    truncated: bool


class JudgeEvidenceBundle(BaseModel):
    user_input: str
    candidate_skills: dict[str, str]
    observed_route: str
    evaluation_target: Literal["quick_turn", "final_output"] = "final_output"
    evidence: list[EvidenceItem]
    expected_output: str | None = None


class QualityJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommended_route: RouteLabel
    route_quality: int = Field(ge=0, le=4)
    process_quality: int = Field(ge=0, le=4)
    output_quality: int = Field(ge=0, le=4)
    overall_quality: int = Field(ge=0, le=4)
    fatal_error: bool = False
    reasons: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)

    @field_validator("reasons", "evidence")
    @classmethod
    def reject_blank_entries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("entries must not be blank")
        return normalized


class JudgeFailure(RuntimeError):
    pass


def bounded_evidence(
    evidence_id: str,
    kind: EvidenceKind,
    content: str,
    *,
    remaining_bytes: int,
) -> EvidenceItem:
    original = content.encode()
    digest = hashlib.sha256(original).hexdigest()
    limit = max(0, min(_EVIDENCE_ITEM_BYTES, remaining_bytes))
    if len(original) <= limit:
        retained = content
        truncated = False
    elif limit == 0:
        retained = f"[omitted: {len(original)} bytes; sha256={digest}]"
        truncated = True
    else:
        marker = f"\n[truncated: {len(original) - limit} bytes omitted; sha256={digest}]\n"
        marker_bytes = len(marker.encode())
        source_budget = max(0, limit - marker_bytes)
        head_bytes = source_budget // 2
        tail_bytes = source_budget - head_bytes
        retained = original[:head_bytes].decode(errors="replace") + marker
        if tail_bytes:
            retained += original[-tail_bytes:].decode(errors="replace")
        truncated = True
    return EvidenceItem(
        id=evidence_id,
        kind=kind,
        content=retained,
        original_bytes=len(original),
        sha256=digest,
        truncated=truncated,
    )


def load_candidate_skill_descriptions(skills_root: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for candidate in CANDIDATE_SKILLS:
        skill_file = skills_root / candidate / "SKILL.md"
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read candidate skill {candidate}: {exc}") from exc
        parts, error = split_skill_markdown(content)
        if parts is None:
            raise ValueError(f"Invalid frontmatter for {candidate}: {error}")
        declared_name = parts.metadata.get("name")
        if declared_name != candidate:
            raise ValueError(f"Candidate directory {candidate} declares name {declared_name!r}")
        description = parts.metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Candidate skill {candidate} has no non-blank description")
        descriptions[candidate] = " ".join(description.split())
    return descriptions


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
        batches = [batch for batch in batches if batch and calls_by_id[batch[0]].message_id != quick_turn.message_id]

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

    artifact_ids: set[str] = set()
    for index, artifact in enumerate(trace.artifacts):
        basename = Path(artifact.path).name or f"artifact-{index}"
        artifact_id = f"artifact[{basename}]"
        if artifact_id in artifact_ids:
            artifact_id = f"artifact[{basename}#{index}]"
        artifact_ids.add(artifact_id)
        raw_items.append(
            (
                artifact_id,
                "artifact",
                json.dumps(artifact.model_dump(), ensure_ascii=False, sort_keys=True),
            )
        )
    if target == "quick_turn" and quick_turn is not None:
        raw_items.append(("quick_turn", "quick_turn", quick_turn.content))
    else:
        raw_items.append(("final_answer", "final_answer", trace.final_answer))

    remaining = _EVIDENCE_TOTAL_BYTES
    items: list[EvidenceItem] = []
    for evidence_id, kind, content in raw_items:
        item = bounded_evidence(
            evidence_id,
            kind,
            content,
            remaining_bytes=remaining,
        )
        items.append(item)
        remaining = max(
            0,
            remaining - min(len(content.encode()), _EVIDENCE_ITEM_BYTES, remaining),
        )

    return JudgeEvidenceBundle(
        user_input=case.input,
        candidate_skills={candidate: skill_descriptions[candidate] for candidate in CANDIDATE_SKILLS},
        observed_route=str(observation.observed_route or "unavailable"),
        evaluation_target=target,
        evidence=items,
        expected_output=case.expected_output,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def build_judge_prompt(bundle: JudgeEvidenceBundle) -> str:
    schema = json.dumps(QualityJudgment.model_json_schema(), ensure_ascii=False, sort_keys=True)
    payload = bundle.model_dump_json()
    return f"""Evaluate only the observable behavior in the evidence bundle below.
Select recommended_route independently from the user request, both candidate descriptions, and observed behavior.
Do not infer hidden reasoning or chain-of-thought. Do not assume unobserved work occurred.
Cite only stable evidence IDs present in the bundle.
Return JSON matching this schema and no prose outside JSON:
{schema}

{_SYSTEMATIC_REVIEW_RUBRIC}

{_PAPER_REVIEW_RUBRIC}

{_NO_SKILL_RUBRIC}

{_COMMON_PROCESS_RUBRIC}

{_SCORE_ANCHORS}

Evidence bundle:
{payload}"""


async def judge_quality(bundle: JudgeEvidenceBundle, model: Any) -> QualityJudgment:
    prompt = build_judge_prompt(bundle)
    if bundle.expected_output:
        prompt += f"\n\n## Expected Output Reference\n\nThe following describes what a good answer should cover. Compare the agent's actual output against this reference when scoring output_quality:\n\n{bundle.expected_output}\n"
    try:
        output = await model.generate(prompt)
    except Exception as exc:
        raise JudgeFailure(f"judge model call failed: {exc}") from exc

    try:
        judgment = QualityJudgment.model_validate_json(_strip_fences(output.completion))
    except (ValidationError, ValueError) as exc:
        repair_prompt = _build_repair_prompt(output.completion, exc, QualityJudgment)
        try:
            repaired_output = await model.generate(repair_prompt)
            judgment = QualityJudgment.model_validate_json(_strip_fences(repaired_output.completion))
        except Exception as repair_exc:
            raise JudgeFailure(f"judge output invalid after format repair: {repair_exc}") from repair_exc

    try:
        _validate_evidence_references(bundle, judgment.evidence)
    except JudgeFailure:
        output_ids = sorted(item.id for item in bundle.evidence if item.kind in _OUTPUT_EVIDENCE_KINDS)
        for reference in output_ids:
            if reference not in judgment.evidence:
                judgment.evidence.append(reference)
        _validate_evidence_references(bundle, judgment.evidence)

    return judgment


def _build_repair_prompt(output: str, error: Exception, schema_model: type[BaseModel]) -> str:
    schema = json.dumps(schema_model.model_json_schema(), ensure_ascii=False, sort_keys=True)
    return f"""format correction only; do not reconsider scores or reasons.
Return only corrected JSON matching this schema:
{schema}
Original output:
{output}
Parse or schema error:
{error}"""


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
