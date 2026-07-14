import json

import pytest
from inspect_ai.model import ModelOutput

from skill_eval.case_schema import RoutingCase
from skill_eval.judge import (
    JudgeFailure,
    bounded_evidence,
    build_judge_evidence,
    judge_quality,
    load_candidate_skill_descriptions,
)
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentArtifact, AgentToolCall, AgentTrace


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelOutput.from_content("fake/judge", next(self.responses))


def valid_judgment_json(evidence=None, **updates):
    payload = {
        "recommended_route": "systematic-literature-review",
        "route_quality": 3,
        "process_quality": 3,
        "output_quality": 3,
        "overall_quality": 3,
        "fatal_error": False,
        "reasons": ["The observable run satisfies the bounded task."],
        "evidence": evidence or ["tool_call[0]", "final_answer"],
    }
    payload.update(updates)
    return json.dumps(payload)


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
        messages=[{"type": "ai", "id": "m1", "content": "Working", "tool_calls": []}],
        tool_calls=[
            AgentToolCall(
                id="t1",
                message_id="m1",
                name="read_file",
                args={"path": "SKILL.md"},
                result="body",
            )
        ],
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
    return RouteObservation(
        observed_route="systematic-literature-review",
        completed=True,
    )


@pytest.fixture
def valid_bundle(routing_case, full_trace, route_observation):
    return build_judge_evidence(
        case=routing_case,
        trace=full_trace,
        observation=route_observation,
        skill_descriptions={
            "systematic-literature-review": "multi-paper",
            "academic-paper-review": "one-paper",
        },
    )


def test_judge_bundle_omits_expected_label_and_rationale(
    routing_case,
    full_trace,
    route_observation,
):
    bundle = build_judge_evidence(
        case=routing_case,
        trace=full_trace,
        observation=route_observation,
        skill_descriptions={
            "systematic-literature-review": "multi-paper",
            "academic-paper-review": "one-paper",
        },
    )

    payload = bundle.model_dump_json()

    assert "expected_route" not in payload
    assert routing_case.rationale not in payload
    assert "message[0]" in payload
    assert "tool_call[0]" in payload
    assert "tool_result[0]" in payload
    assert "artifact[report.md]" in payload
    assert "final_answer" in payload


def test_large_evidence_is_head_tail_truncated_with_hash():
    item = bounded_evidence(
        "tool_result[0]",
        "tool_result",
        "x" * 100_000,
        remaining_bytes=20_000,
    )

    assert item.truncated is True
    assert item.original_bytes == 100_000
    assert len(item.sha256) == 64
    assert "[truncated" in item.content


def test_exhausted_evidence_budget_retains_omission_marker():
    item = bounded_evidence(
        "tool_result[0]",
        "tool_result",
        "observable result",
        remaining_bytes=0,
    )

    assert item.truncated is True
    assert "[omitted" in item.content
    assert item.original_bytes == len("observable result")


def test_load_candidate_skill_descriptions_uses_frontmatter(tmp_path):
    for name, description in [
        ("systematic-literature-review", "Multi-paper synthesis"),
        ("academic-paper-review", "Single-paper review"),
    ]:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nBody\n",
            encoding="utf-8",
        )

    descriptions = load_candidate_skill_descriptions(tmp_path)

    assert descriptions == {
        "systematic-literature-review": "Multi-paper synthesis",
        "academic-paper-review": "Single-paper review",
    }


def test_load_candidate_skill_descriptions_rejects_name_mismatch(tmp_path):
    for name in ["systematic-literature-review", "academic-paper-review"]:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: wrong-name\ndescription: Description\n---\nBody\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="declares name"):
        load_candidate_skill_descriptions(tmp_path)


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
    assert "format correction only; do not reconsider scores or reasons" in model.prompts[1]


@pytest.mark.asyncio
async def test_judge_rejects_second_parse_failure(valid_bundle):
    model = FakeModel(["not json", "still not json"])

    with pytest.raises(JudgeFailure, match="after format repair"):
        await judge_quality(valid_bundle, model)


@pytest.mark.asyncio
async def test_judge_repairs_out_of_range_score_once(valid_bundle):
    model = FakeModel(
        [
            valid_judgment_json(route_quality=5),
            valid_judgment_json(route_quality=4),
        ]
    )

    result = await judge_quality(valid_bundle, model)

    assert result.route_quality == 4
    assert len(model.prompts) == 2


@pytest.mark.asyncio
async def test_judge_rejects_unknown_evidence_without_repair(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["tool_call[999]", "final_answer"])])

    with pytest.raises(JudgeFailure, match="unknown evidence"):
        await judge_quality(valid_bundle, model)
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_judge_requires_trace_evidence(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["final_answer"])])

    with pytest.raises(JudgeFailure, match="trace or tool evidence"):
        await judge_quality(valid_bundle, model)


@pytest.mark.asyncio
async def test_judge_requires_final_answer_or_artifact_evidence(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["tool_call[0]"])])

    with pytest.raises(JudgeFailure, match="final answer or artifact"):
        await judge_quality(valid_bundle, model)
