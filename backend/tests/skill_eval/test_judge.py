import json

import pytest
from inspect_ai.model import ModelOutput
from pydantic import ValidationError

from skill_eval.case_schema import RoutingCase
from skill_eval.judge import (
    JudgeFailure,
    QuickJudgment,
    bounded_evidence,
    build_judge_evidence,
    judge_quality,
    judge_quick_turn,
    load_candidate_skill_descriptions,
)
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentArtifact, AgentToolCall, AgentTrace, QuickTurnCapture


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
        "evidence": evidence or ["tool_chain[0]", "final_answer"],
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
        tool_call_chain=[["t1"]],
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
    assert "message[" not in payload
    assert "tool_call[" not in payload
    assert "tool_result[" not in payload
    assert "tool_chain[0]" in payload
    assert "artifact[report.md]" in payload
    assert "final_answer" in payload


def test_large_evidence_is_head_tail_truncated_with_hash():
    item = bounded_evidence(
        "tool_chain[0]",
        "tool_chain",
        "x" * 100_000,
        remaining_bytes=20_000,
    )

    assert item.truncated is True
    assert item.original_bytes == 100_000
    assert len(item.sha256) == 64
    assert "[truncated" in item.content


def test_exhausted_evidence_budget_retains_omission_marker():
    item = bounded_evidence(
        "tool_chain[0]",
        "tool_chain",
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
    model = FakeModel([valid_judgment_json(evidence=["tool_chain[999]", "final_answer"])])

    with pytest.raises(JudgeFailure, match="unknown evidence"):
        await judge_quality(valid_bundle, model)
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_judge_requires_process_evidence(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["final_answer"])])

    with pytest.raises(JudgeFailure, match="tool chain or error evidence"):
        await judge_quality(valid_bundle, model)


@pytest.mark.asyncio
async def test_judge_auto_adds_output_evidence_when_missing(valid_bundle):
    model = FakeModel([valid_judgment_json(evidence=["tool_chain[0]"])])
    judgment = await judge_quality(valid_bundle, model)
    output_kinds = {"final_answer", "artifact"}
    bundle_ids = {item.id for item in valid_bundle.evidence}
    cited_output = [e for e in judgment.evidence if e in bundle_ids and any(item.kind in output_kinds for item in valid_bundle.evidence if item.id == e)]
    assert cited_output, "judgment should have auto-added output evidence"


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
        quick_turn=QuickTurnCapture(
            message_id="m2",
            skill="systematic-literature-review",
            content="Plan: ...",
            tool_calls=[
                AgentToolCall(
                    id="t2",
                    message_id="m2",
                    name="bash",
                    args={"cmd": "ls"},
                    result="x",
                )
            ],
        ),
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
    captured_output = json.loads(quick_item.content)
    assert captured_output["content"] == "Plan: ..."
    assert captured_output["tool_calls"][0]["name"] == "bash"
    assert captured_output["tool_calls"][0]["result"] == "x"


def test_quick_target_accepts_tool_call_only_output(routing_case, route_observation):
    quick_call = AgentToolCall(
        id="t2",
        message_id="m2",
        name="web_search",
        args={"query": "attention variants"},
        result="results",
    )
    trace = AgentTrace(
        input="Synthesize three papers.",
        final_answer="",
        success=True,
        thread_id="thread-1",
        tool_calls=[
            AgentToolCall(id="t1", message_id="m1", name="read_file", args={"path": "SKILL.md"}, result="skill"),
            quick_call,
        ],
        tool_call_chain=[["t1"], ["t2"]],
        quick_turn=QuickTurnCapture(
            message_id="m2",
            skill="systematic-literature-review",
            content="",
            tool_calls=[quick_call],
        ),
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

    quick_output = json.loads(bundle.evidence[-1].content)
    assert quick_output["content"] == ""
    assert quick_output["tool_calls"] == [quick_call.model_dump()]


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
    assert "first assistant output turn" in model.prompts[0]
    assert "text, tool calls, or both" in model.prompts[0]
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
