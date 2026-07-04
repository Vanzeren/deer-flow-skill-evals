import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from skill_eval.inspect_scorer import skill_assertion_scorer
from skill_eval.trace_schema import AgentToolCall, AgentTrace


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
