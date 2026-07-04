import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from skill_eval.inspect_scorer import trace_integrity_scorer
from skill_eval.trace_schema import AgentTrace


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


@pytest.mark.asyncio
async def test_trace_integrity_scorer_fails_when_trace_missing():
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({}), Target(""))

    assert score.value == 0.0
    assert "Missing agent_trace" in score.explanation


@pytest.mark.asyncio
async def test_trace_integrity_scorer_fails_when_trace_invalid():
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({"agent_trace": {"input": "x"}}), Target(""))

    assert score.value == 0.0
    assert "Invalid AgentTrace" in score.explanation


@pytest.mark.asyncio
async def test_trace_integrity_scorer_passes_for_valid_trace():
    trace = AgentTrace(
        input="input",
        final_answer="answer",
        success=True,
        messages=[{"role": "assistant", "content": "answer"}],
    )
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({"agent_trace": trace.model_dump()}), Target(""))

    assert score.value == 1.0
    assert score.explanation == "Trace is complete."


@pytest.mark.asyncio
async def test_trace_integrity_scorer_fails_on_fatal_error():
    trace = AgentTrace(
        input="input",
        final_answer="answer",
        success=True,
        messages=[{"role": "assistant", "content": "answer"}],
        errors=["fatal: crashed"],
    )
    score_fn = trace_integrity_scorer()

    score = await score_fn(_state({"agent_trace": trace.model_dump()}), Target(""))

    assert score.value == 0.0
    assert "fatal" in score.explanation
