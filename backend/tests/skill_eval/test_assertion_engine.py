from typing import get_args

import pytest
from pydantic import ValidationError

from skill_eval.assertion_engine import ASSERTION_REGISTRY, evaluate_assertion
from skill_eval.case_schema import AssertionName, SkillAssertionSpec, SkillEvalCase
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


def test_skill_eval_case_defaults():
    case = SkillEvalCase(id="case-1", input="Do the task")

    assert case.target is None
    assert case.required_skills == []
    assert case.candidate_skills == []
    assert case.assertions == []
    assert case.tags == []
    assert case.difficulty == "normal"


def test_skill_assertion_accepts_only_resolved_mvp_and_skill_names():
    allowed_names = {
        "tool_called",
        "tool_not_called",
        "output_contains",
        "success_is_true",
        "trace_complete",
        "skill_loaded",
        "skill_used",
        "skill_not_used",
        "skill_applied",
        "skill_not_applied",
    }

    assert set(get_args(AssertionName)) == allowed_names
    assert SkillAssertionSpec(name="skill_used").name == "skill_used"

    with pytest.raises(ValidationError):
        SkillAssertionSpec(name="regex_match")


def test_agent_trace_captures_normalized_evidence_and_raw_ref():
    trace = AgentTrace(
        input="Use the skill",
        final_answer="Done",
        success=True,
        tool_calls=[AgentToolCall(name="bash", args={"cmd": "pwd"})],
        skill_invocations=[SkillInvocation(name="demo", loaded=True, used=True, applied=None, evidence=["read_file loaded SKILL.md"])],
        messages=[{"role": "assistant", "content": "Done"}],
        steps=[{"type": "final"}],
        runtime="mock",
        raw_trace_ref="artifact://trace",
    )

    assert trace.runtime == "mock"
    assert trace.raw_trace_ref == "artifact://trace"
    assert trace.tool_calls[0].name == "bash"
    assert trace.skill_invocations[0].loaded is True
    assert trace.skill_invocations[0].used is True
    assert trace.skill_invocations[0].applied is None
    assert trace.skill_invocations[0].evidence == ["read_file loaded SKILL.md"]


def _valid_trace(**overrides):
    data = {
        "input": "Do it",
        "final_answer": "Done with gcloud run deploy",
        "success": True,
        "messages": [{"role": "assistant", "content": "Done"}],
    }
    data.update(overrides)
    return AgentTrace(**data)


def test_mvp_assertions_are_registered():
    assert set(ASSERTION_REGISTRY) == {
        "tool_called",
        "tool_not_called",
        "output_contains",
        "success_is_true",
        "trace_complete",
        "skill_loaded",
        "skill_used",
        "skill_not_used",
        "skill_applied",
        "skill_not_applied",
    }


def test_tool_called_passes_when_expected_tool_appears():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_called", target="bash"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True
    assert "was called" in result.explanation


def test_tool_called_fails_when_expected_tool_is_absent():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="read_file")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_called", target="bash"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False
    assert "Expected tool `bash`" in result.explanation


def test_tool_not_called_passes_when_forbidden_tool_is_absent():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="read_file")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_not_called", target="write_todos"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_tool_not_called_fails_when_forbidden_tool_is_present():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="write_todos")])

    result = evaluate_assertion(
        SkillAssertionSpec(name="tool_not_called", target="write_todos"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False
    assert "Forbidden tool `write_todos` was called" in result.explanation


def test_output_contains_passes_and_fails():
    trace = _valid_trace(final_answer="Use gcloud run deploy")

    passing = evaluate_assertion(
        SkillAssertionSpec(name="output_contains", target="gcloud run deploy"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="output_contains", target="kubectl apply"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert failing.passed is False


def test_success_is_true_passes_and_fails():
    passing_trace = _valid_trace(success=True)
    failing_trace = _valid_trace(success=False)

    assert (
        evaluate_assertion(
            SkillAssertionSpec(name="success_is_true"),
            passing_trace,
            passing_trace.final_answer,
        ).passed
        is True
    )
    assert (
        evaluate_assertion(
            SkillAssertionSpec(name="success_is_true"),
            failing_trace,
            failing_trace.final_answer,
        ).passed
        is False
    )


def test_trace_complete_passes_for_valid_trace():
    trace = _valid_trace()

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is True


def test_trace_complete_fails_for_empty_input():
    trace = _valid_trace(input="")

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "trace.input is empty" in result.explanation


def test_trace_complete_fails_for_empty_final_answer():
    trace = _valid_trace(final_answer="")

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "trace.final_answer is empty" in result.explanation


def test_trace_complete_fails_without_evidence():
    trace = _valid_trace(messages=[], tool_calls=[], steps=[])

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "trace has no messages, tool_calls, or steps" in result.explanation


def test_trace_complete_fails_on_fatal_error():
    trace = _valid_trace(errors=["fatal: run crashed"])

    result = evaluate_assertion(SkillAssertionSpec(name="trace_complete"), trace, trace.final_answer)

    assert result.passed is False
    assert "fatal" in result.explanation


def test_skill_loaded_passes_when_target_invocation_was_loaded():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", loaded=True)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_loaded", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_skill_loaded_fails_when_target_invocation_was_not_loaded():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", loaded=False)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_loaded", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


def test_skill_used_passes_when_target_invocation_was_used():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", used=True)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_used", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_skill_used_fails_when_target_invocation_was_not_used():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", used=False)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_used", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


def test_skill_not_used_passes_when_target_invocation_was_not_used():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", used=False)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_used", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_skill_not_used_passes_when_target_invocation_is_absent():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="other", used=True)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_used", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_skill_not_used_fails_when_target_invocation_was_used():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", used=True)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_used", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


@pytest.mark.parametrize("applied", [False, None])
def test_skill_applied_fails_when_target_invocation_was_not_applied(applied):
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", applied=applied)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


def test_skill_applied_fails_when_target_invocation_is_absent():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="other", applied=True)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


def test_skill_applied_passes_when_target_invocation_was_applied():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", applied=True)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


def test_skill_not_applied_passes_when_target_invocation_was_not_applied():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", applied=False)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is True


@pytest.mark.parametrize("applied", [True, None])
def test_skill_not_applied_fails_when_target_invocation_is_not_explicitly_false(applied):
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="inspect", applied=applied)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


def test_skill_not_applied_fails_when_any_target_invocation_was_applied():
    trace = _valid_trace(
        skill_invocations=[
            SkillInvocation(name="inspect", applied=False),
            SkillInvocation(name="inspect", applied=True),
        ]
    )

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False


def test_skill_not_applied_fails_when_target_invocation_is_absent():
    trace = _valid_trace(skill_invocations=[SkillInvocation(name="other", applied=False)])

    result = evaluate_assertion(
        SkillAssertionSpec(name="skill_not_applied", target="inspect"),
        trace,
        trace.final_answer,
    )

    assert result.passed is False
