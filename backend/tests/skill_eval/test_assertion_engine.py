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


def test_skill_assertion_accepts_only_resolved_assertion_names():
    allowed_names = {
        "tool_called",
        "tool_not_called",
        "tool_args_contains",
        "tool_args_match",
        "tool_call_order",
        "tool_error_absent",
        "tool_result_contains",
        "tool_result_match",
        "tool_count_under",
        "output_contains",
        "output_not_contains",
        "regex_match",
        "json_valid",
        "success_is_true",
        "trace_complete",
        "skill_loaded",
        "skill_used",
        "skill_not_used",
        "skill_applied",
        "skill_not_applied",
    }

    assert set(get_args(AssertionName)) == allowed_names
    for assertion_name in allowed_names:
        assert SkillAssertionSpec(name=assertion_name).name == assertion_name

    with pytest.raises(ValidationError):
        SkillAssertionSpec(name="unknown_assertion")


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


def test_registered_assertions_match_supported_names():
    assert set(ASSERTION_REGISTRY) == {
        "tool_called",
        "tool_not_called",
        "tool_args_contains",
        "tool_args_match",
        "tool_call_order",
        "tool_error_absent",
        "tool_result_contains",
        "tool_result_match",
        "tool_count_under",
        "output_contains",
        "output_not_contains",
        "regex_match",
        "json_valid",
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


def test_output_not_contains_passes_and_fails():
    trace = _valid_trace(final_answer="Safe deployment complete")

    passing = evaluate_assertion(
        SkillAssertionSpec(name="output_not_contains", target="password"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="output_not_contains", target="deployment"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert failing.passed is False
    assert failing.metadata["target"] == "deployment"


def test_regex_match_passes_and_fails():
    trace = _valid_trace(final_answer="Run id: abc-123")

    passing = evaluate_assertion(
        SkillAssertionSpec(name="regex_match", target=r"[a-z]+-\d+"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="regex_match", target=r"^done$"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["pattern"] == r"[a-z]+-\d+"
    assert failing.passed is False


def test_regex_match_fails_invalid_pattern():
    trace = _valid_trace(final_answer="answer")

    result = evaluate_assertion(
        SkillAssertionSpec(name="regex_match", target="["),
        trace,
        trace.final_answer,
    )

    assert result.passed is False
    assert "Invalid regex" in result.explanation
    assert "error" in result.metadata


def test_json_valid_passes_and_fails():
    valid_trace = _valid_trace(final_answer='{"status":"ok"}')
    invalid_trace = _valid_trace(final_answer="not json")

    passing = evaluate_assertion(
        SkillAssertionSpec(name="json_valid"),
        valid_trace,
        valid_trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="json_valid"),
        invalid_trace,
        invalid_trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["json_type"] == "dict"
    assert failing.passed is False
    assert "error" in failing.metadata


def test_tool_args_contains_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", args={"cmd": "gcloud run deploy app"})])

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_args_contains", target="gcloud run deploy"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_args_contains", target="kubectl apply"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["matched_tool_call"]["name"] == "bash"
    assert failing.passed is False


def test_tool_args_match_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", args={"cmd": "gcloud run deploy app"})])

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_args_match", target=r"gcloud\s+run\s+deploy"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_args_match", target=r"kubectl\s+apply"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["pattern"] == r"gcloud\s+run\s+deploy"
    assert failing.passed is False


def test_tool_call_order_passes_and_fails():
    trace = _valid_trace(
        tool_calls=[
            AgentToolCall(name="read_file"),
            AgentToolCall(name="bash"),
            AgentToolCall(name="present_files"),
        ]
    )

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_call_order", target="read_file,bash"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_call_order", target="bash,read_file"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["observed_order"] == ["read_file", "bash", "present_files"]
    assert failing.passed is False


def test_tool_error_absent_passes_and_fails():
    clean_trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", error=None)])
    failing_trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", error="permission denied")])

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_error_absent", target="bash"),
        clean_trace,
        clean_trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_error_absent", target="bash"),
        failing_trace,
        failing_trace.final_answer,
    )

    assert passing.passed is True
    assert failing.passed is False
    assert failing.metadata["errored_tool_calls"][0]["error"] == "permission denied"


def test_tool_result_contains_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", result="Deployment successful")])

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_result_contains", target="successful"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_result_contains", target="failed"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["matched_tool_call"]["name"] == "bash"
    assert failing.passed is False


def test_tool_result_match_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash", result="revision rev-42 ready")])

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_result_match", target=r"rev-\d+"),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_result_match", target=r"error:\s+"),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["match"] == "rev-42"
    assert failing.passed is False


def test_tool_count_under_passes_and_fails():
    trace = _valid_trace(tool_calls=[AgentToolCall(name="bash"), AgentToolCall(name="read_file")])

    passing = evaluate_assertion(
        SkillAssertionSpec(name="tool_count_under", threshold=3),
        trace,
        trace.final_answer,
    )
    failing = evaluate_assertion(
        SkillAssertionSpec(name="tool_count_under", threshold=2),
        trace,
        trace.final_answer,
    )

    assert passing.passed is True
    assert passing.metadata["observed"] == 2
    assert failing.passed is False
    assert failing.metadata["threshold"] == 2
