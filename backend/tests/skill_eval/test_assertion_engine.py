from typing import get_args

import pytest
from pydantic import ValidationError

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
