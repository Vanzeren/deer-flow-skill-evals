from typing import Literal

from pydantic import BaseModel, Field


AssertionName = Literal[
    "skill_loaded",
    "skill_used",
    "skill_not_used",
    "skill_applied",
    "skill_not_applied",
    "tool_called",
    "tool_not_called",
    "tool_args_contains",
    "tool_args_match",
    "tool_call_order",
    "tool_error_absent",
    "tool_result_contains",
    "tool_result_match",
    "output_contains",
    "output_not_contains",
    "regex_match",
    "json_valid",
    "success_is_true",
    "trace_complete",
    "latency_under",
    "tokens_under",
    "tool_count_under",
    "max_steps_under",
    "no_unexpected_clarification",
]


class SkillAssertionSpec(BaseModel):
    name: AssertionName
    target: str | None = None
    threshold: int | float | None = None
    message: str | None = None


class SkillEvalCase(BaseModel):
    id: str
    input: str
    target: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    candidate_skills: list[str] = Field(default_factory=list)
    assertions: list[SkillAssertionSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["smoke", "normal", "hard"] = "normal"
