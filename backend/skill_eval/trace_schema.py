from typing import Any

from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None


class SkillInvocation(BaseModel):
    name: str
    path: str | None = None
    loaded: bool = False
    used: bool = False
    applied: bool | None = None
    trigger_reason: str | None = None
    evidence: list[str] = Field(default_factory=list)


class AgentTrace(BaseModel):
    input: str
    final_answer: str
    success: bool
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    skill_invocations: list[SkillInvocation] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    runtime: str | None = None
    raw_trace_ref: str | None = None
