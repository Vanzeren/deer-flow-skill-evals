from typing import Any

from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    id: str
    message_id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None


class AgentArtifact(BaseModel):
    path: str
    mime_type: str
    content: str
    original_bytes: int
    sha256: str
    truncated: bool


class QuickTurnCapture(BaseModel):
    message_id: str
    skill: str
    content: str


class AgentTrace(BaseModel):
    input: str
    final_answer: str
    success: bool
    thread_id: str
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    tool_call_chain: list[list[str]] = Field(default_factory=list)
    quick_turn: QuickTurnCapture | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[AgentArtifact] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    runtime: str = "deerflow"
    raw_trace_ref: str | None = None
