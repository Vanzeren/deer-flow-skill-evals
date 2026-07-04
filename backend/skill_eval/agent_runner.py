from typing import Any, Protocol

from pydantic import BaseModel, Field

from skill_eval.trace_schema import AgentTrace


class AgentRunRequest(BaseModel):
    case_id: str | None = None
    user_input: str
    target: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    candidate_skills: list[str] = Field(default_factory=list)
    forced_skills: list[str] | None = None
    sandbox: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResult(BaseModel):
    final_answer: str
    success: bool
    trace: AgentTrace


class AgentRunner(Protocol):
    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        raise NotImplementedError


async def run_agent(request: AgentRunRequest, runner: AgentRunner | None = None) -> AgentRunResult:
    if runner is None:
        from skill_eval.adapters.mock import MockAgentRunner

        runner = MockAgentRunner()

    return await runner.run(request)
