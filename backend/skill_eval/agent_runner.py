from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from skill_eval.case_schema import CANDIDATE_SKILLS
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace

type RunMode = Literal["routing_probe", "full"]
type SandboxMode = Literal["configured", "local"]


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    user_input: str
    mode: RunMode
    model_name: str
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_skills: tuple[str, ...] = CANDIDATE_SKILLS
    timeout_seconds: int = 300
    trace_dir: str | None = None
    sandbox: SandboxMode = "configured"


class AgentRunResult(BaseModel):
    final_answer: str
    success: bool
    trace: AgentTrace
    route_observation: RouteObservation
    thread_id: str


class AgentRunner(Protocol):
    async def run(self, request: AgentRunRequest) -> AgentRunResult: ...
