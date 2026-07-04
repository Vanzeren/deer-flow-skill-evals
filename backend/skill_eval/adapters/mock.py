from skill_eval.agent_runner import AgentRunRequest, AgentRunResult
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


class MockAgentRunner:
    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        selected_skills = request.forced_skills if request.forced_skills is not None else request.candidate_skills
        tool_calls: list[AgentToolCall] = []
        final_answer = "Mock answer."
        lowered = request.user_input.lower()

        if "cloud run" in lowered:
            final_answer = "Use gcloud run deploy to deploy a Cloud Run service."

        if "write todos" in lowered and "do not" not in lowered:
            tool_calls.append(AgentToolCall(name="write_todos", args={"items": ["mock plan"]}))

        trace = AgentTrace(
            input=request.user_input,
            final_answer=final_answer,
            success=True,
            tool_calls=tool_calls,
            skill_invocations=[
                SkillInvocation(
                    name=skill,
                    path=skill,
                    loaded=True,
                    used=True,
                    applied=None,
                    trigger_reason="mock runner loaded candidate skill",
                    evidence=["mock runner selected candidate skill"],
                )
                for skill in selected_skills
            ],
            messages=[
                {"role": "user", "content": request.user_input},
                {"role": "assistant", "content": final_answer},
            ],
            steps=[{"type": "mock_start"}, {"type": "mock_final_answer"}],
            runtime="mock",
        )

        return AgentRunResult(final_answer=final_answer, success=True, trace=trace)
