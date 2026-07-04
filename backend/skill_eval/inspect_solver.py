from inspect_ai.solver import Generate, TaskState, solver

from skill_eval.agent_runner import AgentRunner, AgentRunRequest, run_agent
from skill_eval.case_schema import SkillEvalCase


@solver
def skill_agent_solver(agent_runner: AgentRunner | None = None, skills: list[str] | None = None, sandbox: str | None = "docker"):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case = SkillEvalCase.model_validate(state.metadata.get("case", {}))
        request = AgentRunRequest(
            case_id=case.id,
            user_input=state.input_text,
            target=case.target,
            required_skills=case.required_skills,
            candidate_skills=case.candidate_skills,
            forced_skills=skills,
            sandbox=sandbox,
            metadata={"inspect_sample_id": state.sample_id},
        )

        result = await run_agent(request, runner=agent_runner)

        state.output.completion = result.final_answer
        state.metadata["agent_trace"] = result.trace.model_dump()
        state.metadata["success"] = result.success

        return state

    return solve
