from inspect_ai.solver import Generate, TaskState, solver

from skill_eval.agent_runner import AgentRunner, AgentRunRequest, RunMode
from skill_eval.case_schema import RoutingCase


@solver
def deerflow_solver(
    runner: AgentRunner,
    *,
    mode: RunMode,
    model_name: str,
    timeout_seconds: int,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case = RoutingCase.model_validate(state.metadata.get("case", {}))
        request = AgentRunRequest(
            case_id=case.id,
            user_input=state.input_text,
            mode=mode,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
        )
        result = await runner.run(request)
        state.output.completion = result.final_answer
        state.metadata["agent_trace"] = result.trace.model_dump()
        state.metadata["route_observation"] = result.route_observation.model_dump()
        state.metadata["agent_success"] = result.success
        state.metadata["thread_id"] = result.thread_id
        return state

    return solve
