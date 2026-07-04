import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.solver import TaskState

from evals.skills_eval import skills_eval

from skill_eval.adapters.mock import MockAgentRunner
from skill_eval.agent_runner import AgentRunRequest, run_agent
from skill_eval.inspect_solver import skill_agent_solver

@pytest.mark.asyncio
async def test_mock_runner_returns_agent_trace_for_cloud_run():
    runner = MockAgentRunner()
    request = AgentRunRequest(user_input="How do I deploy a Cloud Run service?", candidate_skills=["skills/gcp-cloud-run"])

    result = await runner.run(request)

    assert result.success is True
    assert "gcloud run deploy" in result.final_answer
    assert result.trace.runtime == "mock"
    assert result.trace.skill_invocations[0].loaded is True
    assert result.trace.messages


@pytest.mark.asyncio
async def test_mock_runner_records_write_todos_when_not_forbidden():
    request = AgentRunRequest(user_input="Please write todos for this task", forced_skills=[])
    result = await run_agent(request, runner=MockAgentRunner())

    assert [call.name for call in result.trace.tool_calls] == ["write_todos"]


@pytest.mark.asyncio
async def test_mock_runner_avoids_write_todos_when_user_forbids_it():
    request = AgentRunRequest(user_input="Create a plan, but do not write todos.", candidate_skills=["skills/no-write-todos-in-pro"])
    result = await run_agent(request, runner=MockAgentRunner())

    assert result.trace.tool_calls == []
    assert result.trace.skill_invocations[0].used is True
    assert result.trace.skill_invocations[0].applied is None


@pytest.mark.asyncio
async def test_skill_agent_solver_writes_completion_and_trace_metadata():
    state = TaskState(
        model="mock-model",
        sample_id="cloud-run-001",
        epoch=1,
        input="How do I deploy a Cloud Run service?",
        target="The answer should mention gcloud run deploy.",
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=""),
        metadata={"case": {"id": "cloud-run-001", "input": "How do I deploy a Cloud Run service?", "candidate_skills": ["skills/gcp-cloud-run"]}},
    )

    async def unused_generate(inner_state):
        return inner_state

    solve = skill_agent_solver(agent_runner=MockAgentRunner(), skills=None, sandbox="docker")
    result_state = await solve(state, unused_generate)

    assert "gcloud run deploy" in result_state.output.completion
    assert result_state.metadata["success"] is True
    assert result_state.metadata["agent_trace"]["runtime"] == "mock"
    assert result_state.metadata["agent_trace"]["skill_invocations"][0]["name"] == "skills/gcp-cloud-run"


def test_demo_inspect_task_constructs_for_each_mode():
    assert skills_eval(case_file="cases/gcp_skills.jsonl", mode="baseline").dataset
    assert skills_eval(case_file="cases/gcp_skills.jsonl", mode="with_skill").dataset
    assert skills_eval(case_file="cases/gcp_skills.jsonl", mode="all_skills").dataset
