import pytest

from skill_eval.adapters.mock import MockAgentRunner
from skill_eval.agent_runner import AgentRunRequest, run_agent


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
