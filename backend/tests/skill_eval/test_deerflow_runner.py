import inspect
import os

import pytest

from skill_eval.adapters.deerflow import DeerFlowAgentRunner


def test_runner_implements_agent_runner_protocol():
    """DeerFlowAgentRunner exposes an async run method per AgentRunner protocol."""
    runner = DeerFlowAgentRunner()
    assert hasattr(runner, "run")
    assert inspect.iscoroutinefunction(runner.run)


SMOKE_TEST_TIMEOUT = 120  # seconds


def _has_config():
    """Check if a valid config.yaml exists for real-agent tests."""
    for path in ["config.yaml", "config.yml", "configure.yml"]:
        if os.path.exists(path) or os.path.exists(os.path.join("..", path)):
            return True
    return False


@pytest.mark.skipif(not _has_config(), reason="No config.yaml found — real-agent smoke test skipped")
def test_runner_smoke_trivial_input():
    """Run a trivial input through the real DeerFlow agent."""
    import asyncio

    from skill_eval.agent_runner import AgentRunRequest

    runner = DeerFlowAgentRunner()
    request = AgentRunRequest(user_input="Say hello in exactly three words.")
    result = asyncio.run(asyncio.wait_for(runner.run(request), timeout=SMOKE_TEST_TIMEOUT))

    assert result.success is True
    assert len(result.final_answer) > 0
    assert result.trace.runtime == "deerflow"
    assert len(result.trace.messages) > 0
    ai_messages = [m for m in result.trace.messages if m.get("type") == "ai"]
    assert len(ai_messages) > 0


@pytest.mark.skipif(not _has_config(), reason="No config.yaml found — real-agent smoke test skipped")
def test_runner_smoke_tool_call():
    """Run an input that triggers a tool call."""
    import asyncio

    from skill_eval.agent_runner import AgentRunRequest

    runner = DeerFlowAgentRunner()
    request = AgentRunRequest(
        user_input="Read the file README.md and tell me what project this is.",
    )
    result = asyncio.run(asyncio.wait_for(runner.run(request), timeout=SMOKE_TEST_TIMEOUT))

    assert result.success is True
    read_calls = [tc for tc in result.trace.tool_calls if tc.name == "read_file"]
    assert len(read_calls) > 0, "Expected at least one read_file tool call"
