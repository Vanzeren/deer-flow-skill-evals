import inspect

from skill_eval.adapters.deerflow import DeerFlowAgentRunner


def test_runner_implements_agent_runner_protocol():
    """DeerFlowAgentRunner exposes an async run method per AgentRunner protocol."""
    runner = DeerFlowAgentRunner()
    assert hasattr(runner, "run")
    assert inspect.iscoroutinefunction(runner.run)
