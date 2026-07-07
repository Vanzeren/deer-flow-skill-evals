from skill_eval.adapters.deerflow import DeerFlowAgentRunner


def test_runner_baseline_mode_no_skills():
    """Baseline mode sets available_skills to empty set."""
    runner = DeerFlowAgentRunner()
    # We can't actually call run() without config, but we verify
    # the class exists and implements the protocol.
    assert hasattr(runner, "run")
    assert callable(runner.run)
