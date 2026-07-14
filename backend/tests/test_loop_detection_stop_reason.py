"""Smoke test: verify that the worker reads ``stop_reason`` from
``runtime.context`` when a guard middleware writes it there during the
run (#4176)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from deerflow.runtime import RunContext, RunManager, RunStatus
from deerflow.runtime.runs.worker import run_agent


@pytest.mark.anyio
async def test_worker_surfaces_stop_reason_from_runtime_context():
    """When a middleware writes ``stop_reason`` to ``runtime.context``,
    the worker picks it up from the live runtime context and surfaces it
    on the run record alongside status=success."""
    run_manager = RunManager()
    record = await run_manager.create("thread-1")

    stop_reason_from_context: list[str | None] = [None]

    class DummyAgent:
        metadata: dict[str, Any] = {"model_name": "test-model"}

        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            # Simulate a guard middleware writing stop_reason to runtime.context.
            runtime = ((config or {}).get("configurable") or {}).get("__pregel_runtime")
            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict):
                ctx["stop_reason"] = "loop_capped"
                stop_reason_from_context[0] = "loop_capped"
            yield {"messages": [AIMessage(content="Done.")]}

    bridge = AsyncMock()
    bridge.publish = AsyncMock()
    bridge.publish_end = AsyncMock()
    bridge.cleanup = AsyncMock()

    def factory(*, config):
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=factory,
        graph_input={"messages": []},
        config={},
    )

    # Verify the middleware actually wrote to the context.
    assert stop_reason_from_context[0] == "loop_capped"

    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.success
    assert fetched.stop_reason == "loop_capped"
