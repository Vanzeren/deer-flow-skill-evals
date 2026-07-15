"""Integration tests: verify that guard middlewares write ``stop_reason`` to
``runtime.context`` and the worker surfaces it on the run record (#4176).

The lead worker calls ``agent.astream()``. During streaming, guard
middlewares (loop detection, token budget) may detect a cap and write
``stop_reason`` into ``runtime.context``. After streaming completes, the
worker reads ``runtime.context["stop_reason"]`` and persists it.

The key invariant: the middleware's ``runtime.context`` IS the worker's
``runtime.context`` — LangGraph surfaces the same dict — so the worker
sees whatever the middleware wrote.

These tests exercise that invariant end-to-end, using real middleware
instances (not hand-written simulations of the write) driven inside
``astream`` to prove the full middleware→context→worker pipeline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from deerflow.runtime import RunContext, RunManager, RunStatus
from deerflow.runtime.runs.worker import run_agent


@pytest.mark.anyio
async def test_worker_surfaces_stop_reason_from_loop_detection():
    """The worker persists ``stop_reason=loop_capped`` when the real
    LoopDetectionMiddleware triggers a hard stop during streaming."""
    from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware

    run_manager = RunManager()
    record = await run_manager.create("thread-1")

    mw = LoopDetectionMiddleware(warn_threshold=1, hard_limit=3, window_size=5)

    class DummyAgent:
        metadata: dict[str, Any] = {"model_name": "test-model"}

        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            runtime = ((config or {}).get("configurable") or {}).get("__pregel_runtime")
            assert runtime is not None, "LangGraph Runtime must be in configurable"

            # Drive the real middleware to a hard stop with repeated identical
            # tool calls.  With hard_limit=3, the 3rd identical call fires the
            # hard stop, triggering the runtime.context write.
            tool_calls = [{"name": "bash", "args": {"command": "ls"}, "id": "c1", "type": "tool_call"}]
            for _ in range(2):
                mw._apply({"messages": [AIMessage(content="", tool_calls=tool_calls)]}, runtime)
            # 3rd call — hard stop fires here.
            mw._apply({"messages": [AIMessage(content="", tool_calls=tool_calls)]}, runtime)

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

    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.success
    assert fetched.stop_reason == "loop_capped"


@pytest.mark.anyio
async def test_worker_surfaces_stop_reason_from_token_budget():
    """The worker persists ``stop_reason=token_capped`` when the real
    TokenBudgetMiddleware triggers a hard stop during streaming."""
    from deerflow.agents.middlewares.token_budget_middleware import TokenBudgetMiddleware
    from deerflow.config.token_budget_config import TokenBudgetConfig

    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    # Use a moderate budget with hard_stop_threshold=0.0 so even
    # modest usage triggers the hard stop immediately.
    config = TokenBudgetConfig(
        enabled=True,
        max_tokens=1000,
        hard_stop_threshold=0.0,
        warn_threshold=0.0,
    )
    mw = TokenBudgetMiddleware(config=config)

    class DummyAgent:
        metadata: dict[str, Any] = {"model_name": "test-model"}

        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            runtime = ((config or {}).get("configurable") or {}).get("__pregel_runtime")
            assert runtime is not None, "LangGraph Runtime must be in configurable"

            # Feed a single AIMessage with token usage that exceeds the tiny budget.
            msg = AIMessage(
                id="msg-budget",
                content="hello",
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )
            mw._apply({"messages": [msg]}, runtime)

            yield {"messages": [AIMessage(content="Budget exceeded, wrapping up.")]}

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

    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.success
    assert fetched.stop_reason == "token_capped"
