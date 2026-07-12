import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint, uuid6
from langgraph.checkpoint.memory import InMemorySaver

from deerflow.runtime.runs.manager import RunManager
from deerflow.runtime.runs.worker import RunContext, _persist_run_duration, run_agent


class _YieldingSaver(InMemorySaver):
    async def aget_tuple(self, config):
        checkpoint_tuple = await super().aget_tuple(config)
        await asyncio.sleep(0)
        return checkpoint_tuple

    async def aput(self, config, checkpoint, metadata, new_versions):
        await asyncio.sleep(0)
        return await super().aput(config, checkpoint, metadata, new_versions)


async def _put_checkpoint(
    checkpointer: InMemorySaver,
    *,
    thread_id: str,
    checkpoint_id: str,
    messages: list[object],
    step: int,
    parent_config: dict | None = None,
) -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = {"messages": messages}
    checkpoint["channel_versions"] = {"messages": step}
    config = parent_config or {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    return await checkpointer.aput(
        config,
        checkpoint,
        {"step": step, "source": "loop", "writes": {"test": {"messages": messages}}, "parents": {}},
        {"messages": step},
    )


@pytest.mark.anyio
async def test_run_duration_survives_a_later_checkpoint() -> None:
    checkpointer = InMemorySaver()
    thread_id = "duration-survives"
    messages = [
        HumanMessage(id="human-1", content="Question", additional_kwargs={"run_id": "run-1"}),
        AIMessage(id="ai-1", content="Answer"),
    ]
    await _put_checkpoint(
        checkpointer,
        thread_id=thread_id,
        checkpoint_id="00000000-0000-6000-8000-000000000001",
        messages=messages,
        step=1,
    )

    await _persist_run_duration(
        checkpointer=checkpointer,
        thread_id=thread_id,
        run_id="run-1",
        duration_seconds=7,
    )

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    duration_checkpoint = await checkpointer.aget_tuple(config)
    assert duration_checkpoint is not None
    persisted_messages = copy.deepcopy(duration_checkpoint.checkpoint["channel_values"]["messages"])
    assert persisted_messages[1].additional_kwargs["turn_duration"] == 7

    await _put_checkpoint(
        checkpointer,
        thread_id=thread_id,
        checkpoint_id=str(uuid6()),
        messages=persisted_messages,
        step=3,
        parent_config=duration_checkpoint.config,
    )

    latest = await checkpointer.aget_tuple(config)
    assert latest is not None
    assert latest.checkpoint["channel_values"]["messages"][1].additional_kwargs["turn_duration"] == 7


@pytest.mark.anyio
async def test_concurrent_run_duration_updates_preserve_both_turns() -> None:
    checkpointer = _YieldingSaver()
    thread_id = "duration-concurrent"
    messages = [
        HumanMessage(id="human-1", content="First", additional_kwargs={"run_id": "run-1"}),
        AIMessage(id="ai-1", content="First answer"),
        HumanMessage(id="human-2", content="Second", additional_kwargs={"run_id": "run-2"}),
        AIMessage(id="ai-2", content="Second answer"),
    ]
    await _put_checkpoint(
        checkpointer,
        thread_id=thread_id,
        checkpoint_id="00000000-0000-6000-8000-000000000001",
        messages=messages,
        step=1,
    )

    await asyncio.gather(
        _persist_run_duration(
            checkpointer=checkpointer,
            thread_id=thread_id,
            run_id="run-1",
            duration_seconds=3,
        ),
        _persist_run_duration(
            checkpointer=checkpointer,
            thread_id=thread_id,
            run_id="run-2",
            duration_seconds=5,
        ),
    )

    latest = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
    assert latest is not None
    persisted_messages = latest.checkpoint["channel_values"]["messages"]
    assert persisted_messages[1].additional_kwargs["turn_duration"] == 3
    assert persisted_messages[3].additional_kwargs["turn_duration"] == 5


@pytest.mark.anyio
async def test_run_duration_checkpoint_preserves_parent_lineage() -> None:
    checkpointer = InMemorySaver()
    thread_id = "duration-parent"
    parent_checkpoint_id = "00000000-0000-6000-8000-000000000001"
    await _put_checkpoint(
        checkpointer,
        thread_id=thread_id,
        checkpoint_id=parent_checkpoint_id,
        messages=[
            HumanMessage(id="human-1", content="Question", additional_kwargs={"run_id": "run-1"}),
            AIMessage(id="ai-1", content="Answer"),
        ],
        step=1,
    )

    await _persist_run_duration(
        checkpointer=checkpointer,
        thread_id=thread_id,
        run_id="run-1",
        duration_seconds=7,
    )

    history = [checkpoint async for checkpoint in checkpointer.alist({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})]
    assert len(history) == 2
    assert history[0].config["configurable"]["checkpoint_id"] != parent_checkpoint_id
    assert history[0].parent_config == {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": parent_checkpoint_id,
        }
    }


@pytest.mark.anyio
async def test_agent_stream_serializes_with_duration_checkpoint_write() -> None:
    checkpointer = _YieldingSaver()
    run_manager = RunManager()
    record = await run_manager.create("duration-stream-lock")
    await _put_checkpoint(
        checkpointer,
        thread_id=record.thread_id,
        checkpoint_id="00000000-0000-6000-8000-000000000001",
        messages=[
            HumanMessage(
                id="human-1",
                content="Question",
                additional_kwargs={"run_id": record.run_id},
            ),
            AIMessage(id="ai-1", content="Answer"),
        ],
        step=1,
    )
    bridge = SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )
    duration_task = None
    finished_during_stream = None

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            nonlocal duration_task, finished_during_stream
            duration_task = asyncio.create_task(
                _persist_run_duration(
                    checkpointer=checkpointer,
                    thread_id=record.thread_id,
                    run_id=record.run_id,
                    duration_seconds=9,
                )
            )
            try:
                await asyncio.wait_for(asyncio.shield(duration_task), timeout=0.05)
            except TimeoutError:
                finished_during_stream = False
            else:
                finished_during_stream = True
            yield {"messages": []}

    def factory(*, config):
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=checkpointer),
        agent_factory=factory,
        graph_input={},
        config={},
    )
    assert duration_task is not None
    await duration_task

    assert finished_during_stream is False
