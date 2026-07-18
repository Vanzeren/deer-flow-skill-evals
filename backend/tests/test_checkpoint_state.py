from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from deerflow.runtime import CheckpointStateAccessor
from deerflow.runtime.checkpoint_mode import CHECKPOINT_MODE_METADATA_KEY, INTERNAL_CHECKPOINT_MODE_KEY


class FakeCheckpointer:
    def __init__(self) -> None:
        self.sync_configs: list[dict[str, Any]] = []
        self.async_configs: list[dict[str, Any]] = []

    def get_tuple(self, config: dict[str, Any]) -> None:
        self.sync_configs.append(config)
        return None

    async def aget_tuple(self, config: dict[str, Any]) -> None:
        self.async_configs.append(config)
        return None


class FakeGraph:
    def __init__(self) -> None:
        self.checkpointer: Any = None
        self.store: Any = None
        self.calls: list[tuple[Any, ...]] = []
        self.sync_history_yields = 0
        self.async_history_yields = 0

    def get_state(self, config: dict[str, Any]) -> SimpleNamespace:
        self.calls.append(("get", config))
        return SimpleNamespace(values={"messages": ["sync"]})

    def get_state_history(self, config: dict[str, Any]):
        self.calls.append(("history", config))
        for index in range(4):
            self.sync_history_yields += 1
            yield SimpleNamespace(values={"index": index})

    def update_state(self, config: dict[str, Any], values: dict[str, Any], *, as_node: str | None = None) -> dict[str, Any]:
        self.calls.append(("update", config, values, as_node))
        return {"updated": values, "as_node": as_node}

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        self.calls.append(("aget", config))
        return SimpleNamespace(values={"messages": ["async"]})

    async def aget_state_history(self, config: dict[str, Any]):
        self.calls.append(("ahistory", config))
        for index in range(4):
            self.async_history_yields += 1
            yield SimpleNamespace(values={"index": index})

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("aupdate", config, values, as_node))
        return {"updated": values, "as_node": as_node}


def _assert_delta_config_is_copied(original: dict[str, Any], forwarded: dict[str, Any]) -> None:
    assert forwarded is not original
    assert forwarded["configurable"] is not original["configurable"]
    assert forwarded["metadata"] is not original["metadata"]
    assert forwarded["configurable"][INTERNAL_CHECKPOINT_MODE_KEY] == "delta"
    assert forwarded["metadata"][CHECKPOINT_MODE_METADATA_KEY] == "delta"


def test_sync_accessor_binds_persistence_guards_operations_and_preserves_input() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    store = object()
    accessor = CheckpointStateAccessor.bind(graph, saver, store=store, mode="delta")
    config = {
        "configurable": {"thread_id": "thread-sync", "checkpoint_ns": ""},
        "metadata": {"caller": "test"},
        "tags": ["preserved"],
    }
    original = deepcopy(config)

    snapshot = accessor.get(config)
    history = accessor.history(config, limit=2)
    update = accessor.update(config, {"messages": ["changed"]}, as_node="tools")

    assert snapshot.values == {"messages": ["sync"]}
    assert [item.values for item in history] == [{"index": 0}, {"index": 1}]
    assert graph.sync_history_yields == 2
    assert update == {"updated": {"messages": ["changed"]}, "as_node": "tools"}
    assert graph.checkpointer is saver
    assert graph.store is store
    assert config == original
    for call in graph.calls:
        _assert_delta_config_is_copied(config, call[1])
    assert graph.calls[-1][2:] == ({"messages": ["changed"]}, "tools")


@pytest.mark.anyio
async def test_async_accessor_binds_persistence_guards_operations_and_preserves_input() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    store = object()
    accessor = CheckpointStateAccessor.bind(graph, saver, store=store, mode="delta")
    config = {
        "configurable": {"thread_id": "thread-async", "checkpoint_ns": ""},
        "metadata": {"caller": "test"},
        "tags": ["preserved"],
    }
    original = deepcopy(config)

    snapshot = await accessor.aget(config)
    history = await accessor.ahistory(config, limit=2)
    update = await accessor.aupdate(config, {"messages": ["changed"]}, as_node="agent")

    assert snapshot.values == {"messages": ["async"]}
    assert [item.values for item in history] == [{"index": 0}, {"index": 1}]
    assert graph.async_history_yields == 2
    assert update == {"updated": {"messages": ["changed"]}, "as_node": "agent"}
    assert graph.checkpointer is saver
    assert graph.store is store
    assert config == original
    for call in graph.calls:
        _assert_delta_config_is_copied(config, call[1])
    assert graph.calls[-1][2:] == ({"messages": ["changed"]}, "agent")


def test_sync_history_zero_limit_guards_without_consuming_a_snapshot() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-sync-zero"}}

    assert accessor.history(config, limit=0) == []
    assert len(saver.sync_configs) == 1
    assert graph.sync_history_yields == 0


@pytest.mark.anyio
async def test_async_history_zero_limit_guards_without_consuming_a_snapshot() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-async-zero"}}

    assert await accessor.ahistory(config, limit=0) == []
    assert len(saver.async_configs) == 1
    assert graph.async_history_yields == 0


@pytest.mark.anyio
async def test_full_accessor_checks_compatibility_before_all_sync_and_async_operations() -> None:
    graph = FakeGraph()
    saver = FakeCheckpointer()
    accessor = CheckpointStateAccessor.bind(graph, saver)
    config = {"configurable": {"thread_id": "thread-full"}}

    accessor.get(config)
    accessor.history(config, limit=1)
    accessor.update(config, {}, as_node=None)
    await accessor.aget(config)
    await accessor.ahistory(config, limit=1)
    await accessor.aupdate(config, {}, as_node=None)

    assert len(saver.sync_configs) == 3
    assert len(saver.async_configs) == 3
    for prepared in [*saver.sync_configs, *saver.async_configs]:
        assert prepared["configurable"][INTERNAL_CHECKPOINT_MODE_KEY] == "full"
        assert CHECKPOINT_MODE_METADATA_KEY not in prepared["metadata"]
    assert config == {"configurable": {"thread_id": "thread-full"}}
