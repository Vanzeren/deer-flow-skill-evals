"""Materialized checkpoint-state access and state-only mutation graphs.

:class:`CheckpointStateAccessor` is the single choke point for thread
checkpoint-state reads and writes. It binds a compiled graph (which carries
the mode-matched channel schema), a checkpointer, and the frozen channel mode:
every operation injects the mode marker into the config and passes the
compatibility gate before touching state. Delta checkpoints store no full
``channel_values`` — raw saver reads see sentinels — so consumers must go
through this accessor instead of calling the checkpointer directly.

:func:`build_state_mutation_graph` compiles a state-only graph (one no-op
node, entry = finish) for wholesale state replacement such as rollback
restore and context compaction: it shares the agent graph's checkpoint
machinery but schedules no pending nodes, so the written head stays idle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deerflow.agents.thread_state import get_thread_state_schema
from deerflow.config.database_config import CheckpointChannelMode
from deerflow.runtime.checkpoint_mode import (
    aensure_checkpoint_mode_compatible,
    ensure_checkpoint_mode_compatible,
    inject_checkpoint_mode,
)


def _finish_state_mutation(_state: dict[str, Any]) -> dict[str, Any]:
    return {}


def build_state_mutation_graph(as_node: str, mode: CheckpointChannelMode) -> Any:
    """Compile a state-only graph whose single writer node finishes immediately.

    ``update_state(..., as_node=...)`` requires the node to be registered in
    the graph; a dedicated single-node graph applies reducer writes and
    finishes, so the mutation checkpoint schedules no agent nodes and has no
    pending ``next`` nodes.
    """
    if not as_node:
        raise ValueError("as_node is required for checkpoint state mutation")
    from langgraph.graph import StateGraph

    builder = StateGraph(get_thread_state_schema(mode))
    builder.add_node(as_node, _finish_state_mutation)
    builder.set_entry_point(as_node)
    builder.set_finish_point(as_node)
    return builder.compile()


@dataclass
class CheckpointStateAccessor:
    graph: Any
    checkpointer: Any
    mode: CheckpointChannelMode

    @classmethod
    def bind(
        cls,
        graph: Any,
        checkpointer: Any,
        *,
        store: Any | None = None,
        mode: CheckpointChannelMode = "full",
    ) -> CheckpointStateAccessor:
        graph.checkpointer = checkpointer
        if store is not None:
            graph.store = store
        return cls(graph=graph, checkpointer=checkpointer, mode=mode)

    def _prepare_config(self, config: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            **config,
            "configurable": dict(config.get("configurable", {})),
            "metadata": dict(config.get("metadata", {})),
        }
        inject_checkpoint_mode(prepared, self.mode)
        return prepared

    def get(self, config: dict[str, Any]) -> Any:
        prepared = self._prepare_config(config)
        ensure_checkpoint_mode_compatible(self.checkpointer, prepared, self.mode)
        return self.graph.get_state(prepared)

    async def aget(self, config: dict[str, Any]) -> Any:
        prepared = self._prepare_config(config)
        await aensure_checkpoint_mode_compatible(self.checkpointer, prepared, self.mode)
        return await self.graph.aget_state(prepared)

    def history(self, config: dict[str, Any], *, limit: int | None = None) -> list[Any]:
        prepared = self._prepare_config(config)
        ensure_checkpoint_mode_compatible(self.checkpointer, prepared, self.mode)
        if limit is not None and limit <= 0:
            return []
        result = []
        for snapshot in self.graph.get_state_history(prepared):
            result.append(snapshot)
            if limit is not None and len(result) >= limit:
                break
        return result

    async def ahistory(self, config: dict[str, Any], *, limit: int | None = None) -> list[Any]:
        prepared = self._prepare_config(config)
        await aensure_checkpoint_mode_compatible(self.checkpointer, prepared, self.mode)
        if limit is not None and limit <= 0:
            return []
        result = []
        async for snapshot in self.graph.aget_state_history(prepared):
            result.append(snapshot)
            if limit is not None and len(result) >= limit:
                break
        return result

    def update(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_config(config)
        ensure_checkpoint_mode_compatible(self.checkpointer, prepared, self.mode)
        return self.graph.update_state(prepared, values, as_node=as_node)

    async def aupdate(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_config(config)
        await aensure_checkpoint_mode_compatible(self.checkpointer, prepared, self.mode)
        return await self.graph.aupdate_state(prepared, values, as_node=as_node)
