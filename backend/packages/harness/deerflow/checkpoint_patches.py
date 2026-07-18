"""Compatibility patches for third-party checkpoint savers.

Lives at the top-level package (not ``deerflow.runtime``) so it can be
imported from ``deerflow.agents.thread_state`` without pulling in the heavy
``deerflow.runtime`` package __init__ (which eagerly imports the runs
machinery). Anchored from ``deerflow.agents.thread_state`` so every process
that builds a DeerFlow graph (gateway, workers, in-process LangGraph
runtime, tests) runs with the fixes in place.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

_PATCH_FLAG = "_deerflow_delta_history_patched"


def _get_delta_channel_history_via_base(self: Any, *, config: Any, channels: Any) -> Any:
    return BaseCheckpointSaver.get_delta_channel_history(self, config=config, channels=channels)


async def _aget_delta_channel_history_via_base(self: Any, *, config: Any, channels: Any) -> Any:
    return await BaseCheckpointSaver.aget_delta_channel_history(self, config=config, channels=channels)


def ensure_inmemory_delta_history_patch() -> None:
    """Fix InMemorySaver dropping writes on full -> delta migrated threads.

    ``InMemorySaver.get_delta_channel_history`` overrides the base walk with a
    single-pass version that, upon reaching the first checkpoint carrying a
    non-empty plain-value blob for a channel, skips that checkpoint's *own*
    pending writes as "subsumed" by the blob. That is only true when the blob
    was written by that same checkpoint. When the version was carried forward
    from an older ancestor - exactly the first superstep after a full -> delta
    migration, where the input write lands on a checkpoint still referencing
    the pre-delta blob version - those pending writes postdate the blob and
    are silently dropped: the first message appended after migration vanishes
    from materialized state.

    Both the base implementation (used by the SQLite savers) and the Postgres
    override collect the terminating checkpoint's writes *before* treating its
    blob as the seed, which is the correct order. This patch delegates
    InMemorySaver to the base implementation - one ``get_tuple`` per ancestor
    instead of a single fused walk, which is fine for dict-backed storage.

    Idempotent. Remove once LangGraph fixes the override upstream.
    """
    if getattr(InMemorySaver, _PATCH_FLAG, False):
        return
    InMemorySaver.get_delta_channel_history = _get_delta_channel_history_via_base  # type: ignore[method-assign]
    InMemorySaver.aget_delta_channel_history = _aget_delta_channel_history_via_base  # type: ignore[method-assign]
    InMemorySaver._deerflow_delta_history_patched = True  # type: ignore[attr-defined]


ensure_inmemory_delta_history_patch()
