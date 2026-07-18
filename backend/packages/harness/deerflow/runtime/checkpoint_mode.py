"""Dual-mode checkpoint channel safety: mode freeze, metadata markers, and the fail-closed gate.

Checkpointer storage runs in ``full`` mode (whole-snapshot channel values) or
``delta`` mode (LangGraph ``DeltaChannel``: sentinel blobs + per-step writes).
The mode is process-frozen at agent-build time, stamped into each checkpoint's
metadata on write, and enforced before every state access: a full-mode process
opening a delta thread raises :class:`CheckpointModeMismatchError` instead of
silently materializing empty state. Delta-mode processes read legacy full
checkpoints transparently, so full -> delta is the supported migration path.
"""

from __future__ import annotations

from typing import Any

from deerflow.config.database_config import CheckpointChannelMode

INTERNAL_CHECKPOINT_MODE_KEY = "__deerflow_checkpoint_channel_mode"
CHECKPOINT_MODE_METADATA_KEY = "deerflow_checkpoint_channel_mode"


class CheckpointModeMismatchError(RuntimeError):
    """Raised before a full-mode graph reads a Delta checkpoint."""


class CheckpointModeReconfigurationError(RuntimeError):
    """Raised when a process attempts to hot-switch its persistence mode."""


_frozen_checkpoint_channel_mode: CheckpointChannelMode | None = None


def freeze_checkpoint_channel_mode(mode: CheckpointChannelMode) -> CheckpointChannelMode:
    global _frozen_checkpoint_channel_mode
    if _frozen_checkpoint_channel_mode is None:
        _frozen_checkpoint_channel_mode = mode
    elif _frozen_checkpoint_channel_mode != mode:
        raise CheckpointModeReconfigurationError("checkpoint_channel_mode is restart-required and cannot change in a running process")
    return _frozen_checkpoint_channel_mode


def inject_checkpoint_mode(config: dict[str, Any], mode: CheckpointChannelMode) -> None:
    configurable = config.setdefault("configurable", {})
    configurable[INTERNAL_CHECKPOINT_MODE_KEY] = mode
    metadata = config.setdefault("metadata", {})
    if mode == "delta":
        metadata[CHECKPOINT_MODE_METADATA_KEY] = "delta"
    else:
        metadata.pop(CHECKPOINT_MODE_METADATA_KEY, None)


def checkpoint_tuple_uses_delta(checkpoint_tuple: Any) -> bool:
    if checkpoint_tuple is None:
        return False
    metadata = getattr(checkpoint_tuple, "metadata", {}) or {}
    if metadata.get(CHECKPOINT_MODE_METADATA_KEY) == "delta":
        return True
    counters = metadata.get("counters_since_delta_snapshot")
    return isinstance(counters, dict) and "messages" in counters


def _raise_if_incompatible(checkpoint_tuple: Any, mode: CheckpointChannelMode) -> None:
    if mode == "full" and checkpoint_tuple_uses_delta(checkpoint_tuple):
        raise CheckpointModeMismatchError("Thread requires delta mode; materialize and convert its checkpoints before using full mode.")


def ensure_checkpoint_mode_compatible(checkpointer: Any, config: dict[str, Any], mode: CheckpointChannelMode) -> None:
    if mode == "delta":
        return
    _raise_if_incompatible(checkpointer.get_tuple(config), mode)


async def aensure_checkpoint_mode_compatible(checkpointer: Any, config: dict[str, Any], mode: CheckpointChannelMode) -> None:
    if mode == "delta":
        return
    _raise_if_incompatible(await checkpointer.aget_tuple(config), mode)
