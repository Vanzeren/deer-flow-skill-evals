"""In-memory, event-driven evidence recorder for skill eval runs.

Spec §6.2: pairing and dedup rules. Nothing here touches LangChain or the
agent; SkillEvalCallback translates callback events into these method calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

ToolCallStatus = Literal["pending", "success", "error", "not_executed"]

_INTERRUPTED_ERROR = "tool end event never arrived (interrupted run)"


@dataclass
class ToolCallRecord:
    tool_call_id: str
    name: str
    args: dict[str, Any]
    status: ToolCallStatus = "pending"
    error: str | None = None
    batch_index: int = -1
    orphan: bool = False
    agent_path: tuple[str, ...] = ("lead",)


@dataclass
class EvalRecorder:
    trace_id: str
    loaded_skills: dict[str, str] = field(default_factory=dict)
    loaded_skill_paths: set[str] = field(default_factory=set)
    delivered_references: set[str] = field(default_factory=set)
    tool_batches: list[list[ToolCallRecord]] = field(default_factory=list)
    _by_tool_call_id: dict[str, ToolCallRecord] = field(default_factory=dict)
    _pending_tool_starts: dict[UUID, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    _seen_activation_message_ids: set[str] = field(default_factory=set)

    def record_batch(self, tool_calls: list[dict[str, Any]], *, run_id: UUID, parent_run_id: UUID | None) -> None:
        batch: list[ToolCallRecord] = []
        for call in tool_calls:
            call_id = call.get("id")
            if not call_id or call_id in self._by_tool_call_id:
                continue
            record = ToolCallRecord(
                tool_call_id=call_id,
                name=call.get("name") or "",
                args=call.get("args") or {},
                batch_index=len(self.tool_batches),
            )
            self._by_tool_call_id[call_id] = record
            batch.append(record)
        if batch:
            self.tool_batches.append(batch)

    def stash_tool_start(self, run_id: UUID, tool_name: str, args: dict[str, Any]) -> None:
        self._pending_tool_starts[run_id] = (tool_name, args)

    def pop_tool_start(self, run_id: UUID) -> tuple[str, dict[str, Any]] | None:
        return self._pending_tool_starts.pop(run_id, None)

    def record_tool_end(self, tool_call_id: str | None, *, ok: bool, error: str | None = None) -> ToolCallRecord:
        record = self._by_tool_call_id.get(tool_call_id or "")
        if record is None:
            record = ToolCallRecord(tool_call_id=tool_call_id or "", name="", args={}, orphan=True)
            self.tool_batches.append([record])
            if tool_call_id:
                self._by_tool_call_id[tool_call_id] = record
        record.status = "success" if ok else "error"
        record.error = error
        return record

    def record_tool_error_match(self, tool_name: str, args: dict[str, Any], error: str) -> ToolCallRecord | None:
        """Best-effort pair an un-IDed tool error with the unique matching pending record.

        Returns the matched record, or None when zero or multiple pending
        records share (name, args) — the caller then falls back to an orphan.
        """
        matches = [record for record in self._by_tool_call_id.values() if record.status == "pending" and record.name == tool_name and record.args == args]
        if len(matches) != 1:
            return None
        record = matches[0]
        record.status = "error"
        record.error = error
        return record

    def record_skill_activation(self, name: str, *, kind: str, message_id: str | None = None) -> None:
        if message_id is not None:
            if message_id in self._seen_activation_message_ids:
                return
            self._seen_activation_message_ids.add(message_id)
        self.loaded_skills.setdefault(name, kind)

    def record_skill_read(self, name: str, path: str) -> None:
        self.loaded_skills.setdefault(name, "read")
        self.loaded_skill_paths.add(path)

    def record_reference(self, rel_path: str) -> None:
        self.delivered_references.add(rel_path)

    def finalize(self) -> None:
        """Drain residual tool starts, then mark batch entries that never executed.

        Leftover on_tool_start stash entries mean the tool ran but its
        end/error event never arrived (e.g. interrupted run) — spec §6.2
        treats them with on_tool_error semantics, not not_executed.
        """
        leftover = list(self._pending_tool_starts.values())
        self._pending_tool_starts.clear()
        for tool_name, args in leftover:
            if self.record_tool_error_match(tool_name, args, _INTERRUPTED_ERROR) is None:
                self.record_tool_end(None, ok=False, error=_INTERRUPTED_ERROR)
        for record in self._by_tool_call_id.values():
            if record.status == "pending":
                record.status = "not_executed"
