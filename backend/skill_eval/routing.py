from collections import defaultdict
from pathlib import PurePosixPath
from typing import Literal, cast

from pydantic import BaseModel, Field

from deerflow.client import StreamEvent
from skill_eval.case_schema import RouteLabel

type EvidenceKind = Literal["described", "load_requested", "loaded", "load_failed"]


class RouteEvidence(BaseModel):
    id: str
    kind: EvidenceKind
    skill: str
    tool_call_id: str
    detail: str | None = None


class RouteObservation(BaseModel):
    observed_route: RouteLabel | Literal["ambiguous"] | None = None
    evidence: list[RouteEvidence] = Field(default_factory=list)
    completed: bool = False
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None


class _PendingCall(BaseModel):
    batch_id: str
    kind: Literal["describe", "load"]
    skill: str


class RoutingObserver:
    def __init__(self, candidates: tuple[str, ...]):
        self._candidates = frozenset(candidates)
        self._pending: dict[str, _PendingCall] = {}
        self._completed_call_ids: set[str] = set()
        self._batch_loads: dict[str, set[str]] = defaultdict(set)
        self._batch_pending_loads: dict[str, set[str]] = defaultdict(set)
        self._evidence: list[RouteEvidence] = []
        self._errors: list[str] = []
        self._observed: RouteLabel | Literal["ambiguous"] | None = None
        self._completed = False

    def feed(self, event: StreamEvent) -> bool:
        if self._completed:
            return True
        if event.type == "messages-tuple" and event.data.get("type") == "ai":
            self._feed_ai(event.data)
        elif event.type == "messages-tuple" and event.data.get("type") == "tool":
            self._feed_tool(event.data)
        elif event.type == "values":
            for message in event.data.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                if message.get("type") == "ai":
                    self._feed_ai(message)
                elif message.get("type") == "tool":
                    self._feed_tool(message)
        elif event.type == "end" and self._observed is None and not self._errors:
            if self._pending_candidate_loads():
                self.fail("stream ended with unresolved candidate skill reads")
            else:
                self._observed = "none"
                self._completed = True
        return self._completed and self._observed != "none"

    def fail(self, message: str) -> None:
        self._errors.append(message)
        self._completed = False
        self._observed = None

    def finalize(self, *, stream_completed: bool, latency_ms: int | None = None) -> RouteObservation:
        if self._observed is None and not self._errors and stream_completed:
            self._observed = "none"
            self._completed = True
        if self._observed is None and not self._errors:
            self.fail("stream stopped before a routing decision")
        return RouteObservation(
            observed_route=self._observed,
            evidence=list(self._evidence),
            completed=self._completed,
            errors=list(self._errors),
            latency_ms=latency_ms,
        )

    def _feed_ai(self, data: dict) -> None:
        batch_id = str(data.get("id") or f"batch-{len(self._pending)}")
        for call in data.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            name = call.get("name")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if not call_id:
                continue
            if call_id in self._pending or call_id in self._completed_call_ids:
                continue
            if name == "describe_skill" and args.get("name") in self._candidates:
                self._pending[call_id] = _PendingCall(
                    batch_id=batch_id,
                    kind="describe",
                    skill=args["name"],
                )
                continue
            skill = self._skill_from_read(name, args)
            if skill:
                self._pending[call_id] = _PendingCall(batch_id=batch_id, kind="load", skill=skill)
                self._batch_pending_loads[batch_id].add(call_id)
                self._record("load_requested", skill, call_id)

    def _feed_tool(self, data: dict) -> None:
        call_id = str(data.get("tool_call_id") or "")
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        self._completed_call_ids.add(call_id)
        content = str(data.get("content") or "")
        failed = content.lstrip().startswith("Error:")
        if pending.kind == "describe":
            if not failed:
                self._record("described", pending.skill, call_id)
            return
        self._batch_pending_loads[pending.batch_id].discard(call_id)
        if failed:
            self._record("load_failed", pending.skill, call_id, content[:500])
        else:
            self._batch_loads[pending.batch_id].add(pending.skill)
            self._record("loaded", pending.skill, call_id)
        if self._batch_pending_loads[pending.batch_id]:
            return
        loaded = self._batch_loads[pending.batch_id]
        if len(loaded) == 1:
            self._observed = cast(RouteLabel, next(iter(loaded)))
            self._completed = True
        elif len(loaded) > 1:
            self._observed = "ambiguous"
            self._completed = True

    def _skill_from_read(self, name: object, args: dict) -> str | None:
        if name not in {"read_file", "read_file_tool"}:
            return None
        path = args.get("path") or args.get("file_path") or args.get("filepath")
        if not isinstance(path, str):
            return None
        pure = PurePosixPath(path)
        parts = pure.parts
        if pure.name != "SKILL.md" or pure.parent.name not in self._candidates or len(parts) < 4 or parts[-4:-2] != ("skills", "public"):
            return None
        return pure.parent.name

    def _pending_candidate_loads(self) -> bool:
        return any(call.kind == "load" for call in self._pending.values())

    def _record(
        self,
        kind: EvidenceKind,
        skill: str,
        call_id: str,
        detail: str | None = None,
    ) -> None:
        self._evidence.append(
            RouteEvidence(
                id=f"route_evidence[{len(self._evidence)}]",
                kind=kind,
                skill=skill,
                tool_call_id=call_id,
                detail=detail,
            )
        )
