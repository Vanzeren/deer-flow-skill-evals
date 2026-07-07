import time
from typing import Any

from skill_eval.agent_runner import AgentRunRequest
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


class DeerFlowTraceAdapter:
    """Pure data converter: accumulates StreamEvents and builds AgentTrace."""

    def __init__(self, request: AgentRunRequest):
        self._tool_calls: dict[str, AgentToolCall] = {}
        self._tool_call_order: list[str] = []
        self._messages: list[dict[str, Any]] = []
        self._chunks_by_msg_id: dict[str, list[str]] = {}
        self._last_ai_msg_id: str = ""
        self._start_time: float = 0.0
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._errors: list[str] = []
        self._raw_events: list[dict[str, Any]] = []
        self._request = request

    def feed(self, event) -> None:
        """Ingest one StreamEvent."""
        self._raw_events.append({"type": event.type, "data": event.data})

        if event.type == "messages-tuple":
            self._feed_message(event.data)
        elif event.type == "end":
            usage = event.data.get("usage", {})
            if usage.get("input_tokens") is not None:
                self._input_tokens = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                self._output_tokens = usage["output_tokens"]

    def build(self, raw_trace_path: str | None = None) -> AgentTrace:
        """Assemble final AgentTrace."""
        if raw_trace_path:
            import json
            from pathlib import Path

            Path(raw_trace_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_trace_path).write_text("\n".join(json.dumps(e) for e in self._raw_events))

        final_answer = ""
        if self._last_ai_msg_id and self._last_ai_msg_id in self._chunks_by_msg_id:
            final_answer = "".join(self._chunks_by_msg_id[self._last_ai_msg_id])

        tool_calls = []
        for tc_id in self._tool_call_order:
            if tc_id in self._tool_calls:
                tool_calls.append(self._tool_calls[tc_id])

        skill_invocations = self._infer_skill_invocations()

        latency_ms = int((time.monotonic() - self._start_time) * 1000) if self._start_time else None

        return AgentTrace(
            input=self._request.user_input,
            final_answer=final_answer,
            success=len(self._errors) == 0,
            tool_calls=tool_calls,
            skill_invocations=skill_invocations,
            messages=self._messages,
            errors=self._errors,
            latency_ms=latency_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            steps=[{"type": e["type"]} for e in self._raw_events],
            runtime="deerflow",
            raw_trace_ref=raw_trace_path,
        )

    def _feed_message(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type")
        if msg_type == "ai":
            self._feed_ai_message(data)
        elif msg_type == "tool":
            self._feed_tool_message(data)
        self._messages.append(data)

    def _feed_ai_message(self, data: dict[str, Any]) -> None:
        msg_id = data.get("id") or ""
        content = data.get("content") or ""

        if msg_id:
            self._last_ai_msg_id = msg_id
            if content:
                self._chunks_by_msg_id.setdefault(msg_id, []).append(content)

        tool_calls_data = data.get("tool_calls") or []
        for tc in tool_calls_data:
            tc_id = tc.get("id") or ""
            if not tc_id:
                continue
            if tc_id not in self._tool_calls:
                self._tool_calls[tc_id] = AgentToolCall(
                    name=tc.get("name", ""),
                    args=tc.get("args", {}),
                )
                self._tool_call_order.append(tc_id)

    def _feed_tool_message(self, data: dict[str, Any]) -> None:
        tc_id = data.get("tool_call_id") or ""
        if tc_id and tc_id in self._tool_calls:
            call = self._tool_calls[tc_id]
            call.result = data.get("content")
            error = data.get("error")
            if error:
                call.error = str(error)

    def _infer_skill_invocations(self) -> list[SkillInvocation]:
        invocations: list[SkillInvocation] = []

        forced = self._request.forced_skills
        if forced is not None:
            loaded_skills = set(forced)
        else:
            loaded_skills = set(self._request.required_skills) | set(self._request.candidate_skills)

        used_skills: set[str] = set()
        for tc_id in self._tool_call_order:
            call = self._tool_calls.get(tc_id)
            if not call or call.name != "read_file":
                continue
            args = call.args or {}
            path = args.get("file_path") or args.get("path") or ""
            for skill_name in loaded_skills:
                if f"/{skill_name}/SKILL.md" in path or f"/{skill_name}/skill.md" in path:
                    used_skills.add(skill_name)
                    break

        for skill_name in sorted(loaded_skills):
            invocations.append(
                SkillInvocation(
                    name=skill_name,
                    path=f"skills/{skill_name}",
                    loaded=True,
                    used=skill_name in used_skills,
                    applied=None,
                    trigger_reason=("read_file SKILL.md" if skill_name in used_skills else "available in context"),
                    evidence=([f"read_file targeting skills/{skill_name}/SKILL.md"] if skill_name in used_skills else []),
                )
            )

        return invocations
