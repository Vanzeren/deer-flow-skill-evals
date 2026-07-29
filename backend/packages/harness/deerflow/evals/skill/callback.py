"""SkillEvalCallback: LangChain callback events -> EvalRecorder.

Spec §6.1. Zero agent modifications: the runner attaches this handler via
config["callbacks"]. Implicit skill hits are judged callback-side by pairing
on_tool_start (tool name + args) with on_tool_end (ToolMessage result) —
this branch has no SKILL_CONTEXT_ENTRY_KEY stamp mechanism (spec §3).
"""

from __future__ import annotations

import html
import posixpath
import re
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage

from deerflow.agents.middlewares.skill_activation_middleware import (
    is_slash_skill_activation_reminder,
)
from deerflow.evals.skill.recorder import EvalRecorder

_PATH_KEYS = ("path", "file_path", "filepath")
_SKILL_NAME_RE = re.compile(r'<skill name="(?P<name>[^"]+)"')


class SkillEvalCallback(AsyncCallbackHandler):
    def __init__(
        self,
        recorder: EvalRecorder,
        *,
        expected_skill: str,
        container_path: str = "/mnt/skills",
        read_tool_names: Iterable[str] = ("read_file", "read", "view", "cat"),
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.expected_skill = expected_skill
        self.read_tool_names = frozenset(read_tool_names)
        prefix = posixpath.normpath(container_path)
        self._skill_path_re = re.compile(rf"^{re.escape(prefix)}/(?P<category>[^/]+)/(?P<skill>[^/]+)/(?P<rel>.+)$")

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        for batch in messages:
            for message in batch:
                if not is_slash_skill_activation_reminder(message):
                    continue
                content = message.content if isinstance(message.content, str) else ""
                match = _SKILL_NAME_RE.search(content)
                if match:
                    self.recorder.record_skill_activation(
                        html.unescape(match.group("name")),
                        kind="slash",
                        message_id=getattr(message, "id", None),
                    )

    async def on_llm_end(self, response, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError, TypeError):
            return
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            self.recorder.record_batch(list(tool_calls), run_id=run_id, parent_run_id=parent_run_id)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = (serialized or {}).get("name") or ""
        self.recorder.stash_tool_start(run_id, tool_name, inputs or {})

    async def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        start = self.recorder.pop_tool_start(run_id) or ("", {})
        tool_name, args = start
        if isinstance(output, ToolMessage):
            ok = output.status != "error" and isinstance(output.content, str) and not output.content.startswith("Error:")
            self.recorder.record_tool_end(
                output.tool_call_id,
                ok=ok,
                error=None if ok else (output.content if isinstance(output.content, str) else None),
            )
            if ok:
                self._judge_skill_path(tool_name, args)
        else:
            self.recorder.record_tool_end(None, ok=True)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        start = self.recorder.pop_tool_start(run_id)
        if start is not None:
            tool_name, args = start
            if self.recorder.record_tool_error_match(tool_name, args, str(error)) is not None:
                return
        self.recorder.record_tool_end(None, ok=False, error=str(error))

    def _judge_skill_path(self, tool_name: str, args: dict[str, Any]) -> None:
        if tool_name not in self.read_tool_names:
            return
        raw = next((args[k] for k in _PATH_KEYS if isinstance(args.get(k), str)), None)
        if not raw:
            return
        match = self._skill_path_re.match(posixpath.normpath(raw))
        if not match:
            return
        skill, rel = match.group("skill"), match.group("rel")
        if rel == "SKILL.md":
            self.recorder.record_skill_read(skill, posixpath.normpath(raw))
        elif skill == self.expected_skill:
            self.recorder.record_reference(posixpath.normpath(rel))
