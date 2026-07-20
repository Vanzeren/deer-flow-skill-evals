import asyncio
import hashlib
import json
import multiprocessing
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Any

from deerflow.client import StreamEvent
from deerflow.config.app_config import (
    get_app_config,
    pop_current_app_config,
    push_current_app_config,
)
from skill_eval.agent_runner import AgentRunRequest, AgentRunResult, SandboxMode
from skill_eval.routing import RouteObservation, RoutingObserver
from skill_eval.trace_schema import AgentArtifact, AgentToolCall, AgentTrace, QuickTurnCapture

_ARTIFACT_HEAD_BYTES = 6_000
_ARTIFACT_TAIL_BYTES = 2_000
_LOCAL_SANDBOX_PROVIDER = "deerflow.sandbox.local:LocalSandboxProvider"
_CHILD_EXIT_GRACE_SECONDS = 5.0

type ClientFactory = Callable[..., Any]
type ChildTarget = Callable[[Connection, dict[str, Any], dict[str, Any]], None]


@contextmanager
def _sandbox_context(mode: SandboxMode) -> Iterator[None]:
    if mode == "configured":
        yield
        return
    app_config = get_app_config()
    sandbox = app_config.sandbox.model_copy(
        update={
            "use": _LOCAL_SANDBOX_PROVIDER,
            "allow_host_bash": True,
        }
    )
    push_current_app_config(app_config.model_copy(update={"sandbox": sandbox}))
    try:
        yield
    finally:
        pop_current_app_config()


class DeerFlowTraceAdapter:
    """Convert a DeerFlow event stream into a compact, stable evaluation trace."""

    def __init__(self, request: AgentRunRequest):
        self._request = request
        self._tool_calls: dict[str, AgentToolCall] = {}
        self._tool_call_order: list[str] = []
        self._messages: list[dict[str, Any]] = []
        self._ai_messages: dict[str, dict[str, Any]] = {}
        self._last_ai_msg_id = ""
        self._artifact_paths: dict[str, None] = {}
        self._start_time = 0.0
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._errors: list[str] = []
        self._raw_events: list[dict[str, Any]] = []
        self._live_raw_trace_path = Path(request.trace_dir) / f"{request.thread_id}.jsonl" if request.trace_dir else None

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return tuple(self._artifact_paths)

    def ai_message_ids(self) -> tuple[str, ...]:
        return tuple(self._ai_messages)

    def ai_message_content(self, message_id: str) -> str:
        message = self._ai_messages.get(message_id)
        return str(message["content"]) if message is not None else ""

    def start(self) -> None:
        self._start_time = time.monotonic()
        if self._live_raw_trace_path is not None:
            try:
                self._live_raw_trace_path.parent.mkdir(parents=True, exist_ok=True)
                self._live_raw_trace_path.write_text("", encoding="utf-8")
            except OSError as exc:
                self.add_error(f"Raw trace initialization failed: {exc}")
                self._live_raw_trace_path = None

    def add_error(self, error: str) -> None:
        if error not in self._errors:
            self._errors.append(error)

    def feed(self, event: StreamEvent) -> None:
        self._raw_events.append({"type": event.type, "data": event.data})
        if self._live_raw_trace_path is not None:
            try:
                with self._live_raw_trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(
                        json.dumps(
                            self._raw_events[-1],
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
            except OSError as exc:
                self.add_error(f"Raw trace append failed: {exc}")
                self._live_raw_trace_path = None
        if event.type == "messages-tuple":
            self._feed_message(event.data)
        elif event.type == "values":
            self._feed_artifacts(event.data)
            self._hydrate_tool_calls(event.data)
        elif event.type == "end":
            usage = event.data.get("usage") or {}
            if usage.get("input_tokens") is not None:
                self._input_tokens = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                self._output_tokens = usage["output_tokens"]

    def build(
        self,
        *,
        thread_id: str,
        artifacts: list[AgentArtifact] | None = None,
        raw_trace_path: str | None = None,
    ) -> AgentTrace:
        if raw_trace_path and Path(raw_trace_path) != self._live_raw_trace_path:
            destination = Path(raw_trace_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                "\n".join(json.dumps(event, ensure_ascii=False, default=str) for event in self._raw_events),
                encoding="utf-8",
            )

        tool_call_chain = [[call["id"] for call in message["tool_calls"]] for message in self._messages if message["type"] == "ai" and message["tool_calls"]]
        final_answer = ""
        if self._last_ai_msg_id:
            final_answer = str(self._ai_messages[self._last_ai_msg_id]["content"])
        latency_ms = int((time.monotonic() - self._start_time) * 1000) if self._start_time else None
        return AgentTrace(
            input=self._request.user_input,
            final_answer=final_answer,
            success=not self._errors,
            thread_id=thread_id,
            tool_calls=[self._tool_calls[call_id] for call_id in self._tool_call_order],
            tool_call_chain=tool_call_chain,
            messages=list(self._messages),
            artifacts=artifacts or [],
            errors=list(self._errors),
            latency_ms=latency_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            runtime="deerflow",
            raw_trace_ref=raw_trace_path,
        )

    def _feed_message(self, data: dict[str, Any]) -> None:
        message_type = data.get("type")
        if message_type == "ai":
            self._feed_ai_message(data)
        elif message_type == "tool":
            self._feed_tool_message(data)

    def _feed_ai_message(self, data: dict[str, Any]) -> None:
        message_id = str(data.get("id") or f"ai-{len(self._ai_messages)}")
        content = str(data.get("content") or "")
        message = self._ai_messages.get(message_id)
        if message is None:
            message = {"type": "ai", "id": message_id, "content": "", "tool_calls": []}
            self._ai_messages[message_id] = message
            self._messages.append(message)
        message["content"] += content
        self._last_ai_msg_id = message_id

        for tool_call in data.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("id") or "")
            if not call_id or call_id in self._tool_calls:
                continue
            call = AgentToolCall(
                id=call_id,
                message_id=message_id,
                name=str(tool_call.get("name") or ""),
                args=tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {},
            )
            self._tool_calls[call_id] = call
            self._tool_call_order.append(call_id)
            message["tool_calls"].append({"id": call.id, "name": call.name, "args": call.args})

    def _feed_tool_message(self, data: dict[str, Any]) -> None:
        call_id = str(data.get("tool_call_id") or "")
        content = data.get("content")
        call = self._tool_calls.get(call_id)
        if call is not None:
            call.result = content
            explicit_error = data.get("error")
            if explicit_error:
                call.error = str(explicit_error)
            elif str(content or "").lstrip().startswith("Error:"):
                call.error = str(content)
        self._messages.append(
            {
                "type": "tool",
                "id": str(data.get("id") or f"tool-{len(self._messages)}"),
                "tool_call_id": call_id,
                "name": str(data.get("name") or (call.name if call else "")),
                "content": content,
                "error": data.get("error"),
            }
        )

    def _hydrate_tool_calls(self, data: dict[str, Any]) -> None:
        for message in data.get("messages") or []:
            if not isinstance(message, dict) or message.get("type") != "ai":
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id") or "")
                args = tool_call.get("args")
                call = self._tool_calls.get(call_id)
                if call is None or not isinstance(args, dict) or not args:
                    continue
                call.args = args
                stored_message = self._ai_messages.get(call.message_id)
                if stored_message is None:
                    continue
                for stored_call in stored_message["tool_calls"]:
                    if stored_call["id"] == call_id:
                        stored_call["args"] = args
                        break

    def _feed_artifacts(self, data: dict[str, Any]) -> None:
        for artifact in data.get("artifacts") or []:
            if isinstance(artifact, str):
                path = artifact
            elif isinstance(artifact, dict):
                path = artifact.get("path")
            else:
                continue
            if isinstance(path, str) and path:
                self._artifact_paths.setdefault(path, None)


class _QuickTurnWatcher:
    """Track the first non-empty AI text turn after a skill-load routing decision."""

    def __init__(self) -> None:
        self.skill: str | None = None
        self.target_id: str | None = None
        self.content: str = ""
        self.complete: bool = False
        self._excluded_ids: set[str] = set()

    def start(self, *, skill: str, existing_message_ids: tuple[str, ...]) -> None:
        self.skill = skill
        self._excluded_ids = set(existing_message_ids)

    def feed(self, event: StreamEvent, adapter: DeerFlowTraceAdapter) -> None:
        if self.skill is None or self.complete:
            return
        if event.type == "end":
            if self.target_id is not None:
                self.content = adapter.ai_message_content(self.target_id)
                self.complete = True
            return
        if event.type != "messages-tuple":
            return
        message_id = str(event.data.get("id") or "")
        if self.target_id is None:
            if event.data.get("type") != "ai" or message_id in self._excluded_ids:
                return
            if adapter.ai_message_content(message_id).strip():
                self.target_id = message_id
            return
        if message_id != self.target_id:
            self.content = adapter.ai_message_content(self.target_id)
            self.complete = True


def snapshot_artifact(client: Any, thread_id: str, path: str) -> AgentArtifact:
    payload = client.get_artifact(thread_id, path)
    if isinstance(payload, tuple):
        raw_content, mime_type = payload
    else:
        raw_content = payload.get("content", b"")
        mime_type = payload.get("mime_type")
    if isinstance(raw_content, str):
        content_bytes = raw_content.encode()
    elif isinstance(raw_content, (bytes, bytearray)):
        content_bytes = bytes(raw_content)
    else:
        content_bytes = str(raw_content).encode()

    truncated = len(content_bytes) > _ARTIFACT_HEAD_BYTES + _ARTIFACT_TAIL_BYTES
    if truncated:
        omitted = len(content_bytes) - _ARTIFACT_HEAD_BYTES - _ARTIFACT_TAIL_BYTES
        retained = content_bytes[:_ARTIFACT_HEAD_BYTES].decode(errors="replace") + f"\n...[{omitted} bytes omitted]...\n" + content_bytes[-_ARTIFACT_TAIL_BYTES:].decode(errors="replace")
    else:
        retained = content_bytes.decode(errors="replace")

    return AgentArtifact(
        path=path,
        mime_type=str(mime_type or "application/octet-stream"),
        content=retained,
        original_bytes=len(content_bytes),
        sha256=hashlib.sha256(content_bytes).hexdigest(),
        truncated=truncated,
    )


def _execute_deerflow(
    request: AgentRunRequest,
    *,
    config_path: str | None = None,
    client_factory: ClientFactory | None = None,
) -> AgentRunResult:
    if client_factory is None:
        from deerflow.client import DeerFlowClient

        client_factory = DeerFlowClient

    adapter = DeerFlowTraceAdapter(request)
    observer = RoutingObserver(request.candidate_skills)
    adapter.start()
    try:
        client = client_factory(
            config_path=config_path,
            model_name=request.model_name,
            available_skills=set(request.candidate_skills),
            subagent_enabled=request.mode == "full",
        )
    except Exception as exc:
        error = f"Failed to create DeerFlowClient: {exc}"
        adapter.add_error(error)
        observer.fail(error)
        return _build_result(request, adapter, observer.finalize(stream_completed=False))

    stream = None
    saw_end = False
    stopped_early = False
    stream_failed = False
    watcher = _QuickTurnWatcher() if request.mode == "quick" else None
    try:
        with _sandbox_context(request.sandbox):
            stream = client.stream(request.user_input, thread_id=request.thread_id)
            for stream_event in stream:
                adapter.feed(stream_event)
                route_ready = observer.feed(stream_event)
                if stream_event.type == "end":
                    saw_end = True
                if request.mode == "routing_probe" and route_ready:
                    stopped_early = True
                    break
                if watcher is not None:
                    if route_ready and watcher.skill is None:
                        decided = observer.decided_route
                        if decided == "ambiguous":
                            stopped_early = True
                            break
                        if decided is not None:
                            watcher.start(skill=decided, existing_message_ids=adapter.ai_message_ids())
                    watcher.feed(stream_event, adapter)
                    if watcher.complete:
                        stopped_early = True
                        break
    except Exception as exc:
        stream_failed = True
        error = f"Stream error: {exc}"
        adapter.add_error(error)
        observer.fail(error)
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception as exc:
                stream_failed = True
                error = f"Stream close error: {exc}"
                adapter.add_error(error)
                observer.fail(error)

    if not saw_end and not stopped_early and not stream_failed:
        error = "Stream ended without an end event"
        adapter.add_error(error)
        observer.fail(error)

    artifacts: list[AgentArtifact] = []
    if request.mode == "full":
        for artifact_path in adapter.artifact_paths:
            try:
                artifacts.append(snapshot_artifact(client, request.thread_id, artifact_path))
            except Exception as exc:
                adapter.add_error(f"Artifact snapshot failed for {artifact_path}: {exc}")

    quick_turn = None
    if watcher is not None and watcher.complete and watcher.skill is not None and watcher.target_id is not None:
        quick_turn = QuickTurnCapture(
            message_id=watcher.target_id,
            skill=watcher.skill,
            content=watcher.content,
        )

    observation = observer.finalize(stream_completed=saw_end)
    for error in observation.errors:
        adapter.add_error(error)
    return _build_result(request, adapter, observation, artifacts=artifacts, quick_turn=quick_turn)


def _build_result(
    request: AgentRunRequest,
    adapter: DeerFlowTraceAdapter,
    observation: RouteObservation,
    *,
    artifacts: list[AgentArtifact] | None = None,
    quick_turn: QuickTurnCapture | None = None,
) -> AgentRunResult:
    raw_trace_path = None
    if request.trace_dir:
        raw_trace_path = str(Path(request.trace_dir) / f"{request.thread_id}.jsonl")
    try:
        trace = adapter.build(
            thread_id=request.thread_id,
            artifacts=artifacts,
            raw_trace_path=raw_trace_path,
        )
    except OSError as exc:
        adapter.add_error(f"Raw trace write failed: {exc}")
        trace = adapter.build(thread_id=request.thread_id, artifacts=artifacts)
    if quick_turn is not None:
        trace = trace.model_copy(update={"quick_turn": quick_turn})
    if request.mode == "full" and trace.success and not trace.final_answer.strip() and not trace.artifacts:
        error = "Full run produced no final answer or artifact"
        trace = trace.model_copy(
            update={
                "success": False,
                "errors": [*trace.errors, error],
            }
        )
    success = trace.success and observation.completed
    if trace.success != success:
        trace = trace.model_copy(update={"success": success})
    return AgentRunResult(
        final_answer=trace.final_answer,
        success=success,
        trace=trace,
        route_observation=observation,
        thread_id=request.thread_id,
    )


def _failed_result(request: AgentRunRequest, error: str) -> AgentRunResult:
    raw_trace_ref = None
    if request.trace_dir:
        candidate = Path(request.trace_dir) / f"{request.thread_id}.jsonl"
        if candidate.exists():
            raw_trace_ref = str(candidate)
    observation = RouteObservation(completed=False, errors=[error])
    trace = AgentTrace(
        input=request.user_input,
        final_answer="",
        success=False,
        thread_id=request.thread_id,
        errors=[error],
        runtime="deerflow",
        raw_trace_ref=raw_trace_ref,
    )
    return AgentRunResult(
        final_answer="",
        success=False,
        trace=trace,
        route_observation=observation,
        thread_id=request.thread_id,
    )


def _run_child(
    connection: Connection,
    request_data: dict[str, Any],
    config: dict[str, Any],
) -> None:
    try:
        request = AgentRunRequest.model_validate(request_data)
        result = _execute_deerflow(request, config_path=config.get("config_path"))
        connection.send({"result": result.model_dump(mode="json")})
    except BaseException as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


class DeerFlowAgentRunner:
    """Run each real DeerFlow sample in a killable spawned process."""

    def __init__(
        self,
        config_path: str | None = None,
        trace_dir: str | None = None,
        sandbox: SandboxMode = "configured",
        *,
        child_target: ChildTarget = _run_child,
    ):
        self._config_path = config_path
        self._trace_dir = trace_dir
        self._sandbox = sandbox
        self._child_target = child_target

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        if request.trace_dir is None and self._trace_dir is not None:
            request = request.model_copy(update={"trace_dir": self._trace_dir})
        if request.sandbox == "configured" and self._sandbox != "configured":
            request = request.model_copy(update={"sandbox": self._sandbox})
        context = multiprocessing.get_context("spawn")
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=self._child_target,
            args=(
                send_connection,
                request.model_dump(mode="json"),
                {"config_path": self._config_path, "trace_dir": request.trace_dir},
            ),
        )
        started = False
        try:
            process.start()
            started = True
            send_connection.close()
            deadline = time.monotonic() + request.timeout_seconds
            ready = await asyncio.to_thread(
                wait_for_connections,
                [receive_connection, process.sentinel],
                request.timeout_seconds,
            )
            payload = None
            if receive_connection in ready:
                try:
                    payload = await asyncio.to_thread(receive_connection.recv)
                except EOFError:
                    pass
            remaining = max(0.0, deadline - time.monotonic())
            join_timeout = _CHILD_EXIT_GRACE_SECONDS if payload is not None else remaining
            await asyncio.to_thread(process.join, join_timeout)
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join)
                if payload is not None:
                    return _failed_result(
                        request,
                        "DeerFlow child did not exit after returning a result",
                    )
                return _failed_result(
                    request,
                    f"DeerFlow run timed out after {request.timeout_seconds}s",
                )
            elif payload is None and receive_connection.poll():
                try:
                    payload = receive_connection.recv()
                except EOFError:
                    pass
            if payload is not None:
                if "result" in payload:
                    return AgentRunResult.model_validate(payload["result"])
                return _failed_result(request, f"DeerFlow child failed: {payload['error']}")
            return _failed_result(
                request,
                f"DeerFlow child exited without a result (exit code {process.exitcode})",
            )
        except asyncio.CancelledError:
            if started and process.is_alive():
                process.terminate()
                await asyncio.shield(asyncio.to_thread(process.join, 1))
                if process.is_alive():
                    process.kill()
                    await asyncio.shield(asyncio.to_thread(process.join))
            raise
        except Exception as exc:
            if started and process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join)
            return _failed_result(request, f"DeerFlow process error: {exc}")
        finally:
            receive_connection.close()
            if not started:
                send_connection.close()
            if started and not process.is_alive():
                process.close()
