import hashlib

from pydantic import BaseModel

from deerflow.client import StreamEvent
from skill_eval.adapters import deerflow as deerflow_module
from skill_eval.adapters.deerflow import (
    DeerFlowTraceAdapter,
    _execute_deerflow,
    snapshot_artifact,
)
from skill_eval.agent_runner import AgentRunRequest


class FakeSandboxConfig(BaseModel):
    use: str
    allow_host_bash: bool = False


class FakeAppConfig(BaseModel):
    sandbox: FakeSandboxConfig


def test_local_sandbox_context_overrides_and_restores(monkeypatch):
    pushed = []
    popped = []
    monkeypatch.setattr(
        deerflow_module,
        "get_app_config",
        lambda: FakeAppConfig(sandbox=FakeSandboxConfig(use="remote.Provider")),
    )
    monkeypatch.setattr(deerflow_module, "push_current_app_config", lambda config: pushed.append(config))
    monkeypatch.setattr(deerflow_module, "pop_current_app_config", lambda: popped.append(True))

    with deerflow_module._sandbox_context("local"):
        assert pushed[-1].sandbox.use == "deerflow.sandbox.local:LocalSandboxProvider"
        assert pushed[-1].sandbox.allow_host_bash is True

    assert popped == [True]


def event(type_: str, data: dict) -> StreamEvent:
    return StreamEvent(type=type_, data=data)


def request(mode: str = "routing_probe") -> AgentRunRequest:
    return AgentRunRequest(
        case_id="case-1",
        user_input="Survey papers",
        mode=mode,
        model_name="default",
        thread_id="thread-1",
    )


def ai_read_skill(skill: str, call_id: str = "t1") -> StreamEvent:
    return event(
        "messages-tuple",
        {
            "type": "ai",
            "id": "m1",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "name": "read_file",
                    "args": {"path": f"/mnt/skills/public/{skill}/SKILL.md"},
                }
            ],
        },
    )


def tool_result(call_id: str = "t1", content: str = "skill body") -> StreamEvent:
    return event(
        "messages-tuple",
        {
            "type": "tool",
            "id": f"result-{call_id}",
            "tool_call_id": call_id,
            "name": "read_file",
            "content": content,
        },
    )


def test_adapter_hydrates_streamed_tool_args_from_values_snapshot():
    adapter = DeerFlowTraceAdapter(request())
    adapter.start()
    adapter.feed(
        event(
            "messages-tuple",
            {
                "type": "ai",
                "id": "m1",
                "tool_calls": [{"id": "r1", "name": "read_file", "args": {}}],
            },
        )
    )
    adapter.feed(
        event(
            "values",
            {
                "messages": [
                    {
                        "type": "ai",
                        "id": "m1",
                        "tool_calls": [
                            {
                                "id": "r1",
                                "name": "read_file",
                                "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"},
                            }
                        ],
                    }
                ]
            },
        )
    )

    trace = adapter.build(thread_id="thread-1")
    assert trace.tool_calls[0].args == {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"}


def test_adapter_persists_each_raw_event_before_build(tmp_path):
    traced_request = request().model_copy(update={"trace_dir": str(tmp_path)})
    adapter = DeerFlowTraceAdapter(traced_request)
    adapter.start()

    adapter.feed(
        event(
            "messages-tuple",
            {"type": "ai", "id": "m1", "content": "partial"},
        )
    )

    raw_trace = tmp_path / "thread-1.jsonl"
    assert raw_trace.exists()
    assert '"content": "partial"' in raw_trace.read_text(encoding="utf-8")


def test_adapter_merges_ai_chunks_by_message_id():
    adapter = DeerFlowTraceAdapter(request())
    adapter.start()
    adapter.feed(event("messages-tuple", {"type": "ai", "id": "m1", "content": "hel"}))
    adapter.feed(event("messages-tuple", {"type": "ai", "id": "m1", "content": "lo"}))

    trace = adapter.build(thread_id="thread-1")

    assert trace.final_answer == "hello"
    assert [message for message in trace.messages if message["type"] == "ai"] == [{"type": "ai", "id": "m1", "content": "hello", "tool_calls": []}]


def test_adapter_records_tool_identity_parent_and_error_prefix():
    adapter = DeerFlowTraceAdapter(request())
    adapter.start()
    adapter.feed(ai_read_skill("systematic-literature-review"))
    adapter.feed(tool_result(content="Error: permission denied"))

    trace = adapter.build(thread_id="thread-1")

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].id == "t1"
    assert trace.tool_calls[0].message_id == "m1"
    assert trace.tool_calls[0].name == "read_file"
    assert trace.tool_calls[0].error == "Error: permission denied"
    assert trace.tool_calls[0].result == "Error: permission denied"


def test_adapter_preserves_tool_order_usage_and_artifact_paths():
    adapter = DeerFlowTraceAdapter(request(mode="full"))
    adapter.start()
    adapter.feed(
        event(
            "messages-tuple",
            {
                "type": "ai",
                "id": "m1",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "name": "read_file", "args": {"path": "a"}},
                    {"id": "t2", "name": "bash", "args": {"command": "pwd"}},
                ],
            },
        )
    )
    adapter.feed(
        event(
            "values",
            {
                "artifacts": [
                    {"path": "/mnt/user-data/outputs/report.md", "title": "Report"},
                    {"path": "/mnt/user-data/outputs/report.md", "title": "Report"},
                    {"path": "/mnt/user-data/outputs/data.json", "title": "Data"},
                ]
            },
        )
    )
    adapter.feed(event("end", {"usage": {"input_tokens": 10, "output_tokens": 5}}))

    trace = adapter.build(thread_id="thread-1")

    assert [call.id for call in trace.tool_calls] == ["t1", "t2"]
    assert adapter.artifact_paths == (
        "/mnt/user-data/outputs/report.md",
        "/mnt/user-data/outputs/data.json",
    )
    assert trace.input_tokens == 10
    assert trace.output_tokens == 5


class ArtifactClient:
    def __init__(self, content: bytes, mime_type: str = "text/plain"):
        self.content = content
        self.mime_type = mime_type

    def get_artifact(self, thread_id: str, path: str) -> dict:
        assert thread_id == "thread-1"
        assert path == "/mnt/user-data/outputs/report.txt"
        return {"content": self.content, "mime_type": self.mime_type}


class TupleArtifactClient(ArtifactClient):
    def get_artifact(self, thread_id: str, path: str) -> tuple[bytes, str]:
        assert thread_id == "thread-1"
        assert path == "/mnt/user-data/outputs/report.txt"
        return self.content, self.mime_type


def test_snapshot_artifact_hashes_original_and_bounds_retained_content():
    content = b"a" * 7_000 + b"z" * 3_000

    artifact = snapshot_artifact(
        ArtifactClient(content),
        "thread-1",
        "/mnt/user-data/outputs/report.txt",
    )

    assert artifact.original_bytes == 10_000
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()
    assert artifact.truncated is True
    assert artifact.content.startswith("a" * 100)
    assert artifact.content.endswith("z" * 100)
    assert len(artifact.content.encode()) < len(content)


def test_snapshot_artifact_accepts_real_client_tuple_contract():
    artifact = snapshot_artifact(
        TupleArtifactClient(b"report", "text/markdown"),
        "thread-1",
        "/mnt/user-data/outputs/report.txt",
    )

    assert artifact.content == "report"
    assert artifact.mime_type == "text/markdown"


class ScriptedStream:
    def __init__(self, client, events):
        self._client = client
        self._events = iter(events)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._events)
        if item is AssertionError:
            raise AssertionError("stream consumed past routing decision")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._client.stream_closed = True


class ScriptedClient:
    def __init__(self, *, events, artifacts=None, **kwargs):
        self.events = events
        self.artifacts = artifacts or {}
        self.options = kwargs
        self.stream_closed = False
        self.artifact_reads = []

    def stream(self, message, *, thread_id):
        assert message == "Survey papers"
        assert thread_id == "thread-1"
        return ScriptedStream(self, self.events)

    def get_artifact(self, thread_id, path):
        self.artifact_reads.append((thread_id, path))
        return self.artifacts[path]


def test_probe_mode_closes_real_stream_after_successful_route_batch():
    holder = {}

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[
                ai_read_skill("systematic-literature-review"),
                tool_result(),
                AssertionError,
            ],
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(), client_factory=client_factory)

    client = holder["client"]
    assert result.route_observation.observed_route == "systematic-literature-review"
    assert result.route_observation.completed is True
    assert client.stream_closed is True
    assert client.options["available_skills"] == {
        "systematic-literature-review",
        "academic-paper-review",
    }
    assert client.options["subagent_enabled"] is False
    assert result.trace.runtime == "deerflow"
    assert result.trace.thread_id == "thread-1"


def test_none_probe_waits_for_end_and_preserves_short_answer():
    holder = {}

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[
                event("messages-tuple", {"type": "ai", "id": "m1", "content": "Direct answer"}),
                event("end", {"usage": {}}),
            ],
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(), client_factory=client_factory)

    assert result.success is True
    assert result.final_answer == "Direct answer"
    assert result.route_observation.observed_route == "none"
    assert holder["client"].stream_closed is True


def test_full_mode_consumes_end_and_snapshots_artifacts():
    holder = {}
    artifact_path = "/mnt/user-data/outputs/report.md"

    def client_factory(**kwargs):
        holder["client"] = ScriptedClient(
            events=[
                ai_read_skill("systematic-literature-review"),
                tool_result(),
                event("values", {"artifacts": [{"path": artifact_path}]}),
                event("messages-tuple", {"type": "ai", "id": "m2", "content": "Final report"}),
                event("end", {"usage": {}}),
            ],
            artifacts={artifact_path: {"content": b"# Report", "mime_type": "text/markdown"}},
            **kwargs,
        )
        return holder["client"]

    result = _execute_deerflow(request(mode="full"), client_factory=client_factory)

    assert result.success is True
    assert result.final_answer == "Final report"
    assert result.trace.artifacts[0].path == artifact_path
    assert result.trace.artifacts[0].content == "# Report"
    assert holder["client"].artifact_reads == [("thread-1", artifact_path)]
    assert holder["client"].options["subagent_enabled"] is True


def test_full_mode_rejects_blank_answer_without_artifacts():
    def client_factory(**kwargs):
        return ScriptedClient(events=[event("end", {"usage": {}})], **kwargs)

    result = _execute_deerflow(
        request(mode="full"),
        client_factory=client_factory,
    )

    assert result.success is False
    assert "no final answer or artifact" in result.trace.errors[-1]


def test_stream_without_end_is_retained_as_infrastructure_failure():
    def client_factory(**kwargs):
        return ScriptedClient(
            events=[event("messages-tuple", {"type": "ai", "id": "m1", "content": "partial"})],
            **kwargs,
        )

    result = _execute_deerflow(request(mode="full"), client_factory=client_factory)

    assert result.success is False
    assert result.route_observation.completed is False
    assert "without an end event" in result.trace.errors[0]


def test_stream_exception_does_not_add_false_missing_end_error():
    def client_factory(**kwargs):
        return ScriptedClient(events=[RuntimeError("connection lost")], **kwargs)

    result = _execute_deerflow(request(mode="full"), client_factory=client_factory)

    assert result.success is False
    assert result.trace.errors == ["Stream error: connection lost"]
    assert result.route_observation.errors == ["Stream error: connection lost"]
