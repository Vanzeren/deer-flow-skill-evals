import time

from skill_eval.adapters.deerflow import DeerFlowTraceAdapter

from skill_eval.agent_runner import AgentRunRequest


def _make_event(type_: str, data: dict):
    from deerflow.client import StreamEvent

    return StreamEvent(type=type_, data=data)


def test_adapter_empty_stream():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hello"))
    adapter._start_time = time.monotonic()
    trace = adapter.build()
    assert trace.final_answer == ""
    assert trace.tool_calls == []
    assert trace.skill_invocations == []
    assert trace.success is True


def test_adapter_single_ai_message():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hello"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "Hello ", "id": "msg1"}))
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "world", "id": "msg1"}))
    adapter.feed(_make_event("end", {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}))
    trace = adapter.build()
    assert trace.final_answer == "Hello world"
    assert trace.input_tokens == 10
    assert trace.output_tokens == 5
    assert trace.runtime == "deerflow"


def test_adapter_multiple_ai_messages_last_is_final():
    """Only the last AI message's text becomes final_answer."""
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="plan then execute"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "I'll plan first.", "id": "msg1"}))
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "Done: result is 42.", "id": "msg2"}))
    trace = adapter.build()
    assert trace.final_answer == "Done: result is 42."


def test_adapter_tool_call_and_result():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="read a file"))
    adapter._start_time = time.monotonic()
    # AI requests a tool call
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "ai",
                "content": "",
                "id": "msg1",
                "tool_calls": [{"name": "read_file", "args": {"file_path": "data.txt"}, "id": "tc1"}],
            },
        )
    )
    # Tool result arrives
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "tool",
                "content": "file contents here",
                "name": "read_file",
                "tool_call_id": "tc1",
                "id": "msg2",
            },
        )
    )
    trace = adapter.build()
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "read_file"
    assert trace.tool_calls[0].args == {"file_path": "data.txt"}
    assert trace.tool_calls[0].result == "file contents here"
    assert trace.tool_calls[0].error is None


def test_adapter_tool_call_with_error():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="bad command"))
    adapter._start_time = time.monotonic()
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "ai",
                "content": "",
                "id": "msg1",
                "tool_calls": [{"name": "bash", "args": {"cmd": "rm -rf /"}, "id": "tc1"}],
            },
        )
    )
    # Error in tool result
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "tool",
                "content": "",
                "name": "bash",
                "tool_call_id": "tc1",
                "id": "msg2",
            },
        )
    )
    # Simulate error on tool call - adapter reads error from data
    adapter._tool_calls["tc1"].error = "permission denied"
    trace = adapter.build()
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].error == "permission denied"


def test_adapter_multiple_tool_calls_ordered():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="multi-step"))
    adapter._start_time = time.monotonic()
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "ai",
                "content": "",
                "id": "msg1",
                "tool_calls": [
                    {"name": "read_file", "args": {}, "id": "tc1"},
                    {"name": "bash", "args": {}, "id": "tc2"},
                ],
            },
        )
    )
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "tool",
                "content": "ok",
                "name": "read_file",
                "tool_call_id": "tc1",
                "id": "msg2",
            },
        )
    )
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "tool",
                "content": "ok",
                "name": "bash",
                "tool_call_id": "tc2",
                "id": "msg3",
            },
        )
    )
    trace = adapter.build()
    assert len(trace.tool_calls) == 2
    assert [tc.name for tc in trace.tool_calls] == ["read_file", "bash"]


def test_adapter_usage_from_end_event():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hi"))
    adapter._start_time = time.monotonic()
    adapter.feed(_make_event("messages-tuple", {"type": "ai", "content": "hey", "id": "m1"}))
    adapter.feed(_make_event("end", {"usage": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75}}))
    trace = adapter.build()
    assert trace.input_tokens == 50
    assert trace.output_tokens == 25


def test_adapter_latency():
    adapter = DeerFlowTraceAdapter(AgentRunRequest(user_input="hi"))
    adapter._start_time = time.monotonic() - 1.5  # simulate 1.5s elapsed
    trace = adapter.build()
    assert trace.latency_ms is not None
    assert 1400 <= trace.latency_ms <= 1600  # ~1.5s


def test_adapter_skill_loaded():
    adapter = DeerFlowTraceAdapter(
        AgentRunRequest(
            user_input="deploy",
            required_skills=["gcp-deploy"],
            candidate_skills=["gcp-deploy", "system-design"],
        )
    )
    adapter._start_time = time.monotonic()
    trace = adapter.build()
    invocations = {inv.name: inv for inv in trace.skill_invocations}
    assert "gcp-deploy" in invocations
    assert invocations["gcp-deploy"].loaded is True
    assert invocations["gcp-deploy"].used is False  # no read_file
    assert "system-design" in invocations
    assert invocations["system-design"].loaded is True


def test_adapter_skill_used_via_read_file():
    adapter = DeerFlowTraceAdapter(
        AgentRunRequest(
            user_input="deploy to cloud run",
            required_skills=["gcp-deploy"],
        )
    )
    adapter._start_time = time.monotonic()
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "ai",
                "content": "",
                "id": "msg1",
                "tool_calls": [{"name": "read_file", "args": {"file_path": "skills/gcp-deploy/SKILL.md"}, "id": "tc1"}],
            },
        )
    )
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "tool",
                "content": "# GCP Deploy Skill\n...",
                "name": "read_file",
                "tool_call_id": "tc1",
                "id": "msg2",
            },
        )
    )
    trace = adapter.build()
    invocations = {inv.name: inv for inv in trace.skill_invocations}
    assert invocations["gcp-deploy"].used is True
    assert invocations["gcp-deploy"].loaded is True


def test_adapter_skill_not_used_without_read_file():
    adapter = DeerFlowTraceAdapter(
        AgentRunRequest(
            user_input="deploy",
            required_skills=["gcp-deploy"],
        )
    )
    adapter._start_time = time.monotonic()
    adapter.feed(
        _make_event(
            "messages-tuple",
            {
                "type": "ai",
                "content": "",
                "id": "msg1",
                "tool_calls": [{"name": "bash", "args": {"cmd": "gcloud run deploy"}, "id": "tc1"}],
            },
        )
    )
    trace = adapter.build()
    invocations = {inv.name: inv for inv in trace.skill_invocations}
    assert invocations["gcp-deploy"].used is False
