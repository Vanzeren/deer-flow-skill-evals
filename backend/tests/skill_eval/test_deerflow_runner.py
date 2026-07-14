import asyncio
import inspect
import os
import signal
import time
from pathlib import Path

import pytest

from skill_eval.adapters.deerflow import DeerFlowAgentRunner
from skill_eval.agent_runner import AgentRunRequest, AgentRunResult
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace


def make_request(**updates) -> AgentRunRequest:
    values = {
        "case_id": "case-1",
        "user_input": "Survey papers",
        "mode": "routing_probe",
        "model_name": "default",
        "thread_id": "thread-1",
        "timeout_seconds": 5,
    }
    values.update(updates)
    return AgentRunRequest(**values)


def child_success(connection, request_data, config):
    request = AgentRunRequest.model_validate(request_data)
    result = AgentRunResult(
        final_answer="done",
        success=True,
        thread_id=request.thread_id,
        route_observation=RouteObservation(observed_route="none", completed=True),
        trace=AgentTrace(
            input=request.user_input,
            final_answer="done",
            success=True,
            thread_id=request.thread_id,
            runtime="deerflow",
        ),
    )
    connection.send({"result": result.model_dump(mode="json")})
    connection.close()


def child_large_result(connection, request_data, config):
    request = AgentRunRequest.model_validate(request_data)
    content = "x" * 2_000_000
    result = AgentRunResult(
        final_answer=content,
        success=True,
        thread_id=request.thread_id,
        route_observation=RouteObservation(observed_route="none", completed=True),
        trace=AgentTrace(
            input=request.user_input,
            final_answer=content,
            success=True,
            thread_id=request.thread_id,
            runtime="deerflow",
        ),
    )
    connection.send({"result": result.model_dump(mode="json")})
    connection.close()


def child_result_then_sleep(connection, request_data, config):
    marker = Path(config["trace_dir"]) / "child.pid"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(os.getpid()), encoding="utf-8")
    request = AgentRunRequest.model_validate(request_data)
    result = AgentRunResult(
        final_answer="done",
        success=True,
        thread_id=request.thread_id,
        route_observation=RouteObservation(observed_route="none", completed=True),
        trace=AgentTrace(
            input=request.user_input,
            final_answer="done",
            success=True,
            thread_id=request.thread_id,
            runtime="deerflow",
        ),
    )
    connection.send({"result": result.model_dump(mode="json")})
    time.sleep(60)


def child_sleep(connection, request_data, config):
    marker = Path(config["trace_dir"]) / "child.pid"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(os.getpid()), encoding="utf-8")
    raw_trace = Path(config["trace_dir"]) / f"{request_data['thread_id']}.jsonl"
    raw_trace.write_text('{"type":"partial"}\n', encoding="utf-8")
    time.sleep(60)


def child_crash(connection, request_data, config):
    raise RuntimeError("child exploded")


def test_runner_implements_agent_runner_protocol():
    runner = DeerFlowAgentRunner()
    assert inspect.iscoroutinefunction(runner.run)


def test_request_rejects_obsolete_assertion_runner_fields():
    with pytest.raises(ValueError):
        AgentRunRequest(
            case_id="case-1",
            user_input="Survey papers",
            mode="routing_probe",
            model_name="default",
            target="hidden label",
        )


@pytest.mark.asyncio
async def test_spawned_runner_returns_serialized_result():
    result = await DeerFlowAgentRunner(child_target=child_success).run(make_request())

    assert result.success is True
    assert result.final_answer == "done"
    assert result.thread_id == "thread-1"
    assert result.route_observation.observed_route == "none"


@pytest.mark.asyncio
async def test_spawned_runner_drains_large_payload_before_joining_child():
    result = await DeerFlowAgentRunner(child_target=child_large_result).run(make_request(timeout_seconds=2))

    assert result.success is True
    assert len(result.final_answer) == 2_000_000


@pytest.mark.asyncio
async def test_spawned_runner_rejects_payload_when_child_does_not_exit(tmp_path):
    result = await DeerFlowAgentRunner(child_target=child_result_then_sleep).run(make_request(timeout_seconds=3, trace_dir=str(tmp_path)))

    assert result.success is False
    assert "did not exit after returning a result" in result.trace.errors[0]
    pid = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_spawned_runner_terminates_timeout_child(tmp_path):
    request = make_request(timeout_seconds=2, trace_dir=str(tmp_path))

    result = await DeerFlowAgentRunner(child_target=child_sleep).run(request)

    assert result.success is False
    assert result.route_observation.completed is False
    assert "timed out after 2s" in result.trace.errors[0]
    assert result.trace.raw_trace_ref == str(tmp_path / "thread-1.jsonl")
    assert (tmp_path / "thread-1.jsonl").exists()
    pid = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_spawned_runner_retains_child_crash():
    result = await DeerFlowAgentRunner(child_target=child_crash).run(make_request())

    assert result.success is False
    assert result.route_observation.completed is False
    assert "exited without a result" in result.trace.errors[0]
    assert result.thread_id == "thread-1"


@pytest.mark.asyncio
async def test_spawned_runner_terminates_child_when_cancelled(tmp_path):
    runner_task = asyncio.create_task(DeerFlowAgentRunner(child_target=child_sleep).run(make_request(timeout_seconds=2, trace_dir=str(tmp_path))))
    marker = tmp_path / "child.pid"
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.02)
    assert marker.exists(), "spawned child did not start"

    runner_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner_task

    pid = int(marker.read_text(encoding="utf-8"))
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
