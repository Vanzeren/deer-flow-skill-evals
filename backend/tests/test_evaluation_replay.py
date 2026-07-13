import json
from dataclasses import replace
from pathlib import Path

import pytest
import replay_provider
from _replay_fixture import REPLAY_MODEL_BLOCK, build_config_yaml, drive_gateway, prepare_hermetic_extras

from app.gateway.app import create_app
from deerflow.evals import (
    BoundaryEventsCheck,
    NoReplayMissesCheck,
    ReplayEvalCase,
    SseShapeGoldenCheck,
)
from deerflow.evals.replay_checks import run_checks
from deerflow.evals.replay_models import Trajectory
from deerflow.evals.replay_report import result_to_json, suite_summary
from deerflow.evals.replay_runner import ReplayRuntime, run_replay_case, run_replay_suite


def _trajectory(*, events=None, misses=None, golden_path="/tmp/golden.json"):
    return Trajectory(
        case_id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        prompt="Create a note and read it back.",
        context={"enable_thinking": True},
        events=events
        if events is not None
        else [
            {"event": "metadata", "keys": ["run_id"]},
            {"event": "end", "keys": ["status"]},
        ],
        replay_misses=misses or [],
        metadata={
            "fixture_path": "/tmp/fixture.json",
            "golden_path": golden_path,
            "generated_at": "2026-07-11T00:00:00Z",
        },
    )


def test_case_defaults_checks_tuple():
    case = ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=(BoundaryEventsCheck(), NoReplayMissesCheck()),
    )
    assert case.id == "write-read-file"
    assert [type(check).__name__ for check in case.checks] == [
        "BoundaryEventsCheck",
        "NoReplayMissesCheck",
    ]


def test_boundary_events_check_passes_for_metadata_to_end():
    result = BoundaryEventsCheck().run(_trajectory())
    assert result.passed is True
    assert result.name == "boundary_events"


def test_boundary_events_check_reports_empty_events():
    result = BoundaryEventsCheck().run(_trajectory(events=[]))
    assert result.passed is False
    assert result.name == "boundary_events"
    assert "replay produced no SSE events" in result.message
    assert result.relevant_slice == []


def test_boundary_events_check_reports_missing_end():
    result = BoundaryEventsCheck().run(_trajectory(events=[{"event": "metadata", "keys": ["run_id"]}]))
    assert result.passed is False
    assert result.relevant_slice == [{"event": "metadata", "keys": ["run_id"]}]
    assert "last event should be end" in result.message


def test_boundary_events_check_reports_both_ends_broken():
    result = BoundaryEventsCheck().run(_trajectory(events=[{"event": "messages", "keys": ["chunk"]}]))
    assert result.passed is False
    assert "first event should be metadata" in result.message
    assert "last event should be end" in result.message
    assert result.relevant_slice == [{"event": "messages", "keys": ["chunk"]}]


def test_no_replay_misses_check_reports_misses():
    result = NoReplayMissesCheck().run(_trajectory(misses=["abc", "def"]))
    assert result.passed is False
    assert result.details["count"] == 2
    assert result.relevant_slice == ["abc", "def"]


def test_no_replay_misses_check_passes_for_empty():
    result = NoReplayMissesCheck().run(_trajectory(misses=[]))
    assert result.passed is True
    assert result.name == "no_replay_misses"
    assert result.message == "no replay misses"


def test_sse_shape_golden_check_reports_first_divergence(tmp_path):
    golden = tmp_path / "write_read_file.ultra.events.json"
    golden.write_text('{"scenario":"write_read_file","mode":"ultra","events":[{"event":"metadata","keys":["run_id"]},{"event":"messages","keys":["chunk"]},{"event":"end","keys":["status"]}]}')
    trajectory = _trajectory(
        events=[
            {"event": "metadata", "keys": ["run_id"]},
            {"event": "updates", "keys": ["delta"]},
            {"event": "end", "keys": ["status"]},
        ],
        golden_path=str(golden),
    )
    result = SseShapeGoldenCheck().run(trajectory)
    assert result.passed is False
    assert result.details["divergence_index"] == 1
    assert result.relevant_slice["got"][0]["event"] == "updates"
    assert result.relevant_slice["want"][0]["event"] == "messages"


def test_sse_shape_golden_check_reports_length_mismatch_after_prefix_match(tmp_path):
    golden = tmp_path / "write_read_file.ultra.events.json"
    golden.write_text('{"scenario":"write_read_file","mode":"ultra","events":[{"event":"metadata","keys":["run_id"]},{"event":"messages","keys":["chunk"]},{"event":"end","keys":["status"]}]}')
    trajectory = _trajectory(
        events=[
            {"event": "metadata", "keys": ["run_id"]},
            {"event": "messages", "keys": ["chunk"]},
        ],
        golden_path=str(golden),
    )
    result = SseShapeGoldenCheck().run(trajectory)
    assert result.passed is False
    assert result.details["got_count"] == 2
    assert result.details["want_count"] == 3
    assert result.details["divergence_index"] == 2
    assert result.relevant_slice["got"] == []
    assert result.relevant_slice["want"][0]["event"] == "end"


def test_sse_shape_golden_check_passes_for_exact_match(tmp_path):
    golden = tmp_path / "write_read_file.ultra.events.json"
    events = [
        {"event": "metadata", "keys": ["run_id"]},
        {"event": "end", "keys": ["status"]},
    ]
    golden.write_text('{"scenario":"write_read_file","mode":"ultra","events":[{"event":"metadata","keys":["run_id"]},{"event":"end","keys":["status"]}]}')
    trajectory = _trajectory(events=events, golden_path=str(golden))
    result = SseShapeGoldenCheck().run(trajectory)
    assert result.passed is True
    assert result.name == "sse_shape_golden"
    assert result.message == "SSE event-shape matches golden"


def test_sse_shape_golden_check_reports_missing_golden_file():
    trajectory = _trajectory(golden_path="/tmp/nonexistent_file.json")
    result = SseShapeGoldenCheck().run(trajectory)
    assert result.passed is False
    assert "cannot load golden" in result.message


def test_sse_shape_golden_check_reports_corrupted_golden_json(tmp_path):
    golden = tmp_path / "corrupted.events.json"
    golden.write_text("not valid json{{{")
    trajectory = _trajectory(golden_path=str(golden))
    result = SseShapeGoldenCheck().run(trajectory)
    assert result.passed is False
    assert "cannot load golden" in result.message


def test_run_checks_preserves_order():
    results = run_checks(_trajectory(), [BoundaryEventsCheck(), NoReplayMissesCheck()])
    assert [result.name for result in results] == ["boundary_events", "no_replay_misses"]
    assert all(result.passed for result in results)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "replay"


def _replay_runtime() -> ReplayRuntime:
    return ReplayRuntime(
        model_block=REPLAY_MODEL_BLOCK,
        build_config_yaml=build_config_yaml,
        prepare_hermetic_extras=prepare_hermetic_extras,
        drive_gateway=drive_gateway,
        create_app=create_app,
        reset_replay_misses=replay_provider.reset_replay_misses,
        replay_misses=replay_provider.replay_misses,
    )


@pytest.mark.no_auto_user
def test_run_replay_case_builds_structured_result(tmp_path, monkeypatch):
    case = ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=(BoundaryEventsCheck(), NoReplayMissesCheck(), SseShapeGoldenCheck()),
    )
    result = run_replay_case(case, tmp_path=tmp_path, monkeypatch=monkeypatch, fixture_dir=FIXTURE_DIR, runtime=_replay_runtime())
    assert result.case_id == "write-read-file"
    assert result.passed is True
    assert result.failure_count == 0
    assert result.trajectory.scenario == "write_read_file"
    assert result.trajectory.events[0]["event"] == "metadata"
    assert result.trajectory.events[-1]["event"] == "end"
    assert "keys" in result.trajectory.events[0]
    assert "data_keys" not in result.trajectory.events[0]
    assert set(result.trajectory.events[0]) == set(_trajectory().events[0])


@pytest.mark.no_auto_user
def test_run_replay_case_resets_misses_between_calls(tmp_path, monkeypatch):
    misses: list[str] = []
    drive_count = 0

    def drive_with_first_call_miss(_app, *, prompt, context):
        nonlocal drive_count
        drive_count += 1
        if drive_count == 1:
            misses.append("first-call-miss")
        return [
            {"event": "metadata", "keys": ["run_id"]},
            {"event": "end", "keys": ["status"]},
        ]

    runtime = replace(
        _replay_runtime(),
        create_app=object,
        drive_gateway=drive_with_first_call_miss,
        reset_replay_misses=misses.clear,
        replay_misses=lambda: list(misses),
    )
    case = ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=(NoReplayMissesCheck(),),
    )

    first_tmp_path = tmp_path / "first"
    second_tmp_path = tmp_path / "second"
    first_tmp_path.mkdir()
    second_tmp_path.mkdir()

    first = run_replay_case(case, tmp_path=first_tmp_path, monkeypatch=monkeypatch, fixture_dir=FIXTURE_DIR, runtime=runtime)
    second = run_replay_case(case, tmp_path=second_tmp_path, monkeypatch=monkeypatch, fixture_dir=FIXTURE_DIR, runtime=runtime)

    assert first.trajectory.replay_misses == ["first-call-miss"]
    assert second.trajectory.replay_misses == []


@pytest.mark.no_auto_user
def test_run_replay_suite_aggregates_counts(tmp_path_factory, monkeypatch):
    case = ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=(BoundaryEventsCheck(), NoReplayMissesCheck()),
    )
    result = run_replay_suite(
        "replay_golden_smoke",
        [case],
        tmp_path_factory=tmp_path_factory,
        monkeypatch=monkeypatch,
        fixture_dir=FIXTURE_DIR,
        runtime=_replay_runtime(),
    )
    assert result.suite_id == "replay_golden_smoke"
    assert result.passed_count == 1
    assert result.failed_count == 0
    assert result.overall_passed is True


@pytest.mark.no_auto_user
def test_sse_shape_golden_check_reports_missing_golden_path():
    trajectory = Trajectory(
        case_id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        prompt="test",
        context={},
        events=[],
        replay_misses=[],
        metadata={"fixture_path": "/tmp/fixture.json", "generated_at": "2026-07-11T00:00:00Z"},
    )
    result = SseShapeGoldenCheck().run(trajectory)
    assert result.passed is False
    assert "cannot load golden" in result.message


@pytest.mark.no_auto_user
def test_run_replay_suite_reports_failure_counts(tmp_path_factory, monkeypatch):
    misses: list[str] = []
    failing_events = [
        {"event": "messages", "keys": ["chunk"]},
    ]

    def failing_drive(_app, *, prompt, context):
        misses.append("stale-hash-abc123")
        return failing_events

    runtime = replace(
        _replay_runtime(),
        create_app=object,
        drive_gateway=failing_drive,
        reset_replay_misses=misses.clear,
        replay_misses=lambda: list(misses),
    )
    case = ReplayEvalCase(
        id="failing-case",
        scenario="write_read_file",
        mode="ultra",
        checks=(BoundaryEventsCheck(), NoReplayMissesCheck()),
    )
    result = run_replay_suite(
        "failing_suite",
        [case],
        tmp_path_factory=tmp_path_factory,
        monkeypatch=monkeypatch,
        fixture_dir=FIXTURE_DIR,
        runtime=runtime,
    )
    assert result.passed_count == 0
    assert result.failed_count == 1
    assert result.overall_passed is False


@pytest.mark.no_auto_user
def test_result_to_json_is_machine_readable(tmp_path, monkeypatch):
    case = ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=(BoundaryEventsCheck(), NoReplayMissesCheck()),
    )
    result = run_replay_case(case, tmp_path=tmp_path, monkeypatch=monkeypatch, fixture_dir=FIXTURE_DIR, runtime=_replay_runtime())
    payload = result_to_json(result)
    parsed = json.loads(payload)
    assert parsed["case_id"] == "write-read-file"
    assert parsed["trajectory"]["scenario"] == "write_read_file"
    assert parsed["checks"][0]["name"] == "boundary_events"


@pytest.mark.no_auto_user
def test_kernel_result_matches_existing_fixture_paths(tmp_path, monkeypatch):
    case = ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=(BoundaryEventsCheck(), NoReplayMissesCheck(), SseShapeGoldenCheck()),
    )
    result = run_replay_case(case, tmp_path=tmp_path, monkeypatch=monkeypatch, fixture_dir=FIXTURE_DIR, runtime=_replay_runtime())
    assert Path(result.trajectory.metadata["fixture_path"]).name == "write_read_file.ultra.json"
    assert Path(result.trajectory.metadata["golden_path"]).name == "write_read_file.ultra.events.json"


def test_suite_summary_mentions_failures():
    from deerflow.evals.replay_models import CheckResult, ReplayEvalResult, ReplayEvalSuiteResult

    trajectory = _trajectory()
    case_result = ReplayEvalResult(
        case_id="write-read-file",
        passed=False,
        trajectory=trajectory,
        checks=[CheckResult(name="boundary_events", passed=False, message="boom")],
        summary="1 check failed",
        failure_count=1,
    )
    suite = ReplayEvalSuiteResult(
        suite_id="replay_golden_smoke",
        case_results=[case_result],
        passed_count=0,
        failed_count=1,
        overall_passed=False,
    )
    assert "replay_golden_smoke: 0 passed, 1 failed" in suite_summary(suite)
