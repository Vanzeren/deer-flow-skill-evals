"""Layer 1 of the record/replay e2e: replay a recorded trace through the **real
gateway** with a deterministic ``ReplayChatModel`` (no API key, no network) and
assert the streamed SSE event sequence matches a committed golden.

This catches backend protocol drift: if a change alters the shape/sequence of
SSE the gateway emits for the recorded scenario, this test goes red. The replay
model serves the recorded assistant turns by input hash, so the agent graph
(write_file -> auto-title -> read_file -> final answer) reproduces offline.

Fixtures are produced by ``scripts/record_gateway.py`` +
``scripts/build_fixture_from_jsonl.py`` (manual, needs a key).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import replay_provider
from _replay_fixture import REPLAY_MODEL_BLOCK, build_config_yaml, drive_gateway, prepare_hermetic_extras

from app.gateway.app import create_app
from deerflow.evals import (
    BoundaryEventsCheck,
    NoReplayMissesCheck,
    ReplayEvalCase,
    ReplayEvalResult,
    ReplayRuntime,
    SseShapeGoldenCheck,
    run_replay_case,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "replay"
GOLDEN_FILENAME = "write_read_file.ultra.events.json"


def _golden_case(*, write_golden: bool) -> ReplayEvalCase:
    checks = [BoundaryEventsCheck(), NoReplayMissesCheck()]
    if not write_golden:
        checks.append(SseShapeGoldenCheck())
    return ReplayEvalCase(
        id="write-read-file",
        scenario="write_read_file",
        mode="ultra",
        checks=tuple(checks),
    )


def _replay_runtime() -> ReplayRuntime:
    return ReplayRuntime(
        model_block=REPLAY_MODEL_BLOCK,
        build_config_yaml=build_config_yaml,
        prepare_hermetic_extras=prepare_hermetic_extras,
        create_app=create_app,
        drive_gateway=drive_gateway,
        reset_replay_misses=replay_provider.reset_replay_misses,
        replay_misses=replay_provider.replay_misses,
    )


def _write_golden_events(events_path: Path, case: ReplayEvalCase, events: list[dict]) -> None:
    events_path.write_text(
        json.dumps(
            {"scenario": case.scenario, "mode": case.mode, "events": events},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_golden_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fixture_dir: Path = FIXTURE_DIR,
) -> tuple[ReplayEvalCase, ReplayEvalResult, Path, bool]:
    write_golden = bool(os.environ.get("DEERFLOW_WRITE_GOLDEN"))
    case = _golden_case(write_golden=write_golden)
    events_path = fixture_dir / GOLDEN_FILENAME
    result = run_replay_case(case, tmp_path=tmp_path, monkeypatch=monkeypatch, fixture_dir=fixture_dir, runtime=_replay_runtime())
    return case, result, events_path, write_golden


@pytest.mark.no_auto_user
def test_replay_write_read_file_ultra_matches_golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case, result, events_path, write_golden = _run_golden_case(tmp_path, monkeypatch)
    if write_golden:
        assert result.passed, "\n".join(f"{check.name}: {check.message} | slice={check.relevant_slice!r}" for check in result.checks if not check.passed)
        _write_golden_events(events_path, case, result.trajectory.events)
        return
    assert result.passed, "\n".join(f"{check.name}: {check.message} | slice={check.relevant_slice!r}" for check in result.checks if not check.passed)


@pytest.mark.no_auto_user
def test_replay_write_golden_mode_recreates_missing_golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture_dir = tmp_path / "fixtures" / "replay"
    fixture_dir.mkdir(parents=True)
    fixture_path = FIXTURE_DIR / "write_read_file.ultra.json"
    fixture_copy = fixture_dir / fixture_path.name
    fixture_copy.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setenv("DEERFLOW_WRITE_GOLDEN", "1")

    case, result, events_path, write_golden = _run_golden_case(tmp_path, monkeypatch, fixture_dir=fixture_dir)

    assert write_golden is True
    assert result.passed is True
    _write_golden_events(events_path, case, result.trajectory.events)
    assert json.loads(events_path.read_text(encoding="utf-8")) == {
        "scenario": case.scenario,
        "mode": case.mode,
        "events": result.trajectory.events,
    }
