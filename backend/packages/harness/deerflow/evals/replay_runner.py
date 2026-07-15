from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from deerflow.evals.replay_checks import run_checks
from deerflow.evals.replay_models import (
    ReplayEvalCase,
    ReplayEvalResult,
    ReplayEvalSuiteResult,
    Trajectory,
)


class ReplayMonkeyPatch(Protocol):
    def setenv(self, name: str, value: str, prepend: str | None = None) -> None: ...

    def setattr(self, target: object, name: str, value: object, raising: bool = True) -> None: ...


class ReplayTmpPathFactory(Protocol):
    def mktemp(self, basename: str, numbered: bool = True) -> Path: ...


@dataclass(frozen=True)
class ReplayRuntime:
    model_block: str
    build_config_yaml: Callable[..., str]
    prepare_hermetic_extras: Callable[..., str]
    drive_gateway: Callable[..., list[dict[str, Any]]]
    create_app: Callable[[], object]
    reset_replay_misses: Callable[[], None]
    replay_misses: Callable[[], list[str]]


def _reset_process_singletons(monkeypatch: ReplayMonkeyPatch) -> None:
    from deerflow.config import app_config as app_config_module
    from deerflow.config import paths as paths_module
    from deerflow.persistence import engine as engine_module

    for module, attr in (
        (app_config_module, "_app_config"),
        (app_config_module, "_app_config_path"),
        (app_config_module, "_app_config_mtime"),
        (paths_module, "_paths"),
        (engine_module, "_engine"),
        (engine_module, "_session_factory"),
    ):
        monkeypatch.setattr(module, attr, None, raising=False)


def run_replay_case(
    case: ReplayEvalCase,
    *,
    tmp_path: Path,
    monkeypatch: ReplayMonkeyPatch,
    fixture_dir: Path,
    runtime: ReplayRuntime,
) -> ReplayEvalResult:
    from deerflow.config import app_config as app_config_module

    fixture_path = fixture_dir / f"{case.scenario}.{case.mode}.json"
    events_path = fixture_dir / f"{case.scenario}.{case.mode}.events.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    home = tmp_path / case.id
    home.mkdir()
    monkeypatch.setenv("DEER_FLOW_HOME", str(home))
    monkeypatch.setenv("DEERFLOW_REPLAY_FIXTURE", str(fixture_path))

    cfg_path = tmp_path / f"{case.id}.config.yaml"
    cfg_path.write_text(runtime.build_config_yaml(model_block=runtime.model_block, home=home), encoding="utf-8")
    monkeypatch.setenv("DEER_FLOW_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH", str(runtime.prepare_hermetic_extras(home)))

    _reset_process_singletons(monkeypatch)
    cfg = app_config_module.get_app_config()
    cfg.database.sqlite_dir = str(home / "db")

    runtime.reset_replay_misses()
    events = runtime.drive_gateway(runtime.create_app(), prompt=fixture["prompt"], context=fixture["context"])
    trajectory = Trajectory(
        case_id=case.id,
        scenario=case.scenario,
        mode=case.mode,
        prompt=fixture["prompt"],
        context=fixture["context"],
        events=events,
        replay_misses=runtime.replay_misses(),
        metadata={
            "fixture_path": str(fixture_path),
            "golden_path": str(events_path),
            "provenance": {
                "generated_at": datetime.now(UTC).isoformat(),
            },
        },
    )
    check_results = run_checks(trajectory, case.checks)
    failure_count = sum(1 for result in check_results if not result.passed)
    summary = "all replay checks passed" if failure_count == 0 else f"{failure_count} replay checks failed"
    return ReplayEvalResult(
        case_id=case.id,
        passed=failure_count == 0,
        trajectory=trajectory,
        checks=check_results,
        summary=summary,
        failure_count=failure_count,
    )


def run_replay_suite(
    suite_id: str,
    cases: list[ReplayEvalCase],
    *,
    tmp_path_factory: ReplayTmpPathFactory,
    monkeypatch: ReplayMonkeyPatch,
    fixture_dir: Path,
    runtime: ReplayRuntime,
) -> ReplayEvalSuiteResult:
    case_results = []
    for case in cases:
        case_tmp = tmp_path_factory.mktemp(f"replay-{case.id}")
        case_results.append(run_replay_case(case, tmp_path=case_tmp, monkeypatch=monkeypatch, fixture_dir=fixture_dir, runtime=runtime))
    passed_count = sum(1 for result in case_results if result.passed)
    failed_count = len(case_results) - passed_count
    return ReplayEvalSuiteResult(
        suite_id=suite_id,
        case_results=case_results,
        passed_count=passed_count,
        failed_count=failed_count,
        overall_passed=failed_count == 0,
    )
