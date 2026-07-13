from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# NOTE: All dataclasses below are frozen=True, which prevents field reassignment
# but does NOT make contained dict/list fields deeply immutable (e.g.
# Trajectory.events.append(...) is still possible). This is intentional for
# data-container usage; do not assume deep constness.


@dataclass(frozen=True)
class ReplayEvalCase:
    id: str
    scenario: str
    mode: str
    checks: tuple[object, ...]
    prompt: str | None = None


@dataclass(frozen=True)
class Trajectory:
    case_id: str
    scenario: str
    mode: str
    prompt: str
    context: dict[str, Any]
    events: list[dict[str, Any]]
    replay_misses: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    relevant_slice: Any = None


@dataclass(frozen=True)
class ReplayEvalResult:
    case_id: str
    passed: bool
    trajectory: Trajectory
    checks: list[CheckResult]
    summary: str
    failure_count: int


@dataclass(frozen=True)
class ReplayEvalSuiteResult:
    suite_id: str
    case_results: list[ReplayEvalResult]
    passed_count: int
    failed_count: int
    overall_passed: bool
