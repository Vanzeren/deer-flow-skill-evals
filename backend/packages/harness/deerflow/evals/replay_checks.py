from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from deerflow.evals.replay_models import CheckResult, Trajectory


class NoReplayMissesCheck:
    name = "no_replay_misses"

    def run(self, trajectory: Trajectory) -> CheckResult:
        misses = trajectory.replay_misses
        if not misses:
            return CheckResult(name=self.name, passed=True, message="no replay misses")
        return CheckResult(
            name=self.name,
            passed=False,
            message=f"replay miss ({len(misses)}): fixture is stale vs current graph",
            details={"count": len(misses)},
            relevant_slice=misses,
        )


class BoundaryEventsCheck:
    name = "boundary_events"

    def run(self, trajectory: Trajectory) -> CheckResult:
        events = trajectory.events
        if not events:
            return CheckResult(name=self.name, passed=False, message="replay produced no SSE events", relevant_slice=[])

        failures: list[str] = []
        if not isinstance(events[0], dict) or events[0].get("event") != "metadata":
            failures.append(f"first event should be metadata, got {events[0]!r}")
        if not isinstance(events[-1], dict) or events[-1].get("event") != "end":
            failures.append(f"last event should be end, got {events[-1]!r}")

        if failures:
            relevant = events[:1]
            if len(events) > 1:
                relevant = relevant + events[-1:]
            return CheckResult(
                name=self.name,
                passed=False,
                message="; ".join(failures),
                relevant_slice=relevant,
            )
        return CheckResult(name=self.name, passed=True, message="metadata/end boundaries intact")


class SseShapeGoldenCheck:
    name = "sse_shape_golden"

    def run(self, trajectory: Trajectory) -> CheckResult:
        try:
            golden_path = Path(trajectory.metadata["golden_path"])
            golden = json.loads(golden_path.read_text(encoding="utf-8"))["events"]
        except (KeyError, FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
            return CheckResult(
                name=self.name,
                passed=False,
                message=f"cannot load golden: {exc}",
                relevant_slice=str(exc),
            )

        if trajectory.events == golden:
            return CheckResult(name=self.name, passed=True, message="SSE event-shape matches golden")

        index = 0
        for got, want in zip(trajectory.events, golden):
            if got != want:
                break
            index += 1

        got_len = len(trajectory.events)
        want_len = len(golden)
        if index == min(got_len, want_len) and got_len != want_len:
            # Prefix match, but lengths differ — one list is a truncated / extended
            # version of the other. Pin divergence at the shorter list's end.
            index = min(got_len, want_len)

        got_window = trajectory.events[index : index + 3]
        want_window = golden[index : index + 3]
        return CheckResult(
            name=self.name,
            passed=False,
            message="SSE event-shape sequence drifted from the golden",
            details={
                "divergence_index": index,
                "got_count": got_len,
                "want_count": want_len,
            },
            relevant_slice={"got": got_window, "want": want_window},
        )


def run_checks(trajectory: Trajectory, checks: Iterable[object]) -> list[CheckResult]:
    return [check.run(trajectory) for check in checks]
