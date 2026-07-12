from __future__ import annotations

import json
from dataclasses import asdict

from deerflow.evals.replay_models import ReplayEvalResult, ReplayEvalSuiteResult


def result_to_json(result: ReplayEvalResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)


def suite_summary(result: ReplayEvalSuiteResult) -> str:
    status = "PASS" if result.overall_passed else "FAIL"
    return f"{result.suite_id}: {result.passed_count} passed, {result.failed_count} failed ({status})"
