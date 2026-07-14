from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from inspect_ai.log import EvalLog
from inspect_ai.scorer import NOANSWER
from pydantic import BaseModel, Field

from skill_eval.case_schema import CANDIDATE_SKILLS, RouteLabel, RoutingCase
from skill_eval.judge import QualityJudgment
from skill_eval.routing import RouteEvidence

type ObservedRoute = RouteLabel | Literal["ambiguous"]

_EXPECTED_LABELS: tuple[RouteLabel, ...] = (*CANDIDATE_SKILLS, "none")
_OBSERVED_LABELS: tuple[ObservedRoute, ...] = (*_EXPECTED_LABELS, "ambiguous")


class RoutingEpochResult(BaseModel):
    case_id: str
    epoch: int
    expected_route: RouteLabel
    observed_route: ObservedRoute | None = None
    infrastructure_error: str | None = None
    evidence: list[RouteEvidence] = Field(default_factory=list)
    log_location: str


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class RoutingMetrics(BaseModel):
    planned_runs: int
    valid_runs: int
    valid_run_rate: float
    confusion: dict[str, dict[str, int]]
    per_class: dict[str, ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_cases: int
    stable_cases: int
    stability_rate: float
    results: list[RoutingEpochResult]


def extract_routing_results(log: EvalLog) -> list[RoutingEpochResult]:
    results: list[RoutingEpochResult] = []
    log_location = str(getattr(log, "location", ""))
    for sample in log.samples or []:
        scores = sample.scores or {}
        score = scores.get("routing_scorer")
        if score is None:
            raise ValueError(f"Sample {sample.id} has no routing_scorer output")
        try:
            case = RoutingCase.model_validate((sample.metadata or {})["case"])
        except Exception as exc:
            raise ValueError(f"Sample {sample.id} has invalid routing case metadata: {exc}") from exc

        metadata = score.metadata or {}
        observation = metadata.get("route_observation") or {}
        evidence = observation.get("evidence") or []
        if score.value == NOANSWER:
            infrastructure_error = str(metadata.get("infrastructure_error") or score.explanation or "routing scorer returned NOANSWER")
            observed_route = None
        else:
            infrastructure_error = None
            observed_route = metadata.get("observed_route") or observation.get("observed_route")
            if observed_route is None:
                raise ValueError(f"Sample {sample.id} has no observed route in routing_scorer metadata")

        results.append(
            RoutingEpochResult(
                case_id=str(metadata.get("case_id") or case.id),
                epoch=sample.epoch,
                expected_route=metadata.get("expected_route") or case.expected_route,
                observed_route=observed_route,
                infrastructure_error=infrastructure_error,
                evidence=evidence,
                log_location=log_location,
            )
        )
    return results


def summarize_routing(
    results: Sequence[RoutingEpochResult],
    *,
    planned_runs: int,
) -> RoutingMetrics:
    confusion = {expected: {observed: 0 for observed in _OBSERVED_LABELS} for expected in _EXPECTED_LABELS}
    valid_results = [result for result in results if result.infrastructure_error is None]
    for result in valid_results:
        if result.observed_route is None:
            raise ValueError(f"Valid routing result {result.case_id} epoch {result.epoch} has no observed route")
        confusion[result.expected_route][result.observed_route] += 1

    per_class: dict[str, ClassMetrics] = {}
    for label in _EXPECTED_LABELS:
        true_positive = confusion[label][label]
        predicted = sum(confusion[expected][label] for expected in _EXPECTED_LABELS)
        actual = sum(confusion[label].values())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=actual,
        )

    grouped: dict[str, list[RoutingEpochResult]] = defaultdict(list)
    for result in results:
        grouped[result.case_id].append(result)
    expected_epochs = planned_runs // len(grouped) if grouped and planned_runs % len(grouped) == 0 else 0
    stable_cases = sum(
        len(case_results) == expected_epochs and expected_epochs > 0 and all(result.infrastructure_error is None for result in case_results) and len({result.observed_route for result in case_results}) == 1
        for case_results in grouped.values()
    )
    valid_runs = len(valid_results)
    class_values = list(per_class.values())
    total_cases = len(grouped)
    return RoutingMetrics(
        planned_runs=planned_runs,
        valid_runs=valid_runs,
        valid_run_rate=valid_runs / planned_runs if planned_runs else 0.0,
        confusion=confusion,
        per_class=per_class,
        macro_precision=sum(metric.precision for metric in class_values) / len(class_values),
        macro_recall=sum(metric.recall for metric in class_values) / len(class_values),
        macro_f1=sum(metric.f1 for metric in class_values) / len(class_values),
        total_cases=total_cases,
        stable_cases=stable_cases,
        stability_rate=stable_cases / total_cases if total_cases else 0.0,
        results=list(results),
    )


def routing_acceptance(metrics: RoutingMetrics) -> bool:
    return metrics.valid_run_rate >= 0.95 and metrics.macro_precision >= 0.80 and metrics.macro_recall >= 0.80


class RunIdentity(BaseModel):
    agent_model: str
    judge_model: str
    inspect_ai_version: str
    deerflow_version: str
    case_file_sha256: str
    skill_file_sha256: dict[str, str]
    runtime_config: dict[str, str]
    started_at: datetime
    ended_at: datetime
    inspect_logs: list[str]


class QualityCaseResult(BaseModel):
    case_id: str
    observed_route: ObservedRoute | None = None
    judgment: QualityJudgment | None = None
    judge_failure: str | None = None
    infrastructure_error: str | None = None
    quality_passed: bool
    label_review_needed: bool
    evidence_log: str


class AcceptanceCheck(BaseModel):
    name: str
    actual: float | int
    threshold: str
    passed: bool


class PocSummary(BaseModel):
    schema_version: Literal["deerflow.agent-routing-poc.v1"] = "deerflow.agent-routing-poc.v1"
    run_id: str
    mode: Literal["smoke", "full"]
    identity: RunIdentity
    routing: RoutingMetrics
    quality_results: list[QualityCaseResult]
    quality_passed_cases: int
    judge_failures: int
    infrastructure_failures: int
    acceptance: list[AcceptanceCheck]
    errors: list[str] = Field(default_factory=list)


def extract_quality_results(log: EvalLog) -> list[QualityCaseResult]:
    results: list[QualityCaseResult] = []
    log_location = str(getattr(log, "location", ""))
    for sample in log.samples or []:
        scores = sample.scores or {}
        routing_score = scores.get("routing_scorer")
        quality_score = scores.get("quality_judge_scorer")
        if routing_score is None:
            raise ValueError(f"Quality sample {sample.id} has no routing_scorer output")
        if quality_score is None:
            raise ValueError(f"Quality sample {sample.id} has no quality_judge_scorer output")
        routing_metadata = routing_score.metadata or {}
        quality_metadata = quality_score.metadata or {}
        observed_route = routing_metadata.get("observed_route")
        infrastructure_error = quality_metadata.get("infrastructure_error")
        judge_failure = quality_metadata.get("judge_failure")
        if infrastructure_error:
            judgment = None
            judge_failure = None
        elif quality_score.value == NOANSWER or judge_failure:
            judgment = None
            judge_failure = str(judge_failure or quality_score.explanation or "quality judge returned NOANSWER")
        else:
            try:
                judgment = QualityJudgment.model_validate(quality_metadata["quality_judgment"])
            except Exception as exc:
                raise ValueError(f"Quality sample {sample.id} has invalid judgment: {exc}") from exc
        results.append(
            QualityCaseResult(
                case_id=str(sample.id),
                observed_route=observed_route,
                judgment=judgment,
                judge_failure=judge_failure,
                infrastructure_error=str(infrastructure_error) if infrastructure_error else None,
                quality_passed=bool(quality_metadata.get("quality_passed", False)),
                label_review_needed=bool(quality_metadata.get("label_review_needed", False)),
                evidence_log=log_location,
            )
        )
    return results


def render_poc_markdown(summary: PocSummary) -> str:
    identity = summary.identity
    lines = [
        f"# DeerFlow Agent Routing POC — {summary.run_id}",
        "",
        f"- Schema: `{summary.schema_version}`",
        f"- Mode: `{summary.mode}`",
        "",
        "## Run identity",
        "",
        f"- Agent model: `{identity.agent_model}`",
        f"- Judge model: `{identity.judge_model}`",
        f"- Inspect AI: `{identity.inspect_ai_version}`",
        f"- DeerFlow: `{identity.deerflow_version}`",
        f"- Case SHA-256: `{identity.case_file_sha256}`",
    ]
    for skill, digest in sorted(identity.skill_file_sha256.items()):
        lines.append(f"- `{skill}` SHA-256: `{digest}`")
    for key, value in sorted(identity.runtime_config.items()):
        lines.append(f"- Runtime `{key}`: `{value}`")
    lines.extend(
        [
            f"- Started: `{identity.started_at.isoformat()}`",
            f"- Ended: `{identity.ended_at.isoformat()}`",
            f"- Inspect logs: {', '.join(f'`{path}`' for path in identity.inspect_logs) or 'none'}",
            f"- Agent execution failures: {summary.infrastructure_failures}",
            f"- Judge failures: {summary.judge_failures}",
            "",
            "## Confusion matrix",
            "",
            "| Expected \\\\ Observed | systematic-literature-review | academic-paper-review | none | ambiguous |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for expected in _EXPECTED_LABELS:
        row = summary.routing.confusion.get(expected, {})
        lines.append(f"| {expected} | " + " | ".join(str(row.get(observed, 0)) for observed in _OBSERVED_LABELS) + " |")
    lines.extend(
        [
            "",
            "## Per-class routing metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in _EXPECTED_LABELS:
        metric = summary.routing.per_class[label]
        lines.append(f"| {label} | {metric.precision:.3f} | {metric.recall:.3f} | {metric.f1:.3f} | {metric.support} |")
    lines.extend(
        [
            f"| **Macro** | **{summary.routing.macro_precision:.3f}** | **{summary.routing.macro_recall:.3f}** | **{summary.routing.macro_f1:.3f}** | — |",
            "",
            f"- Valid runs: {summary.routing.valid_runs}/{summary.routing.planned_runs} ({summary.routing.valid_run_rate:.1%})",
            f"- Stable cases: {summary.routing.stable_cases}/{summary.routing.total_cases} ({summary.routing.stability_rate:.1%})",
            "",
            "## Failed or unstable routing cases",
            "",
        ]
    )
    failures = [result for result in summary.routing.results if result.infrastructure_error or result.observed_route != result.expected_route]
    if failures:
        for result in failures:
            evidence_ids = ", ".join(item.id for item in result.evidence) or "none"
            lines.append(
                f"- `{result.case_id}` epoch {result.epoch}: expected `{result.expected_route}`, observed `{result.observed_route}`, error `{result.infrastructure_error or 'none'}`, evidence {evidence_ids}, log `{result.log_location}`"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Quality judgments", ""])
    if summary.quality_results:
        for result in summary.quality_results:
            if result.infrastructure_error:
                lines.append(f"- `{result.case_id}`: agent execution failure `{result.infrastructure_error}`, log `{result.evidence_log}`")
                continue
            if result.judgment is None:
                lines.append(f"- `{result.case_id}`: judge failure `{result.judge_failure}`, log `{result.evidence_log}`")
                continue
            judgment = result.judgment
            lines.extend(
                [
                    f"### {result.case_id}",
                    "",
                    f"- Observed route: `{result.observed_route}`",
                    f"- Recommended route: `{judgment.recommended_route}`",
                    f"- Scores route/process/output/overall: {judgment.route_quality}/{judgment.process_quality}/{judgment.output_quality}/{judgment.overall_quality}",
                    f"- Fatal error: `{judgment.fatal_error}`",
                    f"- Quality passed: `{result.quality_passed}`",
                    f"- Label review needed: `{result.label_review_needed}`",
                    f"- Reasons: {'; '.join(judgment.reasons)}",
                    f"- Evidence: {', '.join(judgment.evidence)}",
                    f"- Log: `{result.evidence_log}`",
                    "",
                ]
            )
    else:
        lines.append("- Skipped.")

    lines.extend(["", "## Infrastructure and judge errors", ""])
    all_errors = list(summary.errors)
    all_errors.extend(result.judge_failure for result in summary.quality_results if result.judge_failure)
    all_errors.extend(result.infrastructure_error for result in summary.quality_results if result.infrastructure_error)
    if all_errors:
        lines.extend(f"- {error}" for error in all_errors)
    else:
        lines.append("- None.")

    lines.extend(["", "## Acceptance checks", ""])
    for check in summary.acceptance:
        lines.append(f"- [{'x' if check.passed else ' '}] {check.name}: actual `{check.actual}`, threshold `{check.threshold}`")
    return "\n".join(lines) + "\n"
