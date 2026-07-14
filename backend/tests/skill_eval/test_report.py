from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import CORRECT, NOANSWER, Score

from skill_eval.judge import QualityJudgment
from skill_eval.report import (
    AcceptanceCheck,
    ClassMetrics,
    PocSummary,
    QualityCaseResult,
    RoutingEpochResult,
    RoutingMetrics,
    RunIdentity,
    extract_quality_results,
    extract_routing_results,
    render_poc_markdown,
    routing_acceptance,
    summarize_routing,
)
from skill_eval.routing import RouteEvidence


def result(case_id, epoch, expected, observed=None, error=None):
    return RoutingEpochResult(
        case_id=case_id,
        epoch=epoch,
        expected_route=expected,
        observed_route=observed,
        infrastructure_error=error,
        evidence=[],
        log_location=f"logs/{case_id}.eval",
    )


def test_summary_builds_confusion_metrics_and_valid_rate():
    results = [
        result("a", 1, "systematic-literature-review", "systematic-literature-review"),
        result("a", 2, "systematic-literature-review", "academic-paper-review"),
        result("b", 1, "academic-paper-review", "academic-paper-review"),
        result("b", 2, "academic-paper-review", "ambiguous"),
        result("c", 1, "none", "none"),
        result("c", 2, "none", error="timeout"),
    ]

    summary = summarize_routing(results, planned_runs=6)

    assert summary.valid_runs == 5
    assert summary.valid_run_rate == 5 / 6
    assert summary.confusion["systematic-literature-review"]["academic-paper-review"] == 1
    assert summary.confusion["academic-paper-review"]["ambiguous"] == 1
    assert summary.per_class["systematic-literature-review"].recall == 0.5
    assert summary.stable_cases == 0
    assert summary.results == results


def make_balanced_summary(
    *,
    valid_run_rate: float,
    macro_precision: float,
    macro_recall: float,
) -> RoutingMetrics:
    metrics = ClassMetrics(precision=0.8, recall=0.8, f1=0.8, support=1)
    return RoutingMetrics(
        planned_runs=60,
        valid_runs=57,
        valid_run_rate=valid_run_rate,
        confusion={},
        per_class={
            "systematic-literature-review": metrics,
            "academic-paper-review": metrics,
            "none": metrics,
        },
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=0.8,
        total_cases=20,
        stable_cases=20,
        stability_rate=1.0,
        results=[],
    )


def test_acceptance_requires_valid_rate_precision_and_recall():
    passing = make_balanced_summary(
        valid_run_rate=0.95,
        macro_precision=0.8,
        macro_recall=0.8,
    )
    assert routing_acceptance(passing) is True
    assert routing_acceptance(passing.model_copy(update={"valid_run_rate": 0.94})) is False
    assert routing_acceptance(passing.model_copy(update={"macro_precision": 0.79})) is False
    assert routing_acceptance(passing.model_copy(update={"macro_recall": 0.79})) is False


def test_zero_predicted_count_precision_is_zero():
    summary = summarize_routing(
        [result("a", 1, "systematic-literature-review", "none")],
        planned_runs=1,
    )

    assert summary.per_class["academic-paper-review"].precision == 0.0
    assert summary.per_class["academic-paper-review"].recall == 0.0
    assert summary.per_class["academic-paper-review"].f1 == 0.0


def test_case_with_any_infrastructure_result_is_unstable():
    summary = summarize_routing(
        [
            result("a", 1, "none", "none"),
            result("a", 2, "none", error="timeout"),
        ],
        planned_runs=2,
    )

    assert summary.total_cases == 1
    assert summary.stable_cases == 0
    assert summary.stability_rate == 0.0


def test_extract_routing_results_reads_score_metadata_and_log_location():
    sample = SimpleNamespace(
        id="case-1",
        epoch=2,
        metadata={
            "case": {
                "id": "case-1",
                "input": "Survey papers",
                "expected_route": "systematic-literature-review",
                "rationale": "multi-paper",
                "tags": [],
            }
        },
        scores={
            "routing_scorer": Score(
                value=CORRECT,
                metadata={
                    "case_id": "case-1",
                    "expected_route": "systematic-literature-review",
                    "observed_route": "systematic-literature-review",
                    "route_observation": {
                        "observed_route": "systematic-literature-review",
                        "completed": True,
                        "errors": [],
                        "evidence": [
                            {
                                "id": "route_evidence[0]",
                                "kind": "loaded",
                                "skill": "systematic-literature-review",
                                "tool_call_id": "t1",
                            }
                        ],
                    },
                },
            )
        },
    )
    log = SimpleNamespace(samples=[sample], location="logs/run.eval")

    extracted = extract_routing_results(log)

    assert len(extracted) == 1
    assert extracted[0].epoch == 2
    assert extracted[0].observed_route == "systematic-literature-review"
    assert extracted[0].evidence[0].id == "route_evidence[0]"
    assert extracted[0].log_location == "logs/run.eval"


def test_extract_routing_results_marks_noanswer_as_infrastructure_error():
    sample = SimpleNamespace(
        id="case-1",
        epoch=1,
        metadata={
            "case": {
                "id": "case-1",
                "input": "Answer directly",
                "expected_route": "none",
                "rationale": "direct answer",
                "tags": [],
            }
        },
        scores={
            "routing_scorer": Score(
                value=NOANSWER,
                explanation="timeout",
                metadata={"infrastructure_error": "timeout"},
            )
        },
    )

    extracted = extract_routing_results(SimpleNamespace(samples=[sample], location="run.eval"))

    assert extracted[0].observed_route is None
    assert extracted[0].infrastructure_error == "timeout"


def test_extract_routing_results_rejects_missing_scorer_output():
    sample = SimpleNamespace(
        id="case-1",
        epoch=1,
        metadata={
            "case": {
                "id": "case-1",
                "input": "Answer directly",
                "expected_route": "none",
                "rationale": "direct answer",
                "tags": [],
            }
        },
        scores={},
    )

    with pytest.raises(ValueError, match="routing_scorer"):
        extract_routing_results(SimpleNamespace(samples=[sample], location="run.eval"))


def test_extract_quality_results_distinguishes_agent_infrastructure_failure():
    case = {
        "id": "case-1",
        "input": "Survey papers",
        "expected_route": "systematic-literature-review",
        "rationale": "survey",
        "tags": ["quality"],
    }
    sample = SimpleNamespace(
        id="case-1",
        metadata={"case": case},
        scores={
            "routing_scorer": Score(
                value=CORRECT,
                metadata={
                    "route_observation": {
                        "observed_route": "systematic-literature-review",
                        "evidence": [],
                        "completed": True,
                        "errors": [],
                    }
                },
            ),
            "quality_judge_scorer": Score(
                value=NOANSWER,
                explanation="Agent run failed before judging: timed out",
                metadata={"infrastructure_error": "timed out"},
            ),
        },
    )

    results = extract_quality_results(SimpleNamespace(samples=[sample], location="quality.eval"))

    assert results[0].infrastructure_error == "timed out"
    assert results[0].judge_failure is None
    assert results[0].judgment is None


def test_markdown_report_contains_identity_metrics_evidence_and_acceptance():
    epoch = result(
        "case-1",
        1,
        "systematic-literature-review",
        "academic-paper-review",
    )
    epoch.evidence = [
        RouteEvidence(
            id="route_evidence[0]",
            kind="loaded",
            skill="academic-paper-review",
            tool_call_id="t1",
        )
    ]
    routing = summarize_routing([epoch], planned_runs=1)
    now = datetime.now(UTC)
    summary = PocSummary(
        run_id="run-1",
        mode="full",
        identity=RunIdentity(
            agent_model="agent",
            judge_model="mockllm/judge",
            inspect_ai_version="inspect",
            deerflow_version="deerflow",
            case_file_sha256="a" * 64,
            skill_file_sha256={"systematic-literature-review": "b" * 64},
            runtime_config={
                "config_path": "/repo/config.yaml",
                "config_sha256": "c" * 64,
                "sandbox": "local",
                "allow_host_bash": "true",
            },
            started_at=now,
            ended_at=now,
            inspect_logs=["routing.eval", "quality.eval"],
        ),
        routing=routing,
        quality_results=[
            QualityCaseResult(
                case_id="case-1",
                observed_route="academic-paper-review",
                judgment=QualityJudgment(
                    recommended_route="systematic-literature-review",
                    route_quality=2,
                    process_quality=3,
                    output_quality=3,
                    overall_quality=3,
                    reasons=["Wrong route"],
                    evidence=["tool_call[0]", "final_answer"],
                ),
                quality_passed=False,
                label_review_needed=False,
                evidence_log="quality.eval",
            )
        ],
        quality_passed_cases=0,
        judge_failures=0,
        infrastructure_failures=0,
        acceptance=[
            AcceptanceCheck(
                name="macro precision",
                actual=0.0,
                threshold=">= 0.80",
                passed=False,
            )
        ],
    )

    markdown = render_poc_markdown(summary)

    assert "## Run identity" in markdown
    assert "## Confusion matrix" in markdown
    assert "## Per-class routing metrics" in markdown
    assert "route_evidence[0]" in markdown
    assert "## Quality judgments" in markdown
    assert "Wrong route" in markdown
    assert "## Acceptance checks" in markdown
    assert "config_sha256" in markdown
    assert "/repo/config.yaml" in markdown
