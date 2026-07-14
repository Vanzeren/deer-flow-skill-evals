from types import SimpleNamespace

import pytest
from inspect_ai.scorer import CORRECT, Score

from skill_eval.case_schema import CANDIDATE_SKILLS
from skill_eval.dataset_loader import read_routing_cases
from skill_eval.poc import (
    PocConfig,
    PocConfigurationError,
    PreflightRecord,
    exit_code_for,
    preflight,
    run_poc,
)


class FakeClient:
    def __init__(self):
        self.models = [{"name": "default"}]
        self.skills = [
            {"name": "systematic-literature-review", "enabled": True},
            {"name": "academic-paper-review", "enabled": True},
        ]

    def list_models(self):
        return {"models": self.models}

    def list_skills(self, *, enabled_only=False):
        skills = self.skills
        if enabled_only:
            skills = [skill for skill in skills if skill.get("enabled")]
        return {"skills": skills}


@pytest.fixture
def valid_config(tmp_path):
    return PocConfig(
        agent_model="default",
        judge_model="mockllm/judge",
        case_file="cases/literature_skill_routing.jsonl",
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        skills_root="../skills/public",
    )


@pytest.fixture
def preflight_record():
    return PreflightRecord(
        inspect_ai_version="0.3.test",
        deerflow_version="test",
        case_file_sha256="a" * 64,
        skill_file_sha256={skill: "b" * 64 for skill in CANDIDATE_SKILLS},
        runtime_config={"config_path": "auto"},
    )


def routing_score(case):
    return Score(
        value=CORRECT,
        metadata={
            "case_id": case.id,
            "expected_route": case.expected_route,
            "observed_route": case.expected_route,
            "route_observation": {
                "observed_route": case.expected_route,
                "completed": True,
                "errors": [],
                "evidence": [],
            },
        },
    )


def routing_log(cases, epochs=3):
    samples = []
    for case in cases:
        for epoch in range(1, epochs + 1):
            samples.append(
                SimpleNamespace(
                    id=case.id,
                    epoch=epoch,
                    metadata={"case": case.model_dump()},
                    scores={"routing_scorer": routing_score(case)},
                )
            )
    return SimpleNamespace(samples=samples, location="logs/routing.eval")


def quality_log(cases):
    samples = []
    for case in cases:
        judgment = {
            "recommended_route": case.expected_route,
            "route_quality": 3,
            "process_quality": 3,
            "output_quality": 3,
            "overall_quality": 3,
            "fatal_error": False,
            "reasons": ["Observable behavior passes."],
            "evidence": ["message[0]", "final_answer"],
        }
        samples.append(
            SimpleNamespace(
                id=case.id,
                epoch=1,
                metadata={"case": case.model_dump()},
                scores={
                    "routing_scorer": routing_score(case),
                    "quality_judge_scorer": Score(
                        value=3,
                        metadata={
                            "quality_judgment": judgment,
                            "quality_passed": True,
                            "label_review_needed": False,
                        },
                    ),
                },
            )
        )
    return SimpleNamespace(samples=samples, location="logs/quality.eval")


def test_preflight_requires_both_model_inputs(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    with pytest.raises(PocConfigurationError, match="AGENT_MODEL"):
        PocConfig.from_env()


def test_preflight_rejects_missing_candidate_skill(valid_config, monkeypatch):
    fake_client = FakeClient()
    fake_client.skills = [{"name": "systematic-literature-review", "enabled": True}]
    monkeypatch.setattr("skill_eval.poc.get_model", lambda _: object())

    with pytest.raises(PocConfigurationError, match="academic-paper-review"):
        preflight(valid_config, client_factory=lambda **_: fake_client)


def test_preflight_rejects_unknown_agent_model(valid_config, monkeypatch):
    fake_client = FakeClient()
    fake_client.models = [{"name": "other"}]
    monkeypatch.setattr("skill_eval.poc.get_model", lambda _: object())

    with pytest.raises(PocConfigurationError, match="default"):
        preflight(valid_config, client_factory=lambda **_: fake_client)


def test_preflight_records_resolved_config_identity(valid_config, tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("config_version: 24\n", encoding="utf-8")
    config = valid_config.model_copy(update={"config_path": str(config_file)})
    monkeypatch.setattr("skill_eval.poc.get_model", lambda _: object())

    record = preflight(config, client_factory=lambda **_: FakeClient())

    assert record.runtime_config["config_path"] == str(config_file.resolve())
    assert len(record.runtime_config["config_sha256"]) == 64
    assert record.runtime_config["sandbox"] == "local"
    assert record.runtime_config["allow_host_bash"] == "true"


def test_run_poc_calls_routing_three_epochs_and_quality_one(
    monkeypatch,
    valid_config,
    preflight_record,
):
    cases = read_routing_cases(valid_config.case_file)
    logs = [routing_log(cases), quality_log([case for case in cases if "quality" in case.tags])]
    calls = []

    def fake_eval(*args, **kwargs):
        calls.append((args, kwargs))
        return [logs.pop(0)]

    monkeypatch.setattr("skill_eval.poc.inspect_eval", fake_eval)
    monkeypatch.setattr("skill_eval.poc.preflight", lambda config: preflight_record)

    summary, exit_code = run_poc(valid_config)

    assert calls[0][1]["epochs"] == 3
    assert calls[1][1]["epochs"] == 1
    assert all(call[1]["max_samples"] == 1 for call in calls)
    assert exit_code == 0
    assert summary.routing.planned_runs == 60
    assert (valid_config.output_dir / summary.run_id / "summary.json").exists()
    assert (valid_config.output_dir / summary.run_id / "summary.md").exists()


def test_smoke_selects_fixed_three_cases_and_skips_quality(
    monkeypatch,
    valid_config,
    preflight_record,
):
    smoke_ids = {
        "slr-attention-variants-001",
        "paper-review-arxiv-001",
        "none-precision-recall-001",
    }
    cases = [case for case in read_routing_cases(valid_config.case_file) if case.id in smoke_ids]
    calls = []

    def fake_eval(task, **kwargs):
        calls.append((task, kwargs))
        return [routing_log(cases, epochs=1)]

    monkeypatch.setattr("skill_eval.poc.inspect_eval", fake_eval)
    monkeypatch.setattr("skill_eval.poc.preflight", lambda config: preflight_record)

    summary, exit_code = run_poc(valid_config.model_copy(update={"smoke": True}))

    assert len(calls) == 1
    assert calls[0][1]["epochs"] == 1
    assert {sample.id for sample in calls[0][0].dataset} == smoke_ids
    assert summary.mode == "smoke"
    assert summary.quality_results == []
    assert exit_code == 0


def test_incomplete_eval_log_returns_invalid_exit(
    monkeypatch,
    valid_config,
    preflight_record,
):
    cases = read_routing_cases(valid_config.case_file)
    incomplete = routing_log(cases)
    incomplete.samples.pop()
    logs = [incomplete, quality_log([case for case in cases if "quality" in case.tags])]
    monkeypatch.setattr(
        "skill_eval.poc.inspect_eval",
        lambda *args, **kwargs: [logs.pop(0)],
    )
    monkeypatch.setattr("skill_eval.poc.preflight", lambda config: preflight_record)

    summary, exit_code = run_poc(valid_config)

    assert exit_code == 2
    assert any("routing results" in error for error in summary.errors)


def test_exit_codes_separate_quality_failure_and_invalid_evaluation(
    monkeypatch,
    valid_config,
    preflight_record,
):
    cases = read_routing_cases(valid_config.case_file)
    quality = quality_log([case for case in cases if "quality" in case.tags])
    quality.samples[0].scores["quality_judge_scorer"].metadata["quality_passed"] = False
    quality.samples[1].scores["quality_judge_scorer"].metadata["quality_passed"] = False
    logs = [routing_log(cases), quality]
    monkeypatch.setattr(
        "skill_eval.poc.inspect_eval",
        lambda *args, **kwargs: [logs.pop(0)],
    )
    monkeypatch.setattr("skill_eval.poc.preflight", lambda config: preflight_record)

    summary, _ = run_poc(valid_config)

    assert exit_code_for(summary) == 1
    invalid = summary.model_copy(update={"errors": ["missing log"]})
    assert exit_code_for(invalid) == 2
