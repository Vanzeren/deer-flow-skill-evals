import argparse
import hashlib
import importlib.metadata as importlib_metadata
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model
from pydantic import BaseModel

from deerflow.client import DeerFlowClient
from deerflow.config.app_config import AppConfig
from evals.skills_quality_eval import skills_quality_eval
from evals.skills_quick_eval import skills_quick_eval
from evals.skills_routing_eval import skills_routing_eval
from skill_eval.case_schema import CANDIDATE_SKILLS
from skill_eval.dataset_loader import read_routing_cases, validate_poc_suite
from skill_eval.judge import load_candidate_skill_descriptions
from skill_eval.report import (
    AcceptanceCheck,
    PocSummary,
    RunIdentity,
    extract_quality_results,
    extract_quick_results,
    extract_routing_results,
    render_poc_markdown,
    routing_acceptance,
    summarize_routing,
)

_SMOKE_CASE_IDS = {
    "slr-attention-variants-001",
    "paper-review-arxiv-001",
    "none-precision-recall-001",
}


class PocConfigurationError(RuntimeError):
    pass


class PocInvalidEvaluationError(RuntimeError):
    pass


class PocConfig(BaseModel):
    agent_model: str
    judge_model: str
    case_file: Path = Path("cases/literature_skill_routing.jsonl")
    output_dir: Path = Path("eval-results")
    log_dir: Path = Path("logs")
    skills_root: Path = Path("../skills/public")
    config_path: str | None = None
    smoke: bool = False
    quality_mode: Literal["quick", "full", "both"] = "both"

    @classmethod
    def from_env(
        cls,
        *,
        case_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        smoke: bool = False,
        quality_mode: str = "both",
    ) -> "PocConfig":
        agent_model = os.getenv("AGENT_MODEL", "").strip()
        judge_model = os.getenv("JUDGE_MODEL", "").strip()
        missing = [
            name
            for name, value in [
                ("AGENT_MODEL", agent_model),
                ("JUDGE_MODEL", judge_model),
            ]
            if not value
        ]
        if missing:
            raise PocConfigurationError("Required environment variable(s) missing: " + ", ".join(missing))
        values: dict[str, Any] = {
            "agent_model": agent_model,
            "judge_model": judge_model,
            "smoke": smoke,
            "quality_mode": quality_mode,
            "config_path": os.getenv("DEER_FLOW_CONFIG_PATH") or None,
        }
        if case_file is not None:
            values["case_file"] = Path(case_file)
        if output_dir is not None:
            values["output_dir"] = Path(output_dir)
        return cls(**values)


class PreflightRecord(BaseModel):
    inspect_ai_version: str
    deerflow_version: str
    case_file_sha256: str
    skill_file_sha256: dict[str, str]
    runtime_config: dict[str, str]


def _sha256_file(path: Path) -> str:
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as exc:
        raise PocConfigurationError(f"Cannot hash required file {path}: {exc}") from exc


def preflight(
    config: PocConfig,
    *,
    client_factory: Any = DeerFlowClient,
) -> PreflightRecord:
    try:
        cases = read_routing_cases(config.case_file)
        validate_poc_suite(cases)
    except Exception as exc:
        raise PocConfigurationError(f"Invalid routing case suite: {exc}") from exc

    try:
        resolved_config_path = AppConfig.resolve_config_path(config.config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise PocConfigurationError(f"Cannot resolve DeerFlow config: {exc}") from exc

    try:
        client = client_factory(
            config_path=config.config_path,
            model_name=config.agent_model,
            available_skills=set(CANDIDATE_SKILLS),
        )
        models = client.list_models().get("models", [])
    except Exception as exc:
        raise PocConfigurationError(f"Cannot initialize DeerFlow: {exc}") from exc
    model_names = {str(model.get("name") or model.get("model_name")) for model in models if isinstance(model, dict)}
    if config.agent_model not in model_names:
        raise PocConfigurationError(f"Agent model {config.agent_model!r} is not in configured DeerFlow models")

    try:
        skills = client.list_skills(enabled_only=True).get("skills", [])
    except Exception as exc:
        raise PocConfigurationError(f"Cannot list enabled DeerFlow skills: {exc}") from exc
    enabled_skills = {str(skill.get("name")) for skill in skills if isinstance(skill, dict) and skill.get("name")}
    missing_skills = sorted(set(CANDIDATE_SKILLS) - enabled_skills)
    if missing_skills:
        raise PocConfigurationError("Required candidate skill(s) missing or disabled: " + ", ".join(missing_skills))

    try:
        get_model(config.judge_model)
    except Exception as exc:
        raise PocConfigurationError(f"Invalid Inspect judge model {config.judge_model!r}: {exc}") from exc

    try:
        load_candidate_skill_descriptions(config.skills_root)
    except ValueError as exc:
        raise PocConfigurationError(str(exc)) from exc
    skill_hashes = {candidate: _sha256_file(config.skills_root / candidate / "SKILL.md") for candidate in CANDIDATE_SKILLS}
    try:
        inspect_version = importlib_metadata.version("inspect-ai")
        deerflow_version = importlib_metadata.version("deerflow-harness")
    except importlib_metadata.PackageNotFoundError as exc:
        raise PocConfigurationError(f"Required package identity unavailable: {exc}") from exc

    return PreflightRecord(
        inspect_ai_version=inspect_version,
        deerflow_version=deerflow_version,
        case_file_sha256=_sha256_file(config.case_file),
        skill_file_sha256=skill_hashes,
        runtime_config={
            "config_path": str(resolved_config_path),
            "config_sha256": _sha256_file(resolved_config_path),
            "sandbox": "local",
            "allow_host_bash": "true",
        },
    )


def run_poc(config: PocConfig) -> tuple[PocSummary, int]:
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    preflight_record = preflight(config)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = config.output_dir / run_id / "traces"
    routing_task = skills_routing_eval(
        case_file=str(config.case_file),
        agent_model=config.agent_model,
        sample_ids=_SMOKE_CASE_IDS if config.smoke else None,
        trace_dir=trace_dir,
        config_path=config.config_path,
        sandbox="local",
    )
    routing_epochs = 1 if config.smoke else 3
    try:
        routing_logs = inspect_eval(
            routing_task,
            model=None,
            epochs=routing_epochs,
            max_samples=1,
            log_dir=str(config.log_dir),
            fail_on_error=False,
            score_on_error=True,
        )
        routing_log = _single_log(routing_logs, "routing")
    except Exception as exc:
        raise PocInvalidEvaluationError(f"Routing evaluation failed: {exc}") from exc

    quick_log = None
    if config.quality_mode in {"quick", "both"}:
        quick_task = skills_quick_eval(
            case_file=str(config.case_file),
            agent_model=config.agent_model,
            judge_model=config.judge_model,
            skills_root=config.skills_root,
            sample_ids=_SMOKE_CASE_IDS if config.smoke else None,
            trace_dir=trace_dir,
            config_path=config.config_path,
            sandbox="local",
        )
        try:
            quick_logs = inspect_eval(
                quick_task,
                model=None,
                epochs=1,
                max_samples=1,
                log_dir=str(config.log_dir),
                fail_on_error=False,
                score_on_error=True,
            )
            quick_log = _single_log(quick_logs, "quick quality")
        except Exception as exc:
            raise PocInvalidEvaluationError(f"Quick quality evaluation failed: {exc}") from exc

    quality_log = None
    if not config.smoke and config.quality_mode in {"full", "both"}:
        quality_task = skills_quality_eval(
            case_file=str(config.case_file),
            agent_model=config.agent_model,
            judge_model=config.judge_model,
            skills_root=config.skills_root,
            trace_dir=trace_dir,
            config_path=config.config_path,
            sandbox="local",
        )
        try:
            quality_logs = inspect_eval(
                quality_task,
                model=None,
                epochs=1,
                max_samples=1,
                log_dir=str(config.log_dir),
                fail_on_error=False,
                score_on_error=True,
            )
            quality_log = _single_log(quality_logs, "quality")
        except Exception as exc:
            raise PocInvalidEvaluationError(f"Quality evaluation failed: {exc}") from exc

    planned_routing_runs = 3 if config.smoke else 60
    errors: list[str] = []
    try:
        routing_results = extract_routing_results(routing_log)
    except Exception as exc:
        routing_results = []
        errors.append(f"Cannot extract routing results: {exc}")
    routing_metrics = summarize_routing(
        routing_results,
        planned_runs=planned_routing_runs,
    )
    if len(routing_results) != planned_routing_runs:
        errors.append(f"Expected {planned_routing_runs} routing results, found {len(routing_results)}")

    quality_results = []
    if quality_log is not None:
        try:
            quality_results = extract_quality_results(quality_log)
        except Exception as exc:
            errors.append(f"Cannot extract quality results: {exc}")
        if len(quality_results) != 4:
            errors.append(f"Expected 4 quality results, found {len(quality_results)}")

    quick_results = []
    if quick_log is not None:
        try:
            quick_results = extract_quick_results(quick_log)
        except Exception as exc:
            errors.append(f"Cannot extract quick quality results: {exc}")
        expected_quick = 3 if config.smoke else 4
        if len(quick_results) != expected_quick:
            errors.append(f"Expected {expected_quick} quick quality results, found {len(quick_results)}")

    inspect_logs = [str(getattr(routing_log, "location", ""))]
    if quality_log is not None:
        inspect_logs.append(str(getattr(quality_log, "location", "")))
    if quick_log is not None:
        inspect_logs.append(str(getattr(quick_log, "location", "")))
    identity = RunIdentity(
        agent_model=config.agent_model,
        judge_model=config.judge_model,
        inspect_ai_version=preflight_record.inspect_ai_version,
        deerflow_version=preflight_record.deerflow_version,
        case_file_sha256=preflight_record.case_file_sha256,
        skill_file_sha256=preflight_record.skill_file_sha256,
        runtime_config=preflight_record.runtime_config,
        started_at=started_at,
        ended_at=datetime.now(UTC),
        inspect_logs=inspect_logs,
    )
    quality_passed_cases = sum(result.quality_passed for result in quality_results)
    judge_failures = sum(result.judge_failure is not None for result in quality_results)
    infrastructure_failures = sum(result.infrastructure_error is not None for result in quality_results)
    quick_passed_cases = sum(result.quality_passed for result in quick_results)
    quick_turn_missing = sum(result.category == "quick_turn_missing" for result in quick_results)
    quick_judge_failures = sum(result.category == "judge_failure" for result in quick_results)
    quick_infrastructure_failures = sum(result.category == "infrastructure_error" for result in quick_results)
    acceptance = _acceptance_checks(
        routing_metrics,
        quality_passed_cases=quality_passed_cases,
        include_quality=not config.smoke,
        quick_passed_cases=quick_passed_cases,
        include_quick=not config.smoke and config.quality_mode in {"quick", "both"},
    )
    summary = PocSummary(
        run_id=run_id,
        mode="smoke" if config.smoke else "full",
        identity=identity,
        routing=routing_metrics,
        quality_results=quality_results,
        quality_passed_cases=quality_passed_cases,
        judge_failures=judge_failures,
        infrastructure_failures=infrastructure_failures,
        quality_mode=config.quality_mode,
        quick_results=quick_results,
        quick_passed_cases=quick_passed_cases,
        quick_turn_missing=quick_turn_missing,
        quick_judge_failures=quick_judge_failures,
        quick_infrastructure_failures=quick_infrastructure_failures,
        acceptance=acceptance,
        errors=errors,
    )
    _write_summary(config.output_dir / run_id, summary)
    return summary, exit_code_for(summary)


def _single_log(logs: Any, label: str) -> Any:
    if not isinstance(logs, list) or len(logs) != 1:
        count = len(logs) if isinstance(logs, list) else "non-list"
        raise PocInvalidEvaluationError(f"Expected one {label} Inspect log, received {count}")
    return logs[0]


def _acceptance_checks(
    routing_metrics,
    *,
    quality_passed_cases: int,
    include_quality: bool,
    quick_passed_cases: int = 0,
    include_quick: bool = False,
) -> list[AcceptanceCheck]:
    checks = [
        AcceptanceCheck(
            name="valid routing run rate",
            actual=routing_metrics.valid_run_rate,
            threshold=">= 0.95",
            passed=routing_metrics.valid_run_rate >= 0.95,
        ),
        AcceptanceCheck(
            name="macro routing precision",
            actual=routing_metrics.macro_precision,
            threshold=">= 0.80",
            passed=routing_metrics.macro_precision >= 0.80,
        ),
        AcceptanceCheck(
            name="macro routing recall",
            actual=routing_metrics.macro_recall,
            threshold=">= 0.80",
            passed=routing_metrics.macro_recall >= 0.80,
        ),
    ]
    if include_quality:
        checks.append(
            AcceptanceCheck(
                name="quality cases passing all dimension thresholds",
                actual=quality_passed_cases,
                threshold=">= 3 of 4",
                passed=quality_passed_cases >= 3,
            )
        )
    if include_quick:
        checks.append(
            AcceptanceCheck(
                name="quick quality cases passing turn threshold",
                actual=quick_passed_cases,
                threshold=">= 3 of 4",
                passed=quick_passed_cases >= 3,
            )
        )
    return checks


def exit_code_for(summary: PocSummary) -> int:
    expected_routing_runs = 3 if summary.mode == "smoke" else 60
    full_quality_expected = 4 if (summary.mode == "full" and summary.quality_mode in {"full", "both"}) else 0
    quick_expected = (3 if summary.mode == "smoke" else 4) if summary.quality_mode in {"quick", "both"} else 0
    evaluation_invalid = (
        bool(summary.errors)
        or len(summary.routing.results) != expected_routing_runs
        or summary.judge_failures > 0
        or summary.infrastructure_failures > 0
        or len(summary.quality_results) != full_quality_expected
        or len(summary.quick_results) != quick_expected
        or summary.quick_judge_failures > 0
        or summary.quick_infrastructure_failures > 0
        or summary.quick_turn_missing > 0
    )
    if evaluation_invalid:
        return 2
    full_quality_passed = full_quality_expected == 0 or summary.quality_passed_cases >= 3
    quick_passed = quick_expected == 0 or summary.mode == "smoke" or summary.quick_passed_cases >= 3
    return 0 if routing_acceptance(summary.routing) and full_quality_passed and quick_passed else 1


def _write_summary(run_dir: Path, summary: PocSummary) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "summary.json": summary.model_dump_json(indent=2) + "\n",
        "summary.md": render_poc_markdown(summary),
    }
    for filename, content in payloads.items():
        destination = run_dir / filename
        temporary = run_dir / f".{filename}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeerFlow skill-routing evaluation POC")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--case-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--quality-mode", choices=["quick", "full", "both"], default="both")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = PocConfig.from_env(
            case_file=args.case_file,
            output_dir=args.output_dir,
            smoke=args.smoke,
            quality_mode=args.quality_mode,
        )
        summary, exit_code = run_poc(config)
    except (PocConfigurationError, PocInvalidEvaluationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(config.output_dir / summary.run_id)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
