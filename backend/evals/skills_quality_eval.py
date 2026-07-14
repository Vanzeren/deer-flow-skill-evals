from pathlib import Path

from inspect_ai import Task, task

from skill_eval.adapters.deerflow import DeerFlowAgentRunner
from skill_eval.agent_runner import SandboxMode
from skill_eval.dataset_loader import load_routing_samples
from skill_eval.inspect_scorer import quality_judge_scorer, routing_scorer
from skill_eval.inspect_solver import deerflow_solver
from skill_eval.judge import load_candidate_skill_descriptions

_QUALITY_TIMEOUT_SECONDS = 900
_QUALITY_TASK_TIME_LIMIT_SECONDS = 930


@task
def skills_quality_eval(
    case_file: str,
    agent_model: str,
    judge_model: str,
    skills_root: str | Path = "../skills/public",
    trace_dir: str | Path | None = None,
    config_path: str | None = None,
    sandbox: SandboxMode = "configured",
) -> Task:
    samples = load_routing_samples(case_file, tags={"quality"})
    skill_descriptions = load_candidate_skill_descriptions(Path(skills_root))
    runner = DeerFlowAgentRunner(
        config_path=config_path,
        trace_dir=str(trace_dir) if trace_dir is not None else None,
        sandbox=sandbox,
    )
    return Task(
        dataset=samples,
        solver=deerflow_solver(
            runner,
            mode="full",
            model_name=agent_model,
            timeout_seconds=_QUALITY_TIMEOUT_SECONDS,
        ),
        scorer=[
            routing_scorer(),
            quality_judge_scorer(judge_model, skill_descriptions),
        ],
        time_limit=_QUALITY_TASK_TIME_LIMIT_SECONDS,
    )
