from pathlib import Path

from inspect_ai import Task, task

from skill_eval.adapters.deerflow import DeerFlowAgentRunner
from skill_eval.agent_runner import SandboxMode
from skill_eval.dataset_loader import load_routing_samples
from skill_eval.inspect_scorer import quick_turn_scorer, routing_scorer
from skill_eval.inspect_solver import deerflow_solver
from skill_eval.judge import load_candidate_skill_descriptions

_QUICK_TIMEOUT_SECONDS = 300
_QUICK_TASK_TIME_LIMIT_SECONDS = 330


@task
def skills_quick_eval(
    case_file: str,
    agent_model: str,
    judge_model: str,
    skills_root: str | Path = "../skills/public",
    sample_ids: set[str] | None = None,
    trace_dir: str | Path | None = None,
    config_path: str | None = None,
    sandbox: SandboxMode = "configured",
) -> Task:
    if sample_ids is None:
        samples = load_routing_samples(case_file, tags={"quality"})
    else:
        samples = load_routing_samples(case_file)
        samples = [sample for sample in samples if str(sample.id) in sample_ids]
        found = {str(sample.id) for sample in samples}
        if found != sample_ids:
            missing = ", ".join(sorted(sample_ids - found))
            raise ValueError(f"Unknown routing sample id(s): {missing}")
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
            mode="quick",
            model_name=agent_model,
            timeout_seconds=_QUICK_TIMEOUT_SECONDS,
        ),
        scorer=[
            routing_scorer(),
            quick_turn_scorer(judge_model, skill_descriptions),
        ],
        time_limit=_QUICK_TASK_TIME_LIMIT_SECONDS,
    )
