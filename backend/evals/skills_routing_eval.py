from pathlib import Path

from inspect_ai import Task, task

from skill_eval.adapters.deerflow import DeerFlowAgentRunner
from skill_eval.agent_runner import SandboxMode
from skill_eval.dataset_loader import load_routing_samples
from skill_eval.inspect_scorer import routing_scorer
from skill_eval.inspect_solver import deerflow_solver

_ROUTING_TIMEOUT_SECONDS = 180
_ROUTING_TASK_TIME_LIMIT_SECONDS = 210


@task
def skills_routing_eval(
    case_file: str,
    agent_model: str,
    sample_ids: set[str] | None = None,
    trace_dir: str | Path | None = None,
    config_path: str | None = None,
    sandbox: SandboxMode = "configured",
) -> Task:
    samples = load_routing_samples(case_file)
    if sample_ids is not None:
        samples = [sample for sample in samples if str(sample.id) in sample_ids]
        found = {str(sample.id) for sample in samples}
        if found != sample_ids:
            missing = ", ".join(sorted(sample_ids - found))
            raise ValueError(f"Unknown routing sample id(s): {missing}")
    runner = DeerFlowAgentRunner(
        config_path=config_path,
        trace_dir=str(trace_dir) if trace_dir is not None else None,
        sandbox=sandbox,
    )
    return Task(
        dataset=samples,
        solver=deerflow_solver(
            runner,
            mode="routing_probe",
            model_name=agent_model,
            timeout_seconds=_ROUTING_TIMEOUT_SECONDS,
        ),
        scorer=[routing_scorer()],
        time_limit=_ROUTING_TASK_TIME_LIMIT_SECONDS,
    )
