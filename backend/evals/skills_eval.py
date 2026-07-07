from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.scorer import model_graded_qa

from skill_eval.dataset_loader import load_skill_cases
from skill_eval.inspect_scorer import skill_assertion_scorer, trace_integrity_scorer
from skill_eval.inspect_solver import skill_agent_solver


@task
def skills_eval(case_file: str = "cases/gcp_skills.jsonl", mode: str = "with_skill", skills_folder: str = "skills", use_model_graded_qa: bool = False, use_deerflow: bool = False):
    samples = load_skill_cases(case_file)

    if mode == "baseline":
        selected_skills: list[str] | None = []
    elif mode == "with_skill":
        selected_skills = None
    elif mode == "all_skills":
        skill_files = (Path.cwd() / skills_folder).rglob("SKILL.md")
        selected_skills = [str(skill_file.parent) for skill_file in skill_files]
    else:
        raise ValueError("mode must be one of: baseline, with_skill, all_skills")

    scorers = [trace_integrity_scorer(), skill_assertion_scorer()]
    if use_model_graded_qa:
        scorers.append(model_graded_qa())
    agent_runner = None
    if use_deerflow:
        from skill_eval.adapters.deerflow import DeerFlowAgentRunner

        agent_runner = DeerFlowAgentRunner()

    return Task(dataset=samples, solver=skill_agent_solver(agent_runner=agent_runner, skills=selected_skills), scorer=scorers)
