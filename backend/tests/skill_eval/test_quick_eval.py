from types import SimpleNamespace

import pytest

import evals.skills_quick_eval as quick_module
from evals.skills_quick_eval import skills_quick_eval
from skill_eval.agent_runner import AgentRunResult
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace


class ScriptedRunner:
    def __init__(self):
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return AgentRunResult(
            final_answer="done",
            success=True,
            thread_id=request.thread_id,
            route_observation=RouteObservation(observed_route="none", completed=True),
            trace=AgentTrace(
                input=request.user_input,
                final_answer="done",
                success=True,
                thread_id=request.thread_id,
            ),
        )


def descriptions():
    return {
        "systematic-literature-review": "multi-paper",
        "academic-paper-review": "one-paper",
    }


def quick_state():
    return SimpleNamespace(
        input_text="Review the arXiv paper.",
        metadata={
            "case": {
                "id": "quick-1",
                "input": "Review the arXiv paper.",
                "expected_route": "academic-paper-review",
                "rationale": "single paper",
                "tags": ["quality"],
            }
        },
        output=SimpleNamespace(completion=""),
    )


@pytest.mark.asyncio
async def test_quick_task_selects_quality_cases_and_quick_runner(monkeypatch, tmp_path):
    runner = ScriptedRunner()
    runner_options = {}
    monkeypatch.setattr(
        quick_module,
        "DeerFlowAgentRunner",
        lambda **options: runner_options.update(options) or runner,
    )
    monkeypatch.setattr(
        quick_module,
        "load_candidate_skill_descriptions",
        lambda _: descriptions(),
    )
    task = skills_quick_eval(
        case_file="cases/literature_skill_routing.jsonl",
        agent_model="default",
        judge_model="mockllm/judge",
        skills_root="../skills/public",
        trace_dir=tmp_path,
        config_path="eval-config.yaml",
        sandbox="local",
    )
    state = quick_state()

    await task.solver(state, generate=None)

    assert len(task.dataset) == 4
    assert task.time_limit == 330
    assert len(task.scorer) == 2
    assert runner.requests[0].mode == "quick"
    assert runner.requests[0].timeout_seconds == 300
    assert runner_options["trace_dir"] == str(tmp_path)
    assert runner_options["config_path"] == "eval-config.yaml"
    assert runner_options["sandbox"] == "local"


@pytest.mark.asyncio
async def test_quick_task_sample_ids_bypass_quality_tag_filter(monkeypatch, tmp_path):
    smoke_ids = {
        "slr-attention-variants-001",
        "paper-review-arxiv-001",
        "none-precision-recall-001",
    }
    monkeypatch.setattr(quick_module, "DeerFlowAgentRunner", lambda **options: ScriptedRunner())
    monkeypatch.setattr(quick_module, "load_candidate_skill_descriptions", lambda _: descriptions())

    task = skills_quick_eval(
        case_file="cases/literature_skill_routing.jsonl",
        agent_model="default",
        judge_model="mockllm/judge",
        sample_ids=smoke_ids,
    )

    assert {str(sample.id) for sample in task.dataset} == smoke_ids


def test_quick_task_rejects_unknown_sample_id(monkeypatch):
    monkeypatch.setattr(quick_module, "DeerFlowAgentRunner", lambda **options: ScriptedRunner())
    monkeypatch.setattr(quick_module, "load_candidate_skill_descriptions", lambda _: descriptions())

    with pytest.raises(ValueError, match="Unknown routing sample id"):
        skills_quick_eval(
            case_file="cases/literature_skill_routing.jsonl",
            agent_model="default",
            judge_model="mockllm/judge",
            sample_ids={"does-not-exist"},
        )
