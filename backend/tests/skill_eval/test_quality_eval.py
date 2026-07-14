import json

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState

import evals.skills_quality_eval as quality_module
from evals.skills_quality_eval import skills_quality_eval
from skill_eval.agent_runner import AgentRunResult
from skill_eval.inspect_scorer import quality_judge_scorer
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentToolCall, AgentTrace


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelOutput.from_content("fake/judge", self.response)


def judgment_json(**updates):
    payload = {
        "recommended_route": "systematic-literature-review",
        "route_quality": 3,
        "process_quality": 3,
        "output_quality": 3,
        "overall_quality": 3,
        "fatal_error": False,
        "reasons": ["Observable behavior is sound."],
        "evidence": ["tool_call[0]", "final_answer"],
    }
    payload.update(updates)
    return json.dumps(payload)


def quality_state():
    case = {
        "id": "quality-1",
        "input": "Synthesize three papers.",
        "expected_route": "systematic-literature-review",
        "rationale": "PRIVATE HUMAN RATIONALE",
        "tags": ["quality"],
    }
    trace = AgentTrace(
        input=case["input"],
        final_answer="Three-paper synthesis",
        success=True,
        thread_id="thread-1",
        tool_calls=[
            AgentToolCall(
                id="t1",
                message_id="m1",
                name="read_file",
                result="skill body",
            )
        ],
    )
    observation = RouteObservation(
        observed_route="systematic-literature-review",
        completed=True,
    )
    return TaskState(
        model="mock-model",
        sample_id=case["id"],
        epoch=1,
        input=case["input"],
        target=case["expected_route"],
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=trace.final_answer),
        metadata={
            "case": case,
            "agent_trace": trace.model_dump(),
            "route_observation": observation.model_dump(),
        },
    )


def descriptions():
    return {
        "systematic-literature-review": "multi-paper synthesis",
        "academic-paper-review": "single-paper critique",
    }


@pytest.mark.asyncio
async def test_quality_scorer_omits_expected_route_and_rationale_from_prompt(monkeypatch):
    model = FakeModel(judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quality_judge_scorer("fake/judge", descriptions())(
        quality_state(),
        Target("systematic-literature-review"),
    )

    assert score.value == 3
    assert score.metadata["quality_passed"] is True
    assert "expected_route" not in model.prompts[0]
    assert "PRIVATE HUMAN RATIONALE" not in model.prompts[0]


@pytest.mark.asyncio
async def test_quality_pass_requires_each_dimension_not_overall(monkeypatch):
    model = FakeModel(judgment_json(route_quality=2, overall_quality=4))
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quality_judge_scorer("fake/judge", descriptions())(
        quality_state(),
        Target("systematic-literature-review"),
    )

    assert score.value == 4
    assert score.metadata["quality_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["trace", "route"])
async def test_quality_scorer_rejects_failed_agent_run_before_judging(monkeypatch, failure):
    model = FakeModel(judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quality_state()
    if failure == "trace":
        state.metadata["agent_trace"]["success"] = False
        state.metadata["agent_trace"]["errors"] = ["agent failed"]
    else:
        state.metadata["route_observation"]["completed"] = False
        state.metadata["route_observation"]["errors"] = ["route incomplete"]

    score = await quality_judge_scorer("fake/judge", descriptions())(
        state,
        Target("systematic-literature-review"),
    )

    assert score.value == NOANSWER
    assert "infrastructure_error" in score.metadata
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quality_judge_failure_returns_noanswer(monkeypatch):
    model = FakeModel(judgment_json(evidence=["tool_call[999]", "final_answer"]))
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quality_judge_scorer("fake/judge", descriptions())(
        quality_state(),
        Target("systematic-literature-review"),
    )

    assert score.value == NOANSWER
    assert "unknown evidence" in score.metadata["judge_failure"]


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


@pytest.mark.asyncio
async def test_quality_task_selects_four_cases_and_full_runner(monkeypatch, tmp_path):
    runner = ScriptedRunner()
    runner_options = {}
    monkeypatch.setattr(
        quality_module,
        "DeerFlowAgentRunner",
        lambda **options: runner_options.update(options) or runner,
    )
    monkeypatch.setattr(
        quality_module,
        "load_candidate_skill_descriptions",
        lambda _: descriptions(),
    )
    task = skills_quality_eval(
        case_file="cases/literature_skill_routing.jsonl",
        agent_model="default",
        judge_model="mockllm/judge",
        skills_root="../skills/public",
        trace_dir=tmp_path,
        config_path="eval-config.yaml",
        sandbox="local",
    )
    state = quality_state()

    await task.solver(state, generate=None)

    assert len(task.dataset) == 4
    assert task.time_limit == 930
    assert len(task.scorer) == 2
    assert runner.requests[0].mode == "full"
    assert runner.requests[0].timeout_seconds == 900
    assert runner_options["trace_dir"] == str(tmp_path)
    assert runner_options["config_path"] == "eval-config.yaml"
    assert runner_options["sandbox"] == "local"
