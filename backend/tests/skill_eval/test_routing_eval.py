import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Target
from inspect_ai.solver import TaskState

from evals.skills_routing_eval import skills_routing_eval
from skill_eval.agent_runner import AgentRunResult
from skill_eval.inspect_scorer import routing_scorer
from skill_eval.inspect_solver import deerflow_solver
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace


class ScriptedRunner:
    def __init__(self):
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return AgentRunResult(
            final_answer="",
            success=True,
            thread_id=request.thread_id,
            route_observation=RouteObservation(
                observed_route="systematic-literature-review",
                completed=True,
            ),
            trace=AgentTrace(
                input=request.user_input,
                final_answer="",
                success=True,
                thread_id=request.thread_id,
                runtime="deerflow",
            ),
        )


@pytest.fixture
def scripted_runner():
    return ScriptedRunner()


@pytest.fixture
def routing_state():
    case = {
        "id": "route-1",
        "input": "survey papers",
        "expected_route": "systematic-literature-review",
        "rationale": "multi-paper request",
        "tags": [],
    }
    return TaskState(
        model="mock-model",
        sample_id=case["id"],
        epoch=1,
        input=case["input"],
        target=case["expected_route"],
        messages=[],
        output=ModelOutput.from_content(model="mock-model", content=""),
        metadata={"case": case},
    )


@pytest.mark.asyncio
async def test_solver_writes_route_and_trace_metadata_without_hidden_label(
    scripted_runner,
    routing_state,
):
    solver = deerflow_solver(
        scripted_runner,
        mode="routing_probe",
        model_name="default",
        timeout_seconds=180,
    )

    result = await solver(routing_state, generate=None)

    request = scripted_runner.requests[0]
    assert request.user_input == "survey papers"
    assert request.mode == "routing_probe"
    assert request.timeout_seconds == 180
    assert "expected_route" not in request.model_dump()
    assert "rationale" not in request.model_dump()
    assert result.metadata["route_observation"]["observed_route"] == "systematic-literature-review"
    assert result.metadata["agent_trace"]["runtime"] == "deerflow"
    assert result.metadata["agent_success"] is True
    assert result.metadata["thread_id"] == request.thread_id


@pytest.mark.asyncio
async def test_routing_scorer_uses_exact_route_label(routing_state):
    routing_state.metadata["route_observation"] = {
        "observed_route": "systematic-literature-review",
        "evidence": [],
        "completed": True,
        "errors": [],
        "latency_ms": 10,
    }

    score = await routing_scorer()(routing_state, Target("none"))

    assert score.value == CORRECT
    assert score.metadata["expected_route"] == "systematic-literature-review"
    assert score.metadata["observed_route"] == "systematic-literature-review"


@pytest.mark.asyncio
async def test_routing_scorer_marks_ambiguous_incorrect(routing_state):
    routing_state.metadata["route_observation"] = {
        "observed_route": "ambiguous",
        "evidence": [],
        "completed": True,
        "errors": [],
    }

    score = await routing_scorer()(routing_state, Target("systematic-literature-review"))

    assert score.value == INCORRECT


@pytest.mark.asyncio
async def test_routing_scorer_marks_infrastructure_failure_noanswer(routing_state):
    routing_state.metadata["route_observation"] = {
        "observed_route": None,
        "evidence": [],
        "completed": False,
        "errors": ["timeout"],
    }

    score = await routing_scorer()(routing_state, Target("none"))

    assert score.value == NOANSWER
    assert score.metadata["infrastructure_error"] == "timeout"


@pytest.mark.asyncio
async def test_routing_scorer_rejects_invalid_metadata(routing_state):
    routing_state.metadata["route_observation"] = {"observed_route": "unknown"}

    score = await routing_scorer()(routing_state, Target("none"))

    assert score.value == NOANSWER
    assert "infrastructure_error" in score.metadata


def test_routing_task_has_twenty_samples_and_runtime_limit():
    task = skills_routing_eval(
        case_file="cases/literature_skill_routing.jsonl",
        agent_model="default",
    )

    assert len(task.dataset) == 20
    assert task.time_limit == 210
