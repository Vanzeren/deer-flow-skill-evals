import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Target

from skill_eval.inspect_scorer import quick_turn_scorer


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return ModelOutput.from_content("fake/judge", self.response)


def quick_judgment_json(**updates):
    payload = {
        "turn_quality": 3,
        "fatal_error": False,
        "rationale": "The turn follows the loaded skill workflow.",
        "evidence_references": ["tool_chain[0]", "quick_turn"],
    }
    payload.update(updates)
    return json.dumps(payload)


def descriptions():
    return {
        "systematic-literature-review": "multi-paper",
        "academic-paper-review": "one-paper",
    }


def quick_state():
    return SimpleNamespace(
        metadata={
            "case": {
                "id": "quick-1",
                "input": "Review the arXiv paper.",
                "expected_route": "academic-paper-review",
                "rationale": "PRIVATE HUMAN RATIONALE",
                "tags": ["quality"],
            },
            "agent_trace": {
                "input": "Review the arXiv paper.",
                "final_answer": "",
                "success": True,
                "thread_id": "t1",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "message_id": "m1",
                        "name": "read_file",
                        "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"},
                        "result": "skill body",
                        "error": None,
                    }
                ],
                "tool_call_chain": [["tc1"]],
                "quick_turn": {
                    "message_id": "m2",
                    "skill": "academic-paper-review",
                    "content": "I will review the paper along these axes.",
                },
                "messages": [],
                "artifacts": [],
                "errors": [],
            },
            "route_observation": {
                "observed_route": "academic-paper-review",
                "completed": True,
                "errors": [],
                "evidence": [],
            },
            "agent_success": True,
        }
    )


@pytest.mark.asyncio
async def test_quick_scorer_passes_threshold_and_hides_labels(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quick_turn_scorer("fake/judge", descriptions())(quick_state(), Target("academic-paper-review"))

    assert score.value == CORRECT
    assert score.metadata["quality_passed"] is True
    assert "expected_route" not in model.prompts[0]
    assert "PRIVATE HUMAN RATIONALE" not in model.prompts[0]


@pytest.mark.asyncio
async def test_quick_scorer_fails_below_threshold_or_fatal(monkeypatch):
    model = FakeModel(quick_judgment_json(turn_quality=2))
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quick_turn_scorer("fake/judge", descriptions())(quick_state(), Target("academic-paper-review"))

    assert score.value == INCORRECT
    assert score.metadata["quality_passed"] is False


@pytest.mark.asyncio
async def test_quick_scorer_rejects_failed_agent_run(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["agent_trace"]["success"] = False
    state.metadata["agent_trace"]["errors"] = ["agent failed"]

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert "infrastructure_error" in score.metadata
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_skips_none_expected_case(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["case"]["expected_route"] = "none"
    state.metadata["route_observation"]["observed_route"] = "none"

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("none"))

    assert score.value == NOANSWER
    assert score.metadata["not_applicable_none_case"] is True
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_skips_route_mismatch(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["route_observation"]["observed_route"] = "systematic-literature-review"

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert score.metadata["route_mismatch"] is True
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_reports_missing_quick_turn(monkeypatch):
    model = FakeModel(quick_judgment_json())
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)
    state = quick_state()
    state.metadata["agent_trace"]["quick_turn"] = None

    score = await quick_turn_scorer("fake/judge", descriptions())(state, Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert score.metadata["quick_turn_missing"] is True
    assert model.prompts == []


@pytest.mark.asyncio
async def test_quick_scorer_judge_failure_returns_noanswer(monkeypatch):
    model = FakeModel(quick_judgment_json(evidence_references=["tool_chain[9]", "quick_turn"]))
    monkeypatch.setattr("skill_eval.inspect_scorer.get_model", lambda _: model)

    score = await quick_turn_scorer("fake/judge", descriptions())(quick_state(), Target("academic-paper-review"))

    assert score.value == NOANSWER
    assert "unknown evidence" in score.metadata["judge_failure"]
