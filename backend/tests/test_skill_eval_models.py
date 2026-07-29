import pytest

from deerflow.evals.skill.models import EvalMode, SkillEvalCase, spec_from_case_metadata


def _case_metadata(**overrides):
    base = {
        "schema_version": "deerflow.skill-eval/v1",
        "case_id": "pdf-generation-001",
        "mode": "full",
        "expected_skill": {"name": "pdf-generation", "should_trigger": True},
        "references": {"required": ["references/layout.md"], "forbidden": []},
        "output_rubric": [{"name": "task_completion", "weight": 1.0}],
    }
    base.update(overrides)
    return base


def test_spec_from_case_metadata_full():
    spec = spec_from_case_metadata(_case_metadata())
    assert spec.case_id == "pdf-generation-001"
    assert spec.mode == EvalMode.FULL
    assert spec.expected_skill == "pdf-generation"
    assert spec.should_trigger is True
    assert spec.required_references == ("references/layout.md",)
    assert spec.forbidden_references == ()
    assert spec.output_rubric == [{"name": "task_completion", "weight": 1.0}]


def test_spec_defaults():
    meta = _case_metadata()
    del meta["mode"], meta["references"], meta["output_rubric"]
    meta["expected_skill"] = {"name": "x"}
    spec = spec_from_case_metadata(meta)
    assert spec.mode == EvalMode.FAST
    assert spec.should_trigger is True
    assert spec.required_references == ()
    assert spec.output_rubric == []


def test_spec_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        spec_from_case_metadata(_case_metadata(schema_version="nope"))


def test_case_roundtrip_from_dataset_item_shape():
    item = {
        "input": {"messages": [{"role": "user", "content": "hi"}], "files": []},
        "expected_output": {"description": "d"},
        "metadata": _case_metadata(),
    }
    case = SkillEvalCase.model_validate(item)
    assert case.input["messages"][0]["role"] == "user"
    assert spec_from_case_metadata(case.metadata).case_id == "pdf-generation-001"
