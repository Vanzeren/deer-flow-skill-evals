import uuid

from deerflow.evals.skill.assertions import build_eval_report
from deerflow.evals.skill.models import SkillEvalSpec
from deerflow.evals.skill.recorder import EvalRecorder


def _spec(**kw):
    base = dict(
        case_id="c1",
        expected_skill="pdf-generation",
        required_references=("references/layout.md",),
    )
    base.update(kw)
    return SkillEvalSpec(**base)


def _recorder_with(skill=None, refs=(), calls=()):
    rec = EvalRecorder(trace_id="t" * 32)
    if skill:
        rec.record_skill_read(skill, f"/mnt/skills/public/{skill}/SKILL.md")
    for r in refs:
        rec.record_reference(r)
    if calls:
        rec.record_batch(list(calls), run_id=uuid.uuid4(), parent_run_id=None)
    rec.finalize()
    return rec


def test_report_pass_path():
    report = build_eval_report(
        spec=_spec(),
        recorder=_recorder_with("pdf-generation", ("references/layout.md",)),
        agent_result="done",
    )
    assert report["assertions"]["skill_hit"]["passed"] is True
    assert report["assertions"]["references_read"]["passed"] is True
    assert report["subagent_evidence"] == "unavailable"
    assert report["trace_id"] == "t" * 32


def test_missing_reference_fails_with_detail():
    report = build_eval_report(
        spec=_spec(required_references=("references/a.md", "references/b.md")),
        recorder=_recorder_with("pdf-generation", ("references/a.md",)),
        agent_result="x",
    )
    refs = report["assertions"]["references_read"]
    assert refs["passed"] is False
    assert refs["missing"] == ["references/b.md"]


def test_forbidden_reference_fails():
    report = build_eval_report(
        spec=_spec(required_references=(), forbidden_references=("references/draft.md",)),
        recorder=_recorder_with("pdf-generation", ("references/draft.md",)),
        agent_result="x",
    )
    refs = report["assertions"]["references_read"]
    assert refs["passed"] is False
    assert refs["forbidden_hit"] == ["references/draft.md"]


def test_negative_case_should_not_trigger():
    report = build_eval_report(
        spec=_spec(should_trigger=False),
        recorder=_recorder_with("pdf-generation"),
        agent_result="x",
    )
    assert report["assertions"]["skill_hit"]["passed"] is False
    clean = build_eval_report(
        spec=_spec(should_trigger=False),
        recorder=_recorder_with(None),
        agent_result="x",
    )
    assert clean["assertions"]["skill_hit"]["passed"] is True


def test_tool_call_batches_serialized_with_not_executed():
    rec = _recorder_with(calls=[{"name": "a", "args": {"x": 1}, "id": "c1"}])
    report = build_eval_report(spec=_spec(), recorder=rec, agent_result="r")
    batch = report["tool_call_batches"][0][0]
    assert batch == {
        "id": "c1",
        "name": "a",
        "args": {"x": 1},
        "status": "not_executed",
        "agent_path": ["lead"],
    }
    assert report["agent_result"] == "r"
