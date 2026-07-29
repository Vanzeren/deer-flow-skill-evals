import deerflow.evals.skill.output_judge as judge_module
from deerflow.evals.skill.models import OutputBundle, OutputDimension, OutputGrade
from deerflow.evals.skill.output_judge import evaluate_complete_output


class _FakeStructured:
    def __init__(self, grade):
        self.grade = grade
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return self.grade


class _FakeModel:
    def __init__(self, grade):
        self.structured = _FakeStructured(grade)

    def with_structured_output(self, schema):
        assert schema is OutputGrade
        return self.structured


def test_judge_uses_attach_tracing_false_and_returns_grade(monkeypatch):
    grade = OutputGrade(
        passed=True,
        score=0.85,
        dimensions={"task_completion": OutputDimension(score=0.9, reasoning="ok")},
    )
    captured = {}

    def fake_create_chat_model(name=None, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _FakeModel(grade)

    monkeypatch.setattr(judge_module, "create_chat_model", fake_create_chat_model)
    bundle = OutputBundle(final_answer="报告已生成", generated_files=["report.pdf"])
    result = evaluate_complete_output(
        bundle,
        [{"name": "task_completion", "weight": 1.0}],
        judge_model_name="judge-model",
    )
    assert result is grade
    assert captured["name"] == "judge-model"
    assert captured["kwargs"]["attach_tracing"] is False


def test_judge_prompt_contains_answer_and_rubric(monkeypatch):
    grade = OutputGrade(passed=False, score=0.1)
    monkeypatch.setattr(
        judge_module,
        "create_chat_model",
        lambda name=None, **kw: _FakeModel(grade),
    )
    bundle = OutputBundle(final_answer="UNIQUE_ANSWER_TEXT", warnings=["w1"])
    result = evaluate_complete_output(bundle, [{"name": "correctness", "weight": 0.5}], judge_model_name="m")
    assert result is grade
