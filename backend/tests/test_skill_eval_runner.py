import pytest
from langchain_core.messages import AIMessage

import deerflow.evals.skill.runner as runner_module
from deerflow.evals.skill.models import SkillEvalSpec
from deerflow.evals.skill.runner import run_skill_eval_case

pytestmark = pytest.mark.asyncio


class _FakeAgent:
    captured_config = None

    async def ainvoke(self, state, config=None):
        _FakeAgent.captured_config = config
        return {"messages": [AIMessage(content="final answer text")]}


@pytest.fixture
def _patch(monkeypatch):
    monkeypatch.setattr(runner_module, "make_lead_agent", lambda config: _FakeAgent())

    class _Skills:
        container_path = "/mnt/skills"

    class _Summarization:
        skill_file_read_tool_names = ["read_file", "read", "view", "cat"]

    class _AppConfig:
        skills = _Skills()
        summarization = _Summarization()

    monkeypatch.setattr(runner_module, "get_app_config", lambda: _AppConfig())


async def test_runner_builds_worker_equivalent_config(_patch):
    spec = SkillEvalSpec(case_id="c1", expected_skill="pdf-generation")
    report = await run_skill_eval_case(
        spec,
        messages=[{"role": "user", "content": "生成 PDF"}],
        use_langfuse=False,
    )
    cfg = _FakeAgent.captured_config
    assert cfg["configurable"]["thread_id"].startswith("skill-eval-c1-")
    ctx = cfg["context"]
    assert ctx["thread_id"] == cfg["configurable"]["thread_id"]
    assert "run_id" in ctx and "app_config" in ctx
    # Middlewares read runtime.context via configurable["__pregel_runtime"]
    # (mirrors worker.py); without it ThreadDataMiddleware crashes.
    pregel_runtime = cfg["configurable"]["__pregel_runtime"]
    assert pregel_runtime.context["thread_id"] == ctx["thread_id"]
    assert "app_config" in pregel_runtime.context
    assert any(type(c).__name__ == "SkillEvalCallback" for c in cfg["callbacks"])
    meta = cfg["metadata"]
    assert meta["langfuse_trace_name"] == "skill-eval:c1"
    assert meta["eval_case_id"] == "c1"
    assert report["agent_result"] == "final answer text"
    assert report["assertions"]["skill_hit"]["passed"] is False


async def test_runner_disables_ambient_tracing(_patch, monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING", "true")
    spec = SkillEvalSpec(case_id="c2", expected_skill="x")
    await run_skill_eval_case(spec, messages=[{"role": "user", "content": "hi"}], use_langfuse=False)
    import os

    assert os.environ["LANGFUSE_TRACING"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"


async def test_full_mode_invokes_judge_even_when_assertions_fail(_patch, monkeypatch):
    from deerflow.evals.skill.models import OutputGrade

    called = {}

    def fake_judge(bundle, rubric, *, judge_model_name=None):
        called["bundle"] = bundle
        return OutputGrade(passed=False, score=0.2)

    monkeypatch.setattr(runner_module, "evaluate_complete_output", fake_judge)
    spec = SkillEvalSpec(case_id="c3", mode="full", expected_skill="missing-skill")
    report = await run_skill_eval_case(spec, messages=[{"role": "user", "content": "hi"}], use_langfuse=False)
    assert report["assertions"]["skill_hit"]["passed"] is False
    assert report["output_judge"]["score"] == 0.2
    assert called["bundle"].final_answer == "final answer text"


async def test_generated_files_collected_from_successful_write_calls(_patch, monkeypatch):
    import uuid

    from deerflow.evals.skill.models import OutputGrade
    from deerflow.evals.skill.recorder import EvalRecorder

    captured = {}
    real_recorder_init = EvalRecorder.__init__

    def spy_init(self, **kwargs):
        real_recorder_init(self, **kwargs)
        captured["recorder"] = self

    monkeypatch.setattr(runner_module.EvalRecorder, "__init__", spy_init)
    monkeypatch.setattr(
        runner_module,
        "evaluate_complete_output",
        lambda bundle, rubric, *, judge_model_name=None: captured.setdefault("bundle", bundle) or OutputGrade(passed=True, score=1.0),
    )

    class _Agent(_FakeAgent):
        async def ainvoke(self, state, config=None):
            recorder = captured["recorder"]
            recorder.record_batch(
                [
                    {"name": "write_file", "args": {"path": "/mnt/user-data/workspace/chart.png"}, "id": "w1"},
                    {"name": "write_file", "args": {"path": "/mnt/user-data/workspace/broken.png"}, "id": "w2"},
                    {"name": "read_file", "args": {"path": "/x"}, "id": "r1"},
                ],
                run_id=uuid.uuid4(),
                parent_run_id=None,
            )
            recorder.record_tool_end("w1", ok=True)
            recorder.record_tool_end("w2", ok=False, error="disk full")
            return await super().ainvoke(state, config)

    monkeypatch.setattr(runner_module, "make_lead_agent", lambda config: _Agent())
    spec = SkillEvalSpec(case_id="c4", mode="full", expected_skill="x")
    await run_skill_eval_case(spec, messages=[{"role": "user", "content": "hi"}], use_langfuse=False)
    # only successful write_file calls land in the bundle, no duplicates, no reads
    assert captured["bundle"].generated_files == ["/mnt/user-data/workspace/chart.png"]
