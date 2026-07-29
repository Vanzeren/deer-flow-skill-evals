# backend/tests/test_skill_eval_langfuse_sink.py
from deerflow.evals.skill.langfuse_sink import publish_langfuse_scores, score_id


class FakeClient:
    def __init__(self):
        self.scores = []

    def create_score(self, **kwargs):
        self.scores.append(kwargs)


def _report(**kw):
    base = {
        "case_id": "c1",
        "trace_id": "t" * 32,
        "assertions": {
            "skill_hit": {"passed": True, "expected": "pdf-generation", "observed": ["pdf-generation"]},
            "references_read": {
                "passed": False,
                "required": ["references/a.md", "references/b.md"],
                "delivered": ["references/a.md"],
                "missing": ["references/b.md"],
                "forbidden_hit": [],
            },
        },
        "tool_call_batches": [[{"id": "c1", "name": "read_file", "args": {}, "status": "success", "agent_path": ["lead"]}]],
    }
    base.update(kw)
    return base


def test_score_id_stable_and_distinct_per_metric():
    assert score_id("c1", "skill_hit") == score_id("c1", "skill_hit")
    assert score_id("c1", "skill_hit") != score_id("c1", "references_read")
    assert len(score_id("c1", "skill_hit")) == 32


def test_publishes_boolean_and_numeric_scores_with_idempotency_keys():
    client = FakeClient()
    ids = publish_langfuse_scores(client, trace_id="t" * 32, report=_report())
    by_name = {s["name"]: s for s in client.scores}
    assert set(by_name) == {"skill_hit", "references_read", "reference_coverage"}
    assert by_name["skill_hit"]["data_type"] == "BOOLEAN"
    assert by_name["skill_hit"]["value"] == 1.0
    assert by_name["references_read"]["value"] == 0.0
    assert by_name["reference_coverage"]["data_type"] == "NUMERIC"
    assert by_name["reference_coverage"]["value"] == 0.5
    for score in client.scores:
        assert score["trace_id"] == "t" * 32
        assert score["score_id"] == score_id("c1", score["name"])
    assert len(ids) == 3


def test_tool_call_batches_ride_in_score_metadata():
    client = FakeClient()
    publish_langfuse_scores(client, trace_id="t" * 32, report=_report())
    meta = {s["name"]: s.get("metadata") for s in client.scores}
    assert meta["skill_hit"]["tool_call_batches"][0][0]["name"] == "read_file"


def test_empty_required_gives_full_coverage():
    client = FakeClient()
    report = _report()
    report["assertions"]["references_read"]["required"] = []
    publish_langfuse_scores(client, trace_id="t" * 32, report=report)
    cov = next(s for s in client.scores if s["name"] == "reference_coverage")
    assert cov["value"] == 1.0


def test_full_mode_adds_judge_scores_per_dimension():
    client = FakeClient()
    report = _report(
        output_judge={
            "passed": True,
            "score": 0.8,
            "dimensions": {"task_completion": {"score": 0.9, "reasoning": "ok", "evidence": ""}},
            "failure_reasons": [],
        }
    )
    publish_langfuse_scores(client, trace_id="t" * 32, report=report)
    by_name = {s["name"]: s for s in client.scores}
    assert by_name["output_quality"]["value"] == 0.8
    assert by_name["output_task_completion"]["value"] == 0.9
    assert by_name["output_task_completion"]["data_type"] == "NUMERIC"
