"""Tests for deerflow.evals.skill.report_md.render_markdown_report."""

from deerflow.evals.skill.report_md import render_markdown_report


def _report(**kw):
    base = {
        "case_id": "c1",
        "mode": "fast",
        "trace_id": "t" * 32,
        "subagent_evidence": "unavailable",
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
        "tool_call_batches": [
            [{"id": "c1", "name": "read_file", "args": {"path": "/mnt/skills/public/pdf-generation/SKILL.md"}, "status": "success", "agent_path": ["lead"]}],
        ],
        "agent_result": "报告已生成",
    }
    base.update(kw)
    return base


def test_markdown_contains_assertion_results_and_missing_reference():
    md = render_markdown_report(_report())
    assert "# Skill Eval Report: c1" in md
    assert "| skill_hit | ✅ |" in md
    assert "| references_read | ❌ |" in md
    assert "**missing**: `references/b.md`" in md
    assert "read_file" in md and "/mnt/skills/public/pdf-generation/SKILL.md" in md
    assert "报告已生成" in md


def test_markdown_includes_judge_section_when_present():
    report = _report(
        output_judge={
            "passed": True,
            "score": 0.85,
            "dimensions": {"task_completion": {"score": 0.9, "reasoning": "覆盖全部要点"}},
            "failure_reasons": [],
        }
    )
    md = render_markdown_report(report)
    assert "## Output Judge" in md
    assert "| task_completion | 0.90 | 覆盖全部要点 |" in md


def test_markdown_omits_judge_section_when_absent():
    assert "## Output Judge" not in render_markdown_report(_report())


def test_trajectory_groups_parallel_calls_by_batch():
    report = _report(
        tool_call_batches=[
            [
                {"id": "c1", "name": "read_file", "args": {"path": "/a"}, "status": "success", "agent_path": ["lead"]},
                {"id": "c2", "name": "read_file", "args": {"path": "/b"}, "status": "success", "agent_path": ["lead"]},
                {"id": "c3", "name": "ls", "args": {"path": "/c"}, "status": "success", "agent_path": ["lead"]},
            ],
            [{"id": "c4", "name": "bash", "args": {"command": "node x.js"}, "status": "success", "agent_path": ["lead"]}],
        ]
    )
    md = render_markdown_report(report)
    assert "**Batch 0** — 3 parallel calls" in md
    assert "**Batch 1**\n" in md  # single-call batch has no parallel annotation
    # the three parallel calls stay inside the batch 0 group
    batch0_section = md.split("**Batch 0**")[1].split("**Batch 1**")[0]
    assert "/a" in batch0_section and "/b" in batch0_section and "/c" in batch0_section
    assert "node x.js" not in batch0_section
