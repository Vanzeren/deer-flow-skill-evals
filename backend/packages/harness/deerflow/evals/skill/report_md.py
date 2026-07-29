"""Render an eval report dict as a human-readable Markdown document."""

from __future__ import annotations

from typing import Any


def _check(passed: bool) -> str:
    return "✅" if passed else "❌"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Skill Eval Report: {report['case_id']}")
    lines.append("")
    lines.append(f"- mode: `{report['mode']}`")
    lines.append(f"- trace_id: `{report['trace_id']}`")
    lines.append(f"- subagent_evidence: `{report['subagent_evidence']}`")
    lines.append("")

    assertions = report["assertions"]
    skill_hit = assertions["skill_hit"]
    refs = assertions["references_read"]
    lines.append("## Assertions")
    lines.append("")
    lines.append("| Assertion | Result | Detail |")
    lines.append("|---|---|---|")
    observed = ", ".join(f"`{s}`" for s in skill_hit["observed"]) or "—"
    lines.append(f"| skill_hit | {_check(skill_hit['passed'])} | expected `{skill_hit['expected']}`, observed: {observed} |")
    lines.append(f"| references_read | {_check(refs['passed'])} | delivered {len(refs['delivered'])}/{len(refs['required'])} required |")
    lines.append("")
    if refs["required"] or refs["delivered"]:
        lines.append("### References")
        lines.append("")
        lines.append(f"- required: {', '.join(f'`{r}`' for r in refs['required']) or '—'}")
        lines.append(f"- delivered: {', '.join(f'`{r}`' for r in refs['delivered']) or '—'}")
        if refs["missing"]:
            lines.append(f"- **missing**: {', '.join(f'`{r}`' for r in refs['missing'])}")
        if refs["forbidden_hit"]:
            lines.append(f"- **forbidden_hit**: {', '.join(f'`{r}`' for r in refs['forbidden_hit'])}")
        lines.append("")

    judge = report.get("output_judge")
    if judge:
        lines.append("## Output Judge")
        lines.append("")
        lines.append(f"- overall: {_check(judge['passed'])} score **{judge['score']:.2f}**")
        lines.append("")
        if judge.get("dimensions"):
            lines.append("| Dimension | Score | Reasoning |")
            lines.append("|---|---|---|")
            for name, dim in judge["dimensions"].items():
                lines.append(f"| {name} | {dim['score']:.2f} | {dim.get('reasoning', '')} |")
            lines.append("")
        if judge.get("failure_reasons"):
            lines.append("Failure reasons:")
            for reason in judge["failure_reasons"]:
                lines.append(f"- {reason}")
            lines.append("")

    lines.append("## Tool Call Trajectory")
    lines.append("")
    for index, batch in enumerate(report["tool_call_batches"]):
        size = len(batch)
        header = f"**Batch {index}** — {size} parallel calls" if size > 1 else f"**Batch {index}**"
        lines.append(header)
        lines.append("")
        lines.append("| Tool | Status | Args |")
        lines.append("|---|---|---|")
        for call in batch:
            args = call["args"].get("path") or call["args"].get("command") or str(call["args"])
            if len(str(args)) > 80:
                args = str(args)[:77] + "..."
            lines.append(f"| `{call['name']}` | {call['status']} | `{args}` |")
        lines.append("")

    lines.append("## Agent Result")
    lines.append("")
    lines.append(report["agent_result"] or "_(empty)_")
    lines.append("")
    return "\n".join(lines)
