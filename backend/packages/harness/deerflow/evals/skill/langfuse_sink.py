"""Publish eval scores to Langfuse with idempotent score ids. Spec §9.

tool_call_batches ride in score metadata so the trace page shows the
trajectory without opening the local JSON report.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def score_id(case_id: str, metric: str) -> str:
    return hashlib.sha256(f"deerflow:{case_id}:{metric}".encode()).hexdigest()[:32]


def publish_langfuse_scores(client: Any, *, trace_id: str, report: dict) -> list[str]:
    case_id = report["case_id"]
    assertions = report["assertions"]
    published: list[str] = []

    def _send(metric: str, value: float, data_type: str, comment: str, metadata: dict | None = None) -> None:
        sid = score_id(case_id, metric)
        client.create_score(
            name=metric,
            value=value,
            trace_id=trace_id,
            score_id=sid,
            data_type=data_type,
            comment=comment,
            metadata=metadata,
        )
        published.append(sid)

    trajectory_meta = {"tool_call_batches": report.get("tool_call_batches", [])}
    _send(
        "skill_hit",
        1.0 if assertions["skill_hit"]["passed"] else 0.0,
        "BOOLEAN",
        json.dumps(assertions["skill_hit"], ensure_ascii=False),
        metadata=trajectory_meta,
    )
    _send(
        "references_read",
        1.0 if assertions["references_read"]["passed"] else 0.0,
        "BOOLEAN",
        json.dumps(assertions["references_read"], ensure_ascii=False),
    )
    refs = assertions["references_read"]
    required, delivered = refs["required"], refs["delivered"]
    coverage = 1.0 if not required else len(set(required) & set(delivered)) / len(required)
    _send("reference_coverage", coverage, "NUMERIC", f"{len(set(required) & set(delivered))}/{len(required)}")

    judge = report.get("output_judge")
    if judge:
        _send("output_quality", float(judge["score"]), "NUMERIC", json.dumps(judge.get("failure_reasons", []), ensure_ascii=False))
        for dim_name, dim in judge.get("dimensions", {}).items():
            _send(
                f"output_{dim_name}",
                float(dim["score"]),
                "NUMERIC",
                json.dumps({"reasoning": dim.get("reasoning", ""), "evidence": dim.get("evidence", "")}, ensure_ascii=False),
            )
    return published
