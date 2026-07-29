"""Deterministic assertions over EvalRecorder evidence. Spec §8."""

from __future__ import annotations

from deerflow.evals.skill.models import SkillEvalSpec
from deerflow.evals.skill.recorder import EvalRecorder


def build_eval_report(*, spec: SkillEvalSpec, recorder: EvalRecorder, agent_result: str) -> dict:
    skill_hit = spec.expected_skill in recorder.loaded_skills
    required = set(spec.required_references)
    delivered = set(recorder.delivered_references)
    missing = sorted(required - delivered)
    forbidden_hit = sorted(set(spec.forbidden_references) & delivered)
    reference_passed = not missing and not forbidden_hit
    routing_passed = skill_hit if spec.should_trigger else not skill_hit
    return {
        "case_id": spec.case_id,
        "mode": str(spec.mode),
        "trace_id": recorder.trace_id,
        "subagent_evidence": "unavailable",  # spec §6.3: V1 resolved as not viable
        "assertions": {
            "skill_hit": {
                "passed": routing_passed,
                "expected": spec.expected_skill,
                "observed": sorted(recorder.loaded_skills),
            },
            "references_read": {
                "passed": reference_passed,
                "required": sorted(required),
                "delivered": sorted(delivered),
                "missing": missing,
                "forbidden_hit": forbidden_hit,
            },
        },
        "tool_call_batches": [
            [
                {
                    "id": c.tool_call_id,
                    "name": c.name,
                    "args": c.args,
                    "status": c.status,
                    "agent_path": list(c.agent_path),
                }
                for c in batch
            ]
            for batch in recorder.tool_batches
        ],
        "agent_result": agent_result,
    }
