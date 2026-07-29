"""Data models for skill eval cases, specs, and judge outputs.

Mirrors the Langfuse dataset item shape defined in the spec
(docs/superpowers/specs/2026-07-29-skill-eval-callback.md §5).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "deerflow.skill-eval/v1"


class EvalMode(StrEnum):
    FAST = "fast"  # run to completion, deterministic assertions only
    FULL = "full"  # deterministic assertions + LLM output judge


class SkillEvalSpec(BaseModel):
    case_id: str
    mode: EvalMode = EvalMode.FAST
    expected_skill: str
    should_trigger: bool = True
    required_references: tuple[str, ...] = ()
    forbidden_references: tuple[str, ...] = ()
    output_rubric: list[dict[str, Any]] = Field(default_factory=list)


class OutputDimension(BaseModel):
    score: float
    reasoning: str = ""
    evidence: str = ""


class OutputGrade(BaseModel):
    passed: bool
    score: float
    dimensions: dict[str, OutputDimension] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)


class OutputBundle(BaseModel):
    final_answer: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    generated_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SkillEvalCase(BaseModel):
    """Local JSON file / Langfuse dataset item shape."""

    input: dict[str, Any]
    expected_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any]


def spec_from_case_metadata(metadata: dict[str, Any]) -> SkillEvalSpec:
    version = metadata.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version!r} (expected {SCHEMA_VERSION!r})")
    expected = metadata.get("expected_skill") or {}
    references = metadata.get("references") or {}
    return SkillEvalSpec(
        case_id=metadata["case_id"],
        mode=EvalMode(metadata.get("mode", EvalMode.FAST.value)),
        expected_skill=expected["name"],
        should_trigger=expected.get("should_trigger", True),
        required_references=tuple(references.get("required", ())),
        forbidden_references=tuple(references.get("forbidden", ())),
        output_rubric=list(metadata.get("output_rubric", [])),
    )
