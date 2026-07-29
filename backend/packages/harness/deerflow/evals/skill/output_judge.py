"""Full-mode output judge. Spec §10: judge only the final output; routing
and references are already covered by deterministic assertions.
"""

from __future__ import annotations

import json

from deerflow.evals.skill.models import OutputBundle, OutputGrade
from deerflow.models.factory import create_chat_model

_PROMPT_TEMPLATE = """You are grading the final output of an AI agent against a rubric.

## Final answer
{final_answer}

## Generated files
{generated_files}

## Artifact summaries
{artifacts}

## Warnings
{warnings}

## Rubric dimensions (name, weight)
{rubric}

Grade each rubric dimension with a score in [0, 1], a short reasoning, and
concrete evidence from the output. Then give an overall weighted score and
pass/fail. Do NOT re-judge skill routing or reference reading; those are
covered elsewhere.
"""


def evaluate_complete_output(
    bundle: OutputBundle,
    rubric: list[dict],
    *,
    judge_model_name: str | None = None,
) -> OutputGrade:
    model = create_chat_model(judge_model_name, attach_tracing=False)
    grader = model.with_structured_output(OutputGrade)
    prompt = _PROMPT_TEMPLATE.format(
        final_answer=bundle.final_answer,
        generated_files=json.dumps(bundle.generated_files, ensure_ascii=False),
        artifacts=json.dumps(bundle.artifacts, ensure_ascii=False),
        warnings=json.dumps(bundle.warnings, ensure_ascii=False),
        rubric=json.dumps(rubric, ensure_ascii=False),
    )
    return grader.invoke(prompt)
