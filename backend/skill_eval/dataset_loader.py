from pathlib import Path

from inspect_ai.dataset import Sample

from skill_eval.case_schema import SkillEvalCase


def load_skill_cases(path: str, tags: list[str] | None = None, difficulty: str | None = None, required_skill: str | None = None) -> list[Sample]:
    samples: list[Sample] = []

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            case = SkillEvalCase.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Invalid skill eval case at {path}:{line_number}: {exc}") from exc

        if tags and not set(tags).issubset(set(case.tags)):
            continue
        if difficulty and case.difficulty != difficulty:
            continue
        if required_skill and required_skill not in case.required_skills:
            continue

        samples.append(
            Sample(
                id=case.id,
                input=case.input,
                target=case.target or "",
                metadata={"case": case.model_dump()},
            )
        )

    return samples
