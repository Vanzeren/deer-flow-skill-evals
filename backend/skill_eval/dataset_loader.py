from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from inspect_ai.dataset import Sample

from skill_eval.case_schema import CANDIDATE_SKILLS, RoutingCase

_EXPECTED_COUNTS = {
    "systematic-literature-review": 8,
    "academic-paper-review": 6,
    "none": 6,
}


def read_routing_cases(path: str | Path) -> list[RoutingCase]:
    source = Path(path)
    cases: list[RoutingCase] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(RoutingCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid routing case at {source}:{line_number}: {exc}") from exc
    return cases


def load_routing_samples(path: str | Path, *, tags: set[str] | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for case in read_routing_cases(path):
        if tags and not tags.issubset(case.tags):
            continue
        samples.append(
            Sample(
                id=case.id,
                input=case.input,
                target=case.expected_route,
                metadata={"case": case.model_dump()},
            )
        )
    return samples


def validate_poc_suite(cases: Sequence[RoutingCase]) -> None:
    errors: list[str] = []
    if len(cases) != 20:
        errors.append(f"expected 20 cases, found {len(cases)}")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case ids must be unique")
    counts = Counter(case.expected_route for case in cases)
    if dict(counts) != _EXPECTED_COUNTS:
        errors.append(f"expected route counts {_EXPECTED_COUNTS}, found {dict(counts)}")
    quality_count = sum("quality" in case.tags for case in cases)
    if quality_count != 4:
        errors.append(f"expected 4 quality cases, found {quality_count}")
    for case in cases:
        if case.input.lstrip().startswith("/"):
            errors.append(f"{case.id}: slash activation is not an autonomous routing case")
        leaked = [skill for skill in CANDIDATE_SKILLS if skill in case.input]
        if leaked:
            errors.append(f"{case.id}: prompt leaks skill name(s): {', '.join(leaked)}")
    if errors:
        raise ValueError("Invalid POC suite:\n- " + "\n- ".join(errors))
