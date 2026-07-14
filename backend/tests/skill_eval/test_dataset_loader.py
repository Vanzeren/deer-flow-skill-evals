import json
from collections import Counter

import pytest

from skill_eval.case_schema import CANDIDATE_SKILLS, RoutingCase
from skill_eval.dataset_loader import load_routing_samples, read_routing_cases, validate_poc_suite


def test_routing_case_rejects_unknown_label():
    with pytest.raises(ValueError):
        RoutingCase(
            id="bad-route",
            input="Review several papers",
            expected_route="deep-research",
            rationale="Not a benchmark class",
        )


@pytest.mark.parametrize("field", ["id", "input", "rationale"])
def test_routing_case_rejects_blank_required_text(field):
    values = {
        "id": "route-1",
        "input": "Review several papers",
        "expected_route": "systematic-literature-review",
        "rationale": "Multiple-paper synthesis",
    }
    values[field] = " \n "

    with pytest.raises(ValueError, match="must not be blank"):
        RoutingCase(**values)


def test_routing_case_rejects_duplicate_or_blank_tags():
    values = {
        "id": "route-1",
        "input": "Review several papers",
        "expected_route": "systematic-literature-review",
        "rationale": "Multiple-paper synthesis",
    }

    with pytest.raises(ValueError, match="unique"):
        RoutingCase(**values, tags=["quality", " quality "])
    with pytest.raises(ValueError, match="blank"):
        RoutingCase(**values, tags=["quality", " "])


def test_routing_case_rejects_obsolete_assertion_fields():
    with pytest.raises(ValueError):
        RoutingCase(
            id="route-1",
            input="Review several papers",
            expected_route="systematic-literature-review",
            rationale="Multiple-paper synthesis",
            assertions=[],
        )


def test_load_routing_samples_preserves_label_and_private_rationale(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "slr-001",
                "input": "Compare methods across five papers",
                "expected_route": "systematic-literature-review",
                "rationale": "Multiple-paper synthesis",
                "tags": ["implicit"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_routing_samples(path)

    assert len(samples) == 1
    assert samples[0].id == "slr-001"
    assert str(samples[0].target) == "systematic-literature-review"
    assert samples[0].metadata["case"]["rationale"] == "Multiple-paper synthesis"


def test_read_routing_cases_reports_invalid_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"route-1","input":"A","expected_route":"none","rationale":"A"}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"cases\.jsonl:2"):
        read_routing_cases(path)


def test_quality_filter_selects_only_tagged_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {"id": "a", "input": "A", "expected_route": "none", "rationale": "A", "tags": []},
        {"id": "b", "input": "B", "expected_route": "none", "rationale": "B", "tags": ["quality"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert [sample.id for sample in load_routing_samples(path, tags={"quality"})] == ["b"]


def test_committed_poc_suite_is_balanced_and_non_leaking():
    cases = read_routing_cases("cases/literature_skill_routing.jsonl")
    validate_poc_suite(cases)

    assert len(cases) == 20
    assert Counter(case.expected_route for case in cases) == {
        "systematic-literature-review": 8,
        "academic-paper-review": 6,
        "none": 6,
    }
    assert sum("quality" in case.tags for case in cases) == 4
    assert len({case.id for case in cases}) == 20
    for case in cases:
        assert not case.input.lstrip().startswith("/")
        assert all(skill not in case.input for skill in CANDIDATE_SKILLS)
