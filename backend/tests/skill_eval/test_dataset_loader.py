import pytest

from skill_eval.dataset_loader import load_skill_cases


def test_load_skill_cases_preserves_target_and_case_metadata(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text(
        '{"id":"case-1","input":"Say hi","target":"hi","required_skills":["demo"],"candidate_skills":["skills/demo"],"assertions":[{"name":"output_contains","target":"hi"}],"tags":["smoke"],"difficulty":"smoke"}\n',
        encoding="utf-8",
    )

    samples = load_skill_cases(str(case_file))

    assert len(samples) == 1
    assert samples[0].id == "case-1"
    assert samples[0].input == "Say hi"
    assert samples[0].target == "hi"
    assert samples[0].metadata["case"]["assertions"][0]["name"] == "output_contains"


def test_load_skill_cases_ignores_blank_lines(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text('\n{"id":"case-1","input":"Say hi"}\n\n', encoding="utf-8")

    samples = load_skill_cases(str(case_file))

    assert len(samples) == 1


def test_load_skill_cases_reports_invalid_line(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text('{"id":"case-1","input":"ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_skill_cases(str(case_file))

    assert "cases.jsonl:2" in str(exc_info.value)


def test_load_skill_cases_filters_tags_difficulty_and_required_skill(tmp_path):
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text(
        "\n".join(
            [
                '{"id":"keep","input":"A","required_skills":["demo"],"tags":["tool-use","smoke"],"difficulty":"smoke"}',
                '{"id":"drop-tag","input":"B","required_skills":["demo"],"tags":["other"],"difficulty":"smoke"}',
                '{"id":"drop-difficulty","input":"C","required_skills":["demo"],"tags":["tool-use","smoke"],"difficulty":"hard"}',
                '{"id":"drop-skill","input":"D","required_skills":["other"],"tags":["tool-use","smoke"],"difficulty":"smoke"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_skill_cases(str(case_file), tags=["tool-use"], difficulty="smoke", required_skill="demo")

    assert [sample.id for sample in samples] == ["keep"]
