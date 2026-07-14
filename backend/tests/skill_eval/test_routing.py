from deerflow.client import StreamEvent
from skill_eval.case_schema import CANDIDATE_SKILLS
from skill_eval.routing import RoutingObserver


def ai_tools(message_id: str, *calls: dict) -> StreamEvent:
    return StreamEvent(
        type="messages-tuple",
        data={"type": "ai", "id": message_id, "content": "", "tool_calls": list(calls)},
    )


def tool_result(call_id: str, name: str, content: str) -> StreamEvent:
    return StreamEvent(
        type="messages-tuple",
        data={
            "type": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
            "id": f"result-{call_id}",
        },
    )


def values(*messages: dict) -> StreamEvent:
    return StreamEvent(type="values", data={"messages": list(messages)})


def end() -> StreamEvent:
    return StreamEvent(type="end", data={"usage": {}})


def test_describe_then_load_selects_skill_only_after_successful_read():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    assert (
        observer.feed(
            ai_tools(
                "m1",
                {
                    "id": "d1",
                    "name": "describe_skill",
                    "args": {"name": "systematic-literature-review"},
                },
            )
        )
        is False
    )
    assert observer.feed(tool_result("d1", "describe_skill", "description")) is False
    assert (
        observer.feed(
            ai_tools(
                "m2",
                {
                    "id": "r1",
                    "name": "read_file",
                    "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"},
                },
            )
        )
        is False
    )
    assert observer.feed(tool_result("r1", "read_file", "---\nname: systematic-literature-review\n---")) is True

    result = observer.finalize(stream_completed=False)
    assert result.completed is True
    assert result.observed_route == "systematic-literature-review"
    assert [e.kind for e in result.evidence] == ["described", "load_requested", "loaded"]
    assert [e.id for e in result.evidence] == [
        "route_evidence[0]",
        "route_evidence[1]",
        "route_evidence[2]",
    ]


def test_values_snapshot_completes_streamed_tool_args():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(ai_tools("m1", {"id": "r1", "name": "read_file", "args": {}}))

    assert observer.feed(
        values(
            {
                "type": "ai",
                "id": "m1",
                "tool_calls": [
                    {
                        "id": "r1",
                        "name": "read_file",
                        "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"},
                    }
                ],
            },
            {
                "type": "tool",
                "tool_call_id": "r1",
                "content": "---\nname: systematic-literature-review\n---",
            },
        )
    )

    result = observer.finalize(stream_completed=False)
    assert result.observed_route == "systematic-literature-review"
    assert [e.kind for e in result.evidence] == ["load_requested", "loaded"]


def test_describe_without_load_finishes_as_none():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {"id": "d1", "name": "describe_skill", "args": {"name": "academic-paper-review"}},
        )
    )
    observer.feed(tool_result("d1", "describe_skill", "description"))
    observer.feed(end())
    assert observer.finalize(stream_completed=True).observed_route == "none"


def test_failed_skill_read_does_not_select_route():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {
                "id": "r1",
                "name": "read_file",
                "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"},
            },
        )
    )
    assert observer.feed(tool_result("r1", "read_file", "Error: file not found")) is False
    observer.feed(end())
    result = observer.finalize(stream_completed=True)
    assert result.observed_route == "none"
    assert [e.kind for e in result.evidence] == ["load_requested", "load_failed"]


def test_two_successful_loads_in_same_message_are_ambiguous():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {
                "id": "r1",
                "name": "read_file",
                "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"},
            },
            {
                "id": "r2",
                "name": "read_file",
                "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"},
            },
        )
    )
    assert observer.feed(tool_result("r1", "read_file", "skill one")) is False
    assert observer.feed(tool_result("r2", "read_file", "skill two")) is True
    assert observer.finalize(stream_completed=False).observed_route == "ambiguous"


def test_later_batches_do_not_replace_first_route():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {
                "id": "r1",
                "name": "read_file",
                "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"},
            },
        )
    )
    assert observer.feed(tool_result("r1", "read_file", "skill one")) is True
    assert (
        observer.feed(
            ai_tools(
                "m2",
                {
                    "id": "r2",
                    "name": "read_file",
                    "args": {"path": "/mnt/skills/public/academic-paper-review/SKILL.md"},
                },
            )
        )
        is True
    )
    assert observer.finalize(stream_completed=True).observed_route == "systematic-literature-review"


def test_unrelated_read_and_orphan_result_are_not_route_evidence():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {
                "id": "r1",
                "name": "read_file",
                "args": {"path": "/mnt/user-data/workspace/paper.md"},
            },
        )
    )
    observer.feed(tool_result("r1", "read_file", "paper"))
    observer.feed(tool_result("orphan", "read_file", "skill"))
    observer.feed(end())
    result = observer.finalize(stream_completed=True)
    assert result.observed_route == "none"
    assert result.evidence == []


def test_shadow_skill_path_is_not_route_evidence():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {
                "id": "r1",
                "name": "read_file",
                "args": {"path": "/tmp/systematic-literature-review/SKILL.md"},
            },
        )
    )
    observer.feed(tool_result("r1", "read_file", "not the mounted skill"))
    observer.feed(end())
    assert observer.finalize(stream_completed=True).observed_route == "none"


def test_stream_failure_is_not_none():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.fail("stream timed out")
    result = observer.finalize(stream_completed=False)
    assert result.completed is False
    assert result.observed_route is None
    assert result.errors == ["stream timed out"]


def test_unresolved_candidate_read_is_infrastructure_failure():
    observer = RoutingObserver(CANDIDATE_SKILLS)
    observer.feed(
        ai_tools(
            "m1",
            {
                "id": "r1",
                "name": "read_file",
                "args": {"path": "/mnt/skills/public/systematic-literature-review/SKILL.md"},
            },
        )
    )
    observer.feed(end())

    result = observer.finalize(stream_completed=True)

    assert result.completed is False
    assert result.observed_route is None
    assert result.errors == ["stream ended with unresolved candidate skill reads"]
