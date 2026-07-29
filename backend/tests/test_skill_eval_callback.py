# backend/tests/test_skill_eval_callback.py
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from deerflow.evals.skill.callback import SkillEvalCallback
from deerflow.evals.skill.recorder import EvalRecorder

pytestmark = pytest.mark.asyncio

CONTAINER = "/mnt/skills"
SKILL_MD = f"{CONTAINER}/public/pdf-generation/SKILL.md"


def _callback(expected_skill="pdf-generation"):
    recorder = EvalRecorder(trace_id="t" * 32)
    return recorder, SkillEvalCallback(recorder, expected_skill=expected_skill, container_path=CONTAINER)


def _llm_result_with_tool_calls(*ids):
    calls = [{"name": "read_file", "args": {"path": SKILL_MD}, "id": i} for i in ids]
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(content="", tool_calls=calls))]])


async def test_implicit_skill_hit_via_tool_start_end_pairing():
    recorder, cb = _callback()
    rid = uuid.uuid4()
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": SKILL_MD})
    await cb.on_tool_end(
        ToolMessage(content="# PDF Generation", tool_call_id="call_1", name="read_file"),
        run_id=rid,
    )
    assert recorder.loaded_skills == {"pdf-generation": "read"}
    assert SKILL_MD in recorder.loaded_skill_paths


async def test_error_tool_message_does_not_count_as_skill_hit():
    recorder, cb = _callback()
    rid = uuid.uuid4()
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": SKILL_MD})
    await cb.on_tool_end(
        ToolMessage(content="Error: file not found", tool_call_id="call_1", name="read_file", status="error"),
        run_id=rid,
    )
    assert recorder.loaded_skills == {}


async def test_wrong_skill_directory_is_negative():
    recorder, cb = _callback(expected_skill="pdf-generation")
    rid = uuid.uuid4()
    other = f"{CONTAINER}/public/other-skill/SKILL.md"
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": other})
    await cb.on_tool_end(ToolMessage(content="# Other", tool_call_id="c", name="read_file"), run_id=rid)
    assert recorder.loaded_skills == {"other-skill": "read"}  # recorded, but != expected


async def test_custom_category_segment_matches():
    recorder, cb = _callback()
    rid = uuid.uuid4()
    path = f"{CONTAINER}/custom/pdf-generation/SKILL.md"
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": path})
    await cb.on_tool_end(ToolMessage(content="# PDF", tool_call_id="c", name="read_file"), run_id=rid)
    assert recorder.loaded_skills == {"pdf-generation": "read"}


async def test_reference_read_recorded_with_relative_path():
    recorder, cb = _callback()
    rid = uuid.uuid4()
    ref = f"{CONTAINER}/public/pdf-generation/references/layout.md"
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": ref})
    await cb.on_tool_end(ToolMessage(content="layout rules", tool_call_id="c", name="read_file"), run_id=rid)
    assert recorder.delivered_references == {"references/layout.md"}


async def test_reference_of_other_skill_ignored():
    recorder, cb = _callback(expected_skill="pdf-generation")
    rid = uuid.uuid4()
    ref = f"{CONTAINER}/public/other-skill/references/x.md"
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": ref})
    await cb.on_tool_end(ToolMessage(content="x", tool_call_id="c", name="read_file"), run_id=rid)
    assert recorder.delivered_references == set()


async def test_on_llm_end_records_one_batch_and_dedups_replay():
    recorder, cb = _callback()
    result = _llm_result_with_tool_calls("c1", "c2", "c3")
    await cb.on_llm_end(result, run_id=uuid.uuid4())
    await cb.on_llm_end(result, run_id=uuid.uuid4())  # jump_to model replay
    assert len(recorder.tool_batches) == 1
    assert [c.tool_call_id for c in recorder.tool_batches[0]] == ["c1", "c2", "c3"]


async def test_slash_activation_detected_and_deduped():
    recorder, cb = _callback()
    msg = HumanMessage(
        content=f'<slash_skill_activation>\n<skill name="pdf-generation" category="public" path="{SKILL_MD}" sha256="abc">x</skill>\n</slash_skill_activation>',
        id="u1__slash_activation",
        additional_kwargs={"slash_skill_activation": True, "hide_from_ui": True},
    )
    await cb.on_chat_model_start({}, [[msg]], run_id=uuid.uuid4())
    await cb.on_chat_model_start({}, [[msg]], run_id=uuid.uuid4())
    assert recorder.loaded_skills == {"pdf-generation": "slash"}


async def test_non_read_tool_ignored_for_skill_judgment():
    recorder, cb = _callback()
    rid = uuid.uuid4()
    await cb.on_tool_start({"name": "bash"}, "", run_id=rid, inputs={"command": "cat " + SKILL_MD})
    await cb.on_tool_end(ToolMessage(content="# PDF", tool_call_id="c", name="bash"), run_id=rid)
    assert recorder.loaded_skills == {}


async def test_on_tool_error_pairs_with_pending_batch_record():
    recorder, cb = _callback()
    args = {"path": "/x"}
    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="", tool_calls=[{"name": "read_file", "args": args, "id": "c1"}]))]])
    await cb.on_llm_end(result, run_id=uuid.uuid4())
    rid = uuid.uuid4()
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs=args)
    await cb.on_tool_error(RuntimeError("boom"), run_id=rid)
    batch_record = recorder.tool_batches[0][0]
    assert batch_record.tool_call_id == "c1"
    assert batch_record.status == "error"
    assert batch_record.error == "boom"
    all_records = [c for batch in recorder.tool_batches for c in batch]
    assert all_records == [batch_record]  # no orphan record created


async def test_on_tool_error_without_pending_match_falls_back_to_orphan():
    recorder, cb = _callback()
    rid = uuid.uuid4()
    await cb.on_tool_start({"name": "read_file"}, "", run_id=rid, inputs={"path": "/x"})
    await cb.on_tool_error(RuntimeError("boom"), run_id=rid)
    all_records = [c for batch in recorder.tool_batches for c in batch]
    assert len(all_records) == 1
    assert all_records[0].orphan is True
    assert all_records[0].status == "error"
    assert all_records[0].error == "boom"
