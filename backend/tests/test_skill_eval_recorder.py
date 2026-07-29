import uuid

from deerflow.evals.skill.recorder import EvalRecorder


def _recorder():
    return EvalRecorder(trace_id="t" * 32)


def test_record_batch_and_dedup_by_tool_call_id():
    rec = _recorder()
    calls = [
        {"name": "read_file", "args": {"path": "/a"}, "id": "call_1"},
        {"name": "ls", "args": {"path": "/b"}, "id": "call_2"},
    ]
    rec.record_batch(calls, run_id=uuid.uuid4(), parent_run_id=None)
    rec.record_batch(calls, run_id=uuid.uuid4(), parent_run_id=None)  # replayed AIMessage
    assert len(rec.tool_batches) == 1
    assert [c.tool_call_id for c in rec.tool_batches[0]] == ["call_1", "call_2"]
    assert all(c.status == "pending" for c in rec.tool_batches[0])


def test_tool_end_pairs_and_marks_success():
    rec = _recorder()
    rec.record_batch([{"name": "read_file", "args": {}, "id": "call_1"}], run_id=uuid.uuid4(), parent_run_id=None)
    record = rec.record_tool_end("call_1", ok=True)
    assert record.status == "success"


def test_tool_end_error_status():
    rec = _recorder()
    rec.record_batch([{"name": "read_file", "args": {}, "id": "call_1"}], run_id=uuid.uuid4(), parent_run_id=None)
    record = rec.record_tool_end("call_1", ok=False, error="FileNotFoundError")
    assert record.status == "error"
    assert record.error == "FileNotFoundError"


def test_orphan_tool_end_creates_visible_record():
    rec = _recorder()
    record = rec.record_tool_end("call_x", ok=True)
    assert record.orphan is True
    assert record.batch_index == -1
    assert record in [c for batch in rec.tool_batches for c in batch]


def test_finalize_marks_pending_as_not_executed():
    rec = _recorder()
    rec.record_batch(
        [{"name": "a", "args": {}, "id": "c1"}, {"name": "b", "args": {}, "id": "c2"}],
        run_id=uuid.uuid4(),
        parent_run_id=None,
    )
    rec.record_tool_end("c1", ok=True)
    rec.finalize()
    statuses = {c.tool_call_id: c.status for c in rec.tool_batches[0]}
    assert statuses == {"c1": "success", "c2": "not_executed"}


def test_slash_activation_dedup_by_message_id():
    rec = _recorder()
    rec.record_skill_activation("pdf", kind="slash", message_id="m1__slash_activation")
    rec.record_skill_activation("pdf", kind="slash", message_id="m1__slash_activation")
    rec.record_skill_activation("pdf", kind="slash", message_id="m2__slash_activation")
    assert rec.loaded_skills == {"pdf": "slash"}


def test_skill_read_and_reference_recording():
    rec = _recorder()
    rec.record_skill_read("pdf-generation", "/mnt/skills/public/pdf-generation/SKILL.md")
    rec.record_reference("references/layout.md")
    assert rec.loaded_skills == {"pdf-generation": "read"}
    assert "/mnt/skills/public/pdf-generation/SKILL.md" in rec.loaded_skill_paths
    assert rec.delivered_references == {"references/layout.md"}


def test_tool_start_stash_roundtrip():
    rec = _recorder()
    rid = uuid.uuid4()
    rec.stash_tool_start(rid, "read_file", {"path": "/x"})
    assert rec.pop_tool_start(rid) == ("read_file", {"path": "/x"})
    assert rec.pop_tool_start(rid) is None


def test_finalize_drains_residual_tool_starts():
    rec = _recorder()
    rec.record_batch(
        [
            {"name": "read_file", "args": {"path": "/x"}, "id": "c1"},
            {"name": "ls", "args": {"path": "/b"}, "id": "c2"},
        ],
        run_id=uuid.uuid4(),
        parent_run_id=None,
    )
    rec.stash_tool_start(uuid.uuid4(), "read_file", {"path": "/x"})  # matches pending c1
    rec.stash_tool_start(uuid.uuid4(), "bash", {"command": "true"})  # matches nothing
    rec.finalize()
    batch = {c.tool_call_id: c for c in rec.tool_batches[0]}
    assert batch["c1"].status == "error"
    assert batch["c1"].error is not None and "never arrived" in batch["c1"].error
    assert batch["c2"].status == "not_executed"
    orphans = [c for batch in rec.tool_batches for c in batch if c.orphan]
    assert len(orphans) == 1
    assert orphans[0].status == "error"
    assert orphans[0].error is not None and "never arrived" in orphans[0].error
    assert rec._pending_tool_starts == {}
