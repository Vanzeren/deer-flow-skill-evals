# Task 2 Report: Add core case and trace schemas

## Status
Complete.

## TDD Evidence
- RED command: `uv run pytest tests/skill_eval/test_assertion_engine.py -v`
- RED result: failed during collection with `ModuleNotFoundError: No module named 'skill_eval.case_schema'`, confirming the schema package/modules were missing before implementation.
- GREEN command: `uv run pytest tests/skill_eval/test_assertion_engine.py -v`
- GREEN result: `3 passed, 1 warning in 0.27s`.
- Fresh completion verification command: `uv run pytest tests/skill_eval/test_assertion_engine.py -v`
- Fresh completion verification result: `3 passed, 1 warning in 0.26s`.
- Warning observed: existing dependency warning from `langgraph.checkpoint.serde.encrypted` about future `allowed_objects` default change.

## Changed Files
- `skill_eval/__init__.py`
- `skill_eval/case_schema.py`
- `skill_eval/trace_schema.py`
- `tests/skill_eval/test_assertion_engine.py`

## Implementation Notes
- Added package marker for `skill_eval`.
- Added `SkillAssertionSpec` and `SkillEvalCase` Pydantic schemas with the exact assertion-name literals and defaults from the task brief.
- Added `AgentToolCall`, `SkillInvocation`, and `AgentTrace` Pydantic schemas with default factories for mutable collections.
- Added the schema smoke tests exactly covering defaults, assertion-name validation, and trace evidence/raw-reference fields.

## Self-Review
- Verified all requested interfaces exist with the exact class names later tasks will import.
- Verified mutable fields use `Field(default_factory=list)` or `Field(default_factory=dict)` instead of shared mutable defaults.
- Verified no project-wide gates were run.
- Verified committed files are exactly the package marker, two schema modules, and focused schema test file.

## Code Review
- Reviewer: `ReviewTask2Schemas`.
- Result: no Critical, Important, or Minor issues found.
- Assessment: Ready to merge: Yes.

## Commit
- `f644df73d687422495ed679ce50062d802f2a16c` — `feat: add skill eval schemas`

## Concerns
- The brief expected RED to fail with `No module named 'skill_eval'`; because the test path `tests/skill_eval/` exists, Python exposed a namespace package and the observed missing import was `No module named 'skill_eval.case_schema'`. The failure still came from the intended missing schema implementation.
- GREEN test emitted one pre-existing dependency warning from LangGraph; no warnings came from the new schemas/tests.

## Review Fix: AssertionName MVP Scope

### Changed Files
- `backend/skill_eval/case_schema.py`
- `backend/tests/skill_eval/test_assertion_engine.py`
- `.superpowers/sdd/task-2-report.md`

### TDD Evidence
- RED command: `uv run pytest tests/skill_eval/test_assertion_engine.py -v`
- RED result: `1 failed, 2 passed, 1 warning in 0.28s`; `regex_match` was accepted, so `pytest.raises(ValidationError)` did not raise.
- GREEN command: `uv run pytest tests/skill_eval/test_assertion_engine.py -v`
- GREEN result: `3 passed, 1 warning in 0.26s`.
- Warning observed: existing dependency warning from `langgraph.checkpoint.serde.encrypted` about future `allowed_objects` default change.

### Implementation Notes
- Narrowed `AssertionName` to exactly the resolved MVP and skill-related names: `tool_called`, `tool_not_called`, `output_contains`, `success_is_true`, `trace_complete`, `skill_loaded`, `skill_used`, `skill_not_used`, `skill_applied`, `skill_not_applied`.
- Updated the assertion schema smoke test to prove `skill_used` is accepted and `regex_match` is rejected.
- Left unrelated schema fields unchanged.

### Self-Review
- Verified the resolved assertion-name set excludes tool-args, regex, JSON, latency, token, clarification, and other non-MVP names.
- Verified the focused schema test covers both an accepted skill-related name and the rejected non-MVP name from the review finding.
- Verified no project-wide gates were run.
