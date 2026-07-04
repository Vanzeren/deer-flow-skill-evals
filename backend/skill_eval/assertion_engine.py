import json
import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from skill_eval.case_schema import AssertionName, SkillAssertionSpec
from skill_eval.trace_schema import AgentToolCall, AgentTrace, SkillInvocation


class AssertionResult(BaseModel):
    name: str
    passed: bool
    explanation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


AssertionHandler = Callable[[SkillAssertionSpec, AgentTrace, str], AssertionResult]
ASSERTION_REGISTRY: dict[AssertionName, AssertionHandler] = {}


def register_assertion(name: AssertionName) -> Callable[[AssertionHandler], AssertionHandler]:
    def decorator(handler: AssertionHandler) -> AssertionHandler:
        ASSERTION_REGISTRY[name] = handler
        return handler

    return decorator


def evaluate_assertion(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    handler = ASSERTION_REGISTRY.get(assertion.name)
    if handler is None:
        return _fail(assertion, f"Unsupported assertion in MVP: {assertion.name}")

    return handler(assertion, trace, final_answer)


@register_assertion("tool_called")
def evaluate_tool_called(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    called = any(call.name == assertion.target for call in trace.tool_calls)
    if called:
        return _pass(assertion, f"Tool `{assertion.target}` was called.")
    return _fail(assertion, f"Expected tool `{assertion.target}` to be called.")


@register_assertion("tool_not_called")
def evaluate_tool_not_called(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    called = any(call.name == assertion.target for call in trace.tool_calls)
    if not called:
        return _pass(assertion, f"Tool `{assertion.target}` was not called.")
    return _fail(assertion, f"Forbidden tool `{assertion.target}` was called.")


@register_assertion("tool_args_contains")
def evaluate_tool_args_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    for call in trace.tool_calls:
        args_text = _stringify(call.args)
        if target in args_text:
            return _pass(assertion, f"Tool arguments contained `{target}`.", target=target, matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool arguments to contain `{target}`.", target=target, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_args_match")
def evaluate_tool_args_match(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    pattern, error = _compile_pattern(assertion)
    if error is not None:
        return error
    assert pattern is not None
    for call in trace.tool_calls:
        match = pattern.search(_stringify(call.args))
        if match:
            return _pass(assertion, f"Tool arguments matched regex `{pattern.pattern}`.", pattern=pattern.pattern, match=match.group(0), matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool arguments to match regex `{pattern.pattern}`.", pattern=pattern.pattern, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_call_order")
def evaluate_tool_call_order(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    expected = [name.strip() for name in (assertion.target or "").split(",") if name.strip()]
    observed = [call.name for call in trace.tool_calls]
    cursor = 0
    for name in observed:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    if cursor == len(expected):
        return _pass(assertion, f"Tool call order contained `{expected}`.", expected_order=expected, observed_order=observed)
    return _fail(assertion, f"Expected tool call order `{expected}` within observed order `{observed}`.", expected_order=expected, observed_order=observed)


@register_assertion("tool_error_absent")
def evaluate_tool_error_absent(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    calls = _target_tool_calls(assertion, trace)
    errored = [call for call in calls if call.error]
    if not errored:
        return _pass(assertion, "No matching tool errors were present.", checked_tool_calls=[_tool_call_dump(call) for call in calls])
    return _fail(assertion, "Expected no matching tool errors.", errored_tool_calls=[_tool_call_dump(call) for call in errored])


@register_assertion("tool_result_contains")
def evaluate_tool_result_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    for call in trace.tool_calls:
        result_text = _stringify(call.result)
        if target in result_text:
            return _pass(assertion, f"Tool result contained `{target}`.", target=target, matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool result to contain `{target}`.", target=target, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_result_match")
def evaluate_tool_result_match(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    pattern, error = _compile_pattern(assertion)
    if error is not None:
        return error
    assert pattern is not None
    for call in trace.tool_calls:
        match = pattern.search(_stringify(call.result))
        if match:
            return _pass(assertion, f"Tool result matched regex `{pattern.pattern}`.", pattern=pattern.pattern, match=match.group(0), matched_tool_call=_tool_call_dump(call))
    return _fail(assertion, f"Expected tool result to match regex `{pattern.pattern}`.", pattern=pattern.pattern, observed=[_tool_call_dump(call) for call in trace.tool_calls])


@register_assertion("tool_count_under")
def evaluate_tool_count_under(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    threshold = assertion.threshold
    observed = len(trace.tool_calls)
    if threshold is None:
        return _fail(assertion, "tool_count_under requires a threshold.", observed=observed, threshold=None)
    if observed < threshold:
        return _pass(assertion, f"Tool count {observed} was under {threshold}.", observed=observed, threshold=threshold)
    return _fail(assertion, f"Tool count {observed} was not under {threshold}.", observed=observed, threshold=threshold)


@register_assertion("output_contains")
def evaluate_output_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    if target in final_answer:
        return _pass(assertion, f"Output contained `{target}`.")
    return _fail(assertion, f"Expected output to contain `{target}`.")


@register_assertion("output_not_contains")
def evaluate_output_not_contains(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    target = assertion.target or ""
    if target not in final_answer:
        return _pass(assertion, f"Output did not contain `{target}`.", target=target)
    return _fail(assertion, f"Forbidden output `{target}` was present.", target=target)


@register_assertion("regex_match")
def evaluate_regex_match(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    pattern = assertion.target or ""
    try:
        match = re.search(pattern, final_answer)
    except re.error as exc:
        return _fail(assertion, f"Invalid regex `{pattern}`: {exc}", pattern=pattern, error=str(exc))

    if match:
        return _pass(assertion, f"Output matched regex `{pattern}`.", pattern=pattern, match=match.group(0))
    return _fail(assertion, f"Expected output to match regex `{pattern}`.", pattern=pattern)


@register_assertion("json_valid")
def evaluate_json_valid(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    try:
        parsed = json.loads(final_answer)
    except json.JSONDecodeError as exc:
        return _fail(assertion, f"Expected output to be valid JSON: {exc}", error=str(exc))

    return _pass(assertion, "Output is valid JSON.", json_type=type(parsed).__name__)


@register_assertion("success_is_true")
def evaluate_success_is_true(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    if trace.success is True:
        return _pass(assertion, "Trace success is true.")
    return _fail(assertion, "Trace success is not true.")


@register_assertion("trace_complete")
def evaluate_trace_complete(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    failures: list[str] = []
    if not trace.input:
        failures.append("trace.input is empty")
    if not trace.final_answer:
        failures.append("trace.final_answer is empty")
    if trace.success is None:
        failures.append("trace.success is missing")
    if not trace.messages and not trace.tool_calls and not trace.steps:
        failures.append("trace has no messages, tool_calls, or steps")
    if any("fatal" in error.lower() for error in trace.errors):
        failures.append(f"trace contains fatal errors: {trace.errors}")

    if failures:
        return _fail(assertion, "; ".join(failures), failures=failures)
    return _pass(assertion, "Trace is complete.")


@register_assertion("skill_loaded")
def evaluate_skill_loaded(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    if any(invocation.loaded is True for invocation in _target_skill_invocations(assertion, trace)):
        return _pass(assertion, f"Skill `{assertion.target}` was loaded.")
    return _fail(assertion, f"Expected skill `{assertion.target}` to be loaded.")


@register_assertion("skill_used")
def evaluate_skill_used(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    if any(invocation.used is True for invocation in _target_skill_invocations(assertion, trace)):
        return _pass(assertion, f"Skill `{assertion.target}` was used.")
    return _fail(assertion, f"Expected skill `{assertion.target}` to be used.")


@register_assertion("skill_not_used")
def evaluate_skill_not_used(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    if not any(invocation.used is True for invocation in _target_skill_invocations(assertion, trace)):
        return _pass(assertion, f"Skill `{assertion.target}` was not used.")
    return _fail(assertion, f"Forbidden skill `{assertion.target}` was used.")


@register_assertion("skill_applied")
def evaluate_skill_applied(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    if any(invocation.applied is True for invocation in _target_skill_invocations(assertion, trace)):
        return _pass(assertion, f"Skill `{assertion.target}` was applied.")
    return _fail(assertion, f"Expected skill `{assertion.target}` to be explicitly applied.")


@register_assertion("skill_not_applied")
def evaluate_skill_not_applied(assertion: SkillAssertionSpec, trace: AgentTrace, final_answer: str) -> AssertionResult:
    found_target = False
    for invocation in trace.skill_invocations:
        if invocation.name != assertion.target:
            continue
        found_target = True
        if invocation.applied is not False:
            return _fail(assertion, f"Expected skill `{assertion.target}` to be explicitly not applied.")

    if found_target:
        return _pass(assertion, f"Skill `{assertion.target}` was explicitly not applied.")
    return _fail(assertion, f"Expected skill `{assertion.target}` to be explicitly not applied.")


def _target_skill_invocations(assertion: SkillAssertionSpec, trace: AgentTrace) -> Iterable[SkillInvocation]:
    return (invocation for invocation in trace.skill_invocations if invocation.name == assertion.target)


def _tool_call_dump(call: AgentToolCall) -> dict[str, Any]:
    return call.model_dump()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _target_tool_calls(assertion: SkillAssertionSpec, trace: AgentTrace) -> list[AgentToolCall]:
    if not assertion.target or assertion.target not in {call.name for call in trace.tool_calls}:
        return trace.tool_calls
    return [call for call in trace.tool_calls if call.name == assertion.target]


def _compile_pattern(assertion: SkillAssertionSpec) -> tuple[re.Pattern[str] | None, AssertionResult | None]:
    pattern = assertion.target or ""
    try:
        return re.compile(pattern), None
    except re.error as exc:
        return None, _fail(assertion, f"Invalid regex `{pattern}`: {exc}", pattern=pattern, error=str(exc))


def _pass(assertion: SkillAssertionSpec, explanation: str, **metadata: Any) -> AssertionResult:
    return AssertionResult(name=assertion.name, passed=True, explanation=assertion.message or explanation, metadata=metadata)


def _fail(assertion: SkillAssertionSpec, explanation: str, **metadata: Any) -> AssertionResult:
    return AssertionResult(name=assertion.name, passed=False, explanation=assertion.message or explanation, metadata=metadata)
