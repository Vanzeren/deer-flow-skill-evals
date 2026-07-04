import json
import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from skill_eval.case_schema import AssertionName, SkillAssertionSpec
from skill_eval.trace_schema import AgentTrace, SkillInvocation


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


def _pass(assertion: SkillAssertionSpec, explanation: str, **metadata: Any) -> AssertionResult:
    return AssertionResult(name=assertion.name, passed=True, explanation=assertion.message or explanation, metadata=metadata)


def _fail(assertion: SkillAssertionSpec, explanation: str, **metadata: Any) -> AssertionResult:
    return AssertionResult(name=assertion.name, passed=False, explanation=assertion.message or explanation, metadata=metadata)
