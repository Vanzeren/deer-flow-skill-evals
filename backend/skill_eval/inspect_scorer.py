from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from skill_eval.assertion_engine import evaluate_assertion
from skill_eval.case_schema import SkillAssertionSpec, SkillEvalCase
from skill_eval.trace_schema import AgentTrace


@scorer(metrics=[])
def trace_integrity_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if "agent_trace" not in state.metadata:
            return Score(value=0.0, explanation="Missing agent_trace in state.metadata.")

        try:
            trace = AgentTrace.model_validate(state.metadata["agent_trace"])
        except Exception as exc:
            return Score(value=0.0, explanation=f"Invalid AgentTrace: {exc}")

        result = evaluate_assertion(
            SkillAssertionSpec(name="trace_complete"),
            trace,
            trace.final_answer,
        )

        return Score(
            value=1.0 if result.passed else 0.0,
            explanation=result.explanation,
            metadata={"assertion_result": result.model_dump()},
        )

    return score


@scorer(metrics=[])
def skill_assertion_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if "case" not in state.metadata:
            return Score(value=0.0, explanation="Missing case metadata.")
        if "agent_trace" not in state.metadata:
            return Score(value=0.0, explanation="Missing agent_trace metadata.")

        case = SkillEvalCase.model_validate(state.metadata["case"])
        trace = AgentTrace.model_validate(state.metadata["agent_trace"])
        results = [evaluate_assertion(assertion, trace, trace.final_answer) for assertion in case.assertions]
        failures = [result for result in results if not result.passed]

        return Score(
            value=0.0 if failures else 1.0,
            explanation="\n".join(result.explanation for result in failures) if failures else "All assertions passed.",
            metadata={"case_id": case.id, "assertion_results": [result.model_dump() for result in results]},
        )

    return score
