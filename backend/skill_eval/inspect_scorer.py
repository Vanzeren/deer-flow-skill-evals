from inspect_ai.model import get_model
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, Target, scorer
from inspect_ai.solver import TaskState

from skill_eval.case_schema import RoutingCase
from skill_eval.judge import JudgeFailure, build_judge_evidence, judge_quality, judge_quick_turn
from skill_eval.routing import RouteObservation
from skill_eval.trace_schema import AgentTrace


@scorer(metrics=[])
def routing_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            observation = RouteObservation.model_validate(state.metadata["route_observation"])
            case = RoutingCase.model_validate(state.metadata["case"])
        except Exception as exc:
            return Score(
                value=NOANSWER,
                explanation=f"Invalid routing metadata: {exc}",
                metadata={"infrastructure_error": str(exc)},
            )
        if not observation.completed or observation.observed_route is None:
            message = "; ".join(observation.errors) or "incomplete routing observation"
            return Score(
                value=NOANSWER,
                explanation=message,
                metadata={
                    "case_id": case.id,
                    "infrastructure_error": message,
                    "route_observation": observation.model_dump(),
                },
            )
        passed = observation.observed_route == case.expected_route
        return Score(
            value=CORRECT if passed else INCORRECT,
            explanation=(f"expected={case.expected_route} observed={observation.observed_route}"),
            metadata={
                "case_id": case.id,
                "expected_route": case.expected_route,
                "observed_route": observation.observed_route,
                "route_observation": observation.model_dump(),
            },
        )

    return score


@scorer(metrics=[])
def quality_judge_scorer(
    judge_model: str,
    skill_descriptions: dict[str, str],
):
    model = get_model(judge_model)

    async def score(state: TaskState, target: Target) -> Score:
        try:
            case = RoutingCase.model_validate(state.metadata["case"])
            trace = AgentTrace.model_validate(state.metadata["agent_trace"])
            observation = RouteObservation.model_validate(state.metadata["route_observation"])
            infrastructure_errors = []
            if not trace.success:
                infrastructure_errors.extend(trace.errors or ["agent trace failed"])
            if not observation.completed:
                infrastructure_errors.extend(observation.errors or ["route observation incomplete"])
            if state.metadata.get("agent_success") is False:
                infrastructure_errors.append("agent run reported failure")
            if infrastructure_errors:
                message = "; ".join(dict.fromkeys(infrastructure_errors))
                return Score(
                    value=NOANSWER,
                    explanation=message,
                    metadata={"infrastructure_error": message},
                )
            bundle = build_judge_evidence(
                case=case,
                trace=trace,
                observation=observation,
                skill_descriptions=skill_descriptions,
            )
            judgment = await judge_quality(bundle, model)
        except (JudgeFailure, KeyError, ValueError) as exc:
            return Score(
                value=NOANSWER,
                explanation=f"Quality judge failed: {exc}",
                metadata={"judge_failure": str(exc)},
            )

        quality_passed = not judgment.fatal_error and judgment.route_quality >= 3 and judgment.process_quality >= 3 and judgment.output_quality >= 3
        return Score(
            value=judgment.overall_quality,
            explanation="\n".join(judgment.reasons),
            metadata={
                "quality_judgment": judgment.model_dump(),
                "quality_passed": quality_passed,
                "label_review_needed": (judgment.recommended_route != case.expected_route),
            },
        )

    return score


@scorer(metrics=[])
def quick_turn_scorer(
    judge_model: str,
    skill_descriptions: dict[str, str],
):
    model = get_model(judge_model)

    async def score(state: TaskState, target: Target) -> Score:
        try:
            case = RoutingCase.model_validate(state.metadata["case"])
            trace = AgentTrace.model_validate(state.metadata["agent_trace"])
            observation = RouteObservation.model_validate(state.metadata["route_observation"])
        except Exception as exc:
            return Score(
                value=NOANSWER,
                explanation=f"Invalid quick-turn metadata: {exc}",
                metadata={"infrastructure_error": str(exc)},
            )

        infrastructure_errors = []
        if not trace.success:
            infrastructure_errors.extend(trace.errors or ["agent trace failed"])
        if not observation.completed:
            infrastructure_errors.extend(observation.errors or ["route observation incomplete"])
        if state.metadata.get("agent_success") is False:
            infrastructure_errors.append("agent run reported failure")
        if infrastructure_errors:
            message = "; ".join(dict.fromkeys(infrastructure_errors))
            return Score(
                value=NOANSWER,
                explanation=message,
                metadata={"infrastructure_error": message, "case_id": case.id},
            )

        if case.expected_route == "none":
            return Score(
                value=NOANSWER,
                explanation="quick turn not applicable to none-expected case",
                metadata={"not_applicable_none_case": True, "case_id": case.id},
            )
        if observation.observed_route != case.expected_route:
            return Score(
                value=NOANSWER,
                explanation=f"route mismatch: expected={case.expected_route} observed={observation.observed_route}",
                metadata={"route_mismatch": True, "case_id": case.id},
            )
        if trace.quick_turn is None:
            return Score(
                value=NOANSWER,
                explanation="quick turn not captured before the stream ended",
                metadata={"quick_turn_missing": True, "case_id": case.id},
            )

        try:
            bundle = build_judge_evidence(
                case=case,
                trace=trace,
                observation=observation,
                skill_descriptions=skill_descriptions,
                target="quick_turn",
            )
            judgment = await judge_quick_turn(bundle, model)
        except (JudgeFailure, KeyError, ValueError) as exc:
            return Score(
                value=NOANSWER,
                explanation=f"Quick turn judge failed: {exc}",
                metadata={"judge_failure": str(exc), "case_id": case.id},
            )

        quality_passed = not judgment.fatal_error and judgment.turn_quality >= 3
        return Score(
            value=CORRECT if quality_passed else INCORRECT,
            explanation=judgment.rationale,
            metadata={
                "quick_judgment": judgment.model_dump(),
                "quality_passed": quality_passed,
                "case_id": case.id,
            },
        )

    return score
