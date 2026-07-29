"""Single-case skill eval runner + CLI. Spec §4/§7.

    python -m deerflow.evals.skill.runner case.json [--model M] [--judge-model J] \
        [--no-langfuse] [--output report.json]

The runner owns the RunnableConfig: worker-equivalent context keys, a
runner-generated trace_id carried by its own Langfuse CallbackHandler, and
ambient tracing disabled so make_lead_agent does not attach a second handler.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.lead_agent.agent import make_lead_agent
from deerflow.config.app_config import get_app_config
from deerflow.config.tracing_config import get_tracing_config, reset_tracing_config
from deerflow.evals.skill.assertions import build_eval_report
from deerflow.evals.skill.callback import SkillEvalCallback
from deerflow.evals.skill.langfuse_sink import publish_langfuse_scores
from deerflow.evals.skill.models import EvalMode, OutputBundle, SkillEvalCase, SkillEvalSpec, spec_from_case_metadata
from deerflow.evals.skill.output_judge import evaluate_complete_output
from deerflow.evals.skill.recorder import EvalRecorder
from deerflow.evals.skill.report_md import render_markdown_report

logger = logging.getLogger(__name__)


def _disable_ambient_tracing() -> None:
    """Keep make_lead_agent's build_tracing_callbacks() from double-tracing."""
    os.environ["LANGFUSE_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    reset_tracing_config()


def _build_langfuse_handler(trace_id: str):
    """Return (client, handler) or (None, None) when Langfuse is not configured."""
    reset_tracing_config()
    langfuse_cfg = get_tracing_config().langfuse
    if not (langfuse_cfg.public_key and langfuse_cfg.secret_key):
        logger.warning("Langfuse keys not configured; skipping score publish")
        return None, None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    client = Langfuse(
        public_key=langfuse_cfg.public_key,
        secret_key=langfuse_cfg.secret_key,
        host=langfuse_cfg.host,
    )
    return client, LangfuseCallbackHandler(trace_context={"trace_id": trace_id})


def _extract_final_answer(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
            return message.content
    return ""


_WRITE_TOOL_NAMES = frozenset({"write_file", "write", "save_file"})


def _generated_files(recorder: EvalRecorder) -> list[str]:
    """Files successfully written during the run, from trajectory evidence."""
    files: list[str] = []
    for batch in recorder.tool_batches:
        for call in batch:
            if call.name in _WRITE_TOOL_NAMES and call.status == "success":
                path = call.args.get("path")
                if isinstance(path, str) and path not in files:
                    files.append(path)
    return files


async def run_skill_eval_case(
    spec: SkillEvalSpec,
    *,
    messages: list[dict],
    model_name: str | None = None,
    judge_model_name: str | None = None,
    use_langfuse: bool = True,
) -> dict:
    app_config = get_app_config()
    trace_id = uuid.uuid4().hex
    thread_id = f"skill-eval-{spec.case_id}-{trace_id[:8]}"
    recorder = EvalRecorder(trace_id=trace_id)
    callback = SkillEvalCallback(
        recorder,
        expected_skill=spec.expected_skill,
        container_path=app_config.skills.container_path,
        read_tool_names=app_config.summarization.skill_file_read_tool_names,
    )

    _disable_ambient_tracing()
    langfuse_client, langfuse_handler = (None, None)
    if use_langfuse:
        langfuse_client, langfuse_handler = _build_langfuse_handler(trace_id)

    callbacks: list[Any] = ([langfuse_handler] if langfuse_handler else []) + [callback]
    runtime_ctx: dict[str, Any] = {
        "thread_id": thread_id,
        "run_id": f"evalrun-{trace_id[:8]}",
        "app_config": app_config,
    }
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "context": dict(runtime_ctx),
        "callbacks": callbacks,
        "metadata": {
            "langfuse_trace_name": f"skill-eval:{spec.case_id}",
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["skill-eval", f"case:{spec.case_id}", f"mode:{spec.mode}"],
            "eval_case_id": spec.case_id,
        },
    }
    if model_name:
        config["configurable"]["model_name"] = model_name

    # Mirror worker.py: langgraph surfaces runtime.context via the parent
    # runtime stored under configurable["__pregel_runtime"]; without it
    # middlewares see runtime.context=None (e.g. ThreadDataMiddleware).
    from langgraph.runtime import Runtime

    config["configurable"]["__pregel_runtime"] = Runtime(context=cast(Any, runtime_ctx), store=None)

    agent = make_lead_agent(RunnableConfig(**config))
    result = await agent.ainvoke({"messages": messages}, config=RunnableConfig(**config))

    recorder.finalize()
    agent_result = _extract_final_answer(result)
    report = build_eval_report(spec=spec, recorder=recorder, agent_result=agent_result)

    if spec.mode == EvalMode.FULL:
        bundle = OutputBundle(final_answer=agent_result, generated_files=_generated_files(recorder))
        grade = evaluate_complete_output(bundle, spec.output_rubric, judge_model_name=judge_model_name or model_name)
        report["output_judge"] = grade.model_dump()

    if langfuse_client is not None:
        publish_langfuse_scores(langfuse_client, trace_id=trace_id, report=report)
        langfuse_client.flush()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single skill eval case")
    parser.add_argument("case", type=Path, help="Path to case JSON (dataset item shape)")
    parser.add_argument("--model", default=None, help="Agent model name")
    parser.add_argument("--judge-model", default=None, help="Judge model name (full mode)")
    parser.add_argument("--no-langfuse", action="store_true", help="Skip Langfuse trace/scores")
    parser.add_argument("--output", type=Path, default=None, help="Report JSON output path")
    args = parser.parse_args(argv)

    case = SkillEvalCase.model_validate(json.loads(args.case.read_text()))
    spec = spec_from_case_metadata(case.metadata)
    report = asyncio.run(
        run_skill_eval_case(
            spec,
            messages=case.input["messages"],
            model_name=args.model,
            judge_model_name=args.judge_model,
            use_langfuse=not args.no_langfuse,
        )
    )
    output_path = args.output or Path(f"skill-eval-report-{spec.case_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_markdown_report(report))
    print(json.dumps({"case_id": spec.case_id, "trace_id": report["trace_id"], "assertions": {k: v["passed"] for k, v in report["assertions"].items()}}, ensure_ascii=False))
    print(f"report written to {output_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
