#!/usr/bin/env python3
"""Run the v0.1 event-reasoning demonstration with a real frontend LLM.

The frontend only translates natural language into a normalized event graph.
The scheduler then uses the bundled Rule and Knowledge databases to reason
over that graph.  This inference-only release does not include learning code
and does not alter the bundled Rule or Knowledge data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
REQUIRED_FRONTEND_MODEL = "Qwen/Qwen3-8B"
PRESETS: dict[str, str] = {
    "dso": (
        "本报告金额均以百万美元表示，采用年度360天口径。"
        "本年度平均应收账款为120百万美元，净赊销收入为960百万美元。"
        "请计算应收账款周转天数。"
    ),
    "inventory": "某公司库存周转速度明显变慢。请判断是否会形成库存资金占用。",
    "stockout": "公司客户需求波动较大，同时安全库存水平偏低。请判断是否存在缺货暴露。",
    "cash-flow": "某公司库存周转速度明显变慢。请判断是否会对经营现金流形成压力。",
    "missing-condition": "公司预测误差较高。请判断是否存在过量库存风险。",
}


def _bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _require_frontend_configuration() -> None:
    missing = [name for name in ("LLM_BASE_URL", "LLM_API_KEY") if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
    actual_model = os.getenv("LLM_MODEL", "").strip()
    if actual_model != REQUIRED_FRONTEND_MODEL:
        raise RuntimeError(
            f"this demo requires LLM_MODEL={REQUIRED_FRONTEND_MODEL!r}; got {actual_model!r}"
        )
    os.environ.setdefault("MAIN_PART_FRONTEND_IMPLEMENTATION", "semantic_v1")


def _build_launcher(session_id: str):
    _require_frontend_configuration()
    from main_part.frontend_semantic import frontend_from_env
    from main_part.model_launcher import MainPartModelLauncher
    from main_part.rule_vector_encoder import RuleVectorEncoder

    embedding_model = os.getenv("MAIN_PART_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    encoder = RuleVectorEncoder(
        encoder_name_or_path=embedding_model,
        local_files_only=_bool_env("MAIN_PART_EMBEDDING_LOCAL_FILES_ONLY"),
        load_encoder=True,
    )
    launcher = MainPartModelLauncher(
        frontend_adapter=frontend_from_env(),
        vector_encoder=encoder,
        auto_run=True,
        session_id=session_id,
    )
    return launcher


def _build_answer_service():
    """Build the presentation-only LLM layer after the reasoning run.

    The same fixed model is used for both frontend translation and wording.
    This service receives only a decoded, output-gated result payload; it does
    not receive Rule / Knowledge databases or control the Scheduler.
    """
    from main_part.answer_verbalizer import OpenAICompatibleVerbalizer
    from main_part.output_decoder import SemanticOutputDecoder
    from main_part.output_service import SemanticAnswerService

    return SemanticAnswerService(
        decoder=SemanticOutputDecoder(),
        verbalizer=OpenAICompatibleVerbalizer(
            base_url=os.environ["LLM_BASE_URL"],
            model=REQUIRED_FRONTEND_MODEL,
            api_key=os.environ["LLM_API_KEY"],
            timeout=int(os.getenv("LLM_READ_TIMEOUT_SECONDS", "180")),
            include_user_query=False,
            fallback_to_rough_text=True,
            fallback_on_error=True,
        ),
    )


def _compact_span(span: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "span_id",
        "feature_type",
        "core",
        "relation",
        "nature",
        "value",
        "unit",
        "computed_by_rule_id",
        "provenance",
    )
    return {field: span[field] for field in fields if span.get(field) is not None}


def _report(
    question: str,
    result: dict[str, Any],
    launcher: Any,
    answer_result: Any,
    elapsed_seconds: float,
) -> dict[str, Any]:
    frontend_state = result.get("frontend_state", {})
    all_spans = [
        span
        for workspace in launcher.working_memory.event_workspaces.values()
        for span in workspace.get("spans", [])
        if isinstance(span, dict)
    ]
    derived = [span for span in all_spans if span.get("computed_by_rule_id")]
    state = launcher.scheduler.state
    llm = launcher.frontend_adapter.frontend.llm
    return {
        "question": question,
        "frontend_model": llm.model,
        "frontend_spans": [_compact_span(span) for span in frontend_state.get("spans", [])],
        "frontend_events": [
            {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "evidence_span_ids": event.get("evidence_span_ids", []),
            }
            for event in frontend_state.get("events", [])
            if isinstance(event, dict)
        ],
        "derived_spans": [_compact_span(span) for span in derived],
        "scheduler_status": state.status,
        "blocking_reason": state.blocking_reason,
        "task_result": state.control_state.get("task_result", {"status": "unknown"}),
        "final_answer": answer_result.answer,
        "answer_draft": answer_result.draft,
        "answer_payload": answer_result.structured,
        "active_rule_ids": list(state.active_rule_ids),
        "frontend_usage": {
            **asdict(llm.usage),
            "token_usage_note": "Provider-reported fields only; missing or partial usage is not zero cost.",
        },
        "pipeline_timing": result.get("pipeline_timing", {}),
        "runtime_db_paths": dict(launcher.db_paths),
        "runtime_delivery_failures": [item for item in result.get("outputs", [])
                                      if item.get("operation") == "runtime_delivery_failure"],
        "execution_outcome": _execution_outcome(result),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def _print_human_report(report: dict[str, Any]) -> None:
    print("\n=== Reasoning-AI-demo-v0.1 ===")
    print(f"问题: {report['question']}")
    print(f"状态: {report['scheduler_status']}")
    print(f"运行判定: {report['execution_outcome']}")
    if report["runtime_delivery_failures"]:
        print("运行失败: " + json.dumps(report["runtime_delivery_failures"], ensure_ascii=False))
    if report.get("blocking_reason"):
        print(f"阻塞原因: {report['blocking_reason']}")
    print(f"推理状态: {json.dumps(report['task_result'], ensure_ascii=False)}")
    print(f"最终回答: {report['final_answer']}")
    print(f"实际执行规则: {', '.join(report['active_rule_ids']) or '无'}")
    print("前端 Event Graph spans:")
    print(json.dumps(report["frontend_spans"], ensure_ascii=False, indent=2))
    print("推理产生的 spans:")
    print(json.dumps(report["derived_spans"], ensure_ascii=False, indent=2))
    print(
        "前端调用: "
        f"{report['frontend_usage']['calls']} 次, "
        f"输入 {report['frontend_usage']['input_chars']} 字符, "
        f"输出 {report['frontend_usage']['output_chars']} 字符"
    )
    print(f"耗时: {report['elapsed_seconds']} 秒")


def _execution_outcome(result: dict[str, Any]) -> str:
    if any(item.get("operation") == "runtime_delivery_failure" for item in result.get("outputs", [])):
        return "runtime_failed"
    summary = result.get("runtime_summary", {})
    if summary.get("scheduler_status") != "completed":
        return "incomplete"
    status = summary.get("task_result", {}).get("status")
    return status if status in {"solved", "blocked"} else "unresolved"


def _exit_code(outcome: str) -> int:
    # Correct explicit blocking is a completed run, not a solved question.
    return 0 if outcome in {"solved", "blocked"} else 1 if outcome == "runtime_failed" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="a Chinese natural-language question")
    group.add_argument("--preset", choices=sorted(PRESETS), help="run one bundled demonstration question")
    parser.add_argument("--json", action="store_true", help="print the complete report as JSON")
    parser.add_argument("--output", type=Path, help="also write the report to this JSON file")
    args = parser.parse_args()
    question = args.question or PRESETS[args.preset]
    started = time.monotonic()
    launcher = _build_launcher(f"demo_{int(time.time())}")
    initialize_seconds = time.monotonic() - started
    result = launcher.submit_input(question)
    output_started = time.monotonic()
    if _execution_outcome(result) not in {"solved", "blocked"}:
        from main_part.output_service import AnswerResult
        message = "推理运行未完成，不能提供已验证答案。请查看 runtime_delivery_failures 和运行状态。"
        answer_result = AnswerResult(message, message, {"status": "unresolved"})
        output_usage = {"status": "not_called", "provider_usage": None}
    else:
        answer_service = _build_answer_service()
        answer_result = launcher.render_final_answer(
            answer_service, user_query=question, language="zh-CN",
            style_instruction=("Directly state the result first. Then give only the supplied "
                               "supporting facts and the stated uncertainty, if any."),
        )
        output_usage = answer_service.verbalizer.last_call_metadata
    result["pipeline_timing"].update(
        initialize_seconds=initialize_seconds,
        output_presentation_seconds=time.monotonic() - output_started,
        scope="Stage wall times, backend process CPU, and frontend provider usage; not a FLOPs, energy or billing estimate.",
    )
    report = _report(question, result, launcher, answer_result, time.monotonic() - started)
    report["output_llm_usage"] = output_usage
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human_report(report)
    return _exit_code(report["execution_outcome"])


if __name__ == "__main__":
    raise SystemExit(main())
