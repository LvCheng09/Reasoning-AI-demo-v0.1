from __future__ import annotations

"""Backend result state -> semantic decoder -> verbalizer -> final answer.

The functions in this module deliberately sit *after* the Scheduler output
gate.  They never select a Rule, compute a value, or alter the Event Graph.
Their job is to turn the Control-approved result span(s) into a small,
auditable payload for the user-facing renderer.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .output_decoder import SemanticOutputDecoder


@dataclass
class AnswerResult:
    answer: str
    draft: str
    structured: dict[str, Any]


def build_runtime_output_state(
    working_memory: Any,
    *,
    outputs: Sequence[Mapping[str, Any]] | None = None,
    task_result: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build decoder input from the live runtime state without new reasoning.

    ``conclusion_span_ids`` and ``result_spans`` originate from the Executor's
    Control-approved ``output`` operation.  They are therefore the only
    automatic target selectors.  In particular, this does not promote every
    derived span in a workspace into a final answer.
    """
    raw_workspaces = getattr(working_memory, "event_workspaces", {})
    workspaces = raw_workspaces if isinstance(raw_workspaces, Mapping) else {}
    spans_by_id: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for raw_event_id, raw_workspace in workspaces.items():
        if not isinstance(raw_workspace, Mapping):
            continue
        event_id = str(raw_workspace.get("event_id") or raw_event_id or "").strip()
        if not event_id:
            continue
        event_span_ids: list[str] = []
        for raw_span in raw_workspace.get("spans", []) or []:
            if not isinstance(raw_span, Mapping):
                continue
            span = deepcopy(dict(raw_span))
            span_id = str(span.get("span_id") or span.get("id") or "").strip()
            if not span_id:
                continue
            span.setdefault("span_id", span_id)
            spans_by_id[span_id] = span
            event_span_ids.append(span_id)
        for raw_span_id in raw_workspace.get("evidence_span_ids", []) or []:
            span_id = str(raw_span_id or "").strip()
            if span_id and span_id not in event_span_ids:
                event_span_ids.append(span_id)
        events.append(
            {
                "event_id": event_id,
                "event_type": str(raw_workspace.get("event_type") or ""),
                "evidence_span_ids": event_span_ids,
                "relations": [
                    deepcopy(dict(item))
                    for item in raw_workspace.get("relations", []) or []
                    if isinstance(item, Mapping)
                ],
            }
        )

    selected_output: Mapping[str, Any] = {}
    for item in reversed(list(outputs or [])):
        if isinstance(item, Mapping) and str(item.get("operation") or "") == "output":
            selected_output = item
            break

    target_span_ids: list[str] = []
    for raw_span_id in selected_output.get("conclusion_span_ids", []) or []:
        span_id = str(raw_span_id or "").strip()
        if span_id and span_id in spans_by_id and span_id not in target_span_ids:
            target_span_ids.append(span_id)
    for raw_span in selected_output.get("result_spans", []) or []:
        if not isinstance(raw_span, Mapping):
            continue
        span_id = str(raw_span.get("span_id") or raw_span.get("id") or "").strip()
        if span_id and span_id in spans_by_id and span_id not in target_span_ids:
            target_span_ids.append(span_id)

    target_feature_types: list[str] = []
    raw_feature_types = selected_output.get("result_feature_type")
    if isinstance(raw_feature_types, str):
        raw_feature_types = [raw_feature_types]
    for raw_feature_type in raw_feature_types or []:
        feature_type = str(raw_feature_type or "").strip()
        if feature_type and feature_type not in target_feature_types:
            target_feature_types.append(feature_type)

    runtime_state: dict[str, Any] = {
        "event_graph": {"spans": list(spans_by_id.values()), "events": events},
        "output_target_feature_types": target_feature_types,
    }
    status = str((task_result or {}).get("status") or "").strip().lower()
    if status and status != "solved":
        reason = str((task_result or {}).get("reason") or "").strip()
        runtime_state["status"] = "blocked" if status == "blocked" else status
        runtime_state["reason"] = reason or f"task_result:{status}"
        runtime_state["unresolved_conditions"] = [runtime_state["reason"]]
    return runtime_state, target_span_ids


class SemanticAnswerService:
    def __init__(self, *, decoder: SemanticOutputDecoder, verbalizer: Any):
        self.decoder = decoder
        self.verbalizer = verbalizer

    def answer(
        self,
        runtime_state: Mapping[str, Any],
        *,
        user_query: str = "",
        target_feature_types: Optional[Sequence[str]] = None,
        target_span_ids: Optional[Sequence[str]] = None,
        language: str = "zh-CN",
        style_instruction: str = "",
    ) -> AnswerResult:
        decoded = self.decoder.decode(
            runtime_state,
            target_feature_types=target_feature_types,
            target_span_ids=target_span_ids,
        )
        structured = decoded.to_dict()
        answer = self.verbalizer.verbalize(
            structured,
            user_query=user_query,
            language=language,
            style_instruction=style_instruction,
        )
        return AnswerResult(
            answer=answer,
            draft=decoded.rough_text,
            structured=structured,
        )

    def answer_runtime(
        self,
        working_memory: Any,
        *,
        outputs: Sequence[Mapping[str, Any]] | None = None,
        task_result: Mapping[str, Any] | None = None,
        user_query: str = "",
        language: str = "zh-CN",
        style_instruction: str = "",
    ) -> AnswerResult:
        """Render only the final, output-gated state of a Scheduler run."""
        runtime_state, target_span_ids = build_runtime_output_state(
            working_memory,
            outputs=outputs,
            task_result=task_result,
        )
        return self.answer(
            runtime_state,
            user_query=user_query,
            target_span_ids=target_span_ids or None,
            target_feature_types=(
                runtime_state.get("output_target_feature_types") or None
            ) if not target_span_ids else None,
            language=language,
            style_instruction=style_instruction,
        )
