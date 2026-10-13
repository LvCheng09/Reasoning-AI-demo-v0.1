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
    def rejected(reason: str) -> tuple[dict[str, Any], list[str]]:
        return {"event_graph": {"spans": [], "events": []},
                "status": "unresolved", "reason": reason,
                "unresolved_conditions": [reason]}, []

    status = str((task_result or {}).get("status") or "").lower()
    if status != "solved":
        return rejected(str((task_result or {}).get("reason") or f"task_result:{status or 'unknown'}"))

    selected = next((item for item in reversed(list(outputs or []))
                     if isinstance(item, Mapping) and item.get("operation") == "output"), None)
    if selected is None or selected.get("output_gate_checked") is not True:
        return rejected("no_approved_output")
    if selected.get("output_gate_decision") not in (None, "allow_current_candidate"):
        return rejected("output_gate_not_allowed")

    raw_workspaces = getattr(working_memory, "event_workspaces", {})
    workspaces = raw_workspaces if isinstance(raw_workspaces, Mapping) else {}
    scope_ids = {str(selected[key]) for key in ("event_id", "rendered_from_event_id")
                 if selected.get(key)}
    if len(scope_ids) > 1:
        return rejected("conflicting_output_event_scope")
    if scope_ids:
        event_id = next(iter(scope_ids))
        matches = [(key, ws) for key, ws in workspaces.items()
                   if isinstance(ws, Mapping) and str(ws.get("event_id") or key) == event_id]
    else:
        matches = [(key, ws) for key, ws in workspaces.items() if isinstance(ws, Mapping)]
    if len(matches) != 1:
        return rejected("missing_or_ambiguous_output_event_scope")
    key, workspace = matches[0]
    event_id = str(workspace.get("event_id") or key)
    spans_by_id = {}
    for raw_span in workspace.get("spans", []) or []:
        if not isinstance(raw_span, Mapping):
            continue
        span = deepcopy(dict(raw_span))
        span_id = str(span.get("span_id") or span.get("id") or "").strip()
        if not span_id:
            continue
        if span_id in spans_by_id:
            return rejected("duplicate_span_id_in_output_event")
        span["span_id"] = span_id
        spans_by_id[span_id] = span

    target_ids = []
    for raw_id in selected.get("conclusion_span_ids", []) or []:
        span_id = str(raw_id or "").strip()
        if not span_id or span_id not in spans_by_id:
            return rejected("missing_approved_target")
        if span_id not in target_ids:
            target_ids.append(span_id)
    for snapshot in selected.get("result_spans", []) or []:
        if not isinstance(snapshot, Mapping):
            return rejected("invalid_output_result_snapshot")
        span_id = str(snapshot.get("span_id") or snapshot.get("id") or "").strip()
        current = spans_by_id.get(span_id)
        if current is None:
            return rejected("missing_approved_target")
        for field in ("core", "relation", "nature", "feature_type", "value", "unit",
                      "computed_by_rule_id", "evidence_span_ids", "provenance"):
            if field in snapshot and snapshot[field] != current.get(field):
                return rejected("stale_or_conflicting_output_snapshot")
        if span_id not in target_ids:
            target_ids.append(span_id)

    decoder = SemanticOutputDecoder()
    if not target_ids:
        features = selected.get("result_feature_type") or []
        features = [features] if isinstance(features, str) else features
        if not isinstance(features, (list, tuple)):
            return rejected("invalid_output_feature_selector")
        target_ids = [sid for sid, span in spans_by_id.items()
                      if span.get("feature_type") in features and decoder._is_target_output_eligible(span)]
    if not target_ids:
        return rejected("no_approved_target")
    if any(not decoder._is_target_output_eligible(spans_by_id[sid]) for sid in target_ids):
        return rejected("approved_target_not_eligible")

    state = {"event_graph": {"spans": list(spans_by_id.values()), "events": [{
        "event_id": event_id,
        "event_type": str(workspace.get("event_type") or ""),
        "evidence_span_ids": list(spans_by_id),
        "relations": [deepcopy(dict(item)) for item in workspace.get("relations", []) or []
                      if isinstance(item, Mapping)],
    }]}, "output_target_feature_types": []}
    return state, target_ids

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
        decoded = self.decoder.decode(runtime_state, target_span_ids=target_span_ids)
        structured = decoded.to_dict()
        # An unresolved run must not be promoted by a presentation model.
        if not target_span_ids:
            return AnswerResult(decoded.rough_text, decoded.rough_text, structured)
        if decoded.status != "ok" or set(target_span_ids) != {s.span_id for s in decoded.target_spans}:
            rejected = self.decoder.decode({
                "event_graph": {"spans": [], "events": []},
                "unresolved_conditions": ["approved_results_not_fully_decodable"],
            })
            return AnswerResult(rejected.rough_text, rejected.rough_text, rejected.to_dict())
        answer = self.verbalizer.verbalize(
            structured, user_query=user_query, language=language,
            style_instruction=style_instruction,
        )
        return AnswerResult(answer, decoded.rough_text, structured)
