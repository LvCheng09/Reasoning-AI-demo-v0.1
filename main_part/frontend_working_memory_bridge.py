"""Move completed frontend output into working memory.

This file is intentionally small: frontend code calls it after the frontend LLM
has produced spans/events. It records the result in working memory and does not
run the scheduler.
"""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Mapping

from .identity import new_runtime_id
from .frontend_schema import normalize_spans, normalize_events, confidence
from .event_graph_contract import validate_event_graph_contract

from .data_structure import EventHypergraph, WorkingMemoryEntry, WorkingMemoryState


def submit_frontend_state_to_working_memory(
    working_memory: WorkingMemoryState,
    *,
    game_input: Mapping[str, Any],
    frontend_state: Mapping[str, Any],
    input_context: Mapping[str, Any] | None = None,
    session_id: str = "main_runtime",
    source: str = "frontend_llm",
    record_input_entry: bool = True,
) -> dict[str, Any]:
    """Persist frontend spans/events as the next scheduler input.

    RuntimeLauncher watches for entries with payload_type="frontend_ready_input"
    in input_workspace. Event hypergraphs are stored in event_workspace and in
    working_memory.event_workspaces[event_id]. This function never calls scheduler.
    """

    frontend_state = deepcopy(dict(frontend_state))
    frontend_state["spans"] = normalize_spans(frontend_state.get("spans"))
    frontend_state["events"] = normalize_events(frontend_state.get("events"))
    contract = validate_event_graph_contract(frontend_state)
    if not contract["ok"]:
        raise ValueError(f"frontend event graph contract rejected: {contract['errors']}")
    domain = str(game_input.get("domain") or game_input.get("task_domain") or "general").strip() or "general"
    payload = {
        "domain": domain,
        "game_input": dict(game_input or {}),
        "frontend_state": _input_frontend_state(frontend_state),
        "input_context": dict(input_context or {}),
        "status": "frontend_ready",
    }
    entry: WorkingMemoryEntry | None = None
    if record_input_entry:
        entry = WorkingMemoryEntry(
            entry_id=_entry_id(),
            session_id=str(session_id),
            stage="frontend_ready",
            source=str(source),
            payload_type="frontend_ready_input",
            created_at=_utc_now(),
            confidence=confidence(frontend_state.get("confidence", 1.0), field="frontend", default=1.0),
            workspace_zone="input_workspace",
            payload=payload,
            tags=["frontend_ready", str(payload["domain"])],
        )
        working_memory.add(entry, zone="input_workspace")

    spans_by_id = _span_index(frontend_state.get("spans") if isinstance(frontend_state, Mapping) else [])
    event_ids: list[str] = []
    for raw_event in _dict_items(frontend_state.get("events") if isinstance(frontend_state, Mapping) else []):
        event_id = str(raw_event.get("event_id") or raw_event.get("id") or f"event_{len(event_ids)}")
        event_payload = dict(raw_event)
        event_payload["event_id"] = event_id
        span_ids = [str(item) for item in event_payload.get("evidence_span_ids", []) if str(item)]
        event_payload["spans"] = [spans_by_id[item] for item in span_ids if item in spans_by_id]
        if not span_ids and event_payload.get("use_all_spans") is True:
            event_payload["spans"] = list(spans_by_id.values())
            event_payload["evidence_span_ids"] = list(spans_by_id)
        event_payload.setdefault("metadata", {})
        for identity_field in (
            "correlation_group_id",
            "source_template_id",
            "template_id",
            "evidence_provenance_id",
            "source_record_id",
            "source_document_id",
            "source_episode_id",
            "independence_verified",
        ):
            if identity_field in game_input and identity_field not in event_payload["metadata"]:
                event_payload["metadata"][identity_field] = game_input[identity_field]
        event_payload["metadata"].update({"frontend_entry_id": entry.entry_id if entry else ""})
        event_payload["metadata"].setdefault(
            "initial_event_graph",
            {
                "spans": deepcopy(event_payload.get("spans", [])),
                "relations": deepcopy(event_payload.get("relations", [])),
            },
        )
        # Frontend spans are an extensible, already-validated data contract.
        # Converting them through SpanUnit dropped scope, modality/polarity and
        # new semantic identity vectors before the Scheduler could read them.
        event_workspace = working_memory.upsert_event_hypergraph(event_payload, merge=True)
        working_memory.add(
            WorkingMemoryEntry(
                entry_id=_entry_id(),
                session_id=str(session_id),
                stage="frontend_event_hypergraph",
                source=str(source),
                payload_type="frontend_event_hypergraph",
                created_at=_utc_now(),
                confidence=event_payload["confidence"],
                workspace_zone="event_workspace",
                payload={
                    "event_id": event_id,
                    "spans": event_payload.get("spans", []),
                    "relations": event_payload.get("relations", []),
                    "candidate_paths": event_workspace.get("candidate_paths", []),
                    "feedback": event_workspace.get("feedback", []),
                    "metadata": event_workspace.get("metadata", {}),
                },
                tags=["frontend_event", str(payload["domain"]), event_id],
            ),
            zone="event_workspace",
        )
        event_ids.append(event_id)

    return {
        "input_entry": {
            "entry_id": entry.entry_id,
            "session_id": entry.session_id,
            "created_at": entry.created_at,
        } if entry else None,
        "event_ids": event_ids,
        "status": "submitted_to_working_memory",
    }


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _span_index(value: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for span in _dict_items(value):
        span_id = str(span.get("span_id") or span.get("id") or "")
        if span_id:
            result[span_id] = span
    return result


def _input_frontend_state(frontend_state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the frontend source needed for patching without duplicating graphs."""

    payload = dict(frontend_state or {})
    payload.pop("event_workspaces", None)
    return payload


def _entry_id() -> str:
    return new_runtime_id("frontend")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["submit_frontend_state_to_working_memory"]
