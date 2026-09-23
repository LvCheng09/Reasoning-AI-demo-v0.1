"""Versioned contract between a frontend event builder and the reasoning runtime."""

from __future__ import annotations

from typing import Any, Mapping

from .semantic_subgraph import normalize_semantic_relations


EVENT_GRAPH_CONTRACT_VERSION = "event_graph.v1"


def validate_event_graph_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    collections = {}
    for field in ("spans", "events"):
        items = value.get(field, [])
        if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
            errors.append({"status": "invalid_collection", "field": field})
            items = []
        collections[field] = items
    spans, events = collections["spans"], collections["events"]
    if not spans:
        errors.append({"status": "missing_spans"})
    if not events:
        errors.append({"status": "missing_events"})

    span_ids = [str(item.get("span_id") or item.get("id") or "").strip() for item in spans]
    missing_span_ids = [index for index, span_id in enumerate(span_ids) if not span_id]
    if missing_span_ids:
        errors.append({"status": "missing_span_ids", "indices": missing_span_ids})
    duplicates = sorted({span_id for span_id in span_ids if span_id and span_ids.count(span_id) > 1})
    if duplicates:
        errors.append({"status": "duplicate_span_ids", "span_ids": duplicates})

    event_ids = [str(item.get("event_id") or item.get("id") or "").strip() for item in events]
    missing_event_ids = [index for index, event_id in enumerate(event_ids) if not event_id]
    if missing_event_ids:
        errors.append({"status": "missing_event_ids", "indices": missing_event_ids})
    duplicate_event_ids = sorted(
        {event_id for event_id in event_ids if event_id and event_ids.count(event_id) > 1}
    )
    if duplicate_event_ids:
        errors.append({"status": "duplicate_event_ids", "event_ids": duplicate_event_ids})

    known_ids = {span_id for span_id in span_ids if span_id}
    for span in spans:
        span_id = str(span.get("span_id") or span.get("id") or "").strip()
        try:
            relations = normalize_semantic_relations(span)
        except ValueError as exc:
            errors.append({"status": "invalid_semantic_relations", "span_id": span_id, "reason": str(exc)})
            continue
        for relation in relations:
            source_span = relation.get("source_span")
            document = (span.get("metadata") or {}).get("semantic_event", {}).get("source_span")
            if source_span and document and source_span not in document:
                errors.append({"status": "semantic_relation_not_grounded", "span_id": span_id, "source_span": source_span})
    for event in events:
        event_id = str(event.get("event_id") or event.get("id") or "").strip()
        evidence = event.get("evidence_span_ids", [])
        if not isinstance(evidence, list) or any(not isinstance(item, str) or not item.strip() for item in evidence):
            errors.append({"status": "invalid_evidence_span_ids", "event_id": event_id})
            evidence = []
        missing = [item for item in evidence if item not in known_ids]
        if missing:
            errors.append({"status": "unknown_evidence_span_ids", "event_id": event_id, "span_ids": missing})
        if not evidence and event.get("use_all_spans") is not True:
            errors.append({"status": "empty_event_evidence", "event_id": event_id})
        if "use_all_spans" in event and not isinstance(event["use_all_spans"], bool):
            errors.append({"status": "invalid_use_all_spans", "event_id": event_id})
        relations = event.get("relations", [])
        if not isinstance(relations, list):
            errors.append({"status": "invalid_relations", "event_id": event_id})
            relations = []
        for relation in relations:
            if not isinstance(relation, Mapping):
                errors.append({"status": "invalid_relation", "event_id": event_id})
                continue
            source = str(relation.get("source_span_id") or relation.get("source") or "").strip()
            target = str(relation.get("target_span_id") or relation.get("target") or "").strip()
            if not source or not target:
                warnings.append(
                    {
                        "status": "relation_endpoint_missing",
                        "event_id": event_id,
                        "relation_id": str(relation.get("relation_id") or relation.get("id") or ""),
                        "missing_endpoints": [
                            endpoint
                            for endpoint, value in (("source", source), ("target", target))
                            if not value
                        ],
                    }
                )
                continue
            missing = [endpoint for endpoint in (source, target) if endpoint and endpoint not in known_ids]
            if missing:
                warnings.append(
                    {
                        "status": "relation_endpoint_not_yet_resolved",
                        "event_id": event_id,
                        "relation_id": str(relation.get("relation_id") or relation.get("id") or ""),
                        "missing_span_ids": missing,
                    }
                )
    return {
        "ok": not errors,
        "status": "accepted" if not errors else "rejected",
        "version": EVENT_GRAPH_CONTRACT_VERSION,
        "errors": errors,
        "warnings": warnings,
        "span_count": len(spans),
        "event_count": len(events),
    }


__all__ = ["EVENT_GRAPH_CONTRACT_VERSION", "validate_event_graph_contract"]
