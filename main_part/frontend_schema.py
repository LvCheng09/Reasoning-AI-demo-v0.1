"""Normalize frontend containers without discarding evidence references."""

from collections.abc import Mapping
import logging
import math
from typing import Any

from .common.normalize import finite_confidence
from .semantic_subgraph import normalize_semantic_relations

logger = logging.getLogger(__name__)


def records(value: Any, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"frontend {field} must be a list of objects")
    return [dict(item) for item in value]


def metadata(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"frontend {field} must be an object")
    return dict(value)


def confidence(value: Any, *, field: str, default: float = 0.7) -> float:
    try:
        number = float(value)
        valid = math.isfinite(number) and 0 <= number <= 1
    except (TypeError, ValueError):
        valid = False
    if not valid:
        logger.warning("Normalized invalid frontend confidence at %s", field)
    return finite_confidence(value, default)


def normalize_spans(value: Any) -> list[dict[str, Any]]:
    spans = records(value, "spans")
    for index, span in enumerate(spans):
        span_id = str(span.get("span_id") or span.get("id") or f"s{index + 1}").strip()
        span["span_id"] = span_id
        span.setdefault("id", span_id)
        for field in ("core", "relation", "nature"):
            span.setdefault(field, "")
        # A span owns one value/evidence item but may express several explicit
        # semantic triples.  Missing triples are not inferred from its labels.
        span["semantic_relations"] = normalize_semantic_relations(span)
        span.setdefault("text", " ".join(str(span.get(key) or "") for key in ("core", "relation", "nature")).strip())
        span["confidence"] = confidence(span.get("confidence", 0.7), field=f"spans[{index}]")
        if "metadata" in span:
            span["metadata"] = metadata(span["metadata"], f"spans[{index}].metadata")
    return spans


def normalize_events(value: Any) -> list[dict[str, Any]]:
    events = records(value, "events")
    for index, event in enumerate(events):
        workspace = event.get("workspace")
        if workspace is not None and not isinstance(workspace, Mapping):
            raise ValueError("frontend event.workspace must be an object")
        for key, item in (workspace or {}).items():
            event.setdefault(key, item)
        event["event_id"] = str(event.get("event_id") or event.get("id") or f"e{index + 1}").strip()
        evidence = event.get("evidence_span_ids", [])
        if not isinstance(evidence, list) or any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValueError("frontend evidence_span_ids must be a list of nonempty strings")
        event["evidence_span_ids"] = list(dict.fromkeys(item.strip() for item in evidence))
        if "use_all_spans" in event and not isinstance(event["use_all_spans"], bool):
            raise ValueError("frontend use_all_spans must be a boolean")
        for field in ("relations", "candidate_paths", "feedback"):
            event[field] = records(event.get(field, []), f"events[{index}].{field}")
        event["metadata"] = metadata(event.get("metadata", {}), f"events[{index}].metadata")
        event["confidence"] = confidence(event.get("confidence", 0.7), field=f"events[{index}]")
        for relation in event["relations"]:
            for endpoint in ("source", "target"):
                name = f"{endpoint}_span_id"
                item = relation.get(name)
                if item is None or item == "":
                    item = relation.get(endpoint)
                if item is None:
                    item = ""
                if not isinstance(item, str):
                    raise ValueError(f"frontend relation.{name} must be a string")
                if item:
                    relation[name] = item.strip()
            relation["confidence"] = confidence(relation.get("confidence", 0.7), field=f"events[{index}].relation")
            if "metadata" in relation:
                relation["metadata"] = metadata(relation["metadata"], f"events[{index}].relation.metadata")
    return events
