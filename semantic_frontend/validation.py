from __future__ import annotations

"""Structural checks only; no semantic scoring."""
import math
from dataclasses import asdict, dataclass, field
from typing import Any
from .graph import EventGraph, Event, Edge, Mainline, VALID_MODALITIES, VALID_RELATIONS

# ============================================================
# 2. Graph validation
# ============================================================

@dataclass
class ValidationReport:
    dropped_events: int = 0
    dropped_edges: int = 0
    repaired_fields: int = 0
    warnings: list[str] = field(default_factory=list)


class GraphContractValidator:
    """Validate grounding and schema constraints after every LLM call."""

    @staticmethod
    def _grounded(span: str, text: str) -> bool:
        return bool(span and span.strip() and span.strip() in text)

    @staticmethod
    def _confidence(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.5
        if v != v or v in (float("inf"), float("-inf")):
            return 0.5
        return max(0.0, min(1.0, v))

    def validate(self, graph: EventGraph) -> ValidationReport:
        report = ValidationReport()

        valid_events: list[Event] = []
        for event in graph.events:
            if not self._grounded(event.source_span, graph.raw_text):
                report.dropped_events += 1
                report.warnings.append(
                    f"drop {event.id}: source_span is not grounded"
                )
                continue

            if event.modality not in VALID_MODALITIES:
                event.modality = "unknown"
                report.repaired_fields += 1

            event.confidence = self._confidence(event.confidence)
            event.level = max(0, int(event.level or 0))
            valid_events.append(event)

        graph.events = valid_events
        valid_ids = {e.id for e in graph.events}

        for event in graph.events:
            if event.parent_id and event.parent_id not in valid_ids:
                event.parent_id = None
                report.repaired_fields += 1

        valid_edges: list[Edge] = []
        for edge in graph.edges:
            if edge.source not in valid_ids or edge.target not in valid_ids:
                report.dropped_edges += 1
                continue

            if edge.relation not in VALID_RELATIONS:
                edge.relation = "other"
                report.repaired_fields += 1

            if not edge.explicit:
                report.dropped_edges += 1
                report.warnings.append(
                    f"drop {edge.source}->{edge.target}: relation is inferred"
                )
                continue

            if not self._grounded(edge.source_span, graph.raw_text):
                report.dropped_edges += 1
                report.warnings.append(
                    f"drop {edge.source}->{edge.target}: edge span is not grounded"
                )
                continue

            edge.confidence = self._confidence(edge.confidence)
            valid_edges.append(edge)

        graph.edges = valid_edges

        if not graph.mainlines and graph.events:
            graph.mainlines = [
                Mainline(
                    id="M1",
                    title="Mainline",
                    summary="",
                    event_ids=[],
                )
            ]

        mainline_ids = {m.id for m in graph.mainlines}
        fallback = graph.mainlines[0].id if graph.mainlines else "M1"

        for event in graph.events:
            if event.mainline_id not in mainline_ids:
                event.mainline_id = fallback
                report.repaired_fields += 1

        valid_ids = {e.id for e in graph.events}
        for mainline in graph.mainlines:
            mainline.event_ids = [
                eid for eid in mainline.event_ids if eid in valid_ids
            ]

        for event in graph.events:
            m = next(
                (m for m in graph.mainlines if m.id == event.mainline_id),
                None,
            )
            if m and event.id not in m.event_ids:
                m.event_ids.append(event.id)

        graph.validation = asdict(report)
        self._check_structure(graph)
        return report

    def _check_structure(self, graph: EventGraph):
        """Only types, references and structure; never judge semantic correctness."""
        if not isinstance(graph.raw_text, str) or not graph.raw_text.strip():
            raise ValueError("raw_text must be a nonempty string")
        def ids(items, label):
            values = [item.id for item in items]
            if any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
                raise ValueError(f"{label}: empty or duplicate IDs")
            return set(values)
        event_ids = ids(graph.events, "events")
        mainline_ids = ids(graph.mainlines, "mainlines")
        by_id = {e.id: e for e in graph.events}
        for e in graph.events:
            if e.span is not None:
                allowed = {"core", "relation", "nature", "feature_type", "span_role", "value", "unit", "semantic_relations"}
                if not isinstance(e.span, dict) or set(e.span) - allowed:
                    raise ValueError("invalid structured span fields")
                for key, value in e.span.items():
                    if key == "value":
                        if value is not None and not isinstance(value, (str, int, float, bool)):
                            raise ValueError("span value must be a scalar")
                        if isinstance(value, float) and not math.isfinite(value):
                            raise ValueError("span value must be finite")
                    elif not isinstance(value, str):
                        if key != "semantic_relations" or not isinstance(value, list):
                            raise ValueError(f"span {key} must be a string")
                if "semantic_relations" in e.span:
                    for relation in e.span["semantic_relations"]:
                        if not isinstance(relation, dict) or set(relation) - {"core", "relation", "nature", "source_span"}:
                            raise ValueError("invalid semantic relation")
                        if not all(isinstance(relation.get(key), str) and relation[key].strip()
                                   for key in ("core", "relation", "nature")):
                            raise ValueError("semantic relation requires core/relation/nature")
                        if "source_span" in relation and not self._grounded(relation["source_span"], graph.raw_text):
                            raise ValueError("semantic relation source_span not present in source")
            for name in ("canonical_text", "source_span", "mainline_id"):
                if not isinstance(getattr(e, name), str) or not getattr(e, name).strip():
                    raise ValueError(f"{e.id}: invalid {name}")
            if not self._grounded(e.source_span, graph.raw_text):
                raise ValueError(f"{e.id}: source_span not present in source")
            if e.mainline_id not in mainline_ids:
                raise ValueError(f"{e.id}: unknown mainline")
            if type(e.level) is not int or e.level < 0:
                raise ValueError(f"{e.id}: level must be a nonnegative integer")
            if e.parent_id is not None and (not isinstance(e.parent_id, str) or e.parent_id not in event_ids):
                raise ValueError(f"{e.id}: unknown parent")
            if e.modality not in VALID_MODALITIES or e.polarity not in {"positive", "negative", "unknown"}:
                raise ValueError(f"{e.id}: invalid modality or polarity")
            for name in ("event_type", "actor", "object", "state", "direction", "magnitude", "time", "attribution", "condition"):
                if getattr(e, name) is not None and not isinstance(getattr(e, name), str):
                    raise ValueError(f"{e.id}: {name} must be string or null")
            visited = {e.id}
            parent = e.parent_id
            while parent is not None:
                if parent in visited:
                    raise ValueError("cyclic parent hierarchy")
                visited.add(parent)
                if parent not in by_id:
                    raise ValueError("unknown ancestor")
                parent = by_id[parent].parent_id
            if e.parent_id and (by_id[e.parent_id].mainline_id != e.mainline_id or by_id[e.parent_id].level + 1 != e.level):
                raise ValueError("child must share mainline and have parent level + 1")
            if not e.parent_id and e.level != 0:
                raise ValueError("root level must be zero")
        for edge in graph.edges:
            if edge.source not in event_ids or edge.target not in event_ids:
                raise ValueError("unknown edge endpoint")
            if edge.relation not in VALID_RELATIONS or type(edge.explicit) is not bool:
                raise ValueError("invalid relation or explicit flag")
            if not isinstance(edge.source_span, str) or not self._grounded(edge.source_span, graph.raw_text):
                raise ValueError("edge source_span not present in source")
        for mainline in graph.mainlines:
            if not isinstance(mainline.title, str) or not isinstance(mainline.summary, str):
                raise ValueError("mainline title/summary must be strings")
            if not isinstance(mainline.event_ids, list) or any(eid not in event_ids for eid in mainline.event_ids):
                raise ValueError("unknown mainline event reference")
            if len(set(mainline.event_ids)) != len(mainline.event_ids):
                raise ValueError("duplicate mainline event reference")
            if any(by_id[eid].mainline_id != mainline.id for eid in mainline.event_ids):
                raise ValueError("inconsistent mainline membership")
