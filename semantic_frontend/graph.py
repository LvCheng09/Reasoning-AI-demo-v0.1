from __future__ import annotations

"""Text graph data model and decoding, with no LLM or backend dependency."""
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ============================================================
# 1. Event Graph Contract
# ============================================================

VALID_MODALITIES = {
    "request",
    "fact",
    "prediction",
    "possibility",
    "counterfactual",
    "unknown",
}

VALID_RELATIONS = {
    "measures",
    "measured_in",
    "cause",
    "before",
    "after",
    "part_of",
    "contrast",
    "supports",
    "prevents",
    "enables",
    "condition",
    "other",
}


@dataclass
class Event:
    id: str
    canonical_text: str
    mainline_id: str
    level: int = 0
    parent_id: Optional[str] = None
    event_type: Optional[str] = None
    actor: Optional[str] = None
    object: Optional[str] = None
    state: Optional[str] = None
    direction: Optional[str] = None
    magnitude: Optional[str] = None
    time: Optional[str] = None
    modality: str = "fact"
    polarity: str = "unknown"
    attribution: Optional[str] = None
    condition: Optional[str] = None
    source_span: str = ""
    confidence: float = 1.0
    feature: Optional[list[float]] = None
    span: Optional[dict[str, Any]] = None
    embedding_model: str = ""
    vector_version: str = ""


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    source_span: str = ""
    explicit: bool = True
    confidence: float = 1.0


@dataclass
class Mainline:
    id: str
    title: str
    summary: str
    event_ids: list[str] = field(default_factory=list)


@dataclass
class EventGraph:
    doc_id: str
    raw_text: str
    mainlines: list[Mainline] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    version: int = 0
    validation: dict[str, Any] = field(default_factory=dict)

    def event_by_id(self, event_id: str) -> Optional[Event]:
        return next((e for e in self.events if e.id == event_id), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# ============================================================
# 6. Graph construction helpers
# ============================================================

def event_from_dict(data: dict[str, Any]) -> Event:
    if not isinstance(data, dict):
        raise ValueError("event must be an object")
    allowed = set(Event.__dataclass_fields__)
    forbidden = {"feature", "embedding_model", "vector_version"}
    if set(data) - allowed or forbidden.intersection(data):
        raise ValueError("unknown or Python-owned event fields")
    clean = {k: v for k, v in data.items() if k in allowed}
    clean.setdefault("id", f"E_{uuid.uuid4().hex[:8]}")
    clean.setdefault("canonical_text", "")
    clean.setdefault("mainline_id", "M1")
    return Event(**clean)


def edge_from_dict(data: dict[str, Any]) -> Edge:
    allowed = set(Edge.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("invalid edge fields")
    clean = {k: v for k, v in data.items() if k in allowed}
    return Edge(**clean)


def mainline_from_dict(data: dict[str, Any]) -> Mainline:
    allowed = set(Mainline.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("invalid mainline fields")
    clean = {k: v for k, v in data.items() if k in allowed}
    return Mainline(**clean)
