"""Data models and schema objects for the main-part reasoning system."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, RLock
from typing import Any

from .identity import new_runtime_id, stable_content_id


def _tensor_to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_dict_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        payload = _as_dict_payload(item)
        if payload:
            result.append(payload)
    return result


def _normalize_event_claims(claims: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, dict):
            continue
        event_id = str(claim.get("event_id") or "").strip()
        span_id = str(claim.get("span_id") or "").strip()
        usage_policy = str(claim.get("usage_policy") or "reusable").strip().lower()
        if usage_policy not in {"consumable", "capacity"}:
            continue
        amount = float(claim.get("amount", 1.0) or 1.0)
        if not event_id or not span_id or amount <= 0.0:
            raise ValueError("event claims require event_id, span_id, and positive amount")
        normalized.append(
            {
                "event_id": event_id,
                "span_id": span_id,
                "usage_policy": usage_policy,
                "amount": amount,
                "required_slot_id": str(claim.get("required_slot_id") or ""),
                "rule_id": str(claim.get("rule_id") or ""),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            item["event_id"],
            item["span_id"],
            item["usage_policy"],
            item["rule_id"],
            item["required_slot_id"],
        ),
    )


def _event_workspace_span(workspace: dict[str, Any], span_id: str) -> dict[str, Any] | None:
    wanted = str(span_id or "").strip()
    for span in workspace.get("spans", []) or []:
        if isinstance(span, dict) and str(span.get("span_id") or span.get("id") or "").strip() == wanted:
            return span
    return None


def _event_span_capacity(span: dict[str, Any], usage_policy: str) -> float:
    if usage_policy == "consumable":
        return 1.0
    raw_capacity = span.get("capacity", span.get("available_capacity"))
    if raw_capacity is None:
        raise ValueError(
            f"capacity-limited event span requires capacity: {span.get('span_id') or span.get('id') or ''}"
        )
    capacity = float(raw_capacity)
    if capacity <= 0.0:
        raise ValueError("event span capacity must be positive")
    return capacity


def _claim_rejected(reservation_id: str, claim: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "reservation_id": reservation_id,
        "event_id": claim.get("event_id"),
        "span_id": claim.get("span_id"),
    }


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@dataclass
class SpanUnit:
    """Frontend-produced span triple used by matcher and rule.required.

    Text fields are debug labels. Matching should prefer the vector fields when
    they are available.
    """

    span_id: str
    core: str = ""
    relation: str = ""
    nature: str = ""
    text: str = ""
    role: str = ""
    feature_type: str = ""
    value: Any | None = None
    unit: str = ""
    confidence: float = 0.0
    span_core_vector: Any | None = field(default=None, repr=False)
    relation_vector: Any | None = field(default=None, repr=False)
    nature_vector: Any | None = field(default=None, repr=False)
    vector: Any | None = field(default=None, repr=False)
    source: str = "frontend"
    embedding_model: str = ""
    vector_dimension: int = 0
    vector_version: str = ""
    semantic_relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text or " ".join(item for item in (self.core, self.relation, self.nature) if item)
        return {
            "span_id": self.span_id,
            "id": self.span_id,
            "text": text,
            "core": self.core,
            "relation": self.relation,
            "nature": self.nature,
            "role": self.role,
            "span_role": self.role,
            "feature_type": self.feature_type,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "core_vector": _tensor_to_list(self.span_core_vector),
            "span_core_vector": _tensor_to_list(self.span_core_vector),
            "relation_vector": _tensor_to_list(self.relation_vector),
            "nature_vector": _tensor_to_list(self.nature_vector),
            "vector": _tensor_to_list(self.vector),
            "source": self.source,
            "embedding_model": self.embedding_model,
            "vector_dimension": self.vector_dimension,
            "vector_version": self.vector_version,
            "semantic_relations": [dict(item) for item in self.semantic_relations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SpanUnit":
        return cls(
            span_id=str(value.get("span_id") or value.get("id") or ""),
            core=str(value.get("core") or value.get("span_core") or ""),
            relation=str(value.get("relation") or value.get("span_relation") or ""),
            nature=str(value.get("nature") or value.get("span_nature") or ""),
            text=str(value.get("text") or value.get("surface") or ""),
            role=str(value.get("role") or value.get("span_role") or value.get("type") or ""),
            feature_type=str(value.get("feature_type") or value.get("feature_schema_id") or ""),
            value=value.get("value"),
            unit=str(value.get("unit") or ""),
            confidence=float(value.get("confidence", 0.0) or 0.0),
            span_core_vector=_first_present(value, "span_core_vector", "span_core_vec", "core_vector"),
            relation_vector=_first_present(value, "relation_vector", "relation_vec"),
            nature_vector=_first_present(value, "nature_vector", "nature_vec"),
            vector=value.get("vector"),
            source=str(value.get("source") or "frontend"),
            embedding_model=str(value.get("embedding_model") or ""),
            vector_dimension=int(value.get("vector_dimension", 0) or 0),
            vector_version=str(value.get("vector_version") or ""),
            semantic_relations=[dict(item) for item in value.get("semantic_relations", []) if isinstance(item, dict)],
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class EventRelation:
    relation_id: str
    source_span_id: str
    target_span_id: str
    relation_type: str
    core: str = ""
    nature: str = ""
    confidence: float = 0.0
    core_vector: Any | None = field(default=None, repr=False)
    relation_vector: Any | None = field(default=None, repr=False)
    nature_vector: Any | None = field(default=None, repr=False)
    source: str = "frontend"
    embedding_model: str = ""
    vector_dimension: int = 0
    vector_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "id": self.relation_id,
            "source_span_id": self.source_span_id,
            "target_span_id": self.target_span_id,
            "from_span_id": self.source_span_id,
            "to_span_id": self.target_span_id,
            "relation_type": self.relation_type,
            "relation": self.relation_type,
            "core": self.core,
            "nature": self.nature,
            "confidence": self.confidence,
            "core_vector": _tensor_to_list(self.core_vector),
            "relation_vector": _tensor_to_list(self.relation_vector),
            "nature_vector": _tensor_to_list(self.nature_vector),
            "source": self.source,
            "embedding_model": self.embedding_model,
            "vector_dimension": self.vector_dimension,
            "vector_version": self.vector_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EventRelation":
        return cls(
            relation_id=str(value.get("relation_id") or value.get("id") or ""),
            source_span_id=str(value.get("source_span_id") or value.get("from_span_id") or value.get("source") or ""),
            target_span_id=str(value.get("target_span_id") or value.get("to_span_id") or value.get("target") or ""),
            relation_type=str(value.get("relation_type") or value.get("relation") or value.get("debug_relation") or ""),
            core=str(value.get("core") or ""),
            nature=str(value.get("nature") or ""),
            confidence=float(value.get("confidence", 0.0) or 0.0),
            core_vector=_first_present(value, "core_vector"),
            relation_vector=_first_present(value, "relation_vector", "relation_vec"),
            nature_vector=_first_present(value, "nature_vector"),
            source=str(value.get("source") or "frontend"),
            embedding_model=str(value.get("embedding_model") or ""),
            vector_dimension=int(value.get("vector_dimension", 0) or 0),
            vector_version=str(value.get("vector_version") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class EventHypergraph:
    """Current event graph and the context used to reason over it."""

    event_id: str
    spans: list[SpanUnit | dict[str, Any]] = field(default_factory=list)
    relations: list[EventRelation | dict[str, Any]] = field(default_factory=list)
    focus: dict[str, Any] = field(default_factory=dict)
    candidate_paths: list[dict[str, Any]] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    context_items: list[dict[str, Any]] = field(default_factory=list)
    evidence_span_ids: list[str] = field(default_factory=list)
    event_type: str = ""
    actor_display: str = ""
    time_hints: list[str] = field(default_factory=list)
    location_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "version": self.version,
            "event_type": self.event_type,
            "actor_display": self.actor_display,
            "spans": [_as_dict_payload(item) for item in self.spans],
            "relations": [_as_dict_payload(item) for item in self.relations],
            "focus": dict(self.focus),
            "candidate_paths": _as_dict_list(self.candidate_paths),
            "feedback": _as_dict_list(self.feedback),
            "context_items": _as_dict_list(self.context_items),
            "evidence_span_ids": [str(item) for item in self.evidence_span_ids],
            "time_hints": [str(item) for item in self.time_hints],
            "location_hints": [str(item) for item in self.location_hints],
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EventHypergraph":
        workspace = value.get("workspace") if isinstance(value.get("workspace"), dict) else {}
        merged = {**workspace, **value}
        spans = [
            SpanUnit.from_mapping(item) if isinstance(item, dict) else item
            for item in (merged.get("spans") or [])
        ]
        relations = [
            EventRelation.from_mapping(item) if isinstance(item, dict) else item
            for item in (merged.get("relations") or [])
        ]
        metadata = dict(merged.get("metadata") or {})
        return cls(
            event_id=str(merged.get("event_id") or ""),
            spans=spans,
            relations=relations,
            focus=dict(merged.get("focus") or {}),
            candidate_paths=_as_dict_list(merged.get("candidate_paths")),
            feedback=_as_dict_list(merged.get("feedback")),
            context_items=_as_dict_list(merged.get("context_items")),
            evidence_span_ids=[str(item) for item in (merged.get("evidence_span_ids") or [])],
            event_type=str(merged.get("event_type") or ""),
            actor_display=str(merged.get("actor_display") or ""),
            time_hints=[str(item) for item in (merged.get("time_hints") or [])],
            location_hints=[str(item) for item in (merged.get("location_hints") or [])],
            confidence=float(merged.get("confidence", 0.0) or 0.0),
            version=int(merged.get("version", 0) or 0),
            metadata=metadata,
        )


def _event_hypergraph_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, EventHypergraph):
        return value.to_dict()
    payload = _as_dict_payload(value)
    if not payload:
        return {}
    if isinstance(payload.get("workspace"), dict):
        workspace = dict(payload["workspace"])
        payload = {**workspace, **payload}
        payload.pop("workspace", None)
    payload.setdefault("event_id", str(payload.get("event_id") or ""))
    payload.setdefault("spans", [])
    payload.setdefault("relations", [])
    payload.setdefault("focus", {})
    payload["candidate_paths"] = _as_dict_list(payload.get("candidate_paths"))
    payload["feedback"] = _as_dict_list(payload.get("feedback"))
    payload["context_items"] = _as_dict_list(payload.get("context_items"))
    payload.setdefault("evidence_span_ids", [])
    payload.setdefault("time_hints", [])
    payload.setdefault("location_hints", [])
    payload["metadata"] = dict(payload.get("metadata") or {})
    payload["spans"] = _as_dict_list(payload.get("spans"))
    payload["relations"] = _as_dict_list(payload.get("relations"))
    for deprecated in (
        "span_patterns",
        "relation_patterns",
        "known_conditions",
        "needed_conditions",
        "intermediate_states",
        "final_goals",
        "blocked_paths",
        "operations",
        "rules",
        "produced",
        "history",
    ):
        payload.pop(deprecated, None)
    return payload


def _drop_deprecated_event_sections(workspace: dict[str, Any]) -> None:
    for key in (
        "span_patterns",
        "relation_patterns",
        "known_conditions",
        "needed_conditions",
        "intermediate_states",
        "final_goals",
        "blocked_paths",
        "operations",
        "rules",
        "produced",
        "history",
    ):
        workspace.pop(key, None)


@dataclass
class BaseRule:
    rule_id: str
    name: str
    description: str
    function_type: str = "content"
    rule_layer: str = "domain"
    stability_level: str = "adaptive"
    domain: str = "general"
    status: str = "candidate"
    confidence: float = 0.5
    logical_strength: float = 0.5
    empirical_support: float = 0.5
    contradiction_rate: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    scope_tags: list[str] = field(default_factory=list)
    exception_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "function_type": self.function_type,
            "rule_layer": self.rule_layer,
            "stability_level": self.stability_level,
            "domain": self.domain,
            "status": self.status,
            "confidence": self.confidence,
            "logical_strength": self.logical_strength,
            "empirical_support": self.empirical_support,
            "contradiction_rate": self.contradiction_rate,
            "source_ids": self.source_ids,
            "scope_tags": self.scope_tags,
            "exception_patterns": self.exception_patterns,
            "metadata": self.metadata,
        }


@dataclass
class StructuralRule:
    rule_id: str
    name: str
    description: str
    function_type: str = "structural"
    rule_layer: str = "structure"
    stability_level: str = "adaptive"
    domain: str = "general"
    status: str = "active"
    confidence: float = 0.5
    logical_strength: float = 0.5
    empirical_support: float = 0.5
    contradiction_rate: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    scope_tags: list[str] = field(default_factory=list)
    exception_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_types: list[str] = field(default_factory=list)
    proposition_types: list[str] = field(default_factory=list)
    analysis_steps: list[str] = field(default_factory=list)
    mandatory_checks: list[str] = field(default_factory=list)
    preferred_routes: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    reasoning_template: list[str] = field(default_factory=list)
    retrieval_priority: list[str] = field(default_factory=list)
    required_checks: list[str] = field(default_factory=list)
    counterfactual_required: bool = False
    support_count: int = 0
    learned_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "function_type": self.function_type,
            "rule_layer": self.rule_layer,
            "stability_level": self.stability_level,
            "domain": self.domain,
            "status": self.status,
            "confidence": self.confidence,
            "logical_strength": self.logical_strength,
            "empirical_support": self.empirical_support,
            "contradiction_rate": self.contradiction_rate,
            "source_ids": self.source_ids,
            "scope_tags": self.scope_tags,
            "exception_patterns": self.exception_patterns,
            "metadata": self.metadata,
            "task_types": self.task_types,
            "proposition_types": self.proposition_types,
            "analysis_steps": self.analysis_steps,
            "mandatory_checks": self.mandatory_checks,
            "preferred_routes": self.preferred_routes,
            "stop_conditions": self.stop_conditions,
            "risk_tags": self.risk_tags,
            "reasoning_template": self.reasoning_template or self.analysis_steps,
            "retrieval_priority": self.retrieval_priority or self.preferred_routes,
            "required_checks": self.required_checks or self.mandatory_checks,
            "counterfactual_required": self.counterfactual_required,
            "support_count": self.support_count,
            "learned_from": self.learned_from,
        }


@dataclass
class ControlRule:
    rule_id: str
    name: str
    description: str
    task_type: str = "general_reasoning"
    function_type: str = "control"
    rule_layer: str = "control"
    stability_level: str = "adaptive"
    domain: str = "general"
    status: str = "active"
    confidence: float = 0.5
    logical_strength: float = 0.5
    empirical_support: float = 0.5
    contradiction_rate: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    scope_tags: list[str] = field(default_factory=list)
    exception_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    situation_pattern: dict[str, Any] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)
    budget_policy: dict[str, Any] = field(default_factory=dict)
    controller_mode: str = "hybrid"
    memory_policy: str = "archive_on_demand"
    validation_priority: list[str] = field(default_factory=list)
    exploration_rate: float = 0.10
    success_rate: float = 0.5
    support_count: int = 0
    failure_count: int = 0
    risk_level: str = "medium"
    last_updated: str = ""
    learned_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "task_type": self.task_type,
            "name": self.name,
            "description": self.description,
            "function_type": self.function_type,
            "rule_layer": self.rule_layer,
            "stability_level": self.stability_level,
            "domain": self.domain,
            "status": self.status,
            "confidence": self.confidence,
            "logical_strength": self.logical_strength,
            "empirical_support": self.empirical_support,
            "contradiction_rate": self.contradiction_rate,
            "source_ids": self.source_ids,
            "scope_tags": self.scope_tags,
            "exception_patterns": self.exception_patterns,
            "metadata": self.metadata,
            "situation_pattern": self.situation_pattern,
            "source_weights": self.source_weights,
            "budget_policy": self.budget_policy,
            "controller_mode": self.controller_mode,
            "memory_policy": self.memory_policy,
            "validation_priority": self.validation_priority,
            "exploration_rate": self.exploration_rate,
            "success_rate": self.success_rate,
            "support_count": self.support_count,
            "failure_count": self.failure_count,
            "risk_level": self.risk_level,
            "last_updated": self.last_updated,
            "learned_from": self.learned_from,
        }

@dataclass
class WorkingMemoryEntry:
    entry_id: str
    session_id: str
    stage: str
    source: str
    payload_type: str
    created_at: str
    confidence: float
    workspace_zone: str = "memory_workspace"
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "stage": self.stage,
            "source": self.source,
            "payload_type": self.payload_type,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "workspace_zone": self.workspace_zone,
            "payload": self.payload,
            "tags": self.tags,
        }


@dataclass
class WorkingMemoryState:
    zone_names: tuple[str, ...] = (
        "input_workspace",
        "event_workspace",
        "memory_workspace",
    )
    entries: list[WorkingMemoryEntry] = field(default_factory=list)
    zone_entries: dict[str, list[WorkingMemoryEntry]] = field(default_factory=dict)
    event_workspaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    rule_runtime: dict[str, dict[str, Any]] = field(default_factory=dict)
    _event_claim_lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    _event_transaction_locks: dict[str, Lock] = field(default_factory=dict, init=False, repr=False, compare=False)
    _event_claim_reservations: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _event_claim_consumption: dict[tuple[str, str], float] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.zone_entries:
            self.zone_entries = {zone: [] for zone in self.zone_names}
        else:
            for zone in self.zone_names:
                self.zone_entries.setdefault(zone, [])

    def add(self, entry: WorkingMemoryEntry, zone: str | None = None) -> None:
        chosen_zone = _normalize_working_memory_zone(zone or entry.workspace_zone)
        if chosen_zone not in self.zone_entries:
            raise ValueError(f"Unknown workspace zone: {chosen_zone}")
        entry.workspace_zone = chosen_zone
        self.entries.append(entry)
        self.zone_entries[chosen_zone].append(entry)

    def for_session(
        self,
        session_id: str | None = None,
        zone: str | None = None,
    ) -> list[WorkingMemoryEntry]:
        pool = self.entries
        if zone:
            pool = self.zone_entries.get(_normalize_working_memory_zone(zone), [])
        if not session_id:
            return list(pool)
        return [entry for entry in pool if entry.session_id == session_id]

    def zone_counts(self, session_id: str | None = None) -> dict[str, int]:
        if session_id is None:
            return {zone: len(entries) for zone, entries in self.zone_entries.items()}
        return {
            zone: len([entry for entry in entries if entry.session_id == session_id])
            for zone, entries in self.zone_entries.items()
        }

    def query_memory_entries(
        self,
        *,
        session_id: str | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[WorkingMemoryEntry]:
        entries = self.query_entries(
            session_id=session_id,
            zone="memory_workspace",
            limit=None,
            reverse=reverse,
        )
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return entries

    def query_entries(
        self,
        *,
        session_id: str | None = None,
        zone: str | None = None,
        stage: str | None = None,
        source: str | None = None,
        payload_type: str | None = None,
        tags: list[str] | None = None,
        contains_text: str | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[WorkingMemoryEntry]:
        entries = self.for_session(session_id=session_id, zone=zone)
        if stage is not None:
            entries = [entry for entry in entries if entry.stage == stage]
        if source is not None:
            entries = [entry for entry in entries if entry.source == source]
        if payload_type is not None:
            entries = [entry for entry in entries if entry.payload_type == payload_type]
        if tags:
            wanted_tags = {str(tag) for tag in tags}
            entries = [
                entry for entry in entries
                if wanted_tags.issubset({str(tag) for tag in entry.tags})
            ]
        if contains_text:
            needle = str(contains_text)
            entries = [
                entry for entry in entries
                if needle in str(entry.payload) or needle in " ".join(entry.tags)
            ]
        entries = sorted(entries, key=lambda item: (item.created_at, item.entry_id), reverse=reverse)
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return entries

    def query_event_workspace(
        self,
        *,
        event_id: str | None = None,
        span_id: str | None = None,
        rule_id: str | None = None,
        operation: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        event_ids = [str(event_id)] if event_id else list(self.event_workspaces.keys())
        for current_event_id in event_ids:
            workspace = self.event_workspaces.get(current_event_id)
            if not isinstance(workspace, dict):
                continue
            if span_id and not _workspace_references_span(workspace, str(span_id)):
                continue
            if rule_id and not _workspace_references_rule(workspace, str(rule_id)):
                continue
            if operation and not _workspace_references_operation(workspace, str(operation)):
                continue
            results.append({"event_id": current_event_id, "workspace": workspace})
        return results

    def query_span(self, span_id: str | None = None, text: str | None = None) -> list[dict[str, Any]]:
        wanted_span_id = str(span_id or "").strip()
        wanted_text = str(text or "").strip()
        results: list[dict[str, Any]] = []
        for event_id, workspace in self.event_workspaces.items():
            spans = [item for item in workspace.get("spans", []) if isinstance(item, dict)]
            matched_spans = []
            for span in spans:
                current_id = str(span.get("span_id") or span.get("id") or "")
                current_text = str(span.get("text") or span.get("surface") or "")
                if wanted_span_id and current_id != wanted_span_id:
                    continue
                if wanted_text and wanted_text not in current_text:
                    continue
                matched_spans.append(span)
            if not matched_spans:
                continue
            matched_ids = {
                str(span.get("span_id") or span.get("id") or "")
                for span in matched_spans
                if str(span.get("span_id") or span.get("id") or "")
            }
            relations = [
                relation for relation in workspace.get("relations", [])
                if isinstance(relation, dict) and any(_node_references_value(relation, span_id) for span_id in matched_ids)
            ]
            results.append(
                {
                    "event_id": event_id,
                    "spans": matched_spans,
                    "relations": relations,
                    "focus": workspace.get("focus", {}),
                }
            )
        return results

    def query_rule(self, rule_id: str) -> list[dict[str, Any]]:
        wanted = str(rule_id or "").strip()
        if not wanted:
            return []
        results: list[dict[str, Any]] = []
        for event_id, workspace in self.event_workspaces.items():
            matches = []
            for section in ("candidate_paths", "feedback"):
                for item in workspace.get(section, []) or []:
                    if isinstance(item, dict) and _node_references_value(item, wanted):
                        matches.append({"section": section, "item": item})
            if matches:
                results.append({"event_id": event_id, "matches": matches})
        for entry in self.entries:
            if _node_references_value(entry.payload, wanted):
                results.append({"entry_id": entry.entry_id, "workspace_zone": entry.workspace_zone, "entry": entry.to_dict()})
        return results

    def query_operation(self, operation: str) -> list[dict[str, Any]]:
        wanted = str(operation or "").strip()
        if not wanted:
            return []
        results: list[dict[str, Any]] = []
        for event_id, workspace in self.event_workspaces.items():
            for section in ("candidate_paths", "feedback"):
                for item in workspace.get(section, []) or []:
                    if isinstance(item, dict) and str(item.get("operation") or "") == wanted:
                        results.append({"event_id": event_id, "section": section, "item": item})
        for entry in self.entries:
            if _node_has_operation(entry.payload, wanted):
                results.append({"entry_id": entry.entry_id, "workspace_zone": entry.workspace_zone, "entry": entry.to_dict()})
        return results

    def timeline(
        self,
        *,
        session_id: str | None = None,
        zone: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        entries = self.query_entries(session_id=session_id, zone=zone)
        if limit is not None:
            entries = entries[-max(0, int(limit)):]
        return [entry.to_dict() for entry in entries]

    def clear(self) -> None:
        self.entries.clear()
        for zone in self.zone_entries:
            self.zone_entries[zone] = []
        self.event_workspaces.clear()
        self.rule_runtime.clear()
        with self._event_claim_lock:
            self._event_claim_reservations.clear()
            self._event_claim_consumption.clear()
            self._event_transaction_locks.clear()
        self.__post_init__()

    def clear_runtime_workspaces(
        self,
        *,
        keep_memory_workspace: bool = True,
    ) -> None:
        if self.event_workspaces:
            raise RuntimeError(
                "event_workspace still contains events; transfer them to memory_workspace before clearing"
            )
        with self._event_claim_lock:
            if self._event_claim_reservations:
                raise RuntimeError(
                    "event_workspace still has active rule-path transactions; finish or rollback them before clearing"
                )
        keep_zones: set[str] = set()
        if keep_memory_workspace:
            keep_zones.add("memory_workspace")
        self.entries = [
            entry
            for entry in self.entries
            if _normalize_working_memory_zone(entry.workspace_zone) in keep_zones
        ]
        for zone in self.zone_entries:
            if zone in keep_zones:
                self.zone_entries[zone] = [
                    entry
                    for entry in self.zone_entries[zone]
                    if _normalize_working_memory_zone(entry.workspace_zone) in keep_zones
                ]
            else:
                self.zone_entries[zone] = []
        self.event_workspaces.clear()
        with self._event_claim_lock:
            self._event_claim_consumption.clear()
            self._event_transaction_locks.clear()
        self.__post_init__()

    def remove_entries(self, *, zone: str, entry_ids: list[str]) -> dict[str, Any]:
        normalized_zone = _normalize_working_memory_zone(zone)
        wanted = {str(entry_id) for entry_id in entry_ids if str(entry_id)}
        if not wanted:
            return {"zone": normalized_zone, "removed": 0, "entry_ids": []}
        before = len(self.zone_entries.get(normalized_zone, []) or [])
        self.entries = [
            entry
            for entry in self.entries
            if not (
                _normalize_working_memory_zone(entry.workspace_zone) == normalized_zone
                and entry.entry_id in wanted
            )
        ]
        self.zone_entries[normalized_zone] = [
            entry
            for entry in self.zone_entries.get(normalized_zone, [])
            if entry.entry_id not in wanted
        ]
        after = len(self.zone_entries.get(normalized_zone, []) or [])
        return {
            "zone": normalized_zone,
            "removed": before - after,
            "entry_ids": sorted(wanted),
        }

    def transfer_event_workspace_to_memory(
        self,
        *,
        session_id: str,
        event_ids: list[str] | None = None,
        completion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_ids = [
            str(event_id)
            for event_id in (event_ids or list(self.event_workspaces))
            if str(event_id) in self.event_workspaces
        ]
        if not selected_ids:
            return {
                "status": "no_event_workspace_content",
                "transferred": False,
                "event_ids": [],
            }
        with self._event_claim_lock:
            active_claim_event_ids = {
                str(event_id)
                for reservation in self._event_claim_reservations.values()
                for event_id in reservation.get("event_ids", [])
                if str(event_id)
            }
            blocked_event_ids = sorted(set(selected_ids) & active_claim_event_ids)
            if blocked_event_ids:
                raise RuntimeError(
                    "cannot transfer event_workspace with active rule-path reservations: "
                    + ", ".join(blocked_event_ids)
                )
        event_workspaces = {
            event_id: deepcopy(self.event_workspaces[event_id])
            for event_id in selected_ids
        }
        runtime_entries = [
            entry.to_dict()
            for entry in self.zone_entries.get("event_workspace", [])
            if entry.session_id == str(session_id)
        ]
        input_entries = [
            entry.to_dict()
            for entry in self.zone_entries.get("input_workspace", [])
            if entry.session_id == str(session_id)
        ]
        created_at = _utc_now_iso()
        completion_payload = {
            "status": "completed",
            "completed_at": created_at,
            **dict(completion or {}),
        }
        triggered_rule_ids = [
            str(value)
            for value in completion_payload.get("learning_triggered_by_rule_ids", [])
            if str(value)
        ]
        payload = {
            "event_ids": selected_ids,
            "event_workspaces": event_workspaces,
            "event_runtime_entries": runtime_entries,
            "input_entries": input_entries,
            "input_context_summary": _completed_input_context_summary(input_entries),
            "completion": completion_payload,
            "created_at": created_at,
            "updated_at": created_at,
            "revision": 1,
            "rule_mutations": [],
            "learning_state": {
                "status": "pending",
                "transferred_at": created_at,
                "triggered_by_rule_ids": triggered_rule_ids,
            },
        }
        entry = WorkingMemoryEntry(
            entry_id=new_runtime_id("completed_event_memory"),
            session_id=str(session_id),
            stage="event_completed",
            source="event_workspace",
            payload_type="completed_event_package",
            created_at=created_at,
            confidence=1.0,
            workspace_zone="memory_workspace",
            payload=payload,
            tags=["completed_event", "short_term_memory", *selected_ids],
        )
        self.add(entry, zone="memory_workspace")
        stored = any(
            current.entry_id == entry.entry_id
            for current in self.zone_entries.get("memory_workspace", [])
        )
        if not stored:
            raise RuntimeError("completed event package was not stored in memory_workspace")
        for event_id in selected_ids:
            self.event_workspaces.pop(event_id, None)
        with self._event_claim_lock:
            selected_set = set(selected_ids)
            self._event_claim_consumption = {
                key: amount
                for key, amount in self._event_claim_consumption.items()
                if key[0] not in selected_set
            }
            for event_id in selected_ids:
                self._event_transaction_locks.pop(event_id, None)
        if not self.event_workspaces:
            event_entry_ids = [
                current.entry_id
                for current in self.zone_entries.get("event_workspace", [])
                if current.session_id == str(session_id)
            ]
            input_entry_ids = [
                current.entry_id
                for current in self.zone_entries.get("input_workspace", [])
                if current.session_id == str(session_id)
            ]
            self.remove_entries(zone="event_workspace", entry_ids=event_entry_ids)
            self.remove_entries(zone="input_workspace", entry_ids=input_entry_ids)
        return {
            "status": "transferred_event_package",
            "transferred": True,
            "memory_entry_id": entry.entry_id,
            "event_ids": selected_ids,
            "runtime_entry_count": len(runtime_entries),
            "input_entry_count": len(input_entries),
        }

    def apply_memory_workspace_operation(
        self,
        operation: dict[str, Any],
        *,
        rule_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        op = str(operation.get("operation") or "").strip().lower()
        normalized_rule_id = str(rule_id or "").strip()
        if not normalized_rule_id:
            raise ValueError("memory_workspace operation requires the executed rule_id")
        entry_id = str(operation.get("entry_id") or operation.get("id") or "").strip()
        source_ids = [str(value) for value in operation.get("source_entry_ids", []) if str(value)]
        existing = {
            entry.entry_id: entry
            for entry in self.zone_entries.get("memory_workspace", [])
        }
        now = _utc_now_iso()
        reason = str(operation.get("reason") or "").strip()
        if op == "add":
            item = dict(operation.get("item") or {})
            entry_id = entry_id or stable_content_id(
                "memory_workspace",
                {"session_id": session_id, "rule_id": normalized_rule_id, "item": item, "created_at": now},
            )
            if entry_id in existing:
                raise ValueError(f"memory_workspace entry already exists: {entry_id}")
            item.setdefault("revision", 1)
            item.setdefault("created_at", now)
            item["updated_at"] = now
            item.setdefault("rule_mutations", [])
            item["rule_mutations"].append(
                _memory_rule_mutation(normalized_rule_id, op, reason, 0, 1, now)
            )
            entry = WorkingMemoryEntry(
                entry_id=entry_id,
                session_id=str(session_id),
                stage="rule_memory_add",
                source="event_rule",
                payload_type=str(operation.get("payload_type") or "rule_created_memory"),
                created_at=now,
                confidence=float(operation.get("confidence", 1.0) or 1.0),
                workspace_zone="memory_workspace",
                payload=item,
                tags=["short_term_memory", "rule_mutation", normalized_rule_id],
            )
            self.add(entry, zone="memory_workspace")
            return {"status": "memory_entry_added", "entry_id": entry_id, "revision": 1}
        if op == "merge":
            if len(source_ids) < 2:
                raise ValueError("memory_workspace merge requires at least two source_entry_ids")
            missing = [source_id for source_id in source_ids if source_id not in existing]
            if missing:
                raise ValueError(f"memory_workspace merge source not found: {missing}")
            merged_payload = _merge_memory_payloads([existing[source_id].payload for source_id in source_ids])
            target_id = entry_id or stable_content_id(
                "merged_memory_workspace",
                {"source_entry_ids": sorted(source_ids), "rule_id": normalized_rule_id, "payload": merged_payload},
            )
            before_revision = max(int(existing[source_id].payload.get("revision", 1) or 1) for source_id in source_ids)
            merged_payload["revision"] = before_revision + 1
            merged_payload["updated_at"] = now
            merged_payload.setdefault("rule_mutations", []).append(
                _memory_rule_mutation(normalized_rule_id, op, reason, before_revision, before_revision + 1, now)
            )
            self.remove_entries(zone="memory_workspace", entry_ids=source_ids)
            merged_entry = WorkingMemoryEntry(
                entry_id=target_id,
                session_id=str(session_id),
                stage="rule_memory_merge",
                source="event_rule",
                payload_type="merged_event_memory",
                created_at=now,
                confidence=1.0,
                workspace_zone="memory_workspace",
                payload=merged_payload,
                tags=["short_term_memory", "rule_mutation", normalized_rule_id],
            )
            self.add(merged_entry, zone="memory_workspace")
            return {
                "status": "memory_entries_merged",
                "entry_id": target_id,
                "source_entry_ids": source_ids,
                "revision": before_revision + 1,
            }
        if not entry_id or entry_id not in existing:
            raise ValueError(f"memory_workspace entry not found: {entry_id}")
        entry = existing[entry_id]
        before_revision = int(entry.payload.get("revision", 1) or 1)
        if op == "delete":
            self.remove_entries(zone="memory_workspace", entry_ids=[entry_id])
            return {
                "status": "memory_entry_deleted",
                "entry_id": entry_id,
                "audit": _memory_rule_mutation(
                    normalized_rule_id,
                    op,
                    reason,
                    before_revision,
                    before_revision + 1,
                    now,
                ),
            }
        if op != "update":
            raise ValueError(f"unsupported memory_workspace operation: {op}")
        item = dict(operation.get("item") or {})
        updated = deepcopy(entry.payload)
        updated.update(item)
        updated["revision"] = before_revision + 1
        updated["updated_at"] = now
        mutations = list(updated.get("rule_mutations") or [])
        mutations.append(
            _memory_rule_mutation(
                normalized_rule_id,
                op,
                reason,
                before_revision,
                before_revision + 1,
                now,
            )
        )
        updated["rule_mutations"] = mutations
        entry.payload = updated
        entry.stage = "rule_memory_update"
        entry.tags = list(dict.fromkeys([*entry.tags, "rule_mutation", normalized_rule_id]))
        return {
            "status": "memory_entry_updated",
            "entry_id": entry_id,
            "revision": before_revision + 1,
        }



    def get_event_workspace(self, event_id: str) -> dict[str, Any]:
        normalized_event_id = str(event_id or "").strip()
        if not normalized_event_id:
            raise ValueError("event_id is required.")
        workspace = self.event_workspaces.setdefault(
            normalized_event_id,
            {
                "event_id": normalized_event_id,
                "version": 0,
                "event_type": "",
                "actor_display": "",
                "focus": {},
                "spans": [],
                "relations": [],
                "candidate_paths": [],
                "feedback": [],
                "context_items": [],
                "evidence_span_ids": [],
                "time_hints": [],
                "location_hints": [],
                "confidence": 0.0,
                "metadata": {},
            },
        )
        workspace.setdefault("event_id", normalized_event_id)
        workspace.setdefault("version", 0)
        workspace.setdefault("event_type", "")
        workspace.setdefault("actor_display", "")
        workspace.setdefault("focus", {})
        workspace.setdefault("spans", [])
        workspace.setdefault("relations", [])
        workspace.setdefault("candidate_paths", [])
        workspace.setdefault("feedback", [])
        workspace.setdefault("context_items", [])
        workspace.setdefault("evidence_span_ids", [])
        workspace.setdefault("time_hints", [])
        workspace.setdefault("location_hints", [])
        workspace.setdefault("confidence", 0.0)
        workspace.setdefault("metadata", {})
        _drop_deprecated_event_sections(workspace)
        return workspace

    def reserve_event_workspace_claims(
        self,
        claims: list[dict[str, Any]],
        *,
        reservation_id: str,
    ) -> dict[str, Any]:
        """Backward-compatible entry for an event-workspace path transaction."""

        return self.begin_event_workspace_transaction(
            event_ids=[],
            claims=claims,
            reservation_id=reservation_id,
        )

    def begin_event_workspace_transaction(
        self,
        *,
        event_ids: list[str],
        claims: list[dict[str, Any]],
        reservation_id: str,
    ) -> dict[str, Any]:
        """Lock affected events and atomically reserve explicit resource claims."""

        normalized_id = str(reservation_id or "").strip()
        if not normalized_id:
            raise ValueError("event claim reservation_id is required")
        normalized_claims = _normalize_event_claims(claims)
        affected_event_ids = sorted(
            {
                *[str(event_id) for event_id in event_ids if str(event_id)],
                *[claim["event_id"] for claim in normalized_claims],
            }
        )
        if not normalized_claims and not affected_event_ids:
            return {"ok": True, "status": "no_claims", "reservation_id": normalized_id, "claims": []}
        with self._event_claim_lock:
            existing = self._event_claim_reservations.get(normalized_id)
            if existing is not None:
                if existing.get("claims") != normalized_claims or existing.get("event_ids") != affected_event_ids:
                    raise ValueError(f"reservation_id already exists with different claims: {normalized_id}")
                return {"ok": True, "status": "already_reserved", "reservation_id": normalized_id, "claims": normalized_claims}
            event_locks = [
                self._event_transaction_locks.setdefault(event_id, Lock())
                for event_id in affected_event_ids
            ]

        acquired_locks: list[Lock] = []
        for event_id, event_lock in zip(affected_event_ids, event_locks):
            if not event_lock.acquire(blocking=False):
                for acquired_lock in reversed(acquired_locks):
                    acquired_lock.release()
                return {
                    "ok": False,
                    "status": "event_workspace_busy",
                    "reservation_id": normalized_id,
                    "event_id": event_id,
                }
            acquired_locks.append(event_lock)

        def reject(payload: dict[str, Any]) -> dict[str, Any]:
            for event_lock in reversed(acquired_locks):
                event_lock.release()
            return payload

        with self._event_claim_lock:
            requested: dict[tuple[str, str], float] = {}
            span_contracts: dict[tuple[str, str], dict[str, Any]] = {}
            for event_id in affected_event_ids:
                if not isinstance(self.event_workspaces.get(event_id), dict):
                    return reject(
                        {
                            "ok": False,
                            "status": "event_workspace_not_found",
                            "reservation_id": normalized_id,
                            "event_id": event_id,
                        }
                    )
            for claim in normalized_claims:
                key = (claim["event_id"], claim["span_id"])
                workspace = self.event_workspaces.get(claim["event_id"])
                if not isinstance(workspace, dict):
                    return reject(_claim_rejected(normalized_id, claim, "event_workspace_not_found"))
                span = _event_workspace_span(workspace, claim["span_id"])
                if span is None:
                    return reject(_claim_rejected(normalized_id, claim, "claimed_span_not_found"))
                capacity = _event_span_capacity(span, claim["usage_policy"])
                requested[key] = requested.get(key, 0.0) + claim["amount"]
                span_contracts[key] = {"capacity": capacity, "span": span}

            reserved_by_others: dict[tuple[str, str], float] = {}
            for other_id, reservation in self._event_claim_reservations.items():
                if other_id == normalized_id:
                    continue
                for claim in reservation.get("claims", []):
                    key = (str(claim.get("event_id") or ""), str(claim.get("span_id") or ""))
                    reserved_by_others[key] = reserved_by_others.get(key, 0.0) + float(claim.get("amount", 0.0) or 0.0)
            for key, amount in requested.items():
                consumed = self._event_claim_consumption.get(key, 0.0)
                reserved = reserved_by_others.get(key, 0.0)
                capacity = float(span_contracts[key]["capacity"])
                if consumed + reserved + amount > capacity + 1e-12:
                    return reject({
                        "ok": False,
                        "status": "claim_capacity_unavailable",
                        "reservation_id": normalized_id,
                        "event_id": key[0],
                        "span_id": key[1],
                        "capacity": capacity,
                        "consumed": consumed,
                        "reserved": reserved,
                        "requested": amount,
                    })

            self._event_claim_reservations[normalized_id] = {
                "event_ids": affected_event_ids,
                "claims": normalized_claims,
                "event_locks": acquired_locks,
                "snapshots": {
                    event_id: deepcopy(self.event_workspaces[event_id])
                    for event_id in affected_event_ids
                },
            }
        return {"ok": True, "status": "reserved", "reservation_id": normalized_id, "claims": normalized_claims}

    def commit_event_workspace_claims(self, *, reservation_id: str) -> dict[str, Any]:
        return self.commit_event_workspace_transaction(reservation_id=reservation_id)

    def commit_event_workspace_transaction(self, *, reservation_id: str) -> dict[str, Any]:
        normalized_id = str(reservation_id or "").strip()
        with self._event_claim_lock:
            reservation = self._event_claim_reservations.pop(normalized_id, None)
            if reservation is None:
                return {"ok": False, "status": "reservation_not_found", "reservation_id": normalized_id}
            for claim in reservation.get("claims", []):
                key = (str(claim.get("event_id") or ""), str(claim.get("span_id") or ""))
                self._event_claim_consumption[key] = self._event_claim_consumption.get(key, 0.0) + float(claim.get("amount", 0.0) or 0.0)
        for event_lock in reversed(reservation.get("event_locks", [])):
            event_lock.release()
        return {"ok": True, "status": "committed", "reservation_id": normalized_id}

    def rollback_event_workspace_claims(self, *, reservation_id: str) -> dict[str, Any]:
        return self.rollback_event_workspace_transaction(reservation_id=reservation_id)

    def rollback_event_workspace_transaction(self, *, reservation_id: str) -> dict[str, Any]:
        """Restore event workspaces changed by a failed selected path."""

        normalized_id = str(reservation_id or "").strip()
        with self._event_claim_lock:
            reservation = self._event_claim_reservations.pop(normalized_id, None)
            if reservation is None:
                return {"ok": False, "status": "reservation_not_found", "reservation_id": normalized_id}
            for event_id, snapshot in (reservation.get("snapshots") or {}).items():
                restored = deepcopy(snapshot)
                restored["version"] = int(restored.get("version", 0) or 0) + 1
                self.event_workspaces[str(event_id)] = restored
        for event_lock in reversed(reservation.get("event_locks", [])):
            event_lock.release()
        return {"ok": True, "status": "rolled_back", "reservation_id": normalized_id}

    def upsert_event_workspace(
        self,
        event_id: str,
        workspace: dict[str, Any] | EventHypergraph,
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        current = self.get_event_workspace(event_id)
        if isinstance(workspace, EventHypergraph):
            supplied_keys = set(workspace.to_dict())
        else:
            raw_workspace = dict(workspace or {})
            supplied_keys = set(raw_workspace)
            nested_workspace = raw_workspace.get("workspace")
            if isinstance(nested_workspace, dict):
                supplied_keys.update(nested_workspace)
        incoming = _event_hypergraph_payload(workspace)
        if merge:
            for key, value in incoming.items():
                if key == "event_id" or key not in supplied_keys:
                    continue
                if key == "metadata" and isinstance(value, dict):
                    current_metadata = (
                        dict(current.get("metadata") or {})
                        if isinstance(current.get("metadata"), dict)
                        else {}
                    )
                    current_metadata.update(value)
                    current[key] = current_metadata
                else:
                    current[key] = value
        else:
            current.clear()
            current.update(incoming)
            current["event_id"] = str(event_id or incoming.get("event_id") or "").strip()
        current["version"] = int(current.get("version", 0) or 0) + 1
        return current

    def upsert_event_hypergraph(
        self,
        event: EventHypergraph | dict[str, Any],
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        payload = _event_hypergraph_payload(event)
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("event_id is required.")
        return self.upsert_event_workspace(event_id, payload, merge=merge)

    def commit_calculated_workspace(self, event_id: str, workspace: dict[str, Any], *, expected_version: int) -> None:
        """Commit one staged produced in this synchronous workspace owner."""
        current = self.get_event_workspace(event_id)
        if int(current.get("version", 0)) != expected_version:
            raise ValueError("event workspace changed during calculation")
        from copy import deepcopy
        self.event_workspaces[event_id] = deepcopy(workspace)

    def apply_event_workspace_operation(
        self,
        event_id: str,
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        workspace = self.get_event_workspace(event_id)
        op = str((operation or {}).get("operation") or "").strip().lower()
        target = str((operation or {}).get("target") or "spans").strip()
        if target in {"produced", "history", "rules", "operations", "blocked_paths", "span_patterns", "relation_patterns"}:
            raise ValueError(f"Runtime field {target!r} is not allowed in event_workspace.")
        if target not in {"spans", "relations", "feedback", "candidate_paths", "context_items"}:
            raise ValueError(f"Unsupported event workspace target: {target}")
        item = dict((operation or {}).get("item") or {})
        item_id = str((operation or {}).get("id") or item.get("id") or item.get("span_id") or item.get("relation_id") or item.get("rule_id") or "")
        collection = workspace.setdefault(target, [])
        if not isinstance(collection, list):
            collection = []
            workspace[target] = collection

        if op == "add":
            collection.append(item)
        elif op == "update":
            for index, existing in enumerate(collection):
                existing_id = _workspace_item_id(existing)
                if existing_id and existing_id == item_id:
                    updated = dict(existing)
                    updated.update(item)
                    collection[index] = updated
                    break
            else:
                collection.append(item)
        elif op == "delete":
            workspace[target] = [
                existing for existing in collection
                if _workspace_item_id(existing) != item_id
            ]
        elif op == "merge":
            merged = dict(item)
            source_ids = {str(value) for value in (operation or {}).get("source_ids", [])}
            workspace[target] = [
                existing for existing in collection
                if _workspace_item_id(existing) not in source_ids
            ]
            workspace[target].append(merged)
        elif op == "split":
            source_id = item_id
            new_items = [
                dict(value) for value in (operation or {}).get("items", [])
                if isinstance(value, dict)
            ]
            workspace[target] = [
                existing for existing in collection
                if _workspace_item_id(existing) != source_id
            ]
            workspace[target].extend(new_items)
        else:
            raise ValueError(f"Unsupported event workspace operation: {op}")

        workspace["version"] = int(workspace.get("version", 0) or 0) + 1
        return workspace

    def add_event_context_records(
        self,
        event_id: str,
        records: list[dict[str, Any]],
        *,
        source: str,
        current_round: int,
        retention_rule_id: str,
        expire_after_unused_rounds: int,
    ) -> dict[str, Any]:
        if not retention_rule_id:
            raise ValueError("event context retention requires a rule_id")
        if expire_after_unused_rounds not in {2, 3}:
            raise ValueError("event context expiry must be two or three unused rounds")
        workspace = self.get_event_workspace(event_id)
        existing = {
            str(item.get("context_id") or ""): item
            for item in workspace.get("context_items", [])
            if isinstance(item, dict)
        }
        added = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            source_record_id = str(
                record.get("memory_id")
                or record.get("memory_entry_id")
                or record.get("knowledge_id")
                or record.get("id")
                or ""
            ).strip()
            context_id = stable_content_id(
                "event_context",
                {"event_id": event_id, "source": source, "source_record_id": source_record_id, "record": record},
            )
            item = {
                "context_id": context_id,
                "source": str(source),
                "source_record_id": source_record_id,
                "record_version": str(record.get("version") or record.get("vector_version") or ""),
                "payload": deepcopy(record),
                "retrieved_at": _utc_now_iso(),
                "retrieved_round": int(current_round),
                "last_used_round": None,
                "unused_rounds": 0,
                "retention_rule_id": str(retention_rule_id),
                "expire_after_unused_rounds": int(expire_after_unused_rounds),
                "status": "active",
            }
            if context_id in existing:
                existing[context_id].update(item)
                continue
            workspace.setdefault("context_items", []).append(item)
            existing[context_id] = item
            added += 1
        workspace["version"] = int(workspace.get("version", 0) or 0) + 1
        return {
            "status": "event_context_added",
            "event_id": event_id,
            "added": added,
            "total": len(workspace.get("context_items", [])),
        }

    def advance_event_context_round(
        self,
        event_id: str,
        *,
        current_round: int,
        used_context_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        workspace = self.get_event_workspace(event_id)
        used = {str(context_id) for context_id in (used_context_ids or []) if str(context_id)}
        kept: list[dict[str, Any]] = []
        removed: list[str] = []
        for raw_item in workspace.get("context_items", []) or []:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            context_id = str(item.get("context_id") or "")
            if context_id in used:
                item["last_used_round"] = int(current_round)
                item["unused_rounds"] = 0
                item["status"] = "active"
                kept.append(item)
                continue
            if item.get("pinned") or item.get("evidence_in_use"):
                kept.append(item)
                continue
            item["unused_rounds"] = int(item.get("unused_rounds", 0) or 0) + 1
            threshold = int(item.get("expire_after_unused_rounds", 0) or 0)
            retention_rule_id = str(item.get("retention_rule_id") or "").strip()
            if retention_rule_id and threshold in {2, 3} and item["unused_rounds"] >= threshold:
                removed.append(context_id)
                continue
            kept.append(item)
        workspace["context_items"] = kept
        if removed:
            workspace["version"] = int(workspace.get("version", 0) or 0) + 1
        return {
            "status": "event_context_round_advanced",
            "event_id": event_id,
            "used": sorted(used),
            "removed_context_ids": removed,
            "remaining": len(kept),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "zones": {
                zone: [entry.to_dict() for entry in entries]
                for zone, entries in self.zone_entries.items()
            },
            "zone_counts": self.zone_counts(),
            "event_workspaces": self.event_workspaces,
            "rule_runtime": deepcopy(self.rule_runtime),
        }


def _completed_input_context_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    input_entry_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        input_entry_ids.append(str(entry.get("entry_id") or ""))
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        context = payload.get("input_context")
        if isinstance(context, dict):
            contexts.append(deepcopy(context))
    return {
        "input_entry_ids": [entry_id for entry_id in input_entry_ids if entry_id],
        "contexts": contexts,
    }


def _memory_rule_mutation(
    rule_id: str,
    operation: str,
    reason: str,
    before_version: int,
    after_version: int,
    created_at: str,
) -> dict[str, Any]:
    payload = {
        "rule_id": str(rule_id),
        "operation": str(operation),
        "reason": str(reason),
        "before_version": int(before_version),
        "after_version": int(after_version),
        "created_at": str(created_at),
    }
    return {
        "mutation_id": stable_content_id("memory_mutation", payload),
        **payload,
    }


def _merge_memory_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "event_ids": [],
        "event_workspaces": {},
        "event_runtime_entries": [],
        "rule_mutations": [],
    }
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        merged["event_ids"] = list(
            dict.fromkeys([*merged["event_ids"], *[str(value) for value in payload.get("event_ids", []) if str(value)]])
        )
        if isinstance(payload.get("event_workspaces"), dict):
            merged["event_workspaces"].update(deepcopy(payload["event_workspaces"]))
        merged["event_runtime_entries"].extend(
            deepcopy([value for value in payload.get("event_runtime_entries", []) if isinstance(value, dict)])
        )
        merged["rule_mutations"].extend(
            deepcopy([value for value in payload.get("rule_mutations", []) if isinstance(value, dict)])
        )
        for key in ("completion", "input_context_summary", "learning_state", "created_at"):
            if key in payload and key not in merged:
                merged[key] = deepcopy(payload[key])
    return merged


def _workspace_item_id(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("id", "span_id", "relation_id", "rule_id", "event_id", "context_id"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _workspace_references_span(workspace: dict[str, Any], span_id: str) -> bool:
    return any(
        _node_references_value(workspace.get(section, []), span_id)
        for section in ("spans", "relations", "focus", "candidate_paths", "feedback")
    )


def _workspace_references_rule(workspace: dict[str, Any], rule_id: str) -> bool:
    return any(
        _node_references_value(workspace.get(section, []), rule_id)
        for section in ("candidate_paths", "feedback")
    )


def _workspace_references_operation(workspace: dict[str, Any], operation: str) -> bool:
    return any(
        _node_has_operation(workspace.get(section, []), operation)
        for section in ("candidate_paths", "feedback")
    )


def _node_references_value(node: Any, value: str) -> bool:
    wanted = str(value)
    if not wanted:
        return False
    if isinstance(node, dict):
        return any(_node_references_value(key, wanted) or _node_references_value(item, wanted) for key, item in node.items())
    if isinstance(node, (list, tuple, set)):
        return any(_node_references_value(item, wanted) for item in node)
    return str(node) == wanted


def _node_has_operation(node: Any, operation: str) -> bool:
    wanted = str(operation)
    if not wanted:
        return False
    if isinstance(node, dict):
        if str(node.get("operation") or "") == wanted:
            return True
        return any(_node_has_operation(item, wanted) for item in node.values())
    if isinstance(node, (list, tuple, set)):
        return any(_node_has_operation(item, wanted) for item in node)
    return False


def _normalize_working_memory_zone(zone: Any) -> str:
    normalized = str(zone or "").strip()
    if normalized in {
        "input_workspace",
        "memory_workspace",
        "event_workspace",
    }:
        return normalized
    raise ValueError(f"Unknown workspace zone: {normalized or '<empty>'}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    record_id: str
    record_type: str
    topic: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    event: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    lifecycle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "topic": self.topic,
            "entities": self.entities,
            "state": self.state,
            "event": self.event,
            "decision": self.decision,
            "uncertainty": self.uncertainty,
            "provenance": self.provenance,
            "lifecycle": self.lifecycle,
        }


@dataclass
class PartialMemoryRecord(MemoryRecord):
    completion_ratio: float = 0.0
    source: str = "input"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["completion_ratio"] = self.completion_ratio
        payload["source"] = self.source
        return payload


@dataclass
class ResolvedMemoryRecord(MemoryRecord):
    resolution_status: str = "candidate"
    resolved_at: str = ""
    trace_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["resolution_status"] = self.resolution_status
        payload["resolved_at"] = self.resolved_at
        payload["trace_ids"] = self.trace_ids
        return payload


@dataclass
class ReasoningNetworkNode:
    node_id: str
    node_type: str
    content: str
    confidence: float = 0.0
    timestamp: str = ""
    rule_id: str | None = None
    proposition_id: str | None = None
    event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Any | None = field(default=None, repr=False)
    embedding_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "rule_id": self.rule_id,
            "proposition_id": self.proposition_id,
            "event_ids": self.event_ids,
            "metadata": self.metadata,
            "vector": _tensor_to_list(self.vector),
            "embedding_model": self.embedding_model,
        }


@dataclass
class ReasoningNetworkEdge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class LongTermMemoryRecord:
    record_id: str
    created_at: str
    unit_type: str
    source_session_id: str = ""
    event_ids: list[str] = field(default_factory=list)
    proposition_ids: list[str] = field(default_factory=list)
    span_keys: list[str] = field(default_factory=list)
    reasoning_network: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "created_at": self.created_at,
            "unit_type": self.unit_type,
            "source_session_id": self.source_session_id,
            "event_ids": self.event_ids,
            "proposition_ids": self.proposition_ids,
            "span_keys": self.span_keys,
            "reasoning_network": self.reasoning_network,
            "payload": self.payload,
            "metadata": self.metadata,
        }


@dataclass
class LongTermMemoryStore:
    records: dict[str, LongTermMemoryRecord] = field(default_factory=dict)
    time_index: list[str] = field(default_factory=list)
    event_index: dict[str, list[str]] = field(default_factory=dict)
    proposition_index: dict[str, list[str]] = field(default_factory=dict)
    span_index: dict[str, list[str]] = field(default_factory=dict)
    episodic_index: list[str] = field(default_factory=list)
    semantic_index: list[str] = field(default_factory=list)
    rule_index: list[str] = field(default_factory=list)
    meta_index: list[str] = field(default_factory=list)

    def _layer_key_for_record(self, record: LongTermMemoryRecord) -> str:
        layer = str(record.metadata.get("memory_layer", "")).strip().lower()
        if layer in {"episodic", "semantic", "rule", "meta"}:
            return layer
        unit = str(record.unit_type).strip().lower()
        if unit in {"rule", "rules", "rule_memory"}:
            return "rule"
        if unit in {"semantic", "knowledge", "kb"}:
            return "semantic"
        if unit in {"meta", "controller", "quality"}:
            return "meta"
        return "episodic"

    def _append_layer_index(self, layer: str, record_id: str) -> None:
        if layer == "semantic":
            self.semantic_index.append(record_id)
            return
        if layer == "rule":
            self.rule_index.append(record_id)
            return
        if layer == "meta":
            self.meta_index.append(record_id)
            return
        self.episodic_index.append(record_id)

    def add(self, record: LongTermMemoryRecord) -> None:
        self.records[record.record_id] = record
        self.time_index.append(record.record_id)
        for event_id in record.event_ids:
            self.event_index.setdefault(event_id, []).append(record.record_id)
        for proposition_id in record.proposition_ids:
            self.proposition_index.setdefault(proposition_id, []).append(record.record_id)
        for span_key in record.span_keys:
            self.span_index.setdefault(span_key, []).append(record.record_id)
        self._append_layer_index(self._layer_key_for_record(record), record.record_id)

    def remove(self, record_id: str) -> LongTermMemoryRecord | None:
        record = self.records.pop(record_id, None)
        if record is None:
            return None
        self.time_index = [item for item in self.time_index if item != record_id]
        self.episodic_index = [item for item in self.episodic_index if item != record_id]
        self.semantic_index = [item for item in self.semantic_index if item != record_id]
        self.rule_index = [item for item in self.rule_index if item != record_id]
        self.meta_index = [item for item in self.meta_index if item != record_id]
        for index in (self.event_index, self.proposition_index, self.span_index):
            for key in list(index.keys()):
                index[key] = [value for value in index[key] if value != record_id]
                if not index[key]:
                    index.pop(key, None)
        return record

    def rebuild_indexes(self) -> None:
        self.time_index = []
        self.event_index = {}
        self.proposition_index = {}
        self.span_index = {}
        self.episodic_index = []
        self.semantic_index = []
        self.rule_index = []
        self.meta_index = []
        for record_id, record in self.records.items():
            self.time_index.append(record_id)
            for event_id in record.event_ids:
                self.event_index.setdefault(event_id, []).append(record_id)
            for proposition_id in record.proposition_ids:
                self.proposition_index.setdefault(proposition_id, []).append(record_id)
            for span_key in record.span_keys:
                self.span_index.setdefault(span_key, []).append(record_id)
            self._append_layer_index(self._layer_key_for_record(record), record_id)

    def clear(self) -> None:
        self.records.clear()
        self.time_index.clear()
        self.event_index.clear()
        self.proposition_index.clear()
        self.span_index.clear()
        self.episodic_index.clear()
        self.semantic_index.clear()
        self.rule_index.clear()
        self.meta_index.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": {key: value.to_dict() for key, value in self.records.items()},
            "time_index": self.time_index,
            "event_index": self.event_index,
            "proposition_index": self.proposition_index,
            "span_index": self.span_index,
            "memory_layers": {
                "episodic": self.episodic_index,
                "semantic": self.semantic_index,
                "rule": self.rule_index,
                "meta": self.meta_index,
            },
        }


@dataclass
class ReasoningRule:
    rule_id: str
    name: str
    description: str
    function_type: str = "content"
    rule_layer: str = "domain"
    stability_level: str = "adaptive"
    domain: str = "general"
    status: str = "active"
    confidence: float = 0.5
    logical_strength: float = 0.5
    empirical_support: float = 0.5
    contradiction_rate: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    scope_tags: list[str] = field(default_factory=list)
    exception_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    required_proposition_types: list[str] = field(default_factory=list)
    required_relation_types: list[str] = field(default_factory=list)
    required_conditions: list[str] = field(default_factory=list)
    required: list[dict[str, Any]] = field(default_factory=list)
    produced_conditions: list[str] = field(default_factory=list)
    produced: list[dict[str, Any]] = field(default_factory=list)
    cue_terms: list[str] = field(default_factory=list)
    required_heads: list[str] = field(default_factory=list)
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    conclusion_type: str = "candidate"
    conclusion_template: str = "{summary}"
    conclusion_distribution: dict[str, float] = field(default_factory=dict)
    complexity: float = 0.5
    priority: float = 1.0
    confidence_prior: float = 0.5
    application_count: float = 0.0
    success_count: float = 0.0
    failure_count: float = 0.0
    support_count: int = 0
    learned_from: list[str] = field(default_factory=list)
    execution_contract_version: int = 1
    require_policy: dict[str, Any] = field(default_factory=dict)
    activation_spec: dict[str, Any] = field(default_factory=dict)
    q_semantics: dict[str, Any] = field(default_factory=dict)
    io_contract: dict[str, Any] = field(default_factory=dict)
    effect_contract: dict[str, Any] = field(default_factory=dict)
    composition_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "function_type": self.function_type,
            "rule_layer": self.rule_layer,
            "stability_level": self.stability_level,
            "domain": self.domain,
            "status": self.status,
            "confidence": self.confidence,
            "logical_strength": self.logical_strength,
            "empirical_support": self.empirical_support,
            "contradiction_rate": self.contradiction_rate,
            "source_ids": self.source_ids,
            "scope_tags": self.scope_tags,
            "exception_patterns": self.exception_patterns,
            "metadata": self.metadata,
            "required_proposition_types": self.required_proposition_types,
            "required_relation_types": self.required_relation_types,
            "required_conditions": self.required_conditions,
            "required": self.required,
            "produced_conditions": self.produced_conditions,
            "produced": self.produced,
            "cue_terms": self.cue_terms,
            "required_heads": self.required_heads,
            "hard_constraints": self.hard_constraints,
            "conclusion_type": self.conclusion_type,
            "conclusion_template": self.conclusion_template,
            "conclusion_distribution": self.conclusion_distribution,
            "complexity": self.complexity,
            "priority": self.priority,
            "confidence_prior": self.confidence_prior,
            "application_count": self.application_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "support_count": self.support_count,
            "learned_from": self.learned_from,
            "execution_contract_version": self.execution_contract_version,
            "require_policy": deepcopy(self.require_policy),
            "activation_spec": deepcopy(self.activation_spec),
            "q_semantics": deepcopy(self.q_semantics),
            "io_contract": deepcopy(self.io_contract),
            "effect_contract": deepcopy(self.effect_contract),
            "composition_contract": deepcopy(self.composition_contract),
        }

@dataclass
class RuleMatch:
    rule_id: str
    proposition_id: str
    score: float
    matched_cues: list[str]
    satisfied_constraints: list[str]
    violated_constraints: list[str]
    derived_statement: str
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    coverage_score: float = 0.0
    support_score: float = 0.0
    step_index: int = 0
    rule_layer: str = ""
    condition_signature: str = ""
    produced_conditions: list[str] = field(default_factory=list)
    lineage_rule_ids: list[str] = field(default_factory=list)
    support_edges: list[str] = field(default_factory=list)
    proposition_vector: Any | None = field(default=None, repr=False)
    rule_vector: Any | None = field(default=None, repr=False)
    derived_vector: Any | None = field(default=None, repr=False)
    embedding_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "proposition_id": self.proposition_id,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "coverage_score": self.coverage_score,
            "support_score": self.support_score,
            "rule_layer": self.rule_layer,
            "matched_cues": self.matched_cues,
            "satisfied_constraints": self.satisfied_constraints,
            "violated_constraints": self.violated_constraints,
            "derived_statement": self.derived_statement,
            "step_index": self.step_index,
            "condition_signature": self.condition_signature,
            "produced_conditions": self.produced_conditions,
            "lineage_rule_ids": self.lineage_rule_ids,
            "support_edges": self.support_edges,
            "proposition_vector": _tensor_to_list(self.proposition_vector),
            "rule_vector": _tensor_to_list(self.rule_vector),
            "derived_vector": _tensor_to_list(self.derived_vector),
            "embedding_model": self.embedding_model,
        }

@dataclass
class ProblemDynamicSlot:
    slot_id: str
    name: str
    value: Any
    source: str
    confidence: float
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class ProblemState:
    problem_id: str
    source_proposition_ids: list[str]
    source_event_ids: list[str]
    goal: str
    current_stage: str
    status: str
    confidence: float
    known_state: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    dynamic_slots: list[ProblemDynamicSlot] = field(default_factory=list)
    selected_rule_ids: list[str] = field(default_factory=list)
    active_chain_ids: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "source_proposition_ids": self.source_proposition_ids,
            "source_event_ids": self.source_event_ids,
            "goal": self.goal,
            "current_stage": self.current_stage,
            "status": self.status,
            "confidence": self.confidence,
            "known_state": self.known_state,
            "open_questions": self.open_questions,
            "constraints": self.constraints,
            "dynamic_slots": [item.to_dict() for item in self.dynamic_slots],
            "selected_rule_ids": self.selected_rule_ids,
            "active_chain_ids": self.active_chain_ids,
            "history": self.history,
            "metadata": self.metadata,
        }


@dataclass
class RuleChainStep:
    step_id: str
    rule_id: str
    rule_layer: str
    required_conditions: list[str]
    satisfied_conditions: list[str]
    missing_conditions: list[str]
    produced_conditions: list[str]
    score: float
    status: str
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "rule_id": self.rule_id,
            "rule_layer": self.rule_layer,
            "required_conditions": self.required_conditions,
            "satisfied_conditions": self.satisfied_conditions,
            "missing_conditions": self.missing_conditions,
            "produced_conditions": self.produced_conditions,
            "score": self.score,
            "status": self.status,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass
class CandidateRuleChain:
    chain_id: str
    problem_id: str
    target_goal: str
    steps: list[RuleChainStep]
    produced_state: dict[str, Any]
    missing_conditions: list[str]
    confidence: float
    status: str
    audit_status: str = "pending"
    audit_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "problem_id": self.problem_id,
            "target_goal": self.target_goal,
            "steps": [item.to_dict() for item in self.steps],
            "produced_state": self.produced_state,
            "missing_conditions": self.missing_conditions,
            "confidence": self.confidence,
            "status": self.status,
            "audit_status": self.audit_status,
            "audit_notes": self.audit_notes,
            "metadata": self.metadata,
        }

__all__ = ['SpanUnit', 'EventRelation', 'EventHypergraph', 'BaseRule', 'StructuralRule', 'ControlRule', 'WorkingMemoryEntry', 'WorkingMemoryState', 'MemoryRecord', 'PartialMemoryRecord', 'ResolvedMemoryRecord', 'ReasoningNetworkNode', 'ReasoningNetworkEdge', 'LongTermMemoryRecord', 'LongTermMemoryStore', 'ReasoningRule', 'RuleMatch', 'ProblemDynamicSlot', 'ProblemState', 'RuleChainStep', 'CandidateRuleChain']
