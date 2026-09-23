"""Unified short-term memory workspace policies.

The memory workspace stores completed event packages. Persistent memory DB
records are not copied into this workspace during retrieval.
"""

from __future__ import annotations
from .common.normalize import safe_float as shared_safe_float

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .data_structure import WorkingMemoryEntry
from .identity import stable_content_hash
from .memory_anchors import build_structural_memory_anchors, score_structural_memory_anchors
from .vector_contracts import cosine_similarity_strict



def select_memory_workspace_context(
    entries: Sequence[WorkingMemoryEntry],
    *,
    query_vector: Any,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    top_k: int = 6,
) -> dict[str, Any]:
    vector = _number_list(query_vector)
    if not vector:
        return {
            "ok": False,
            "status": "input_context_vector_failed",
            "records": [],
            "errors": [{"status": "missing_query_vector"}],
        }
    ranked: list[tuple[float, WorkingMemoryEntry]] = []
    errors: list[dict[str, Any]] = []
    for entry in entries:
        best_score: float | None = None
        for candidate in _event_package_vectors(entry.payload):
            candidate_model = str(candidate.get("embedding_model") or "")
            candidate_version = str(candidate.get("vector_version") or "")
            if query_embedding_model and candidate_model and candidate_model != query_embedding_model:
                continue
            if query_vector_version and candidate_version and candidate_version != query_vector_version:
                continue
            similarity = cosine_similarity_strict(vector, candidate.get("vector"))
            if not similarity.get("ok"):
                errors.append(
                    {
                        "entry_id": entry.entry_id,
                        "status": similarity.get("status"),
                        "left_dimension": similarity.get("left_dimension"),
                        "right_dimension": similarity.get("right_dimension"),
                    }
                )
                continue
            score = float(similarity.get("score") or 0.0)
            best_score = score if best_score is None else max(best_score, score)
        if best_score is not None:
            ranked.append((best_score, entry))
    ranked.sort(key=lambda item: (item[0], item[1].created_at, item[1].entry_id), reverse=True)
    records = [
        {
            "memory_entry_id": entry.entry_id,
            "score": round(score, 6),
            **_compact_memory_context(entry.payload),
        }
        for score, entry in ranked[: max(1, int(top_k))]
    ]
    return {
        "ok": True,
        "status": "memory_workspace_context_ready" if records else "no_vector_compatible_memory",
        "records": records,
        "errors": errors,
    }


def _event_package_vectors(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    workspaces = payload.get("event_workspaces") if isinstance(payload.get("event_workspaces"), Mapping) else {}
    for workspace in workspaces.values():
        if not isinstance(workspace, Mapping):
            continue
        for section in ("spans", "relations"):
            for item in workspace.get(section, []) or []:
                if not isinstance(item, Mapping):
                    continue
                for key in ("vector", "core_vector", "span_core_vector", "nature_vector", "relation_vector"):
                    vector = _number_list(item.get(key))
                    if vector:
                        vectors.append(
                            {
                                "vector": vector,
                                "embedding_model": item.get("embedding_model"),
                                "vector_version": item.get("vector_version"),
                            }
                        )
    return vectors


def _compact_memory_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    workspaces = payload.get("event_workspaces") if isinstance(payload.get("event_workspaces"), Mapping) else {}
    events: list[dict[str, Any]] = []
    for event_id, workspace in workspaces.items():
        if not isinstance(workspace, Mapping):
            continue
        events.append(
            {
                "event_id": str(event_id),
                "event_type": str(workspace.get("event_type") or ""),
                "spans": [
                    {
                        key: span.get(key)
                        for key in ("span_id", "core", "relation", "nature", "text", "confidence")
                        if span.get(key) not in (None, "")
                    }
                    for span in workspace.get("spans", [])[:24]
                    if isinstance(span, Mapping)
                ],
                "relations": [
                    {
                        key: relation.get(key)
                        for key in ("relation_id", "source_span_id", "target_span_id", "relation_type", "relation", "confidence")
                        if relation.get(key) not in (None, "")
                    }
                    for relation in workspace.get("relations", [])[:24]
                    if isinstance(relation, Mapping)
                ],
            }
        )
    return {
        "event_ids": [str(value) for value in payload.get("event_ids", []) if str(value)],
        "events": events,
        "completion": dict(payload.get("completion") or {}),
    }


def _event_package_signatures(payload: Mapping[str, Any]) -> set[str]:
    """Return text-free structural and trajectory signatures.

    Semantic novelty is evaluated separately with vectors. These signatures
    only describe graph topology and structured runtime outcomes.
    """

    signatures: set[str] = set()
    workspaces = payload.get("event_workspaces") if isinstance(payload.get("event_workspaces"), Mapping) else {}
    for event_id, workspace in workspaces.items():
        if not isinstance(workspace, Mapping):
            continue
        if workspace.get("event_type"):
            signatures.add(_signature("event_type", workspace.get("event_type")))
        spans = [span for span in workspace.get("spans", []) or [] if isinstance(span, Mapping)]
        span_positions = {
            str(span.get("span_id") or span.get("id") or ""): index
            for index, span in enumerate(spans)
            if str(span.get("span_id") or span.get("id") or "")
        }
        signatures.add(_signature("graph_shape", len(spans), len(workspace.get("relations", []) or [])))
        for span in workspace.get("spans", []) or []:
            if not isinstance(span, Mapping):
                continue
            metadata = span.get("metadata") if isinstance(span.get("metadata"), Mapping) else {}
            role = next(
                (
                    str(value).strip()
                    for value in (
                        span.get("role"),
                        metadata.get("causal_role"),
                        metadata.get("slot_role"),
                        metadata.get("semantic_role"),
                    )
                    if str(value or "").strip()
                ),
                "",
            )
            if role:
                signatures.add(_signature("span_role", role))
        for relation in workspace.get("relations", []) or []:
            if isinstance(relation, Mapping):
                signatures.add(
                    _signature(
                        "relation_shape",
                        span_positions.get(str(relation.get("source_span_id") or ""), -1),
                        span_positions.get(str(relation.get("target_span_id") or ""), -1),
                    )
                )
        for path in workspace.get("candidate_paths", []) or []:
            if isinstance(path, Mapping):
                signatures.add(_signature("path", path.get("rule_id"), path.get("status")))
        for feedback in workspace.get("feedback", []) or []:
            if isinstance(feedback, Mapping):
                metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), Mapping) else {}
                evidence_identity = (
                    metadata.get("evidence_identity")
                    if isinstance(metadata.get("evidence_identity"), Mapping)
                    else {}
                )
                signatures.add(
                    _signature(
                        "feedback",
                        feedback.get("feedback_signal") or feedback.get("status"),
                        metadata.get("evidence_provenance_id")
                        or evidence_identity.get("evidence_cluster_id")
                        or evidence_identity.get("structural_fingerprint"),
                    )
                )
    for runtime_entry in payload.get("event_runtime_entries", []) or []:
        if isinstance(runtime_entry, Mapping):
            signatures.update(_runtime_entry_signatures(runtime_entry))
    return {value for value in signatures if value}


def _long_term_signatures(records: Sequence[Mapping[str, Any]]) -> set[str]:
    signatures: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        signatures.add(_signature("memory_kind", record.get("memory_kind")))
        graph = record.get("event_hypergraph") if isinstance(record.get("event_hypergraph"), Mapping) else {}
        graph_workspaces = graph.get("event_workspaces") if isinstance(graph.get("event_workspaces"), Mapping) else {}
        if not graph_workspaces and graph:
            graph_workspaces = {str(graph.get("event_id") or record.get("memory_id") or "memory_event"): graph}
        graph_metadata = graph.get("metadata") if isinstance(graph.get("metadata"), Mapping) else {}
        completed_package = (
            graph_metadata.get("completed_event_package")
            if isinstance(graph_metadata.get("completed_event_package"), Mapping)
            else {}
        )
        runtime_entries = (
            graph_metadata.get("runtime_entries")
            if isinstance(graph_metadata.get("runtime_entries"), list)
            else completed_package.get("event_runtime_entries")
            if isinstance(completed_package.get("event_runtime_entries"), list)
            else record.get("event_runtime_entries")
            if isinstance(record.get("event_runtime_entries"), list)
            else []
        )
        signatures.update(
            _event_package_signatures(
                {
                    "event_workspaces": graph_workspaces,
                    "event_runtime_entries": runtime_entries,
                }
            )
        )
    return {value for value in signatures if value}


def _compare_event_package_to_long_term(
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    vector_overlap_threshold: float,
) -> dict[str, Any]:
    current_signatures = _event_package_signatures(payload)
    current_anchors = _event_package_anchor_projection(payload)
    current_vectors = _event_package_vectors(payload)
    closest: dict[str, Any] | None = None
    vector_errors: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        historical_signatures = _long_term_signatures([record])
        matched = current_signatures & historical_signatures
        structural_familiarity = (
            len(matched) / len(current_signatures) if current_signatures else 1.0
        )
        vector_comparison = _compare_package_vectors_to_record(
            payload,
            current_anchors=current_anchors,
            record=record,
            vector_overlap_threshold=vector_overlap_threshold,
        )
        vector_errors.extend(vector_comparison["errors"])
        vector_familiarity = vector_comparison["familiarity"]
        familiarity = (
            0.7 * structural_familiarity + 0.3 * vector_familiarity
            if vector_familiarity is not None
            else structural_familiarity
        )
        candidate = {
            "memory_id": str(record.get("memory_id") or ""),
            "familiarity": familiarity,
            "matched_signatures": matched,
            "vector": vector_comparison,
        }
        if closest is None or candidate["familiarity"] > closest["familiarity"]:
            closest = candidate

    matched_signatures = closest["matched_signatures"] if closest else set()
    novel_signatures = current_signatures - matched_signatures
    structural_novelty = 1.0 - (
        len(matched_signatures) / len(current_signatures)
        if current_signatures
        else 1.0
    )
    vector_comparison = closest["vector"] if closest else {
        "status": "no_long_term_memory",
        "familiarity": None,
        "matches": 0,
        "compared": 0,
    }
    vector_familiarity = vector_comparison.get("familiarity")
    vector_novelty = 1.0 - vector_familiarity if vector_familiarity is not None else None
    novelty_ratio = 1.0 - closest["familiarity"] if closest else 1.0
    return {
        "novelty_ratio": round(novelty_ratio, 6),
        "structural_novelty_ratio": round(structural_novelty, 6),
        "vector_novelty_ratio": round(vector_novelty, 6) if vector_novelty is not None else None,
        "signature_count": len(current_signatures),
        "matched_signature_count": len(matched_signatures),
        "novel_signature_count": len(novel_signatures),
        "novel_feedback_signature_count": sum(
            1 for signature in novel_signatures if signature.startswith("feedback:")
        ),
        "novel_runtime_signature_count": sum(
            1
            for signature in novel_signatures
            if signature.startswith(("runtime:", "runtime_fact:"))
        ),
        "current_vector_count": len(current_vectors),
        "historical_vector_count": len(_long_term_vectors(records)),
        "vector_compared_count": int(vector_comparison.get("compared") or 0),
        "vector_overlap_count": int(vector_comparison.get("matches") or 0),
        "vector_overlap_threshold": vector_overlap_threshold,
        "vector_comparison_status": str(vector_comparison.get("status") or ""),
        "closest_memory_id": str(closest.get("memory_id") or "") if closest else "",
        "vector_errors": vector_errors[:32],
    }


def _event_package_anchor_projection(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    workspaces = payload.get("event_workspaces") if isinstance(payload.get("event_workspaces"), Mapping) else {}
    best: list[dict[str, Any]] = []
    for workspace in workspaces.values():
        if not isinstance(workspace, Mapping):
            continue
        projection = build_structural_memory_anchors(workspace)
        anchors = projection.get("anchors") if projection.get("status") == "structural_anchors_ready" else []
        if len(anchors) > len(best):
            best = [dict(item) for item in anchors if isinstance(item, Mapping)]
    return best


def _compare_package_vectors_to_record(
    payload: Mapping[str, Any],
    *,
    current_anchors: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    vector_overlap_threshold: float,
) -> dict[str, Any]:
    historical_anchors = [
        dict(item)
        for item in record.get("retrieval_anchors", []) or []
        if isinstance(item, Mapping)
    ]
    if not historical_anchors:
        graph = record.get("event_hypergraph") if isinstance(record.get("event_hypergraph"), Mapping) else {}
        projection = build_structural_memory_anchors(graph) if graph else {}
        if projection.get("status") == "structural_anchors_ready":
            historical_anchors = [dict(item) for item in projection.get("anchors", [])]
    if current_anchors and historical_anchors:
        scored = score_structural_memory_anchors(current_anchors, historical_anchors)
        if scored.get("ok"):
            score = min(1.0, max(0.0, float(scored.get("score") or 0.0)))
            return {
                "status": "structural_anchor_exact_comparison",
                "familiarity": score,
                "matches": int(scored.get("matched_anchors") or 0) if score >= vector_overlap_threshold else 0,
                "compared": len(current_anchors),
                "errors": [],
            }

    current_vectors = _event_package_vectors(payload)
    historical_vectors = _long_term_vectors([record])
    matches = 0
    compared = 0
    errors: list[dict[str, Any]] = []
    for current in current_vectors:
        best_score: float | None = None
        for historical in historical_vectors:
            if str(current.get("embedding_model") or "") != str(historical.get("embedding_model") or ""):
                continue
            if str(current.get("vector_version") or "") != str(historical.get("vector_version") or ""):
                continue
            similarity = cosine_similarity_strict(current.get("vector"), historical.get("vector"))
            if not similarity.get("ok"):
                errors.append(
                    {
                        "status": similarity.get("status"),
                        "left_dimension": similarity.get("left_dimension"),
                        "right_dimension": similarity.get("right_dimension"),
                    }
                )
                continue
            score = float(similarity.get("score") or 0.0)
            best_score = score if best_score is None else max(best_score, score)
        if best_score is None:
            continue
        compared += 1
        if best_score >= vector_overlap_threshold:
            matches += 1
    return {
        "status": "legacy_vector_exact_comparison" if compared else "vectors_not_comparable",
        "familiarity": matches / compared if compared else None,
        "matches": matches,
        "compared": compared,
        "errors": errors,
    }


def _long_term_vectors(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    for record in records:
        record_vector = _number_list(record.get("memory_vector") or record.get("vector"))
        if record_vector:
            vectors.append(
                {
                    "vector": record_vector,
                    "embedding_model": record.get("embedding_model"),
                    "vector_version": record.get("vector_version"),
                }
            )
        graph = record.get("event_hypergraph") if isinstance(record.get("event_hypergraph"), Mapping) else {}
        if graph:
            vectors.extend(
                _event_package_vectors(
                    {"event_workspaces": {str(graph.get("event_id") or record.get("memory_id") or "memory_event"): graph}}
                )
            )
    return vectors


def _runtime_entry_signatures(entry: Mapping[str, Any]) -> set[str]:
    payload = entry.get("payload") if isinstance(entry.get("payload"), Mapping) else {}
    payload_type = str(entry.get("payload_type") or "runtime_entry")
    signatures = {
        _signature("runtime", payload_type, entry.get("stage"), entry.get("source"))
    }
    scalar_fact_keys = {
        "control_status",
        "correction_status",
        "failure_type",
        "feedback_signal",
        "next_phase",
        "operation",
        "outcome_status",
        "output_gate_decision",
        "output_status",
        "phase",
        "rule_id",
        "selected_rule_id",
        "status",
    }
    list_fact_keys = {
        "active_event_ids",
        "active_rule_ids",
        "executed_rule_ids",
        "matched_rule_ids",
        "pending_rule_ids",
        "rule_path_rule_ids",
        "rules_applied",
    }

    for key_text, value in _iter_runtime_fields(payload):
        if key_text in scalar_fact_keys and isinstance(value, (str, int, float, bool)):
            signatures.add(_signature("runtime_fact", payload_type, key_text, value))
        elif key_text in list_fact_keys and isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, (str, int, float, bool)) and str(item):
                    signatures.add(_signature("runtime_fact", payload_type, key_text, item))
    return {value for value in signatures if value}


def _feedback_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    feedback_count = 0
    failure_count = 0
    high_value_feedback = False
    workspaces = payload.get("event_workspaces") if isinstance(payload.get("event_workspaces"), Mapping) else {}
    for workspace in workspaces.values():
        if not isinstance(workspace, Mapping):
            continue
        for feedback in workspace.get("feedback", []) or []:
            if not isinstance(feedback, Mapping):
                continue
            feedback_count += 1
            states = {
                str(feedback.get(key) or "").strip().lower()
                for key in (
                    "status",
                    "feedback_signal",
                    "feedback_type",
                    "outcome_status",
                    "correction_status",
                )
                if str(feedback.get(key) or "").strip()
            }
            if states.intersection({"negative", "missing", "contaminated", "correction", "corrected", "expert"}):
                high_value_feedback = True
            if states.intersection({"failed", "error", "blocked", "rollback", "missing", "contaminated"}):
                failure_count += 1
    for entry in payload.get("event_runtime_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        runtime_states = _runtime_state_values(entry)
        if runtime_states.intersection({"failed", "error", "blocked", "rollback", "missing", "contaminated"}):
            failure_count += 1
    return {
        "feedback_count": feedback_count,
        "failure_count": failure_count,
        "high_value_feedback": high_value_feedback,
    }


def _runtime_state_values(entry: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    state_keys = {
        "correction_status",
        "failure_type",
        "feedback_signal",
        "outcome_status",
        "output_status",
        "status",
    }

    for key_text, value in _iter_runtime_fields(entry):
        if key_text in state_keys and isinstance(value, (str, int, float, bool)):
            normalized = str(value).strip().lower()
            if normalized:
                values.add(normalized)
    return values


_RUNTIME_SCAN_SKIPPED_FIELDS = {
    "candidate_paths",
    "candidate_rules",
    "completed_event_package",
    "event_graph",
    "event_hypergraph",
    "event_runtime_entries",
    "event_workspace",
    "event_workspaces",
    "initial_event_graph",
    "memory_entries",
    "produced",
    "relations",
    "required",
    "result_event_graph",
    "runtime_entries",
    "spans",
    "target_event_graph",
}


def _iter_runtime_fields(
    root: Any,
    *,
    max_depth: int = 6,
    max_nodes: int = 2048,
    max_sequence_items: int = 256,
):
    """Yield bounded runtime fields without reopening complete graph payloads.

    Event topology and semantic vectors are compared by dedicated paths above.
    Runtime scanning is limited to control, matcher, executor, and status facts.
    """

    stack: list[tuple[Any, int]] = [(root, 0)]
    visited: set[int] = set()
    scanned_nodes = 0
    while stack and scanned_nodes < max_nodes:
        node, depth = stack.pop()
        if not isinstance(node, (Mapping, list, tuple, set)):
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        scanned_nodes += 1
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_text = str(key)
                yield key_text, value
                if depth >= max_depth or key_text in _RUNTIME_SCAN_SKIPPED_FIELDS:
                    continue
                if "vector" in key_text.lower() and isinstance(value, (list, tuple)):
                    continue
                if isinstance(value, (Mapping, list, tuple, set)):
                    stack.append((value, depth + 1))
        elif depth < max_depth:
            items = list(node) if isinstance(node, set) else node
            for value in list(items)[:max_sequence_items]:
                if isinstance(value, (Mapping, list, tuple, set)):
                    stack.append((value, depth + 1))


def _signature(kind: str, *values: Any) -> str:
    parts = [str(value).strip() for value in values if str(value or "").strip()]
    if not parts:
        return ""
    return f"{kind}:{stable_content_hash(parts, length=32)}"


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None
    return None


def _number_list(value: Any) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            return None
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return None
    return result or None


def _safe_float(value: Any, default: float) -> float:
    return shared_safe_float(value, float(default))


__all__ = [
    "select_memory_workspace_context",
]
