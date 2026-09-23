from __future__ import annotations
from .common.math_utils import cosine as shared_cosine
from .common.normalize import as_list as shared_as_list
from .common.normalize import safe_float as shared_safe_float

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .vector_contracts import cosine_similarity_strict


def extract_event_method_signature(event: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the event-side method signature used by EVL comparison."""

    workspace = event.get("workspace") if isinstance(event.get("workspace"), Mapping) else {}
    metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}
    legacy_context = metadata.get("legacy_context") if isinstance(metadata.get("legacy_context"), Mapping) else {}
    spans = _as_list(event.get("spans") or event.get("evidence_span_ids"))
    span_terms = _top_terms([_stable_text(item) for item in spans], limit=18)
    condition_terms = _top_terms(
        [
            *_texts_from_items(_as_list(metadata.get("known_facts"))),
            *_texts_from_items(_as_list(metadata.get("missing_links"))),
            *_texts_from_items(_as_list(metadata.get("intermediate_states"))),
            *_texts_from_items(_as_list(legacy_context.get("known_conditions"))),
            *_texts_from_items(_as_list(legacy_context.get("needed_conditions"))),
            *_texts_from_items(_as_list(legacy_context.get("intermediate_states"))),
        ],
        limit=18,
    )
    rule_ids = _dedupe(
        [
            *_rule_ids_from_items(_as_list(event.get("candidate_paths"))),
            *_rule_ids_from_items(_as_list(workspace.get("candidate_paths"))),
        ]
    )
    conclusion_terms = _top_terms(
        [
            *_texts_from_items(_as_list(metadata.get("final_goals"))),
            *_texts_from_items(_as_list(legacy_context.get("final_goals"))),
            *_texts_from_items(_as_list(event.get("candidate_paths"))),
            *_texts_from_items(_as_list(workspace.get("candidate_paths"))),
        ],
        limit=12,
    )
    relations = _as_list(event.get("relations") or workspace.get("relations"))
    relation_terms = _top_terms(
        [
            _stable_text(
                {
                    "source": item.get("source_span_id") or item.get("source"),
                    "relation": item.get("relation") or item.get("relation_type"),
                    "target": item.get("target_span_id") or item.get("target"),
                }
            )
            for item in relations
            if isinstance(item, Mapping)
        ],
        limit=18,
    )
    rule_trace = workspace.get("rule_trace") if isinstance(workspace.get("rule_trace"), Mapping) else {}
    path_roles = _dedupe(
        [
            *[str(item.get("learning_role") or "") for item in _as_list(event.get("candidate_paths")) if isinstance(item, Mapping)],
            *[str(item.get("learning_role") or "") for item in _as_list(workspace.get("candidate_paths")) if isinstance(item, Mapping)],
            *[str(item) for item in _as_list(rule_trace.get("executed_rule_ids"))],
        ]
    )
    feedback_signal = _feedback_signal_from_payload(event)
    topology = _graph_topology_signature(event)
    return {
        "span_terms": span_terms,
        "condition_terms": condition_terms,
        "relation_terms": relation_terms,
        "rule_ids": rule_ids,
        "conclusion_terms": conclusion_terms,
        "path_roles": path_roles,
        "feedback_signal": feedback_signal,
        **topology,
        "debug_only_text": True,
        "method_text": " ".join([
            *span_terms,
            *condition_terms,
            *relation_terms,
            *rule_ids,
            *conclusion_terms,
            *path_roles,
        ]),
        "vectors": _extract_vectors(event),
    }


def find_similar_memory_methods(
    event: Mapping[str, Any],
    *,
    extractor: Any = None,
    memory_records: Sequence[Any] | None = None,
    db_paths: Mapping[str, str] | None = None,
    top_k: int = 8,
    min_similarity: float = 0.18,
    signature: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find similar prior event/method records from in-memory store and memory.sqlite."""

    event_signature = dict(signature) if signature is not None else extract_event_method_signature(event)
    candidates: list[dict[str, Any]] = []
    for record in memory_records or []:
        candidates.append(_memory_record_to_candidate(record))
    if extractor is not None:
        store = getattr(extractor, "long_term_memory", None)
        records = getattr(store, "records", None)
        if isinstance(records, Mapping):
            candidates.extend(_memory_record_to_candidate(record) for record in records.values())
        elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            candidates.extend(_memory_record_to_candidate(record) for record in records)
    memory_db_path = str((db_paths or {}).get("memory") or "").strip()
    if memory_db_path:
        candidates.extend(
            _load_memory_db_candidates(
                memory_db_path,
                query_vectors=event_signature.get("vectors") or [],
                limit=max(30, top_k * 8),
            )
        )

    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        signature_vectors = event_signature.get("vectors")
        candidate_vectors = candidate.get("vectors")
        vector_similarity = _max_vector_similarity(
            signature_vectors if isinstance(signature_vectors, list) else [],
            candidate_vectors if isinstance(candidate_vectors, list) else [],
        )
        relation_similarity = _jaccard(
            set(event_signature.get("relation_topology") or []),
            set(candidate.get("relation_topology") or []),
        )
        path_similarity = _jaccard(
            set(event_signature.get("path_topology") or []),
            set(candidate.get("path_topology") or []),
        )
        shape_similarity = _graph_shape_similarity(
            event_signature,
            candidate,
        )
        conclusion_similarity = vector_similarity if vector_similarity is not None else 0.0
        if vector_similarity is not None:
            similarity = (
                0.60 * vector_similarity
                + 0.20 * relation_similarity
                + 0.12 * path_similarity
                + 0.08 * shape_similarity
            )
            similarity_kind = "vector_plus_structure"
        else:
            similarity = (
                0.50 * relation_similarity
                + 0.30 * path_similarity
                + 0.20 * shape_similarity
            )
            similarity_kind = "structure_only"
        shared_rules = sorted(set(event_signature.get("rule_ids") or []) & set(candidate.get("rule_ids") or []))
        if shared_rules:
            similarity = min(1.0, similarity + min(0.20, 0.08 * len(shared_rules)))
            similarity_kind = f"{similarity_kind}_plus_rule_id"
        if similarity < min_similarity:
            continue
        cid = str(
            candidate.get("memory_id")
            or candidate.get("record_id")
            or _structure_identity(candidate)
        )
        if cid in seen:
            continue
        seen.add(cid)
        ranked.append({
            **candidate,
            "similarity": round(similarity, 4),
            "similarity_kind": similarity_kind,
            "vector_similarity": round(vector_similarity, 4) if vector_similarity is not None else None,
            "relation_similarity": round(relation_similarity, 4),
            "path_similarity": round(path_similarity, 4),
            "shape_similarity": round(shape_similarity, 4),
            "conclusion_similarity": round(conclusion_similarity, 4),
            "shared_rule_ids": shared_rules,
        })
    ranked.sort(
        key=lambda item: (
            float(item.get("similarity") or 0.0),
            float(item.get("success_score") or 0.0),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def compare_event_with_memory_methods(
    event: Mapping[str, Any],
    *,
    extractor: Any = None,
    memory_records: Sequence[Any] | None = None,
    db_paths: Mapping[str, str] | None = None,
    candidate_score: float | None = None,
    own_best_score: float | None = None,
    min_similarity: float = 0.18,
) -> dict[str, Any]:
    """Read-only EVL report comparing the current event method with memory.

    EVL has no write authority and does not execute rules. It only builds an
    evaluation packet; auto-learning is the only layer allowed to turn this
    report into learn / skip / update-rule actions.
    """

    method_signature = extract_event_method_signature(event)
    similar = find_similar_memory_methods(
        event,
        extractor=extractor,
        memory_records=memory_records,
        db_paths=db_paths,
        min_similarity=min_similarity,
        signature=method_signature,
    )
    memory_best = max((float(item.get("success_score") or 0.0) for item in similar), default=0.0)
    memory_avg = _mean(float(item.get("success_score") or 0.0) for item in similar)
    memory_evidence = _group_memory_evidence(
        similar,
        current_signal=str(method_signature.get("feedback_signal") or ""),
    )
    candidate = _safe_float(candidate_score, _event_result_score(event))
    own = _safe_float(own_best_score, memory_best)

    if not similar:
        evaluation_status = "no_similar_memory_method"
        learning_signal = "candidate_method_novel"
    elif candidate >= max(own, memory_best) + 0.03:
        evaluation_status = "candidate_method_beats_memory"
        learning_signal = "candidate_method_useful"
    elif candidate + 0.02 >= memory_best and memory_avg < 0.66:
        evaluation_status = "candidate_method_not_worse_and_memory_unstable"
        learning_signal = "candidate_method_useful"
    elif own < 0.45 and candidate >= 0.50:
        evaluation_status = "own_method_failed_candidate_viable"
        learning_signal = "candidate_method_useful"
    else:
        evaluation_status = "similar_memory_method_not_worse"
        learning_signal = "existing_method_sufficient"

    confidence_delta = candidate - max(own, memory_best)
    return {
        "module": "evl",
        "permission": "read_only",
        "evaluation_status": evaluation_status,
        "learning_signal": learning_signal,
        "candidate_score": round(candidate, 4),
        "own_best_score": round(own, 4),
        "memory_best_score": round(memory_best, 4),
        "memory_avg_score": round(memory_avg, 4),
        "confidence_delta": round(confidence_delta, 4),
        "has_similar_method": bool(similar),
        "similar_methods": similar,
        "memory_evidence": memory_evidence,
        "method_signature": method_signature,
    }


def _load_memory_db_candidates(
    memory_db_path: str,
    *,
    query_vectors: Sequence[Any],
    limit: int,
) -> list[dict[str, Any]]:
    path = Path(memory_db_path).expanduser()
    if not path.exists() or not query_vectors:
        return []
    try:
        from .persistence import memory_db
    except ImportError as exc:
        raise RuntimeError("memory structural retrieval backend is unavailable") from exc
    records = memory_db.search_memory_anchor_vectors(
        path,
        query_vectors=query_vectors[:8],
        top_k=limit,
    )
    return [
        {
            **_memory_record_to_candidate(record),
            "source": "memory.sqlite",
        }
        for record in records
    ]


def _memory_record_to_candidate(record: Any) -> dict[str, Any]:
    payload = _public_payload(record)
    state = payload.get("state") if isinstance(payload.get("state"), Mapping) else {}
    rule_uses = _as_list(payload.get("rule_uses") or state.get("rule_uses"))
    if not rule_uses:
        rule_uses = _as_list(payload.get("memory_rule_uses"))
    event_graph = payload.get("event_hypergraph") if isinstance(payload.get("event_hypergraph"), Mapping) else {}
    workspace = event_graph.get("workspace") if isinstance(event_graph.get("workspace"), Mapping) else {}
    metadata = event_graph.get("metadata") if isinstance(event_graph.get("metadata"), Mapping) else {}
    legacy_context = metadata.get("legacy_context") if isinstance(metadata.get("legacy_context"), Mapping) else {}
    execution_trace = payload.get("execution_trace") if isinstance(payload.get("execution_trace"), Mapping) else {}
    rule_trace = (
        execution_trace.get("rule_trace")
        if isinstance(execution_trace.get("rule_trace"), Mapping)
        else workspace.get("rule_trace")
        if isinstance(workspace.get("rule_trace"), Mapping)
        else {}
    )
    relation_terms = _top_terms(
        [_stable_text(item) for item in _as_list(event_graph.get("relations"))],
        limit=18,
    )
    conclusion_terms = _top_terms(
        [
            *_texts_from_items(_as_list(metadata.get("final_goals"))),
            *_texts_from_items(_as_list(legacy_context.get("final_goals"))),
            *_texts_from_items(_as_list(event_graph.get("candidate_paths"))),
            *_texts_from_items(_as_list(workspace.get("candidate_paths"))),
        ],
        limit=12,
    )
    path_roles = _dedupe(
        [
            *[str(item.get("learning_role") or "") for item in _as_list(workspace.get("candidate_paths")) if isinstance(item, Mapping)],
            *[str(item) for item in _as_list(rule_trace.get("executed_rule_ids"))],
        ]
    )
    method_parts = [
        payload.get("text", ""),
        payload.get("outcome", ""),
        _stable_text(state),
        *(_stable_text(item) for item in _as_list(payload.get("memory_stages"))),
        *(_stable_text(item) for item in rule_uses),
    ]
    confidence = _safe_float(payload.get("confidence"), _safe_float(getattr(record, "confidence", 0.0), 0.5))
    rule_scores = [
        _stage_score(
            item.get("rule_confidence"),
            item.get("logical_strength"),
            item.get("empirical_support"),
            item.get("contradiction_rate"),
        )
        for item in rule_uses
        if isinstance(item, Mapping)
    ]
    return {
        "memory_id": str(payload.get("memory_id") or payload.get("record_id") or getattr(record, "record_id", "")),
        "method_text": " ".join(str(part) for part in method_parts if part),
        "vectors": _extract_vectors(payload),
        "confidence": confidence,
        "success_score": max([confidence, *rule_scores] or [0.5]),
        "rule_ids": _rule_ids_from_items(rule_uses),
        "relation_terms": relation_terms,
        "conclusion_terms": conclusion_terms,
        "path_roles": path_roles,
        "feedback_signal": _feedback_signal_from_payload(payload),
        "rule_trace": dict(rule_trace),
        "source": "long_term_memory",
        **_graph_topology_signature(event_graph),
        "debug_only_text": True,
    }


def _graph_topology_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    workspace = value.get("workspace") if isinstance(value.get("workspace"), Mapping) else {}
    spans = [item for item in _as_list(value.get("spans") or workspace.get("spans")) if isinstance(item, Mapping)]
    relations = [
        item
        for item in _as_list(value.get("relations") or workspace.get("relations"))
        if isinstance(item, Mapping)
    ]
    span_positions = {
        str(span.get("span_id") or span.get("id") or index): index
        for index, span in enumerate(spans)
    }
    relation_topology: list[str] = []
    for relation in relations:
        source = str(relation.get("source_span_id") or relation.get("source") or "")
        target = str(relation.get("target_span_id") or relation.get("target") or "")
        source_position = span_positions.get(source, -1)
        target_position = span_positions.get(target, -1)
        has_vector = int(bool(_extract_vectors(relation)))
        relation_topology.append(f"{source_position}>{target_position}:v{has_vector}")

    rule_trace = workspace.get("rule_trace") if isinstance(workspace.get("rule_trace"), Mapping) else {}
    rule_sequences: list[list[str]] = []
    executed = [str(item) for item in _as_list(rule_trace.get("executed_rule_ids")) if str(item)]
    if executed:
        rule_sequences.append(executed)
    for path in [
        *_as_list(value.get("candidate_paths")),
        *_as_list(workspace.get("candidate_paths")),
    ]:
        if not isinstance(path, Mapping):
            continue
        rule_ids = _rule_ids_from_items([path])
        if rule_ids:
            rule_sequences.append(rule_ids)
    path_topology = [
        f"{source}>{target}"
        for sequence in rule_sequences
        for source, target in zip(sequence, sequence[1:])
    ]
    path_topology.extend(f"length:{len(sequence)}" for sequence in rule_sequences)
    return {
        "span_count": len(spans),
        "relation_count": len(relations),
        "relation_topology": sorted(set(relation_topology)),
        "path_topology": sorted(set(path_topology)),
    }


def _graph_shape_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    span_score = _count_similarity(left.get("span_count"), right.get("span_count"))
    relation_score = _count_similarity(left.get("relation_count"), right.get("relation_count"))
    return 0.5 * span_score + 0.5 * relation_score


def _count_similarity(left: Any, right: Any) -> float:
    left_count = max(0, int(_safe_float(left, 0.0)))
    right_count = max(0, int(_safe_float(right, 0.0)))
    if left_count == right_count == 0:
        return 0.0
    return min(left_count, right_count) / max(left_count, right_count)


def _structure_identity(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "span_count": value.get("span_count"),
            "relation_count": value.get("relation_count"),
            "relation_topology": value.get("relation_topology") or [],
            "path_topology": value.get("path_topology") or [],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "structure_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _group_memory_evidence(
    similar: Sequence[dict[str, Any]],
    *,
    current_signal: str,
) -> dict[str, Any]:
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    side: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    for item in similar:
        signal = str(item.get("feedback_signal") or "").strip().lower()
        roles = {str(role) for role in item.get("path_roles") or []}
        if signal == "positive":
            positive.append(item)
        elif signal in {"negative", "missing", "contaminated"}:
            negative.append(item)
        if any(
            token in role
            for role in roles
            for token in ("selection", "opportunity", "avoidance", "unreached", "side")
        ):
            side.append(item)
        if (
            max(
                float(item.get("similarity") or 0.0),
                float(item.get("relation_similarity") or 0.0),
                float(item.get("path_similarity") or 0.0),
            ) >= 0.30
            and (
                float(item.get("conclusion_similarity") or 0.0) < 0.20
                or (
                    current_signal in {"positive", "negative"}
                    and signal in {"positive", "negative"}
                    and current_signal != signal
                )
            )
        ):
            boundary.append(item)
    support_count = len(positive)
    contradiction_count = len(negative)
    outcome_consistency = (
        max(support_count, contradiction_count) / (support_count + contradiction_count)
        if support_count + contradiction_count
        else 0.0
    )
    return {
        "current_feedback_signal": current_signal,
        "positive_memory_ids": [str(item.get("memory_id") or "") for item in positive],
        "negative_memory_ids": [str(item.get("memory_id") or "") for item in negative],
        "side_memory_ids": [str(item.get("memory_id") or "") for item in side],
        "boundary_memory_ids": [str(item.get("memory_id") or "") for item in boundary],
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "side_evidence_count": len(side),
        "boundary_evidence_count": len(boundary),
        "outcome_consistency": round(outcome_consistency, 4),
        "mean_structural_similarity": round(
            _mean(
                max(
                    float(item.get("relation_similarity") or 0.0),
                    float(item.get("path_similarity") or 0.0),
                )
                for item in similar
            ),
            4,
        ),
    }


def _feedback_signal_from_payload(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    workspace = payload.get("workspace") if isinstance(payload.get("workspace"), Mapping) else {}
    execution_trace = payload.get("execution_trace") if isinstance(payload.get("execution_trace"), Mapping) else {}
    for value in (
        metadata.get("feedback_signal"),
        workspace.get("feedback_signal"),
        execution_trace.get("feedback_signal"),
    ):
        signal = str(value or "").strip().lower()
        if signal:
            return signal
    for feedback in [
        *_as_list(workspace.get("feedback")),
        *_as_list(execution_trace.get("feedback")),
    ]:
        if isinstance(feedback, Mapping):
            signal = str(feedback.get("feedback_signal") or feedback.get("signal") or "").strip().lower()
            if signal:
                return signal
    return ""


def _stage_score(confidence: Any, logical_strength: Any, empirical_support: Any, contradiction_rate: Any) -> float:
    score = (
        _safe_float(confidence, 0.5) * 0.35
        + _safe_float(logical_strength, 0.5) * 0.30
        + _safe_float(empirical_support, 0.5) * 0.25
        + (1.0 - _safe_float(contradiction_rate, 0.0)) * 0.10
    )
    return round(max(0.0, min(1.0, score)), 4)


def _event_result_score(event: Mapping[str, Any]) -> float:
    workspace = event.get("workspace") if isinstance(event.get("workspace"), Mapping) else {}
    scores = [_safe_float(event.get("confidence"), 0.0)]
    scores.extend(_safe_float(item.get("confidence"), 0.0) for item in _walk_semantic_mappings(workspace))
    scores.extend(_safe_float(item.get("score"), 0.0) for item in _walk_semantic_mappings(workspace))
    return max(scores or [0.0])


def _extract_vectors(value: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in _walk_semantic_mappings(value):
        for key in (
            "vector",
            "embedding",
            "span_vector",
            "triple_vector",
            "required_vector",
            "produced_vector",
            "condition_vector",
            "memory_vector",
            "core_vector",
            "relation_vector",
            "nature_vector",
            "rule_vector",
            "conclusion_vector",
            "vector_query_vector",
        ):
            vector = _as_float_vector(item.get(key))
            if vector:
                vectors.append(vector)
        vector_parts = item.get("vectors")
        if isinstance(vector_parts, list):
            direct = _as_float_vector(vector_parts)
            if direct:
                vectors.append(direct)
            else:
                for part in vector_parts:
                    vector = _as_float_vector(part)
                    if vector:
                        vectors.append(vector)
        if len(vectors) >= 24:
            break
    return vectors[:24]


_SEMANTIC_ROOT_KEYS = {
    "spans",
    "evidence_spans",
    "relations",
    "candidate_paths",
    "workspace",
    "event_hypergraph",
    "event_workspaces",
    "feedback",
    "supervision",
    "rule_trace",
    "execution_trace",
    "execution_results",
    "output",
    "completion",
}
_NON_SEMANTIC_RUNTIME_KEYS = {
    "event_runtime_entries",
    "runtime_entries",
    "input_entries",
    "scheduler_records",
    "scheduler_history",
    "state_history",
    "completed_event_package",
}


def _walk_semantic_mappings(value: Any, *, max_mappings: int = 768) -> Iterable[dict[str, Any]]:
    """Walk event semantics while excluding scheduler and transfer bookkeeping."""

    if not isinstance(value, Mapping):
        return
    root = dict(value)
    selected_roots = [root]
    selected_roots.extend(root[key] for key in _SEMANTIC_ROOT_KEYS if key in root)
    stack: list[Any] = list(reversed(selected_roots))
    seen_mappings: set[int] = set()
    visited = 0
    while stack and visited < max_mappings:
        current = stack.pop()
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen_mappings:
                continue
            seen_mappings.add(marker)
            item = current if isinstance(current, dict) else dict(current)
            visited += 1
            yield item
            for key, nested in reversed(list(item.items())):
                if str(key) in _NON_SEMANTIC_RUNTIME_KEYS:
                    continue
                if isinstance(nested, Mapping):
                    stack.append(nested)
                elif isinstance(nested, list) and not _as_float_vector(nested):
                    stack.extend(reversed(nested[:128]))
        elif isinstance(current, list) and not _as_float_vector(current):
            stack.extend(reversed(current[:128]))


def _as_float_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        return None
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    while len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    out: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            return None
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out if len(out) >= 3 else None


def _max_vector_similarity(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float | None:
    scores = [
        score
        for lvec in left
        for rvec in right
        for score in [_cosine(lvec, rvec)]
        if score is not None
    ]
    if not scores:
        return None
    return max(scores)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    return shared_cosine(left, right, invalid=None, positive_only=True)


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _walk_mappings(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mappings(item)


def _public_payload(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
            return dict(value) if isinstance(value, Mapping) else {}
        except Exception:
            return {}
    payload = getattr(record, "payload", None)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _texts_from_items(items: Sequence[Any]) -> list[str]:
    return [_stable_text(item) for item in items]


def _rule_ids_from_items(items: Sequence[Any]) -> list[str]:
    rule_ids: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            if item.get("rule_id"):
                rule_ids.append(str(item["rule_id"]))
            if item.get("strategy_rule_id"):
                rule_ids.append(str(item["strategy_rule_id"]))
            rule_ids.extend(str(rid) for rid in _as_list(item.get("rule_ids")) if rid)
            rule_ids.extend(_rule_ids_from_items(_as_list(item.get("path_rule_trace"))))
    return _dedupe(rule_ids)


def _top_terms(texts: Sequence[str], *, limit: int) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_token_set(text))
    return [term for term, _count in counter.most_common(limit)]


def _token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in str(text or "").replace("_", " ").split()
        if len(token.strip()) >= 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _as_list(value: Any) -> list[Any]:
    return shared_as_list(value, expand_sets=False)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    return shared_safe_float(value, default)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
