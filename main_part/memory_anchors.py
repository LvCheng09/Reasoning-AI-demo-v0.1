from __future__ import annotations
from .common.math_utils import cosine as shared_cosine

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .vector_contracts import canonical_embedding_model, coerce_vector, cosine_similarity_strict


SPAN_VECTOR_FIELDS = ("core_vector", "relation_vector", "nature_vector")


def build_structural_memory_anchors(
    graph: Mapping[str, Any],
    *,
    max_span_anchors: int = 8,
    max_relation_anchors: int = 8,
) -> dict[str, Any]:
    spans = [dict(item) for item in graph.get("spans", []) or [] if isinstance(item, Mapping)]
    relations = _graph_relations(graph)
    span_by_id = {
        span_id: span
        for span in spans
        if (span_id := _span_id(span))
    }
    focus_ids = _graph_focus_ids(graph, set(span_by_id))
    evidence_ids = _graph_evidence_ids(graph, set(span_by_id))
    in_degree = {span_id: 0 for span_id in span_by_id}
    out_degree = {span_id: 0 for span_id in span_by_id}
    for relation in relations:
        source = _relation_endpoint(relation, "source")
        target = _relation_endpoint(relation, "target")
        if source in out_degree:
            out_degree[source] += 1
        if target in in_degree:
            in_degree[target] += 1

    ranked_spans: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []
    warnings: list[str] = []
    errors: list[str] = []
    contracts: set[tuple[str, int, str]] = set()
    for index, span in enumerate(spans):
        span_id = _span_id(span)
        vectors = {field: coerce_vector(span.get(field)) for field in SPAN_VECTOR_FIELDS}
        if not span_id or not all(vectors.values()):
            warnings.append(f"spans[{index}].structural_anchor_incomplete")
            continue
        contract = _contract(span)
        if contract is None:
            errors.append(f"spans[{index}].vector_contract_missing")
            continue
        if any(len(vector) != contract[1] for vector in vectors.values()):
            errors.append(f"spans[{index}].vector_dimension_mismatch")
            continue
        contracts.add(contract)
        degree = in_degree.get(span_id, 0) + out_degree.get(span_id, 0)
        rank = (
            int(span_id in focus_ids),
            int(span_id in evidence_ids),
            int(in_degree.get(span_id, 0) > out_degree.get(span_id, 0)),
            degree,
        )
        ranked_spans.append((rank, -index, {"span": span, "vectors": vectors, "contract": contract}))

    if len(contracts) > 1:
        errors.append("event_graph_vector_contract_inconsistent")
    if errors:
        return _anchor_result("vector_contract_failed", [], errors, warnings)
    if not ranked_spans:
        return _anchor_result(
            "structural_anchors_unavailable",
            [],
            ["event_graph_structural_anchors_missing"],
            warnings,
        )

    ranked_spans.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = ranked_spans[: max(1, int(max_span_anchors))]
    selected_ids = {_span_id(item[2]["span"]) for item in selected}
    anchors: list[dict[str, Any]] = []
    for anchor_index, (_, _, item) in enumerate(selected):
        span = item["span"]
        span_id = _span_id(span)
        model, dimension, version = item["contract"]
        anchors.append(
            {
                "anchor_id": f"span:{span_id}",
                "anchor_index": anchor_index,
                "anchor_kind": "span",
                "span_id": span_id,
                "structural_role": _span_structural_role(
                    span_id,
                    focus_ids=focus_ids,
                    evidence_ids=evidence_ids,
                    in_degree=in_degree,
                    out_degree=out_degree,
                ),
                "core_vector": list(item["vectors"]["core_vector"]),
                "relation_vector": list(item["vectors"]["relation_vector"]),
                "nature_vector": list(item["vectors"]["nature_vector"]),
                "embedding_model": model,
                "vector_dimension": dimension,
                "vector_version": version,
            }
        )

    relation_contract = next(iter(contracts))
    ranked_relations: list[tuple[int, int, dict[str, Any]]] = []
    for index, relation in enumerate(relations):
        source = _relation_endpoint(relation, "source")
        target = _relation_endpoint(relation, "target")
        vector = coerce_vector(relation.get("relation_vector"))
        if source not in selected_ids or target not in selected_ids or not vector:
            continue
        contract = _contract(relation) or relation_contract
        if contract != relation_contract or len(vector) != relation_contract[1]:
            errors.append(f"relations[{index}].vector_contract_mismatch")
            continue
        endpoint_rank = (
            int(source in focus_ids)
            + int(target in focus_ids)
            + int(source in evidence_ids)
            + int(target in evidence_ids)
        )
        ranked_relations.append((endpoint_rank, -index, {"relation": relation, "vector": vector}))
    if errors:
        return _anchor_result("vector_contract_failed", [], errors, warnings)
    ranked_relations.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for offset, (_, _, item) in enumerate(
        ranked_relations[: max(0, int(max_relation_anchors))],
        start=len(anchors),
    ):
        relation = item["relation"]
        relation_id = str(
            relation.get("relation_id")
            or relation.get("id")
            or f"relation_{offset}"
        )
        anchors.append(
            {
                "anchor_id": f"relation:{relation_id}",
                "anchor_index": offset,
                "anchor_kind": "relation",
                "relation_id": relation_id,
                "source_span_id": _relation_endpoint(relation, "source"),
                "target_span_id": _relation_endpoint(relation, "target"),
                "relation_vector": list(item["vector"]),
                "embedding_model": relation_contract[0],
                "vector_dimension": relation_contract[1],
                "vector_version": relation_contract[2],
            }
        )
    return _anchor_result("structural_anchors_ready", anchors, [], warnings)


def score_structural_memory_anchors(
    query_anchors: Sequence[Mapping[str, Any]],
    candidate_anchors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    query_spans = [item for item in query_anchors if item.get("anchor_kind") == "span"]
    candidate_spans = [item for item in candidate_anchors if item.get("anchor_kind") == "span"]
    if not query_spans or not candidate_spans:
        return {"ok": False, "score": 0.0, "coverage": 0.0, "matched_anchors": 0}

    span_pairs: list[tuple[float, int, int]] = []
    for query_index, query in enumerate(query_spans):
        for candidate_index, candidate in enumerate(candidate_spans):
            if _contract(query) != _contract(candidate):
                continue
            scores = [
                _cosine(query.get(field), candidate.get(field))
                for field in SPAN_VECTOR_FIELDS
            ]
            if all(score is not None for score in scores):
                span_pairs.append((sum(scores) / len(scores), query_index, candidate_index))
    span_matches = _greedy_matches(span_pairs)
    if not span_matches:
        return {"ok": False, "score": 0.0, "coverage": 0.0, "matched_anchors": 0}
    span_mapping = {
        str(query_spans[q].get("span_id") or ""): str(candidate_spans[c].get("span_id") or "")
        for _, q, c in span_matches
    }
    span_score = sum(item[0] for item in span_matches) / len(span_matches)
    span_coverage = len(span_matches) / len(query_spans)

    query_relations = [item for item in query_anchors if item.get("anchor_kind") == "relation"]
    candidate_relations = [
        item for item in candidate_anchors if item.get("anchor_kind") == "relation"
    ]
    relation_matches: list[tuple[float, int, int]] = []
    for query_index, query in enumerate(query_relations):
        for candidate_index, candidate in enumerate(candidate_relations):
            if _contract(query) != _contract(candidate):
                continue
            vector_score = _cosine(query.get("relation_vector"), candidate.get("relation_vector"))
            if vector_score is None:
                continue
            endpoint_score = 0.0
            if span_mapping.get(str(query.get("source_span_id") or "")) == str(
                candidate.get("source_span_id") or ""
            ):
                endpoint_score += 0.5
            if span_mapping.get(str(query.get("target_span_id") or "")) == str(
                candidate.get("target_span_id") or ""
            ):
                endpoint_score += 0.5
            relation_matches.append(
                (0.7 * vector_score + 0.3 * endpoint_score, query_index, candidate_index)
            )
    relation_selected = _greedy_matches(relation_matches)
    relation_coverage = (
        len(relation_selected) / len(query_relations) if query_relations else 1.0
    )
    relation_score = (
        sum(item[0] for item in relation_selected) / len(relation_selected)
        if relation_selected
        else 0.0 if query_relations else 1.0
    )
    coverage = 0.75 * span_coverage + 0.25 * relation_coverage
    score = (0.75 * span_score + 0.25 * relation_score) * coverage
    return {
        "ok": True,
        "score": float(score),
        "coverage": float(coverage),
        "matched_anchors": len(span_matches) + len(relation_selected),
        "span_matches": len(span_matches),
        "relation_matches": len(relation_selected),
    }


def _anchor_result(
    status: str,
    anchors: list[dict[str, Any]],
    errors: Sequence[str],
    warnings: Sequence[str],
) -> dict[str, Any]:
    contracts = {_contract(item) for item in anchors if _contract(item) is not None}
    contract = next(iter(contracts)) if len(contracts) == 1 else None
    return {
        "status": status,
        "anchors": deepcopy(anchors),
        "anchor_count": len(anchors),
        "embedding_model": contract[0] if contract else "",
        "vector_dimension": contract[1] if contract else 0,
        "vector_version": contract[2] if contract else "",
        "errors": list(dict.fromkeys(str(item) for item in errors if str(item))),
        "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item))),
    }


def _span_structural_role(
    span_id: str,
    *,
    focus_ids: set[str],
    evidence_ids: set[str],
    in_degree: Mapping[str, int],
    out_degree: Mapping[str, int],
) -> str:
    if span_id in focus_ids:
        return "focus"
    if span_id in evidence_ids:
        return "evidence"
    if in_degree.get(span_id, 0) > out_degree.get(span_id, 0):
        return "sink"
    if in_degree.get(span_id, 0) + out_degree.get(span_id, 0) > 1:
        return "bridge"
    return "context"


def _referenced_span_ids(value: Any, valid_ids: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            found.update(_referenced_span_ids(item, valid_ids))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(_referenced_span_ids(item, valid_ids))
    else:
        candidate = str(value or "")
        if candidate in valid_ids:
            found.add(candidate)
    return found


def _graph_relations(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    sources: list[Any] = [graph.get("relations", [])]
    for event in graph.get("events", []) or []:
        if isinstance(event, Mapping):
            sources.append(event.get("relations", []))
    for source in sources:
        for item in source or []:
            if not isinstance(item, Mapping):
                continue
            relation = dict(item)
            key = (
                str(relation.get("relation_id") or relation.get("id") or ""),
                _relation_endpoint(relation, "source"),
                _relation_endpoint(relation, "target"),
            )
            if key in seen:
                continue
            seen.add(key)
            relations.append(relation)
    return relations


def _graph_focus_ids(graph: Mapping[str, Any], valid_ids: set[str]) -> set[str]:
    found = _referenced_span_ids(graph.get("focus"), valid_ids)
    for event in graph.get("events", []) or []:
        if isinstance(event, Mapping):
            found.update(_referenced_span_ids(event.get("focus"), valid_ids))
    return found


def _graph_evidence_ids(graph: Mapping[str, Any], valid_ids: set[str]) -> set[str]:
    found = {
        str(item)
        for item in graph.get("evidence_span_ids", []) or []
        if str(item) in valid_ids
    }
    for event in graph.get("events", []) or []:
        if not isinstance(event, Mapping):
            continue
        found.update(
            str(item)
            for item in event.get("evidence_span_ids", []) or []
            if str(item) in valid_ids
        )
    return found


def _greedy_matches(
    pairs: Sequence[tuple[float, int, int]],
) -> list[tuple[float, int, int]]:
    selected: list[tuple[float, int, int]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for score, left, right in sorted(pairs, reverse=True):
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        selected.append((score, left, right))
    return selected


def _contract(value: Mapping[str, Any]) -> tuple[str, int, str] | None:
    model = canonical_embedding_model(value.get("embedding_model"))
    version = str(value.get("vector_version") or "").strip()
    try:
        dimension = int(value.get("vector_dimension") or 0)
    except (TypeError, ValueError):
        return None
    if not model or not version or dimension <= 0:
        return None
    return model, dimension, version


def _cosine(left: Any, right: Any) -> float | None:
    return shared_cosine(coerce_vector(left), coerce_vector(right), invalid=None)


def _span_id(span: Mapping[str, Any]) -> str:
    return str(span.get("span_id") or span.get("id") or "").strip()


def _relation_endpoint(relation: Mapping[str, Any], side: str) -> str:
    return str(
        relation.get(f"{side}_span_id")
        or relation.get(f"{side}_id")
        or relation.get(side)
        or ""
    ).strip()


__all__ = ["build_structural_memory_anchors", "score_structural_memory_anchors"]
