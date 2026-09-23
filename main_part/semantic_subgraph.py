"""Local semantic subgraphs: vectors are semantic data, text is debug evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .identity import stable_content_id
from .vector_contracts import canonical_embedding_model, coerce_vector, rescale_cosine_strict


TRIPLE_TEXT_FIELDS = ("core", "relation", "nature")
TRIPLE_VECTOR_FIELDS = ("core_vector", "relation_vector", "nature_vector")


def _vector(value: Any) -> list[float] | None:
    result = coerce_vector(value)
    return result if result and any(abs(item) > 1e-12 for item in result) else None


def vector_node_id(vector: Any, *, embedding_model: str, vector_version: str) -> str:
    values = _vector(vector)
    if not values:
        raise ValueError("semantic node requires a nonzero vector")
    model, version = canonical_embedding_model(embedding_model), str(vector_version or "").strip()
    if not model or not version:
        raise ValueError("semantic node requires embedding_model and vector_version")
    return stable_content_id("knode", {"embedding_model": model, "vector_version": version, "vector": values}, length=32)


def normalize_semantic_relations(span: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize explicit frontend triples without inventing a legacy triple.

    A frontend may initially provide labels for vectorization.  Such labels stay
    in ``debug`` until the launcher attaches vectors.  Runtime consumers fail
    closed if vectors are absent.
    """
    raw = span.get("semantic_relations")
    if raw is None:
        raw = (span.get("metadata") or {}).get("semantic_relations", [])
    if not isinstance(raw, list):
        raise ValueError("semantic_relations must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"semantic_relations[{index}] must be an object")
        debug = item.get("debug") if isinstance(item.get("debug"), Mapping) else {}
        relation = {"debug": {field: str(debug.get(field) or item.get(field) or "").strip() for field in TRIPLE_TEXT_FIELDS}}
        if not all(relation["debug"].values()):
            raise ValueError(f"semantic_relations[{index}] requires core/relation/nature debug labels")
        for field in TRIPLE_VECTOR_FIELDS:
            if field in item and item[field] is not None:
                values = _vector(item[field])
                if not values:
                    raise ValueError(f"semantic_relations[{index}].{field} is invalid")
                relation[field] = values
        model = canonical_embedding_model(item.get("embedding_model") or span.get("embedding_model"))
        version = str(item.get("vector_version") or span.get("vector_version") or "").strip()
        if any(field in relation for field in TRIPLE_VECTOR_FIELDS):
            if not all(field in relation for field in TRIPLE_VECTOR_FIELDS):
                raise ValueError(f"semantic_relations[{index}] requires all three vectors once vectorized")
            dimensions = {len(relation[field]) for field in TRIPLE_VECTOR_FIELDS}
            if len(dimensions) != 1 or not model or not version:
                raise ValueError(f"semantic_relations[{index}] has an invalid vector contract")
            relation.update({"embedding_model": model, "vector_version": version, "vector_dimension": dimensions.pop()})
            relation.update({
                "core_node_id": vector_node_id(relation["core_vector"], embedding_model=model, vector_version=version),
                "relation_node_id": vector_node_id(relation["relation_vector"], embedding_model=model, vector_version=version),
                "nature_node_id": vector_node_id(relation["nature_vector"], embedding_model=model, vector_version=version),
            })
            key = (relation["core_node_id"], relation["relation_node_id"], relation["nature_node_id"])
        else:
            key = tuple(relation["debug"].values())
        if key in seen:
            continue
        seen.add(key)
        source_span = item.get("source_span")
        if source_span is not None:
            if not isinstance(source_span, str) or not source_span.strip():
                raise ValueError(f"semantic_relations[{index}].source_span must be a nonempty string")
            relation["source_span"] = source_span.strip()
        normalized.append(relation)
    return normalized


def has_required_subgraph(span: Mapping[str, Any], required_components: Sequence[Mapping[str, Any]], *, minimum_score: float = 0.995) -> tuple[bool, list[dict[str, Any]]]:
    """Prove every component with same-contract vector comparison; never text."""
    available = normalize_semantic_relations(span)
    evidence: list[dict[str, Any]] = []
    for component in required_components:
        model = canonical_embedding_model(component.get("embedding_model"))
        version = str(component.get("vector_version") or "").strip()
        vectors = {field: _vector(component.get(field)) for field in TRIPLE_VECTOR_FIELDS}
        if not model or not version or not all(vectors.values()):
            return False, []
        matched = None
        for candidate in available:
            if candidate.get("embedding_model") != model or candidate.get("vector_version") != version:
                continue
            scores = {field: rescale_cosine_strict(candidate[field], vectors[field]) for field in TRIPLE_VECTOR_FIELDS}
            if all(result["ok"] and float(result["score"]) >= minimum_score for result in scores.values()):
                matched = {"knowledge_edge_id": str(component.get("edge_id") or component.get("knowledge_id") or ""), "candidate_node_ids": {field: candidate.get(field.replace("_vector", "_node_id"), "") for field in TRIPLE_VECTOR_FIELDS}, "scores": {field: round(float(result["score"]), 6) for field, result in scores.items()}}
                break
        if matched is None:
            return False, []
        evidence.append(matched)
    return bool(required_components), evidence


__all__ = ["TRIPLE_TEXT_FIELDS", "TRIPLE_VECTOR_FIELDS", "has_required_subgraph", "normalize_semantic_relations", "vector_node_id"]
