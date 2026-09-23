from __future__ import annotations

from typing import Any, Mapping, Sequence

from .identity import stable_content_hash
from .memory_anchors import build_structural_memory_anchors
from .persistence import knowledge_db, memory_db
from .vector_contracts import canonical_embedding_model, coerce_vector


def build_event_query(
    event_workspace: Mapping[str, Any] | None,
    *,
    extra_text: str = "",
    keys: Sequence[str] | None = None,
    query_vector: Any = None,
    query_embedding_model: str = "",
    query_vector_version: str = "",
) -> dict[str, Any]:
    workspace = dict(event_workspace or {})
    text_chunks: list[str] = [str(extra_text or "")]
    collected_keys: list[str] = [str(item) for item in (keys or []) if str(item).strip()]
    vector = query_vector
    embedding_model = str(query_embedding_model or "")
    vector_version = str(query_vector_version or "")

    for span in _iter_dicts(workspace.get("spans")):
        for key in ("text", "core", "nature", "relation", "debug_text"):
            value = str(span.get(key) or "").strip()
            if value:
                text_chunks.append(value)
                collected_keys.append(value)
        if vector is None:
            vector = span.get("span_core_vec") or span.get("core_vector") or span.get("vector")
        if not embedding_model:
            embedding_model = str(span.get("embedding_model") or "").strip()
        if not vector_version:
            vector_version = str(span.get("vector_version") or "").strip()

    for relation in _iter_dicts(workspace.get("relations")):
        value = str(relation.get("relation") or relation.get("relation_type") or relation.get("debug_relation") or "").strip()
        if value:
            text_chunks.append(value)
            collected_keys.append(value)
        if vector is None:
            vector = relation.get("vector") or relation.get("relation_vector")
        if not embedding_model:
            embedding_model = str(relation.get("embedding_model") or "").strip()
        if not vector_version:
            vector_version = str(relation.get("vector_version") or "").strip()

    for item in [*_iter_dicts(workspace.get("candidate_paths")), *_iter_dicts(workspace.get("feedback"))]:
        value = str(item.get("debug_text") or item.get("reason") or item.get("feedback_signal") or item.get("status") or "").strip()
        if value:
            text_chunks.append(value)
            collected_keys.append(value)

    return {
        "query": " ".join(chunk for chunk in text_chunks if chunk).strip(),
        "keys": _unique(collected_keys)[:64],
        "query_vector": vector,
        "query_embedding_model": embedding_model,
        "query_vector_version": vector_version,
    }


def retrieve_knowledge(
    db_paths: Mapping[str, str] | None,
    *,
    event_workspace: Mapping[str, Any] | None = None,
    query: str = "",
    keys: Sequence[str] | None = None,
    query_vector: Any = None,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    top_k: int = 16,
) -> list[dict[str, Any]]:
    db_path = _db_path(db_paths, "knowledge")
    if not db_path:
        return []
    payload = build_event_query(
        event_workspace,
        extra_text=query,
        keys=keys,
        query_vector=query_vector,
        query_embedding_model=query_embedding_model,
        query_vector_version=query_vector_version,
    )
    if not payload["query_vector"]:
        raise ValueError("knowledge ANN retrieval requires an event query vector")
    knowledge_db.ensure_knowledge_ann_index(db_path)
    records = knowledge_db.search_knowledge_ann(
        db_path,
        query={
            "vector": payload["query_vector"],
            "embedding_model": payload["query_embedding_model"],
            "vector_version": payload["query_vector_version"],
        },
        top_k=top_k,
    )
    return [{**record, "retrieval_source": "knowledge_db"} for record in records]


def knowledge_record_to_event_context(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a DB hit into vector-only runtime context.

    Knowledge text fields are persistence/debug material. Runtime matching uses
    only the stored vectors and their embedding contract.
    """

    knowledge_id = str(record.get("knowledge_id") or record.get("id") or "").strip()
    if not knowledge_id:
        raise ValueError("knowledge context record is missing knowledge_id")

    vectors = {
        field: vector
        for field in ("core_vector", "relation_vector", "nature_vector")
        if (vector := coerce_vector(record.get(field)))
    }
    if not vectors:
        raise ValueError(f"knowledge {knowledge_id} has no usable vector")

    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(
            f"knowledge {knowledge_id} has inconsistent vector dimensions: {sorted(dimensions)}"
        )
    vector_dimension = dimensions.pop()
    declared_dimension = int(record.get("vector_dimension") or 0)
    if declared_dimension and declared_dimension != vector_dimension:
        raise ValueError(
            f"knowledge {knowledge_id} vector_dimension={declared_dimension} "
            f"does not match stored vectors={vector_dimension}"
        )

    embedding_model = canonical_embedding_model(record.get("embedding_model"))
    vector_version = str(record.get("vector_version") or "").strip()
    if not embedding_model or not vector_version:
        raise ValueError(f"knowledge {knowledge_id} has an incomplete vector contract")

    edge_id = str(record.get("edge_id") or knowledge_id).strip()
    core_node_id = str(record.get("core_node_id") or "").strip()
    relation_node_id = str(record.get("relation_node_id") or "").strip()
    nature_node_id = str(record.get("nature_node_id") or "").strip()
    if not all((core_node_id, relation_node_id, nature_node_id)):
        raise ValueError(f"knowledge {knowledge_id} is missing vector-node identities")
    statement_span = {
        "span_id": knowledge_id,
        "span_role": "knowledge_evidence",
        "knowledge_id": knowledge_id,
        "source": "knowledge_db",
        "embedding_model": embedding_model,
        "vector_dimension": vector_dimension,
        "vector_version": vector_version,
        **vectors,
    }
    for key in ("knowledge_type", "confidence", "priority", "score"):
        if record.get(key) not in (None, ""):
            statement_span[key] = record.get(key)

    spans = [statement_span]
    core_vector = vectors.get("core_vector")
    nature_vector = vectors.get("nature_vector")
    if core_vector:
        spans.append(
            {
                "span_id": core_node_id,
                "span_role": "knowledge_node",
                "knowledge_id": knowledge_id,
                "knowledge_node_role": "core",
                "core_vector": core_vector,
                "embedding_model": embedding_model,
                "vector_dimension": vector_dimension,
                "vector_version": vector_version,
                "source": "knowledge_db",
            }
        )
    if nature_vector:
        spans.append(
            {
                "span_id": nature_node_id,
                "span_role": "knowledge_node",
                "knowledge_id": knowledge_id,
                "knowledge_node_role": "nature",
                "core_vector": nature_vector,
                "nature_vector": nature_vector,
                "embedding_model": embedding_model,
                "vector_dimension": vector_dimension,
                "vector_version": vector_version,
                "source": "knowledge_db",
            }
        )
    relations: list[dict[str, Any]] = []
    relation_vector = vectors.get("relation_vector")
    if core_vector and nature_vector and relation_vector:
        relations.append(
            {
                "relation_id": edge_id,
                "source_span_id": core_node_id,
                "target_span_id": nature_node_id,
                "relation_vector": relation_vector,
                "knowledge_id": knowledge_id,
                "source": "knowledge_db",
                "embedding_model": embedding_model,
                "vector_dimension": vector_dimension,
                "vector_version": vector_version,
            }
        )

    return {
        "id": knowledge_id,
        "knowledge_id": knowledge_id,
        "retrieval_source": "knowledge_db",
        "embedding_model": embedding_model,
        "vector_dimension": vector_dimension,
        "vector_version": vector_version,
        "edge_id": edge_id,
        "core_node_id": core_node_id,
        "relation_node_id": relation_node_id,
        "nature_node_id": nature_node_id,
        "knowledge_edge": {
            "edge_id": edge_id,
            "core_node_id": core_node_id,
            "relation_node_id": relation_node_id,
            "nature_node_id": nature_node_id,
            "core_vector": core_vector,
            "relation_vector": relation_vector,
            "nature_vector": nature_vector,
            "embedding_model": embedding_model,
            "vector_version": vector_version,
            "vector_dimension": vector_dimension,
            "debug": dict(record.get("debug") or {}),
        },
        "span_schema": dict((record.get("metadata") or {}).get("span_schema") or {}),
        "spans": spans,
        "relations": relations,
        **{
            key: record.get(key)
            for key in ("knowledge_type", "confidence", "priority", "score")
            if record.get(key) not in (None, "")
        },
    }


def retrieve_knowledge_for_missing_requirements(
    db_paths: Mapping[str, str] | None,
    *,
    matcher_result: Mapping[str, Any],
    event_workspace: Mapping[str, Any],
    top_k_per_query: int = 3,
    max_records: int = 8,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Retrieve the smallest useful knowledge set for failed rule requirements."""

    db_path = _db_path(db_paths, "knowledge")
    if not db_path:
        return {"status": "knowledge_db_not_configured", "records": [], "queries": []}
    summary = knowledge_db.summarize_database(db_path)
    indexed_vectors = int((summary.get("ann") or {}).get("indexed_vectors") or 0)
    if indexed_vectors <= 0:
        return {
            "status": "knowledge_index_unavailable",
            "records": [],
            "queries": [],
            "knowledge_records": int(summary.get("knowledge_records") or 0),
            "indexed_vectors": indexed_vectors,
        }

    queries = _missing_requirement_queries(matcher_result)
    if not queries:
        return {"status": "no_missing_requirement_vectors", "records": [], "queries": []}
    contract = build_event_query(event_workspace)
    existing_ids = {
        str(item.get("source_record_id") or (item.get("payload") or {}).get("knowledge_id") or "")
        for item in event_workspace.get("context_items", []) or []
        if isinstance(item, Mapping)
    }
    selected: dict[str, dict[str, Any]] = {}
    query_audit: list[dict[str, Any]] = []
    for query in queries:
        query_vector = query.get("query_vector")
        if not query_vector:
            continue
        records = knowledge_db.search_knowledge_ann(
            db_path,
            query={
                "vector": query_vector,
                "embedding_model": query.get("embedding_model") or contract.get("query_embedding_model") or "",
                "vector_version": query.get("vector_version") or contract.get("query_vector_version") or "",
            },
            top_k=max(1, int(top_k_per_query)),
            update_usage=False,
        )
        accepted_ids: list[str] = []
        for record in records:
            knowledge_id = str(record.get("knowledge_id") or record.get("id") or "")
            if not knowledge_id or knowledge_id in existing_ids:
                continue
            if float(record.get("score") or 0.0) < max(0.0, float(min_score)):
                continue
            previous = selected.get(knowledge_id)
            if previous is None or float(record.get("score") or 0.0) > float(previous.get("score") or 0.0):
                selected[knowledge_id] = record
            accepted_ids.append(knowledge_id)
        query_audit.append(
            {
                **{key: value for key, value in query.items() if key != "query_vector"},
                "hit_ids": accepted_ids,
            }
        )
        if len(selected) >= max(1, int(max_records)):
            break
    ranked = sorted(
        selected.values(),
        key=lambda item: (float(item.get("score") or 0.0), str(item.get("knowledge_id") or item.get("id") or "")),
        reverse=True,
    )[: max(1, int(max_records))]
    return {
        "status": "knowledge_candidates_ready" if ranked else "no_new_knowledge_candidates",
        "records": ranked,
        "queries": query_audit,
        "indexed_vectors": indexed_vectors,
    }


def build_missing_knowledge_queries(matcher_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose matcher-derived expansion requests for control-rule decisions."""

    return _missing_requirement_queries(matcher_result)


def _missing_requirement_queries(matcher_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for produced in _iter_dicts(matcher_result.get("produced")):
        for rejected in _iter_dicts(produced.get("rejected_rules")):
            rule = rejected.get("rule") if isinstance(rejected.get("rule"), Mapping) else {}
            outcome = rejected.get("match") if isinstance(rejected.get("match"), Mapping) else {}
            metadata = outcome.get("metadata") if isinstance(outcome.get("metadata"), Mapping) else {}
            for required_outcome in _iter_dicts(metadata.get("per_required")):
                required = required_outcome.get("required") if isinstance(required_outcome.get("required"), Mapping) else {}
                details = required_outcome.get("details") if isinstance(required_outcome.get("details"), Mapping) else {}
                condition_by_id = {
                    str(slot.get("span_id") or slot.get("slot_id") or ""): slot
                    for slot in _iter_dicts(required.get("condition"))
                }
                for slot_match in _iter_dicts(details.get("slot_matches")):
                    if slot_match.get("matched"):
                        continue
                    slot_id = str(slot_match.get("slot_id") or "")
                    slot = condition_by_id.get(slot_id) or {}
                    for field in ("core_vector", "relation_vector", "nature_vector"):
                        vector = coerce_vector(slot.get(field))
                        key = (str(rule.get("rule_id") or ""), slot_id, field)
                        if vector and key not in seen:
                            seen.add(key)
                            queries.append(
                                {
                                    "kind": "missing_required_span",
                                    "rule_id": str(rule.get("rule_id") or ""),
                                    "required_id": str(required.get("required_id") or ""),
                                    "slot_id": slot_id,
                                    "vector_field": field,
                                    "query_vector": vector,
                                    "embedding_model": str(slot.get("embedding_model") or rule.get("embedding_model") or ""),
                                    "vector_version": str(slot.get("vector_version") or rule.get("vector_version") or ""),
                                }
                            )
                relation_by_id = {
                    str(relation.get("relation_id") or relation.get("id") or ""): relation
                    for relation in _iter_dicts(required.get("relation_patterns"))
                }
                for relation_match in _iter_dicts(details.get("relation_matches")):
                    if relation_match.get("matched"):
                        continue
                    relation_id = str(
                        relation_match.get("required_relation_id")
                        or relation_match.get("relation_id")
                        or ""
                    )
                    relation = relation_by_id.get(relation_id) or {}
                    vector = coerce_vector(relation.get("relation_vector"))
                    key = (str(rule.get("rule_id") or ""), relation_id, "relation_vector")
                    if vector and key not in seen:
                        seen.add(key)
                        queries.append(
                            {
                                "kind": "missing_required_relation",
                                "rule_id": str(rule.get("rule_id") or ""),
                                "required_id": str(required.get("required_id") or ""),
                                "relation_id": relation_id,
                                "vector_field": "relation_vector",
                                "query_vector": vector,
                                "embedding_model": str(relation.get("embedding_model") or rule.get("embedding_model") or ""),
                                "vector_version": str(relation.get("vector_version") or rule.get("vector_version") or ""),
                            }
                        )
    return queries


def retrieve_memory(
    db_paths: Mapping[str, str] | None,
    *,
    event_workspace: Mapping[str, Any] | None = None,
    query: str = "",
    keys: Sequence[str] | None = None,
    query_vector: Any = None,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    top_k: int = 16,
    memory_cache: Any = None,
) -> list[dict[str, Any]]:
    db_path = _db_path(db_paths, "memory")
    if not db_path:
        return []
    anchor_projection = build_structural_memory_anchors(dict(event_workspace or {}))
    payload = build_event_query(
        event_workspace,
        extra_text=query,
        keys=keys,
        query_vector=query_vector,
        query_embedding_model=query_embedding_model,
        query_vector_version=query_vector_version,
    )
    structural_query = anchor_projection["status"] == "structural_anchors_ready"
    if not structural_query and not payload["query_vector"]:
        raise ValueError(
            "memory retrieval requires complete structural anchors or an explicit legacy query vector"
        )
    cache_key = stable_content_hash(
        {
            "retrieval_mode": "structural_anchor_exact_scan" if structural_query else "legacy_ann",
            "anchors": anchor_projection["anchors"] if structural_query else [],
            "vector": None if structural_query else payload["query_vector"],
            "embedding_model": payload["query_embedding_model"],
            "vector_version": payload["query_vector_version"],
            "top_k": int(top_k),
        },
        length=48,
    )
    cache_read_error = ""
    if memory_cache is not None:
        try:
            cached_records = memory_cache.get_records(cache_key)
        except Exception as exc:
            cached_records = None
            cache_read_error = repr(exc)
        if _valid_cached_memory_records(
            cached_records,
            query_vector_version=payload["query_vector_version"],
        ):
            return [
                {**record, "retrieval_source": "memory_redis"}
                for record in cached_records[:top_k]
            ]
    if structural_query:
        records = memory_db.search_memory_structural(
            db_path,
            query_anchors=anchor_projection["anchors"],
            top_k=top_k,
        )
    else:
        memory_db.ensure_memory_ann_index(db_path)
        records = memory_db.search_memory_ann(
            db_path,
            query_vector=payload["query_vector"],
            query_embedding_model=payload["query_embedding_model"],
            query_vector_version=payload["query_vector_version"],
            top_k=top_k,
        )
    cache_write_error = ""
    if memory_cache is not None:
        try:
            memory_cache.set_records(cache_key, records)
        except Exception as exc:
            cache_write_error = repr(exc)
    return [
        {
            **record,
            "retrieval_source": "memory_db",
            "memory_cache_status": (
                "write_failed"
                if cache_write_error
                else "read_failed"
                if cache_read_error
                else "refreshed"
                if memory_cache is not None
                else "not_configured"
            ),
            **({"memory_cache_error": cache_write_error or cache_read_error} if cache_write_error or cache_read_error else {}),
        }
        for record in records
    ]


def retrieve_knowledge_and_memory(
    db_paths: Mapping[str, str] | None,
    *,
    event_workspace: Mapping[str, Any] | None = None,
    query: str = "",
    keys: Sequence[str] | None = None,
    query_vector: Any = None,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    top_k: int = 16,
    memory_cache: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "knowledge": retrieve_knowledge(
            db_paths,
            event_workspace=event_workspace,
            query=query,
            keys=keys,
            query_vector=query_vector,
            query_embedding_model=query_embedding_model,
            query_vector_version=query_vector_version,
            top_k=top_k,
        ),
        "memory": retrieve_memory(
            db_paths,
            event_workspace=event_workspace,
            query=query,
            keys=keys,
            query_vector=query_vector,
            query_embedding_model=query_embedding_model,
            query_vector_version=query_vector_version,
            top_k=top_k,
            memory_cache=memory_cache,
        ),
    }


def _db_path(db_paths: Mapping[str, str] | None, key: str) -> str:
    if not isinstance(db_paths, Mapping):
        return ""
    return str(db_paths.get(key) or db_paths.get(f"{key}_db") or "").strip()


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _valid_cached_memory_records(
    records: Any,
    *,
    query_vector_version: str,
) -> bool:
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        if not isinstance(record, Mapping):
            return False
        if not str(record.get("memory_id") or record.get("id") or "").strip():
            return False
        record_version = str(record.get("vector_version") or "").strip()
        if query_vector_version and record_version and record_version != query_vector_version:
            return False
    return True
