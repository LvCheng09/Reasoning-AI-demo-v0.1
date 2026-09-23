"""Unified three-database persistence wrapper for knowledge, rules, and memories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import knowledge_db
from . import memory_db
from . import rule_db


DB_KINDS = ("knowledge", "rule", "memory")


def normalize_db_paths(
    db_path: Mapping[str, str | Path] | None = None,
    *,
    knowledge_db_path: str | Path | None = None,
    rule_db_path: str | Path | None = None,
    memory_db_path: str | Path | None = None,
) -> dict[str, str]:
    explicit = {
        "knowledge": knowledge_db_path,
        "rule": rule_db_path,
        "memory": memory_db_path,
    }
    if any(value is not None for value in explicit.values()):
        missing = [kind for kind, value in explicit.items() if value is None]
        if missing:
            raise ValueError(
                "knowledge_db_path, rule_db_path, and memory_db_path must all be provided together."
            )
        return {
            kind: str(Path(value).expanduser())
            for kind, value in explicit.items()
            if value is not None
        }
    if isinstance(db_path, Mapping):
        missing = [kind for kind in DB_KINDS if kind not in db_path]
        if missing:
            raise ValueError(
                f"db_path mapping must include keys {DB_KINDS}; missing: {', '.join(missing)}"
            )
        return {kind: str(Path(db_path[kind]).expanduser()) for kind in DB_KINDS}
    raise ValueError(
        "Three database paths are required. Provide either a mapping with keys "
        "('knowledge', 'rule', 'memory') or the explicit knowledge_db_path/rule_db_path/memory_db_path trio."
    )


def _merge_summary(prefix: str, summary: dict[str, Any], target: dict[str, Any]) -> None:
    for key, value in summary.items():
        if key == "db_path":
            target[f"{prefix}_db_path"] = value
        else:
            target[f"{prefix}_{key}"] = value


def _merge_by_attr(items: list[Any], attr_name: str) -> list[Any]:
    merged: dict[str, Any] = {}
    for item in items:
        key = str(getattr(item, attr_name, "") or "")
        if key:
            merged[key] = item
    return list(merged.values())


def _knowledge_signature(payload: dict[str, Any], *keys: str) -> str:
    chunks: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
        elif isinstance(value, (list, tuple, set)):
            chunks.extend(str(item).strip() for item in value if str(item).strip())
    return " ".join(chunks).strip()


def _with_knowledge_vectors(extractor: Any, records: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        payload = dict(record)
        if enabled:
            if payload.get("core_vector") is None:
                signature = _knowledge_signature(
                    payload,
                    "core",
                )
                if signature:
                    payload["core_vector"] = extractor._encode_text_vector(signature)
            if payload.get("nature_vector") is None:
                signature = _knowledge_signature(
                    payload,
                    "nature",
                )
                if signature:
                    payload["nature_vector"] = extractor._encode_text_vector(signature)
            if payload.get("relation_vector") is None:
                signature = _knowledge_signature(
                    payload,
                    "relation",
                )
                if signature:
                    payload["relation_vector"] = extractor._encode_text_vector(signature)
            payload["embedding_model"] = str(getattr(extractor, "encoder_name_or_path", "") or "")
        result.append(payload)
    return result


def _memory_signature(payload: dict[str, Any], *keys: str) -> str:
    chunks: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
        elif isinstance(value, (list, tuple, set)):
            chunks.extend(str(item).strip() for item in value if str(item).strip())
    return " ".join(chunks).strip()


def _with_memory_vectors(extractor: Any, records: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        payload = dict(record)
        if enabled and payload.get("memory_vector") is None and payload.get("vector") is None:
            signature = _memory_signature(
                payload,
                "text",
                "input_summary",
                "outcome",
                "heads",
                "relations",
                "context_keys",
                "selected_rule_ids",
            )
            if signature:
                payload["memory_vector"] = extractor._encode_text_vector(signature)
        result.append(payload)
    return result



def _rule_signature(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in (
        "rule_id", "name", "description", "function_type", "rule_layer", "domain",
        "conclusion_template", "learn_from",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            chunks.append(str(value))
    for key in (
        "required_proposition_types", "proposition_types", "required_relation_types",
        "required_conditions", "produced_conditions", "cue_terms", "required_heads",
        "scope_tags", "task_types", "risk_tags", "analysis_steps", "mandatory_checks",
        "preferred_routes", "validation_priority",
    ):
        values = payload.get(key)
        if isinstance(values, (list, tuple, set)):
            chunks.extend(str(item) for item in values if str(item).strip())
        elif values not in (None, ""):
            chunks.append(str(values))
    for item in payload.get("required", []) or []:
        if isinstance(item, dict):
            chunks.extend(
                str(item.get(key, "")).strip()
                for key in (
                    "condition",
                    "relation",
                    "required_type",
                    "vector_query_text",
                    "source_preference",
                )
                if str(item.get(key, "")).strip()
            )
    for item in payload.get("produced", []) or []:
        if isinstance(item, dict):
            chunks.extend(
                str(item.get(key, "")).strip()
                for key in (
                    "conclusion",
                    "produced_type",
                    "confidence_policy",
                )
                if str(item.get(key, "")).strip()
            )
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        chunks.extend(str(v) for v in metadata.values() if isinstance(v, (str, int, float)))
    return " ".join(item.strip() for item in chunks if item and str(item).strip())


def _with_rule_vectors(extractor: Any, rules: list[Any], enabled: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rule in rules:
        payload = dict(rule.to_dict()) if hasattr(rule, "to_dict") else dict(rule)
        if enabled and payload.get("vector") is None and payload.get("rule_vector") is None:
            signature = _rule_signature(payload)
            if signature:
                payload["rule_vector"] = extractor._encode_text_vector(signature)
                payload["vector"] = payload["rule_vector"]
                payload["embedding_model"] = str(getattr(extractor, "encoder_name_or_path", "") or "")
        if enabled:
            required_items: list[dict[str, Any]] = []
            for item in payload.get("required", []) or []:
                if not isinstance(item, dict):
                    continue
                item_payload = dict(item)
                condition = str(item_payload.get("condition") or "").strip()
                relation = str(item_payload.get("relation") or "").strip()
                vector_query_text = str(item_payload.get("vector_query_text") or "").strip()
                if condition and item_payload.get("condition_vector") is None:
                    item_payload["condition_vector"] = extractor._encode_text_vector(condition)
                if relation and item_payload.get("relation_vector") is None:
                    item_payload["relation_vector"] = extractor._encode_text_vector(relation)
                if vector_query_text and item_payload.get("vector_query_vector") is None:
                    item_payload["vector_query_vector"] = extractor._encode_text_vector(vector_query_text)
                required_items.append(item_payload)
            if required_items:
                payload["required"] = required_items
            produced_items: list[dict[str, Any]] = []
            for item in payload.get("produced", []) or []:
                if not isinstance(item, dict):
                    continue
                item_payload = dict(item)
                conclusion = str(item_payload.get("conclusion") or "").strip()
                if conclusion and item_payload.get("conclusion_vector") is None:
                    item_payload["conclusion_vector"] = extractor._encode_text_vector(conclusion)
                produced_items.append(item_payload)
            if produced_items:
                payload["produced"] = produced_items
        result.append(payload)
    return result
def load_reasoning_databases(
    extractor: Any,
    db_path: Mapping[str, str | Path] | None = None,
    *,
    knowledge_db_path: str | Path | None = None,
    rule_db_path: str | Path | None = None,
    memory_db_path: str | Path | None = None,
    replace: bool = False,
    max_knowledge_records: int | None = 512,
    max_memory_records: int | None = 512,
    max_rule_records: int | None = 4096,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(
        db_path,
        knowledge_db_path=knowledge_db_path,
        rule_db_path=rule_db_path,
        memory_db_path=memory_db_path,
    )
    load_summary: dict[str, Any] = {
        "db_paths": dict(db_paths),
        "replace": bool(replace),
    }
    _merge_summary(
        "knowledge",
        {
            "db_path": db_paths["knowledge"],
            "loaded_knowledge_records": len(
                [
                    extractor._build_knowledge_record(item)
                    for item in knowledge_db.load_knowledge_records(
                        db_paths["knowledge"],
                        limit=max_knowledge_records,
                    )
                ]
            ),
            "loaded_memories": 0,
            "loaded_content_rules": 0,
            "loaded_structural_rules": 0,
            "loaded_control_rules": 0,
            "replace": bool(replace),
        },
        load_summary,
    )
    loaded_knowledge = [
        extractor._build_knowledge_record(item)
        for item in knowledge_db.load_knowledge_records(
            db_paths["knowledge"],
            limit=max_knowledge_records,
        )
    ]
    if replace:
        extractor.knowledge_base = loaded_knowledge
    else:
        extractor.knowledge_base = _merge_by_attr([*extractor.knowledge_base, *loaded_knowledge], "id")
    if hasattr(extractor, "clear_retrieval_cache"):
        extractor.clear_retrieval_cache()
    content_payloads = rule_db.load_rule_records(
        db_paths["rule"],
        rule_kind="content",
        limit=max_rule_records,
    )
    structural_payloads = rule_db.load_rule_records(
        db_paths["rule"],
        rule_kind="structural",
        limit=max_rule_records,
    )
    control_payloads = rule_db.load_rule_records(
        db_paths["rule"],
        rule_kind="control",
        limit=max_rule_records,
    )
    loaded_content_rules = extractor._normalize_rule_bank(content_payloads) if content_payloads else []
    loaded_structural_rules = extractor._normalize_structural_rule_bank(structural_payloads) if structural_payloads else []
    loaded_control_rules = extractor._normalize_control_rule_bank(control_payloads) if control_payloads else []
    if replace:
        if loaded_content_rules:
            extractor.rule_bank = loaded_content_rules
        if loaded_structural_rules:
            extractor.structural_rule_bank = loaded_structural_rules
        if loaded_control_rules:
            extractor.control_rule_bank = loaded_control_rules
    else:
        extractor.rule_bank = _merge_by_attr([*extractor.rule_bank, *loaded_content_rules], "rule_id")
        extractor.structural_rule_bank = _merge_by_attr([*extractor.structural_rule_bank, *loaded_structural_rules], "rule_id")
        extractor.control_rule_bank = _merge_by_attr([*extractor.control_rule_bank, *loaded_control_rules], "rule_id")
    extractor.sync_rule_store()
    _merge_summary(
        "rule",
        {
            "db_path": db_paths["rule"],
            "loaded_knowledge_records": 0,
            "loaded_memories": 0,
            "loaded_content_rules": len(loaded_content_rules),
            "loaded_structural_rules": len(loaded_structural_rules),
            "loaded_control_rules": len(loaded_control_rules),
            "replace": bool(replace),
        },
        load_summary,
    )
    _merge_summary(
        "memory",
        {
            "db_path": db_paths["memory"],
            "loaded_knowledge_records": 0,
            "loaded_memories": len(
                [
                    extractor._build_memory_entry(item)
                    for item in memory_db.load_memory_records(
                        db_paths["memory"],
                        limit=max_memory_records,
                    )
                ]
            ),
            "loaded_content_rules": 0,
            "loaded_structural_rules": 0,
            "loaded_control_rules": 0,
            "replace": bool(replace),
        },
        load_summary,
    )
    loaded_memories = [
        extractor._build_memory_entry(item)
        for item in memory_db.load_memory_records(
            db_paths["memory"],
            limit=max_memory_records,
        )
    ]
    if hasattr(extractor, "register_memories"):
        extractor.register_memories(
            loaded_memories,
            replace=bool(replace),
            session_id="global_retrieved_memory",
        )
    load_summary["loaded_knowledge_records"] = len(getattr(extractor, "knowledge_base", []) or [])
    if hasattr(extractor, "_collect_retrieved_memories_from_working_memory"):
        load_summary["loaded_memories"] = len(
            extractor._collect_retrieved_memories_from_working_memory(session_id="global_retrieved_memory")
        )
    else:
        load_summary["loaded_memories"] = len(loaded_memories)
    load_summary["loaded_content_rules"] = len(getattr(extractor, "rule_bank", []) or [])
    load_summary["loaded_structural_rules"] = len(getattr(extractor, "structural_rule_bank", []) or [])
    load_summary["loaded_control_rules"] = len(getattr(extractor, "control_rule_bank", []) or [])
    return load_summary


def connect_reasoning_databases(
    extractor: Any,
    db_path: Mapping[str, str | Path] | None = None,
    *,
    knowledge_db_path: str | Path | None = None,
    rule_db_path: str | Path | None = None,
    memory_db_path: str | Path | None = None,
    seed_if_empty: bool = False,
    replace_seed: bool = False,
    encode_seed_vectors: bool = False,
    max_knowledge_records: int | None = 512,
    max_memory_records: int | None = 512,
    max_rule_records: int | None = 4096,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(
        db_path,
        knowledge_db_path=knowledge_db_path,
        rule_db_path=rule_db_path,
        memory_db_path=memory_db_path,
    )
    load_summary = load_reasoning_databases(
        extractor,
        db_paths,
        replace=False,
        max_knowledge_records=max_knowledge_records,
        max_memory_records=max_memory_records,
        max_rule_records=max_rule_records,
    )
    return {
        "db_paths": dict(db_paths),
        "seeded": False,
        "seed_summary": None,
        "seed_disabled": True,
        "load_summary": load_summary,
    }


def save_reasoning_databases(
    extractor: Any,
    db_path: Mapping[str, str | Path] | None = None,
    *,
    knowledge_db_path: str | Path | None = None,
    rule_db_path: str | Path | None = None,
    memory_db_path: str | Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(
        db_path,
        knowledge_db_path=knowledge_db_path,
        rule_db_path=rule_db_path,
        memory_db_path=memory_db_path,
    )
    embedding_model = str(getattr(extractor, "encoder_name_or_path", "") or "")
    encode_vectors = getattr(extractor, "encoder", None) is not None
    content_rules = _with_rule_vectors(extractor, list(getattr(extractor, "rule_bank", []) or []), encode_vectors)
    structural_rules = _with_rule_vectors(extractor, list(getattr(extractor, "structural_rule_bank", []) or []), encode_vectors)
    control_rules = _with_rule_vectors(extractor, list(getattr(extractor, "control_rule_bank", []) or []), encode_vectors)
    summary: dict[str, Any] = {
        "db_paths": dict(db_paths),
        "replace": bool(replace),
    }
    _merge_summary(
        "knowledge",
        knowledge_db.upsert_knowledge_records(
            db_paths["knowledge"],
            list(getattr(extractor, "knowledge_base", []) or []),
            replace=replace,
            embedding_model=embedding_model,
        ),
        summary,
    )
    _merge_summary(
        "rule_content",
        rule_db.upsert_rule_records(
            db_paths["rule"],
            content_rules,
            rule_kind="content",
            replace_kind=replace,
            embedding_model=embedding_model,
        ),
        summary,
    )
    _merge_summary(
        "rule_structural",
        rule_db.upsert_rule_records(
            db_paths["rule"],
            structural_rules,
            rule_kind="structural",
            replace_kind=replace,
            embedding_model=embedding_model,
        ),
        summary,
    )
    _merge_summary(
        "rule_control",
        rule_db.upsert_rule_records(
            db_paths["rule"],
            control_rules,
            rule_kind="control",
            replace_kind=replace,
            embedding_model=embedding_model,
        ),
        summary,
    )
    _merge_summary(
        "memory",
        memory_db.upsert_memory_records(
            db_paths["memory"],
            list(
                (
                    extractor._collect_retrieved_memories_from_working_memory(session_id="global_retrieved_memory")
                    if hasattr(extractor, "_collect_retrieved_memories_from_working_memory")
                    else []
                )
            ),
            replace=replace,
            embedding_model=embedding_model,
        ),
        summary,
    )
    return summary


def summarize_databases(
    db_path: Mapping[str, str | Path] | None = None,
    *,
    knowledge_db_path: str | Path | None = None,
    rule_db_path: str | Path | None = None,
    memory_db_path: str | Path | None = None,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(
        db_path,
        knowledge_db_path=knowledge_db_path,
        rule_db_path=rule_db_path,
        memory_db_path=memory_db_path,
    )
    knowledge = knowledge_db.summarize_database(db_paths["knowledge"])
    rule = rule_db.summarize_database(db_paths["rule"])
    memory = memory_db.summarize_database(db_paths["memory"])
    return {
        "db_paths": dict(db_paths),
        "knowledge": knowledge,
        "rule": rule,
        "memory": memory,
        "knowledge_records": int(knowledge.get("knowledge_records", 0)),
        "rules": int(rule.get("rules", 0)),
        "reasoning_memories": int(memory.get("reasoning_memories", 0)),
        "vectors": int(knowledge.get("vectors", 0))
        + int(rule.get("vectors", 0))
        + int(memory.get("vectors", 0)),
    }


def search_knowledge_records(
    db_path: Mapping[str, str | Path],
    *,
    query: str = "",
    keys: list[str] | None = None,
    query_vector: Any = None,
    top_k: int = 128,
) -> list[dict[str, Any]]:
    db_paths = normalize_db_paths(db_path)
    return knowledge_db.search_knowledge_records(
        db_paths["knowledge"],
        query=query,
        keys=keys,
        query_vector=query_vector,
        top_k=top_k,
    )


def search_knowledge_ann(
    db_path: Mapping[str, str | Path],
    *,
    query: dict[str, Any],
    top_k: int = 10,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    db_paths = normalize_db_paths(db_path)
    return knowledge_db.search_knowledge_ann(
        db_paths["knowledge"],
        query=query,
        top_k=top_k,
        limit=limit,
    )


def record_knowledge_usage(
    db_path: Mapping[str, str | Path],
    knowledge_ids: list[str],
    *,
    useful: bool = False,
    failed: bool = False,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(db_path)
    return knowledge_db.record_knowledge_usage(
        db_paths["knowledge"],
        knowledge_ids,
        useful=useful,
        failed=failed,
    )


def maintain_knowledge_records(
    db_path: Mapping[str, str | Path],
    *,
    retire_failure_rate: float = 0.60,
    low_usage_threshold: int = 0,
    delete_retired: bool = False,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(db_path)
    return knowledge_db.reprioritize_knowledge_records(
        db_paths["knowledge"],
        retire_failure_rate=retire_failure_rate,
        low_usage_threshold=low_usage_threshold,
        delete_retired=delete_retired,
    )


def search_memory_records(
    db_path: Mapping[str, str | Path],
    *,
    query: str = "",
    keys: list[str] | None = None,
    query_vector: Any = None,
    top_k: int = 64,
) -> list[dict[str, Any]]:
    db_paths = normalize_db_paths(db_path)
    return memory_db.search_memory_records(
        db_paths["memory"],
        query=query,
        keys=keys,
        query_vector=query_vector,
        top_k=top_k,
    )


def cleanup_memory_records(
    db_path: Mapping[str, str | Path],
    *,
    stale_days: int = 60,
    min_utility_score: float = 0.22,
    prune_ratio: float = 0.10,
    min_records_to_keep: int = 128,
    dry_run: bool = False,
) -> dict[str, Any]:
    db_paths = normalize_db_paths(db_path)
    return memory_db.cleanup_memory_records(
        db_paths["memory"],
        stale_days=stale_days,
        min_utility_score=min_utility_score,
        prune_ratio=prune_ratio,
        min_records_to_keep=min_records_to_keep,
        dry_run=dry_run,
    )


def search_rule_records(
    db_path: Mapping[str, str | Path],
    *,
    query: str = "",
    keys: list[str] | None = None,
    query_vector: Any = None,
    rule_kind: str | None = None,
    top_k: int = 256,
) -> list[dict[str, Any]]:
    db_paths = normalize_db_paths(db_path)
    return rule_db.search_rule_records(
        db_paths["rule"],
        query=query,
        keys=keys,
        query_vector=query_vector,
        rule_kind=rule_kind,
        top_k=top_k,
    )
