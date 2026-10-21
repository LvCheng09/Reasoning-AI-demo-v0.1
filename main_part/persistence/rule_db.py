"""Persistent rule DB with required and produced child tables."""

from __future__ import annotations
from . import base as shared_db
from ..common.math_utils import cosine as shared_cosine
from ..common.normalize import as_list as shared_as_list

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from ..identity import stable_text_id as _stable_id
from ..rule_layers import normalize_rule_layer
from ..vector_contracts import canonical_embedding_model, coerce_vector, cosine_similarity_strict
from .ann_index import (
    ann_index_summary,
    delete_lsh_record,
    query_lsh_index,
    rebuild_lsh_index,
    upsert_lsh_vector,
)


SCHEMA_VERSION = 8
SCHEMA_NAME = "rule_required_produced_vector_tag_index_runtime_activation_v8"
RULE_ANN_INDEX_NAME = "rules"
_RULE_LAYER_BY_KIND = {
    "content": "domain",
    "structural": "structure",
    "control": "control",
    "matcher_rule": "matcher_rule",
    "meta": "control",
}

_RUNTIME_INACTIVE_STATUSES = (
    "candidate",
    "candidates",
    "instance_seed",
    "recurring_pattern_candidate",
    "causal_pattern_candidate",
    "modified_candidate",
    "copy_ready",
    "draft",
    "inactive",
    "deprecated",
    "superseded",
    "archived",
    "deleted",
    "rejected",
    "blocked",
    "shadow",
)
_RUNTIME_INACTIVE_PROMOTION_STATUSES = (
    "candidate",
    "pending",
    "pending_validation",
    "shadow",
    "rejected",
    "deprecated",
    "superseded",
    "failed",
)
_RUNTIME_INACTIVE_VALIDATION_STATUSES = (
    "candidate",
    "pending",
    "pending_validation",
    "shadow",
    "unverified",
    "rejected",
    "failed",
)


def _rule_payload_runtime_active(payload: Mapping[str, Any]) -> bool:
    """Whether a stored rule may participate in formal inference."""

    status = str(payload.get("status") or payload.get("revision_status") or "active").strip().lower()
    active_slot = str(payload.get("active_slot") or "active").strip().lower()
    promotion = str(payload.get("promotion_status") or "").strip().lower()
    validation = str(payload.get("validation_status") or "").strip().lower()
    return (
        status not in _RUNTIME_INACTIVE_STATUSES
        and active_slot == "active"
        and promotion not in _RUNTIME_INACTIVE_PROMOTION_STATUSES
        and validation not in _RUNTIME_INACTIVE_VALIDATION_STATUSES
    )


def _runtime_active_rule_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    payload = f"{prefix}payload_json"
    inactive = ",".join(repr(value) for value in _RUNTIME_INACTIVE_STATUSES)
    inactive_promotion = ",".join(repr(value) for value in _RUNTIME_INACTIVE_PROMOTION_STATUSES)
    inactive_validation = ",".join(repr(value) for value in _RUNTIME_INACTIVE_VALIDATION_STATUSES)
    return (
        f"LOWER(COALESCE(json_extract({payload}, '$.status'), "
        f"json_extract({payload}, '$.revision_status'), 'active')) NOT IN ({inactive}) "
        f"AND LOWER(COALESCE(json_extract({payload}, '$.active_slot'), 'active')) = 'active' "
        f"AND LOWER(COALESCE(json_extract({payload}, '$.promotion_status'), '')) NOT IN ({inactive_promotion}) "
        f"AND LOWER(COALESCE(json_extract({payload}, '$.validation_status'), '')) NOT IN ({inactive_validation})"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    return shared_db.connect(db_path, foreign_keys=True)


def _connect_readonly_or_create(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    return shared_db.connect(path, readonly=path.exists())


def _json_dumps(value: Any) -> str:
    return shared_db.json_dumps(value)


def _json_loads(value: str | None, default: Any) -> Any:
    return shared_db.json_loads(value, default)


def _blob_to_vector(value: bytes | bytearray | memoryview | None) -> list[float] | None:
    return shared_db.blob_to_vector(value)


def _vector_to_list(value: Any) -> list[float] | None:
    return coerce_vector(value)


def _vector_to_blob(value: Any) -> bytes | None:
    return shared_db.vector_to_blob(value, reject_zero=True, inclusive_zero=False)


def _first_vector(*values: Any) -> Any:
    for value in values:
        if _vector_to_list(value):
            return value
    return None


def _cosine(left: Any, right: Any) -> float:
    return shared_cosine(left, right, positive_only=True)


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return shared_db.table_columns(conn, table)


def _ensure_payload_json_column(conn: sqlite3.Connection) -> None:
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "rules" not in tables:
        return
    if "payload_json" in _table_columns(conn, "rules"):
        return
    conn.execute("ALTER TABLE rules ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    tables = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if table not in tables:
        return
    if column in _table_columns(conn, table):
        return
    shared_db.ensure_column(conn, table, column, definition)
    conn.commit()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rules (
            rule_id TEXT PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            rule_vector BLOB,
            embedding_model TEXT NOT NULL DEFAULT '',
            vector_dimension INTEGER NOT NULL DEFAULT 0,
            vector_version TEXT NOT NULL DEFAULT '',
            rule_kind TEXT NOT NULL DEFAULT 'content',
            complexity REAL NOT NULL DEFAULT 0.5,
            priority REAL NOT NULL DEFAULT 1.0,
            confidence REAL NOT NULL DEFAULT 0.5,
            logical_strength REAL NOT NULL DEFAULT 0.5,
            empirical_support REAL NOT NULL DEFAULT 0.5,
            contradiction_rate REAL NOT NULL DEFAULT 0.0,
            application_count REAL NOT NULL DEFAULT 0.0,
            success_count REAL NOT NULL DEFAULT 0.0,
            failure_count REAL NOT NULL DEFAULT 0.0,
            support_count INTEGER NOT NULL DEFAULT 0,
            learn_from TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_shadow_stats (
            rule_id TEXT PRIMARY KEY,
            application_count REAL NOT NULL DEFAULT 0.0,
            success_count REAL NOT NULL DEFAULT 0.0,
            failure_count REAL NOT NULL DEFAULT 0.0,
            semantic_success_count REAL NOT NULL DEFAULT 0.0,
            semantic_failure_count REAL NOT NULL DEFAULT 0.0,
            path_success_count REAL NOT NULL DEFAULT 0.0,
            path_failure_count REAL NOT NULL DEFAULT 0.0,
            effective_evidence_weight REAL NOT NULL DEFAULT 0.0,
            last_case_id TEXT NOT NULL DEFAULT '',
            last_status TEXT NOT NULL DEFAULT '',
            stats_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
        )
        """
    )
    _ensure_column(conn, "rule_shadow_stats", "semantic_failure_count", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "rule_shadow_stats", "path_failure_count", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "rule_shadow_stats", "effective_evidence_weight", "REAL NOT NULL DEFAULT 0.0")
    _ensure_column(conn, "rule_shadow_stats", "stats_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "rule_shadow_stats", "created_at", "TEXT NOT NULL DEFAULT ''")
    conn.execute(
        "UPDATE rule_shadow_stats SET created_at = updated_at WHERE created_at = ''"
    )
    _ensure_payload_json_column(conn)
    _ensure_column(conn, "rules", "rule_vector", "BLOB")
    _ensure_column(conn, "rules", "embedding_model", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "rules", "vector_dimension", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "rules", "vector_version", "TEXT NOT NULL DEFAULT ''")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_tags (
            rule_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            source_field TEXT NOT NULL,
            PRIMARY KEY(rule_id, tag, source_field),
            FOREIGN KEY(rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rule_tags_lookup
        ON rule_tags(tag, rule_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rules_runtime_visibility
        ON rules(
            LOWER(COALESCE(json_extract(payload_json, '$.status'), json_extract(payload_json, '$.revision_status'), 'active')),
            LOWER(COALESCE(json_extract(payload_json, '$.active_slot'), 'active')),
            LOWER(COALESCE(json_extract(payload_json, '$.promotion_status'), '')),
            LOWER(COALESCE(json_extract(payload_json, '$.validation_status'), ''))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rules_runtime_activation
        ON rules(
            LOWER(COALESCE(json_extract(payload_json, '$.runtime_activation.policy'), json_extract(payload_json, '$.metadata.repair_activation_policy'), '')),
            COALESCE(json_extract(payload_json, '$.runtime_activation.target_rule_id'), json_extract(payload_json, '$.metadata.repair_target_rule_id'), ''),
            COALESCE(json_extract(payload_json, '$.runtime_activation.missing_structure_signature'), json_extract(payload_json, '$.metadata.missing_structure_signature'), ''),
            COALESCE(json_extract(payload_json, '$.runtime_activation.repair_family_key'), json_extract(payload_json, '$.metadata.repair_family_key'), '')
        )
        """
    )
    tag_backfill_key = "rule_tag_index_backfilled_v1"
    tag_backfill = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?",
        (tag_backfill_key,),
    ).fetchone()
    if tag_backfill is None:
        for source_field in ("tags", "scope_tags"):
            conn.execute(
                f"""
                INSERT OR IGNORE INTO rule_tags(rule_id, tag, source_field)
                SELECT rules.rule_id, LOWER(TRIM(tag_items.value)), ?
                FROM rules, json_each(rules.payload_json, '$.{source_field}') AS tag_items
                WHERE json_valid(rules.payload_json)
                  AND json_type(rules.payload_json, '$.{source_field}') = 'array'
                  AND TRIM(CAST(tag_items.value AS TEXT)) <> ''
                """,
                (source_field,),
            )
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
            (tag_backfill_key, _now_iso()),
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS required (
            required_id TEXT PRIMARY KEY,
            rule_id TEXT NOT NULL,
            condition TEXT NOT NULL DEFAULT '',
            condition_vector BLOB,
            condition_priority REAL NOT NULL DEFAULT 1.0,
            relation TEXT NOT NULL DEFAULT '',
            relation_vector BLOB,
            required_type TEXT NOT NULL DEFAULT 'hard',
            vector_query_text TEXT NOT NULL DEFAULT '',
            vector_query_vector BLOB,
            match_threshold REAL NOT NULL DEFAULT 0.5,
            source_preference TEXT NOT NULL DEFAULT 'current',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS produced (
            produced_id TEXT PRIMARY KEY,
            rule_id TEXT NOT NULL,
            conclusion TEXT NOT NULL DEFAULT '',
            conclusion_vector BLOB,
            conclusion_priority REAL NOT NULL DEFAULT 1.0,
            produced_type TEXT NOT NULL DEFAULT 'hypothesis',
            confidence_policy TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
        )
        """
    )
    _ensure_column(conn, "required", "required_type", "TEXT NOT NULL DEFAULT 'hard'")
    _ensure_column(conn, "required", "condition_vector", "BLOB")
    _ensure_column(conn, "required", "relation_vector", "BLOB")
    _ensure_column(conn, "required", "vector_query_text", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "required", "vector_query_vector", "BLOB")
    _ensure_column(conn, "required", "match_threshold", "REAL NOT NULL DEFAULT 0.5")
    _ensure_column(conn, "required", "source_preference", "TEXT NOT NULL DEFAULT 'current'")
    _ensure_column(conn, "produced", "produced_type", "TEXT NOT NULL DEFAULT 'hypothesis'")
    _ensure_column(conn, "produced", "conclusion_vector", "BLOB")
    _ensure_column(conn, "produced", "confidence_policy", "TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_kind_priority ON rules(rule_kind, priority DESC, confidence DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_vector_contract ON rules(embedding_model, vector_dimension, vector_version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_required_rule ON required(rule_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_required_condition ON required(condition)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_required_type ON required(required_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_required_source ON required(source_preference)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_produced_rule ON produced(rule_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_produced_conclusion ON produced(conclusion)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_produced_type ON produced(produced_type)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_family_state (
            family_key TEXT PRIMARY KEY,
            domain TEXT NOT NULL DEFAULT 'general',
            rule_layer TEXT NOT NULL DEFAULT 'domain',
            target_rule_id TEXT NOT NULL DEFAULT '',
            maturity TEXT NOT NULL DEFAULT '',
            potential_score REAL NOT NULL DEFAULT 0.0,
            effective_observations REAL NOT NULL DEFAULT 0.0,
            state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_evolution_events (
            proposal_id TEXT PRIMARY KEY,
            family_key TEXT NOT NULL DEFAULT '',
            mutation_type TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending_validation',
            source_rule_ids_json TEXT NOT NULL DEFAULT '[]',
            candidate_rule_ids_json TEXT NOT NULL DEFAULT '[]',
            proposal_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_family_target "
        "ON rule_family_state(target_rule_id, rule_layer)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_evolution_family "
        "ON rule_evolution_events(family_key, mutation_type, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_shadow_status "
        "ON rule_shadow_stats(last_status, application_count DESC)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_name', ?)",
        (SCHEMA_NAME,),
    )
    conn.commit()


def _payload(rule: Any) -> dict[str, Any]:
    from ..rule_execution_contract import validate_rule_contract
    if hasattr(rule, "to_dict"):
        payload = dict(rule.to_dict())
    else:
        payload = dict(rule or {})
    validate_rule_contract(payload)
    return payload


def _as_list(value: Any) -> list[Any]:
    return shared_as_list(value, expand_sets=True)


def _source_preference_text(value: Any) -> str:
    if value in (None, ""):
        return "current"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item).strip() for item in value if str(item).strip()) or "current"
    return str(value).strip() or "current"


def _debug_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(
                    " ".join(
                        str(item.get(key) or "").strip()
                        for key in ("core", "relation", "nature", "text", "debug_text")
                        if str(item.get(key) or "").strip()
                    )
                )
            else:
                parts.append(str(item).strip())
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        return " ".join(
            str(value.get(key) or "").strip()
            for key in ("debug_text", "core", "relation", "nature", "text")
            if str(value.get(key) or "").strip()
        ).strip()
    return str(value).strip()


def _coerce_required(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = payload.get("required")
    if raw_items is None:
        raw_items = []
        for condition in _as_list(payload.get("required_conditions")):
            raw_items.append({"condition": condition, "condition_priority": 1.0, "relation": ""})
        for relation in _as_list(payload.get("required_relation_types")):
            raw_items.append({"condition": "", "condition_priority": 0.8, "relation": relation})
        for head in _as_list(payload.get("required_heads")):
            raw_items.append({"condition": head, "condition_priority": 0.7, "relation": "required_head"})
    items: list[dict[str, Any]] = []
    for item in _as_list(raw_items):
        if isinstance(item, dict):
            normalized = dict(item)
            condition_value = item.get("condition")
            condition = _debug_text(condition_value) or str(item.get("debug_text") or "").strip()
            relation = str(item.get("relation") or "").strip()
            priority = float(item.get("condition_priority", item.get("priority", 1.0)))
            match_threshold = float(item.get("match_threshold", item.get("threshold", 0.5)))
            source_preference = _source_preference_text(item.get("source_preference", item.get("source", "current")))
        else:
            normalized = {}
            condition = str(item).strip()
            relation = ""
            priority = 1.0
            match_threshold = 0.5
            source_preference = "current"
        if not condition and not relation and not _condition_has_structural_payload(normalized):
            continue
        normalized.update({
            "condition": normalized.get("condition") if isinstance(normalized.get("condition"), list) else condition,
            "condition_text": condition,
            "condition_priority": priority,
            "relation": relation,
            "match_threshold": max(0.0, min(match_threshold, 1.0)),
            "source_preference": source_preference,
        })
        items.append(normalized)
    return items


def _condition_has_structural_payload(value: Mapping[str, Any]) -> bool:
    conditions = value.get("condition")
    if not isinstance(conditions, list):
        return False
    for condition in conditions:
        if not isinstance(condition, Mapping):
            continue
        if any(
            _vector_to_list(condition.get(field))
            for field in ("core_vector", "relation_vector", "nature_vector")
        ):
            return True
    return False


def _validate_required_relation_endpoints(payload: dict[str, Any]) -> None:
    rule_id = str(payload.get("rule_id") or payload.get("id") or "").strip()
    for required_index, required in enumerate(_as_list(payload.get("required"))):
        if not isinstance(required, dict):
            continue
        condition_span_ids = {
            str(span.get("span_id") or span.get("id") or "").strip()
            for span in _as_list(required.get("condition"))
            if isinstance(span, dict)
            and str(span.get("span_id") or span.get("id") or "").strip()
        }
        for relation_index, relation in enumerate(_as_list(required.get("relation_patterns"))):
            if not isinstance(relation, dict):
                continue
            source_span_id = str(relation.get("source_span_id") or relation.get("source_slot") or "").strip()
            target_span_id = str(relation.get("target_span_id") or relation.get("target_slot") or "").strip()
            invalid_endpoints = [
                endpoint
                for endpoint in (source_span_id, target_span_id)
                if not endpoint or endpoint not in condition_span_ids
            ]
            if invalid_endpoints:
                raise ValueError(
                    "Invalid relation endpoints for "
                    f"rule={rule_id or '<unknown>'} required={required_index} relation={relation_index}: "
                    f"{invalid_endpoints} are not condition span ids"
                )


def _coerce_produced(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = payload.get("produced")
    if raw_items is None:
        raw_items = []
        for condition in _as_list(payload.get("produced_conditions")):
            raw_items.append({"conclusion": condition, "conclusion_priority": 1.0})
        template = str(payload.get("conclusion_template") or "").strip()
        conclusion_type = str(payload.get("conclusion_type") or "").strip()
        if template:
            raw_items.append({"conclusion": template, "conclusion_priority": 1.0})
        elif conclusion_type:
            raw_items.append({"conclusion": conclusion_type, "conclusion_priority": 0.8})
    items: list[dict[str, Any]] = []
    for item in _as_list(raw_items):
        if isinstance(item, dict):
            normalized = dict(item)
            operation = str(normalized.get("operation") or normalized.get("op") or "").strip()
            conclusion_value = normalized.get("conclusion")
            conclusion = _debug_text(conclusion_value)
            priority = float(item.get("conclusion_priority", item.get("priority", 1.0)))
            default_type = "action" if operation else "hypothesis"
            produced_type = str(item.get("produced_type") or item.get("type") or default_type).strip() or default_type
            confidence_policy = str(item.get("confidence_policy") or "").strip()
        else:
            normalized = {}
            conclusion = str(item).strip()
            priority = 1.0
            produced_type = "hypothesis"
            confidence_policy = ""
        if not conclusion and not operation:
            continue
        normalized.update({
            "conclusion": normalized.get("conclusion") if isinstance(normalized.get("conclusion"), list) else conclusion,
            "conclusion_text": conclusion,
            "conclusion_priority": priority,
            "produced_type": produced_type if produced_type in {"fact", "hypothesis", "action", "missing_slot", "warning"} else "hypothesis",
            "confidence_policy": confidence_policy,
        })
        if operation:
            normalized["operation"] = operation
        items.append(normalized)
    return items


def _clean_required_payload_item(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    for key in (
        "span_pattern",
        "vector_query_text",
    ):
        cleaned.pop(key, None)
    for span in cleaned.get("condition") if isinstance(cleaned.get("condition"), list) else []:
        if isinstance(span, dict):
            for key in ("vector", "span_vector", "span_core_vec", "relation_vec", "nature_vec", "slot_id"):
                span.pop(key, None)
    return cleaned


def _clean_produced_payload_item(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    for key in ("span_pattern", "vector_query_text"):
        cleaned.pop(key, None)
    for span in cleaned.get("conclusion") if isinstance(cleaned.get("conclusion"), list) else []:
        if isinstance(span, dict):
            for key in ("vector", "span_vector", "span_core_vec", "relation_vec", "nature_vec", "slot_id"):
                span.pop(key, None)
    return cleaned


def _row_to_payload(
    row: sqlite3.Row,
    required_rows: Sequence[sqlite3.Row],
    produced_rows: Sequence[sqlite3.Row],
) -> dict[str, Any]:
    payload = dict(_json_loads(row["payload_json"] if "payload_json" in row.keys() else None, {}))
    stored_payload_required = _as_list(payload.get("required"))
    required_items = [
        {
            "condition": str(item["condition"] or ""),
            "condition_priority": float(item["condition_priority"]),
            "relation": str(item["relation"] or ""),
            "match_threshold": float(item["match_threshold"]),
            "source_preference": str(item["source_preference"] or "current"),
        }
        for item in required_rows
    ]
    # payload_json is the canonical ordered structure. The child tables are
    # searchable projections and are sorted independently, so positional merges
    # can attach one condition's scalar fields to another condition.
    if stored_payload_required:
        required_items = [dict(item) for item in stored_payload_required if isinstance(item, dict)]
    stored_payload_produced = _as_list(payload.get("produced"))
    produced_items = [
        {
            "conclusion": str(item["conclusion"] or ""),
            "conclusion_vector": _blob_to_vector(item["conclusion_vector"]),
            "conclusion_priority": float(item["conclusion_priority"]),
            "produced_type": str(item["produced_type"] or "hypothesis"),
            "confidence_policy": str(item["confidence_policy"] or ""),
        }
        for item in produced_rows
    ]
    if stored_payload_produced:
        produced_items = [dict(item) for item in stored_payload_produced if isinstance(item, dict)]
    produced_conditions = [
        _debug_text(item.get("conclusion"))
        for item in produced_items
        if _debug_text(item.get("conclusion"))
    ]
    rule_kind = str(row["rule_kind"] or "content")
    description = str(row["description"] or row["rule_id"])
    conclusion_template = str(payload.get("conclusion_template") or (produced_conditions[0] if produced_conditions else "{summary}"))
    payload.update({
        "rule_id": str(row["rule_id"]),
        "name": str(payload.get("name") or description),
        "description": description,
        "rule_vector": _blob_to_vector(row["rule_vector"]) if "rule_vector" in row.keys() else _vector_to_list(payload.get("vector")),
        "embedding_model": str(row["embedding_model"] or payload.get("embedding_model") or "") if "embedding_model" in row.keys() else str(payload.get("embedding_model") or ""),
        "vector_dimension": int(row["vector_dimension"] or payload.get("vector_dimension") or 0) if "vector_dimension" in row.keys() else int(payload.get("vector_dimension") or 0),
        "vector_version": str(row["vector_version"] or payload.get("vector_version") or "") if "vector_version" in row.keys() else str(payload.get("vector_version") or ""),
        "function_type": str(payload.get("function_type") or rule_kind),
        "rule_kind": rule_kind,
        "rule_layer": normalize_rule_layer(payload.get("rule_layer") or _RULE_LAYER_BY_KIND.get(rule_kind, rule_kind)),
        "status": str(payload.get("status") or "active"),
        "domain": str(payload.get("domain") or "general"),
        "complexity": float(row["complexity"]),
        "priority": float(row["priority"]),
        "confidence": float(row["confidence"]),
        "confidence_prior": float(row["confidence"]),
        "logical_strength": float(row["logical_strength"]),
        "empirical_support": float(row["empirical_support"]),
        "contradiction_rate": float(row["contradiction_rate"]),
        "application_count": float(row["application_count"]),
        "success_count": float(row["success_count"]),
        "failure_count": float(row["failure_count"]),
        "support_count": int(row["support_count"]),
        "learned_from": _json_loads(row["learn_from"], []),
        "required": required_items,
        "produced": produced_items,
        "required_conditions": [item.get("condition") for item in required_items if item.get("condition")],
        "required_relation_types": [item.get("relation") for item in required_items if item.get("relation")],
        "produced_conditions": produced_conditions,
        "conclusion_type": str(payload.get("conclusion_type") or (produced_conditions[0] if produced_conditions else "candidate")),
        "conclusion_template": conclusion_template,
    })
    return payload


def database_is_empty(db_path: str | Path) -> bool:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        row = conn.execute("SELECT COUNT(*) AS count FROM rules").fetchone()
        return int(row["count"]) == 0
    finally:
        conn.close()


_RULE_LAYER_VALUE_MIGRATIONS = {
    "common": "common_sense",
    "commonsense": "common_sense",
    "auto_learn_policy": "auto_learning",
}
_RULE_LAYER_FIELDS = {"domain", "layer", "rule_layer", "target_rule_layer"}
_RULE_LAYER_LIST_FIELDS = {"preferred_rule_tags", "scope_tags", "tags"}


def inspect_rule_layer_contract(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path).expanduser()
    if not path.is_file():
        return {
            "ok": False,
            "status": "rule_db_not_found",
            "db_path": str(path),
            "deprecated_rule_count": 0,
            "invalid_payload_rule_ids": [],
        }
    conn = _connect_readonly_or_create(path)
    try:
        rows = conn.execute("SELECT rule_id, payload_json FROM rules ORDER BY rule_id").fetchall()
    finally:
        conn.close()
    deprecated: list[dict[str, Any]] = []
    invalid_payload_rule_ids: list[str] = []
    for row in rows:
        rule_id = str(row["rule_id"])
        try:
            payload = json.loads(str(row["payload_json"] or "{}"))
        except json.JSONDecodeError:
            invalid_payload_rule_ids.append(rule_id)
            continue
        changes = _canonicalize_rule_layer_payload(payload, mutate=False)
        if changes:
            deprecated.append({"rule_id": rule_id, "changes": changes})
    ok = not deprecated and not invalid_payload_rule_ids
    return {
        "ok": ok,
        "status": "canonical" if ok else "migration_required",
        "db_path": str(path),
        "rule_count": len(rows),
        "deprecated_rule_count": len(deprecated),
        "deprecated_rules": deprecated,
        "invalid_payload_rule_ids": invalid_payload_rule_ids,
    }


def migrate_rule_layer_contract(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"rule DB does not exist: {path}")
    conn = _connect(path)
    changed_rule_ids: list[str] = []
    changed_field_count = 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT rule_id, payload_json FROM rules ORDER BY rule_id").fetchall()
        updates: list[tuple[str, str, str]] = []
        now = _now_iso()
        for row in rows:
            rule_id = str(row["rule_id"])
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"rule {rule_id!r} has invalid payload_json") from exc
            changes = _canonicalize_rule_layer_payload(payload, mutate=True)
            if not changes:
                continue
            changed_rule_ids.append(rule_id)
            changed_field_count += len(changes)
            updates.append((_json_dumps(payload), now, rule_id))
        conn.executemany(
            "UPDATE rules SET payload_json = ?, updated_at = ? WHERE rule_id = ?",
            updates,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    audit = inspect_rule_layer_contract(path)
    if not audit.get("ok"):
        raise RuntimeError(f"rule layer migration did not produce a canonical DB: {audit}")
    return {
        "status": "migrated" if changed_rule_ids else "already_canonical",
        "db_path": str(path),
        "changed_rule_count": len(changed_rule_ids),
        "changed_field_count": changed_field_count,
        "changed_rule_ids": changed_rule_ids,
        "post_migration_audit": audit,
    }


def _canonicalize_rule_layer_payload(
    value: Any,
    *,
    mutate: bool,
    path: str = "$",
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in _RULE_LAYER_FIELDS and isinstance(item, str):
                replacement = _RULE_LAYER_VALUE_MIGRATIONS.get(item.strip().lower())
                if replacement:
                    changes.append({"path": item_path, "before": item, "after": replacement})
                    if mutate:
                        value[key] = replacement
                    continue
            if key in _RULE_LAYER_LIST_FIELDS and isinstance(item, list):
                for index, list_item in enumerate(item):
                    if not isinstance(list_item, str):
                        continue
                    replacement = _RULE_LAYER_VALUE_MIGRATIONS.get(list_item.strip().lower())
                    if not replacement:
                        continue
                    changes.append(
                        {
                            "path": f"{item_path}[{index}]",
                            "before": list_item,
                            "after": replacement,
                        }
                    )
                    if mutate:
                        item[index] = replacement
            changes.extend(_canonicalize_rule_layer_payload(item, mutate=mutate, path=item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            changes.extend(
                _canonicalize_rule_layer_payload(item, mutate=mutate, path=f"{path}[{index}]")
            )
    return changes


def upsert_rule_records(
    db_path: str | Path,
    rules: Sequence[Any],
    *,
    rule_kind: str,
    replace_kind: bool = False,
    embedding_model: str = "",
    preserve_stats: bool = True,
    _connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    owns_connection = _connection is None
    conn = _connection or _connect(db_path)
    try:
        if owns_connection:
            _init_schema(conn)
        if replace_kind:
            rows = conn.execute("SELECT rule_id FROM rules WHERE rule_kind = ?", (rule_kind,)).fetchall()
            for row in rows:
                deleted_rule_id = str(row["rule_id"])
                conn.execute("DELETE FROM rules WHERE rule_id = ?", (deleted_rule_id,))
                delete_lsh_record(conn, index_name=RULE_ANN_INDEX_NAME, record_id=deleted_rule_id)
        saved = 0
        required_count = 0
        produced_count = 0
        now = _now_iso()
        for rule in rules:
            payload = _payload(rule)
            _canonicalize_rule_layer_payload(payload, mutate=True)
            rule_id = str(payload.get("rule_id") or payload.get("id") or "").strip()
            if not rule_id:
                continue
            _validate_required_relation_endpoints(payload)
            description = str(
                payload.get("description")
                or payload.get("name")
                or payload.get("conclusion_template")
                or rule_id
            )
            required_items = _coerce_required(payload)
            produced_items = _coerce_produced(payload)
            rule_vector = _vector_to_list(_first_vector(payload.get("rule_vector"), payload.get("vector")))
            effective_embedding_model = canonical_embedding_model(payload.get("embedding_model") or embedding_model or "")
            vector_dimension = len(rule_vector or [])
            vector_version = str(payload.get("vector_version") or payload.get("embedding_version") or "")
            payload_to_store = dict(payload)
            payload_to_store["rule_id"] = rule_id
            payload_to_store["description"] = description
            payload_to_store["function_type"] = str(payload_to_store.get("function_type") or rule_kind)
            payload_to_store["rule_kind"] = rule_kind
            payload_to_store["rule_layer"] = normalize_rule_layer(
                payload_to_store.get("rule_layer") or _RULE_LAYER_BY_KIND.get(rule_kind, rule_kind)
            )
            payload_to_store["rule_vector"] = rule_vector
            payload_to_store["vector_dimension"] = vector_dimension
            payload_to_store["vector_version"] = vector_version
            payload_to_store.pop("vector", None)
            payload_to_store["required"] = [_clean_required_payload_item(item) for item in required_items]
            payload_to_store["produced"] = [_clean_produced_payload_item(item) for item in produced_items]
            if payload.get("execution_contract_version") == 2:
                # Child SQL tables are projections. Preserve the exact versioned
                # executable contract; legacy normalization changes thresholds
                # and injects fields, invalidating activation identities.
                payload_to_store["required"] = payload.get("required", [])
                payload_to_store["produced"] = payload.get("produced", [])
            payload_to_store["required_conditions"] = [_debug_text(item["condition"]) for item in required_items if _debug_text(item["condition"])]
            payload_to_store["required_relation_types"] = [item["relation"] for item in required_items if item["relation"]]
            payload_to_store["produced_conditions"] = [_debug_text(item["conclusion"]) for item in produced_items if _debug_text(item["conclusion"])]
            if effective_embedding_model:
                payload_to_store["embedding_model"] = effective_embedding_model
            conn.execute(
                """
                INSERT INTO rules(
                    rule_id, description, rule_vector, embedding_model, vector_dimension, vector_version,
                    rule_kind, complexity, priority, confidence,
                    logical_strength, empirical_support, contradiction_rate,
                    application_count, success_count, failure_count, support_count,
                    learn_from, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    description=excluded.description,
                    rule_vector=COALESCE(excluded.rule_vector, rules.rule_vector),
                    embedding_model=COALESCE(NULLIF(excluded.embedding_model, ''), rules.embedding_model),
                    vector_dimension=CASE WHEN excluded.rule_vector IS NOT NULL THEN excluded.vector_dimension ELSE rules.vector_dimension END,
                    vector_version=COALESCE(NULLIF(excluded.vector_version, ''), rules.vector_version),
                    rule_kind=excluded.rule_kind,
                    complexity=excluded.complexity,
                    priority=excluded.priority,
                    confidence=excluded.confidence,
                    logical_strength=excluded.logical_strength,
                    empirical_support=excluded.empirical_support,
                    contradiction_rate=excluded.contradiction_rate,
                    application_count=CASE WHEN ? THEN rules.application_count ELSE excluded.application_count END,
                    success_count=CASE WHEN ? THEN rules.success_count ELSE excluded.success_count END,
                    failure_count=CASE WHEN ? THEN rules.failure_count ELSE excluded.failure_count END,
                    support_count=CASE WHEN ? THEN rules.support_count ELSE excluded.support_count END,
                    learn_from=CASE WHEN ? THEN rules.learn_from ELSE excluded.learn_from END,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    rule_id,
                    description,
                    _vector_to_blob(rule_vector),
                    effective_embedding_model,
                    vector_dimension,
                    vector_version,
                    rule_kind,
                    float(payload.get("complexity", 0.5)),
                    float(payload.get("priority", 1.0)),
                    float(payload.get("confidence", payload.get("confidence_prior", 0.5))),
                    float(payload.get("logical_strength", 0.5)),
                    float(payload.get("empirical_support", 0.5)),
                    float(payload.get("contradiction_rate", 0.0)),
                    float(payload.get("application_count", 0.0)),
                    float(payload.get("success_count", 0.0)),
                    float(payload.get("failure_count", 0.0)),
                    int(payload.get("support_count", 0)),
                    _json_dumps(payload.get("learned_from", payload.get("learn_from", [])) or []),
                    _json_dumps(payload_to_store),
                    now,
                    now,
                    bool(preserve_stats),
                    bool(preserve_stats),
                    bool(preserve_stats),
                    bool(preserve_stats),
                    bool(preserve_stats),
                ),
            )
            conn.execute("DELETE FROM rule_tags WHERE rule_id = ?", (rule_id,))
            for source_field in ("tags", "scope_tags"):
                raw_tags = payload_to_store.get(source_field)
                tag_values = raw_tags if isinstance(raw_tags, (list, tuple, set)) else [raw_tags]
                for tag in dict.fromkeys(
                    str(value).strip().lower()
                    for value in tag_values
                    if str(value or "").strip()
                ):
                    conn.execute(
                        "INSERT INTO rule_tags(rule_id, tag, source_field) VALUES (?, ?, ?)",
                        (rule_id, tag, source_field),
                    )
            stored_vector = conn.execute(
                "SELECT rule_vector, embedding_model, vector_version FROM rules WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            if _rule_payload_runtime_active(payload_to_store):
                upsert_lsh_vector(
                    conn,
                    index_name=RULE_ANN_INDEX_NAME,
                    record_id=rule_id,
                    vector_field="rule_vector",
                    vector=stored_vector["rule_vector"] if stored_vector is not None else None,
                    embedding_model=str(stored_vector["embedding_model"] or "") if stored_vector is not None else "",
                    vector_version=str(stored_vector["vector_version"] or "") if stored_vector is not None else "",
                    now=now,
                )
            else:
                delete_lsh_record(conn, index_name=RULE_ANN_INDEX_NAME, record_id=rule_id)
            conn.execute("DELETE FROM required WHERE rule_id = ?", (rule_id,))
            for index, item in enumerate(required_items):
                conn.execute(
                    """
                    INSERT INTO required(
                        required_id, rule_id,
                        condition, condition_vector, condition_priority,
                        relation, relation_vector,
                        required_type, vector_query_text, vector_query_vector,
                        match_threshold, source_preference,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("required", rule_id, index, item.get("condition_text") or _debug_text(item["condition"]), item["relation"], item.get("required_type", "hard")),
                        rule_id,
                        item.get("condition_text") or _debug_text(item["condition"]),
                        _vector_to_blob(item.get("condition_vector")),
                        item["condition_priority"],
                        item["relation"],
                        _vector_to_blob(item.get("relation_vector")),
                        item.get("required_type", "hard"),
                        item.get("vector_query_text", ""),
                        _vector_to_blob(item.get("vector_query_vector")),
                        item["match_threshold"],
                        item["source_preference"],
                        now,
                        now,
                    ),
                )
                required_count += 1
            conn.execute("DELETE FROM produced WHERE rule_id = ?", (rule_id,))
            for index, item in enumerate(produced_items):
                conn.execute(
                    """
                    INSERT INTO produced(
                        produced_id, rule_id, conclusion, conclusion_vector, conclusion_priority,
                        produced_type, confidence_policy, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("produced", rule_id, index, item.get("conclusion_text") or _debug_text(item["conclusion"]), item["produced_type"]),
                        rule_id,
                        item.get("conclusion_text") or _debug_text(item["conclusion"]),
                        _vector_to_blob(item.get("conclusion_vector")),
                        item["conclusion_priority"],
                        item["produced_type"],
                        item["confidence_policy"],
                        now,
                        now,
                    ),
                )
                produced_count += 1
            saved += 1
        if owns_connection:
            conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "saved": saved,
            f"saved_{rule_kind}_rules": saved,
            f"saved_{rule_kind}_required": required_count,
            f"saved_{rule_kind}_produced": produced_count,
            "schema": SCHEMA_NAME,
        }
    except Exception:
        if owns_connection:
            conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()


def upsert_rule_content_records(
    db_path: str | Path,
    rules: Sequence[Any],
    *,
    rule_kind: str,
    replace_kind: bool = False,
    embedding_model: str = "",
) -> dict[str, Any]:
    return upsert_rule_records(
        db_path,
        rules,
        rule_kind=rule_kind,
        replace_kind=replace_kind,
        embedding_model=embedding_model,
        preserve_stats=True,
    )


def upsert_rule_content_transaction(
    db_path: str | Path,
    mutations: Sequence[Mapping[str, Any]],
    *,
    evolution_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply rule mutations and their evolution audit events atomically."""

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        summaries: list[dict[str, Any]] = []
        for mutation in mutations:
            records = mutation.get("rules")
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                raise TypeError("rule mutation rules must be a sequence")
            rule_kind = str(mutation.get("rule_kind") or "").strip()
            if not rule_kind:
                raise ValueError("rule mutation requires rule_kind")
            summaries.append(
                upsert_rule_records(
                    db_path,
                    records,
                    rule_kind=rule_kind,
                    replace_kind=bool(mutation.get("replace_kind", False)),
                    embedding_model=str(mutation.get("embedding_model") or ""),
                    preserve_stats=bool(mutation.get("preserve_stats", True)),
                    _connection=conn,
                )
            )
        event_count = _upsert_rule_evolution_events_with_connection(
            conn,
            evolution_events,
            require_proposal_id=True,
        )
        conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "transactional": True,
            "mutation_count": len(summaries),
            "evolution_event_count": event_count,
            "summaries": summaries,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_rule_usage_stats(
    db_path: str | Path,
    updates: Sequence[Mapping[str, Any]],
    *,
    mode: str = "increment",
) -> dict[str, Any]:
    """Update runtime counters without changing rule content or child tables."""

    normalized_mode = str(mode or "increment").strip().lower()
    if normalized_mode not in {"increment", "set"}:
        raise ValueError("mode must be 'increment' or 'set'")
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        updated = 0
        now = _now_iso()
        for item in updates:
            rule_id = str(item.get("rule_id") or item.get("id") or "").strip()
            if not rule_id:
                continue
            application_count = float(item.get("application_count", 0.0) or 0.0)
            success_count = float(item.get("success_count", 0.0) or 0.0)
            failure_count = float(item.get("failure_count", 0.0) or 0.0)
            support_count = int(item.get("support_count", 0) or 0)
            learned_from = item.get("learned_from", item.get("learn_from"))
            if normalized_mode == "set":
                cursor = conn.execute(
                    """
                    UPDATE rules
                    SET application_count = ?,
                        success_count = ?,
                        failure_count = ?,
                        support_count = ?,
                        learn_from = COALESCE(?, learn_from),
                        updated_at = ?
                    WHERE rule_id = ?
                    """,
                    (
                        application_count,
                        success_count,
                        failure_count,
                        support_count,
                        _json_dumps(learned_from) if learned_from is not None else None,
                        now,
                        rule_id,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE rules
                    SET application_count = application_count + ?,
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        support_count = support_count + ?,
                        updated_at = ?
                    WHERE rule_id = ?
                    """,
                    (
                        application_count,
                        success_count,
                        failure_count,
                        support_count,
                        now,
                        rule_id,
                    ),
                )
            updated += int(cursor.rowcount or 0)
        conn.commit()
        return {"db_path": str(Path(db_path).expanduser()), "updated_rule_stats": updated, "mode": normalized_mode}
    finally:
        conn.close()



def update_rule_shadow_stats(
    db_path: str | Path,
    updates: Sequence[Mapping[str, Any]],
    *,
    mode: str = "increment",
) -> dict[str, Any]:
    """Persist candidate evidence without changing formal runtime counters."""

    normalized_mode = str(mode or "increment").strip().lower()
    if normalized_mode not in {"increment", "set"}:
        raise ValueError("mode must be 'increment' or 'set'")
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        now = _now_iso()
        saved = 0
        skipped_missing_rules: list[str] = []
        numeric_fields = (
            "application_count",
            "success_count",
            "failure_count",
            "semantic_success_count",
            "semantic_failure_count",
            "path_success_count",
            "path_failure_count",
            "effective_evidence_weight",
        )
        for raw in updates:
            if not isinstance(raw, Mapping):
                continue
            update = dict(raw)
            rule_id = str(update.get("rule_id") or update.get("id") or "").strip()
            if not rule_id:
                continue
            if conn.execute("SELECT 1 FROM rules WHERE rule_id = ?", (rule_id,)).fetchone() is None:
                skipped_missing_rules.append(rule_id)
                continue
            existing = conn.execute(
                "SELECT * FROM rule_shadow_stats WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            old = dict(existing) if existing is not None else {}
            created_at = str(old.get("created_at") or now)
            values: dict[str, float] = {}
            for field in numeric_fields:
                previous = float(old.get(field, 0.0) or 0.0)
                incoming = float(update.get(field, 0.0) or 0.0)
                if normalized_mode == "set" and field not in update:
                    values[field] = previous
                else:
                    values[field] = incoming if normalized_mode == "set" else previous + incoming

            previous_stats = _json_loads(str(old.get("stats_json") or "{}"), {})
            stats = dict(previous_stats) if isinstance(previous_stats, Mapping) else {}
            incoming_stats = update.get("stats") if isinstance(update.get("stats"), Mapping) else {}
            stats.update(dict(incoming_stats))
            stats.update(values)
            applications = max(0.0, values["application_count"])
            stats["success_rate"] = values["success_count"] / applications if applications else 0.0
            stats["semantic_success_rate"] = (
                values["semantic_success_count"] / applications if applications else 0.0
            )
            stats["path_success_rate"] = (
                values["path_success_count"] / applications if applications else 0.0
            )
            last_case_id = str(
                update.get("last_case_id")
                or update.get("case_id")
                or old.get("last_case_id")
                or ""
            )
            last_status = str(
                update.get("last_status")
                or update.get("status")
                or old.get("last_status")
                or ""
            )
            conn.execute(
                """
                INSERT INTO rule_shadow_stats (
                    rule_id, application_count, success_count, failure_count,
                    semantic_success_count, semantic_failure_count,
                    path_success_count, path_failure_count, effective_evidence_weight,
                    last_case_id, last_status, stats_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    application_count=excluded.application_count,
                    success_count=excluded.success_count,
                    failure_count=excluded.failure_count,
                    semantic_success_count=excluded.semantic_success_count,
                    semantic_failure_count=excluded.semantic_failure_count,
                    path_success_count=excluded.path_success_count,
                    path_failure_count=excluded.path_failure_count,
                    effective_evidence_weight=excluded.effective_evidence_weight,
                    last_case_id=excluded.last_case_id,
                    last_status=excluded.last_status,
                    stats_json=excluded.stats_json,
                    updated_at=excluded.updated_at
                """,
                (
                    rule_id,
                    values["application_count"],
                    values["success_count"],
                    values["failure_count"],
                    values["semantic_success_count"],
                    values["semantic_failure_count"],
                    values["path_success_count"],
                    values["path_failure_count"],
                    values["effective_evidence_weight"],
                    last_case_id,
                    last_status,
                    _json_dumps(stats),
                    created_at,
                    now,
                ),
            )

            aggregate = conn.execute(
                "SELECT * FROM rule_shadow_stats WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            rule_row = conn.execute(
                "SELECT payload_json FROM rules WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            if aggregate is not None and rule_row is not None:
                payload = _json_loads(rule_row["payload_json"], {})
                if not isinstance(payload, dict):
                    payload = {}
                applications = float(aggregate["application_count"] or 0.0)
                successes = float(aggregate["success_count"] or 0.0)
                failures = float(aggregate["failure_count"] or 0.0)
                semantic_successes = float(aggregate["semantic_success_count"] or 0.0)
                semantic_failures = float(aggregate["semantic_failure_count"] or 0.0)
                path_successes = float(aggregate["path_success_count"] or 0.0)
                path_failures = float(aggregate["path_failure_count"] or 0.0)
                payload["shadow_validation_stats"] = {
                    "application_count": applications,
                    "success_count": successes,
                    "failure_count": failures,
                    "semantic_success_count": semantic_successes,
                    "semantic_failure_count": semantic_failures,
                    "path_success_count": path_successes,
                    "path_failure_count": path_failures,
                    "effective_evidence_weight": float(aggregate["effective_evidence_weight"] or 0.0),
                    "success_beta_alpha": successes + 1.0,
                    "success_beta_beta": failures + 1.0,
                    "posterior_mean_success": (successes + 1.0) / (successes + failures + 2.0),
                    "semantic_success_rate": (semantic_successes / applications) if applications > 0 else 0.0,
                    "path_success_rate": (path_successes / applications) if applications > 0 else 0.0,
                    "last_case_id": str(aggregate["last_case_id"] or ""),
                    "last_status": str(aggregate["last_status"] or ""),
                    "updated_at": str(aggregate["updated_at"] or now),
                    "evidence_scope": "shadow_path_outcome_not_causal_credit",
                }
                conn.execute(
                    "UPDATE rules SET payload_json = ?, updated_at = ? WHERE rule_id = ?",
                    (_json_dumps(payload), now, rule_id),
                )
            saved += 1
        conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "saved": saved,
            "updated_rule_shadow_stats": saved,
            "mode": normalized_mode,
            "table": "rule_shadow_stats",
            "skipped_missing_rule_ids": sorted(set(skipped_missing_rules)),
        }
    finally:
        conn.close()


def load_rule_shadow_stats(
    db_path: str | Path,
    *,
    rule_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        params: list[Any] = []
        where = ""
        if rule_ids:
            ids = [str(item) for item in rule_ids if str(item)]
            if ids:
                where = " WHERE rule_id IN (" + ",".join("?" for _ in ids) + ")"
                params.extend(ids)
        rows = conn.execute(
            "SELECT * FROM rule_shadow_stats" + where + " ORDER BY rule_id",
            params,
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["stats"] = _json_loads(str(item.get("stats_json") or "{}"), {})
            result.append(item)
        return result
    finally:
        conn.close()


def upsert_rule_family_states(
    db_path: str | Path,
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist the latest evidence state for each learned rule family."""

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        now = _now_iso()
        saved = 0
        for raw in states:
            if not isinstance(raw, Mapping):
                continue
            state = dict(raw)
            family_key = str(state.get("family_key") or "").strip()
            if not family_key:
                continue
            target_ids = [str(item) for item in state.get("target_rule_ids", []) or [] if str(item)]
            potential = state.get("rule_potential") if isinstance(state.get("rule_potential"), Mapping) else {}
            existing = conn.execute(
                "SELECT created_at FROM rule_family_state WHERE family_key = ?",
                (family_key,),
            ).fetchone()
            created_at = str(existing["created_at"] or now) if existing is not None else now
            conn.execute(
                """
                INSERT INTO rule_family_state(
                    family_key, domain, rule_layer, target_rule_id, maturity,
                    potential_score, effective_observations, state_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(family_key) DO UPDATE SET
                    domain=excluded.domain,
                    rule_layer=excluded.rule_layer,
                    target_rule_id=excluded.target_rule_id,
                    maturity=excluded.maturity,
                    potential_score=excluded.potential_score,
                    effective_observations=excluded.effective_observations,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    family_key,
                    str(state.get("domain") or "general"),
                    str(state.get("rule_layer") or "domain"),
                    target_ids[0] if target_ids else "",
                    str(state.get("maturity") or ""),
                    float(potential.get("score", 0.0) or 0.0),
                    float(state.get("effective_observations", 0.0) or 0.0),
                    _json_dumps(state),
                    created_at,
                    now,
                ),
            )
            saved += 1
        conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "saved": saved,
            "table": "rule_family_state",
        }
    finally:
        conn.close()


def _upsert_rule_evolution_events_with_connection(
    conn: sqlite3.Connection,
    proposals: Sequence[Mapping[str, Any]],
    *,
    require_proposal_id: bool = False,
) -> int:
    now = _now_iso()
    saved = 0
    for raw in proposals:
        if not isinstance(raw, Mapping):
            if require_proposal_id:
                raise TypeError("rule evolution event must be a mapping")
            continue
        proposal = dict(raw)
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        if not proposal_id:
            if require_proposal_id:
                raise ValueError("rule evolution event requires proposal_id")
            continue
        existing = conn.execute(
            "SELECT created_at FROM rule_evolution_events WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        created_at = str(existing["created_at"] or now) if existing is not None else now
        conn.execute(
            """
            INSERT INTO rule_evolution_events(
                proposal_id, family_key, mutation_type, status,
                source_rule_ids_json, candidate_rule_ids_json, proposal_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                family_key=excluded.family_key,
                mutation_type=excluded.mutation_type,
                status=excluded.status,
                source_rule_ids_json=excluded.source_rule_ids_json,
                candidate_rule_ids_json=excluded.candidate_rule_ids_json,
                proposal_json=excluded.proposal_json,
                updated_at=excluded.updated_at
            """,
            (
                proposal_id,
                str(proposal.get("family_key") or ""),
                str(proposal.get("mutation_type") or ""),
                str(proposal.get("status") or "pending_validation"),
                _json_dumps(proposal.get("source_rule_ids") or []),
                _json_dumps(proposal.get("candidate_rule_ids") or []),
                _json_dumps(proposal),
                created_at,
                now,
            ),
        )
        saved += 1
    return saved


def upsert_rule_evolution_events(
    db_path: str | Path,
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist deterministic rule-evolution proposals as an auditable log."""

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        saved = _upsert_rule_evolution_events_with_connection(conn, proposals)
        conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "saved": saved,
            "table": "rule_evolution_events",
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_rule_family_states(
    db_path: str | Path,
    *,
    family_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        params: list[Any] = []
        where = ""
        keys = [str(item) for item in family_keys or [] if str(item)]
        if keys:
            where = " WHERE family_key IN (" + ",".join("?" for _ in keys) + ")"
            params.extend(keys)
        rows = conn.execute(
            "SELECT state_json FROM rule_family_state" + where + " ORDER BY family_key",
            params,
        ).fetchall()
        return [
            state
            for row in rows
            for state in [_json_loads(row["state_json"], {})]
            if isinstance(state, dict)
        ]
    finally:
        conn.close()


def load_rule_evolution_events(
    db_path: str | Path,
    *,
    proposal_ids: Sequence[str] | None = None,
    family_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        clauses: list[str] = []
        params: list[Any] = []
        ids = [str(item) for item in proposal_ids or [] if str(item)]
        families = [str(item) for item in family_keys or [] if str(item)]
        if ids:
            clauses.append("proposal_id IN (" + ",".join("?" for _ in ids) + ")")
            params.extend(ids)
        if families:
            clauses.append("family_key IN (" + ",".join("?" for _ in families) + ")")
            params.extend(families)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = conn.execute(
            "SELECT proposal_json FROM rule_evolution_events" + where + " ORDER BY updated_at, proposal_id",
            params,
        ).fetchall()
        return [
            proposal
            for row in rows
            for proposal in [_json_loads(row["proposal_json"], {})]
            if isinstance(proposal, dict)
        ]
    finally:
        conn.close()


def sync_rule_learning_evidence_stats(
    db_path: str | Path,
    updates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist effective learning support without changing runtime counters.

    Auto-Learn collapses correlated observations before this call. The supplied
    support count is therefore an absolute effective-evidence count, not an
    increment. Rule content upserts intentionally preserve all statistics, so
    this separate interface is the only path that may synchronize learning
    support after a content update.
    """

    requested: dict[str, int] = {}
    for item in updates:
        rule_id = str(item.get("rule_id") or item.get("id") or "").strip()
        if not rule_id:
            continue
        support_count = int(item.get("support_count", 0) or 0)
        if support_count < 0:
            raise ValueError(f"support_count must be non-negative for {rule_id}")
        requested[rule_id] = support_count
    if not requested:
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "updated_rule_learning_evidence_stats": 0,
        }

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        now = _now_iso()
        updated = 0
        for rule_id, support_count in requested.items():
            row = conn.execute(
                "SELECT payload_json FROM rules WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"cannot synchronize learning evidence for missing rule {rule_id}")
            payload = _json_loads(row["payload_json"], {})
            if not isinstance(payload, dict):
                payload = {}
            payload["support_count"] = support_count
            cursor = conn.execute(
                """
                UPDATE rules
                SET support_count = ?, payload_json = ?, updated_at = ?
                WHERE rule_id = ?
                """,
                (support_count, _json_dumps(payload), now, rule_id),
            )
            updated += int(cursor.rowcount or 0)
        conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "updated_rule_learning_evidence_stats": updated,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_rule_learning_parameters(
    db_path: str | Path,
    updates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Update learned confidence, priority, and boundaries without changing rule logic."""

    requested = {
        str(item.get("rule_id") or item.get("id") or "").strip(): dict(item)
        for item in updates
        if isinstance(item, Mapping)
        and str(item.get("rule_id") or item.get("id") or "").strip()
    }
    if not requested:
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "updated_rule_learning_parameters": 0,
        }
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        current = _load_rule_payloads(
            conn,
            rule_ids=list(requested),
            limit=None,
        )
        mutations_by_kind: dict[str, list[dict[str, Any]]] = {}
        for existing in current:
            rule_id = str(existing.get("rule_id") or "")
            update = requested.get(rule_id)
            if update is None:
                continue
            merged = dict(existing)
            for field in (
                "confidence",
                "priority",
                "logical_strength",
                "empirical_support",
                "contradiction_rate",
                "reliability_posterior",
                "selection_posterior",
                "parameter_learning_history",
            ):
                if field in update:
                    merged[field] = update[field]
            merged["required"] = _merge_learned_required_boundaries(
                existing.get("required"),
                update.get("required"),
            )
            kind = str(existing.get("rule_kind") or "content")
            mutations_by_kind.setdefault(kind, []).append(merged)
        summaries = []
        for rule_kind, records in mutations_by_kind.items():
            summaries.append(
                upsert_rule_records(
                    db_path,
                    records,
                    rule_kind=rule_kind,
                    replace_kind=False,
                    preserve_stats=True,
                    _connection=conn,
                )
            )
        conn.commit()
        updated = sum(int(item.get("saved") or 0) for item in summaries)
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "updated_rule_learning_parameters": updated,
            "summaries": summaries,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _merge_learned_required_boundaries(current: Any, update: Any) -> list[dict[str, Any]]:
    current_items = [dict(item) for item in current or [] if isinstance(item, Mapping)]
    update_items = [dict(item) for item in update or [] if isinstance(item, Mapping)]
    by_id = {
        str(item.get("condition_id") or item.get("required_id") or ""): item
        for item in update_items
        if str(item.get("condition_id") or item.get("required_id") or "")
    }
    for index, item in enumerate(current_items):
        key = str(item.get("condition_id") or item.get("required_id") or "")
        source = by_id.get(key) if key else update_items[index] if index < len(update_items) else None
        if not isinstance(source, Mapping):
            continue
        for field in ("match_threshold", "boundary_learning"):
            if field in source:
                item[field] = source[field]
        source_relations = [
            dict(relation)
            for relation in source.get("relation_patterns", []) or []
            if isinstance(relation, Mapping)
        ]
        if not source_relations:
            continue
        relation_by_id = {
            str(relation.get("relation_id") or ""): relation
            for relation in source_relations
            if str(relation.get("relation_id") or "")
        }
        current_relations = [
            dict(relation)
            for relation in item.get("relation_patterns", []) or []
            if isinstance(relation, Mapping)
        ]
        for relation_index, relation in enumerate(current_relations):
            relation_id = str(relation.get("relation_id") or "")
            learned = (
                relation_by_id.get(relation_id)
                if relation_id
                else source_relations[relation_index]
                if relation_index < len(source_relations)
                else None
            )
            if not isinstance(learned, Mapping):
                continue
            for field in ("match_threshold", "boundary_learning"):
                if field in learned:
                    relation[field] = learned[field]
        item["relation_patterns"] = current_relations
    return current_items


def _load_rule_payloads(
    conn: sqlite3.Connection,
    *,
    rule_kind: str | None = None,
    rule_layer: str | None = None,
    rule_ids: Sequence[str] | None = None,
    embedding_model: str | None = None,
    vector_dimension: int | None = None,
    vector_version: str | None = None,
    min_priority: float | None = None,
    min_confidence: float | None = None,
    max_contradiction_rate: float | None = None,
    runtime_active_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    sql = "SELECT * FROM rules"
    clauses: list[str] = []
    if rule_kind:
        clauses.append("rule_kind = ?")
        params.append(rule_kind)
    if rule_layer:
        normalized_layer = normalize_rule_layer(rule_layer)
        compatible_layers = [normalized_layer]
        clauses.append(
            "LOWER(COALESCE(json_extract(payload_json, '$.rule_layer'), '')) "
            f"IN ({','.join('?' for _ in compatible_layers)})"
        )
        params.extend(compatible_layers)
    normalized_ids = [str(rule_id) for rule_id in (rule_ids or []) if str(rule_id)]
    if rule_ids is not None and not normalized_ids:
        return []
    if normalized_ids:
        clauses.append(f"rule_id IN ({','.join('?' for _ in normalized_ids)})")
        params.extend(normalized_ids)
    if embedding_model:
        clauses.append("embedding_model = ?")
        params.append(canonical_embedding_model(embedding_model))
    if vector_dimension is not None:
        clauses.append("vector_dimension = ?")
        params.append(int(vector_dimension))
    if vector_version:
        clauses.append("vector_version = ?")
        params.append(str(vector_version))
    if min_priority is not None:
        clauses.append("priority >= ?")
        params.append(float(min_priority))
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(float(min_confidence))
    if max_contradiction_rate is not None:
        clauses.append("contradiction_rate <= ?")
        params.append(float(max_contradiction_rate))
    if runtime_active_only:
        clauses.append(_runtime_active_rule_sql())
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY priority DESC, confidence DESC, rule_id"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    selected_ids = [str(row["rule_id"]) for row in rows]
    required_by_rule: dict[str, list[sqlite3.Row]] = {rule_id: [] for rule_id in selected_ids}
    produced_by_rule: dict[str, list[sqlite3.Row]] = {rule_id: [] for rule_id in selected_ids}
    if selected_ids:
        placeholders = ",".join("?" for _ in selected_ids)
        for child in conn.execute(
            f"SELECT * FROM required WHERE rule_id IN ({placeholders}) ORDER BY rule_id, condition_priority DESC, required_id",
            selected_ids,
        ).fetchall():
            required_by_rule.setdefault(str(child["rule_id"]), []).append(child)
        for child in conn.execute(
            f"SELECT * FROM produced WHERE rule_id IN ({placeholders}) ORDER BY rule_id, conclusion_priority DESC, produced_id",
            selected_ids,
        ).fetchall():
            produced_by_rule.setdefault(str(child["rule_id"]), []).append(child)
    return [
        _row_to_payload(
            row,
            required_by_rule.get(str(row["rule_id"]), []),
            produced_by_rule.get(str(row["rule_id"]), []),
        )
        for row in rows
    ]


def load_rule_records(
    db_path: str | Path,
    *,
    rule_kind: str | None = None,
    rule_layer: str | None = None,
    rule_ids: Sequence[str] | None = None,
    embedding_model: str | None = None,
    vector_dimension: int | None = None,
    vector_version: str | None = None,
    min_priority: float | None = None,
    min_confidence: float | None = None,
    max_contradiction_rate: float | None = None,
    runtime_active_only: bool = False,
    limit: int | None = 4096,
) -> list[dict[str, Any]]:
    path = Path(db_path).expanduser()
    existed = path.exists()
    conn = _connect_readonly_or_create(path)
    try:
        if not existed:
            _init_schema(conn)
        return _load_rule_payloads(
            conn,
            rule_kind=rule_kind,
            rule_layer=rule_layer,
            rule_ids=rule_ids,
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
            vector_version=vector_version,
            min_priority=min_priority,
            min_confidence=min_confidence,
            max_contradiction_rate=max_contradiction_rate,
            runtime_active_only=runtime_active_only,
            limit=limit,
        )
    finally:
        conn.close()


def load_rule_ids_by_tags(
    db_path: str | Path,
    *,
    required_tags: Sequence[str] | None = None,
    required_any_tags: Sequence[str] | None = None,
    excluded_tags: Sequence[str] | None = None,
    rule_layer: str | None = None,
    activation_policies: Sequence[str] | None = None,
    repair_target_rule_ids: Sequence[str] | None = None,
    missing_structure_signatures: Sequence[str] | None = None,
    repair_family_keys: Sequence[str] | None = None,
    runtime_active_only: bool = True,
) -> list[str]:
    """Resolve structural selector tags before ANN candidate retrieval."""

    required = _normalized_rule_tags(required_tags)
    required_any = _normalized_rule_tags(required_any_tags)
    excluded = _normalized_rule_tags(excluded_tags)
    activation_policies = [str(value).strip().lower() for value in (activation_policies or []) if str(value).strip()]
    repair_target_rule_ids = [str(value).strip() for value in (repair_target_rule_ids or []) if str(value).strip()]
    missing_structure_signatures = [str(value).strip() for value in (missing_structure_signatures or []) if str(value).strip()]
    repair_family_keys = [str(value).strip() for value in (repair_family_keys or []) if str(value).strip()]
    conn = _connect(Path(db_path).expanduser())
    try:
        _init_schema(conn)
        params: list[Any] = []
        clauses: list[str] = []
        if rule_layer:
            clauses.append("LOWER(COALESCE(json_extract(r.payload_json, '$.rule_layer'), '')) = ?")
            params.append(normalize_rule_layer(rule_layer))
        if runtime_active_only:
            clauses.append(_runtime_active_rule_sql("r"))
        activation_paths = {
            "activation_policies": "LOWER(COALESCE(json_extract(r.payload_json, '$.runtime_activation.policy'), json_extract(r.payload_json, '$.metadata.repair_activation_policy'), ''))",
            "repair_target_rule_ids": "COALESCE(json_extract(r.payload_json, '$.runtime_activation.target_rule_id'), json_extract(r.payload_json, '$.metadata.repair_target_rule_id'), '')",
            "missing_structure_signatures": "COALESCE(json_extract(r.payload_json, '$.runtime_activation.missing_structure_signature'), json_extract(r.payload_json, '$.metadata.missing_structure_signature'), '')",
            "repair_family_keys": "COALESCE(json_extract(r.payload_json, '$.runtime_activation.repair_family_key'), json_extract(r.payload_json, '$.metadata.repair_family_key'), '')",
        }
        for field, values in (
            ("activation_policies", activation_policies),
            ("repair_target_rule_ids", repair_target_rule_ids),
            ("missing_structure_signatures", missing_structure_signatures),
            ("repair_family_keys", repair_family_keys),
        ):
            if values:
                clauses.append(f"{activation_paths[field]} IN ({','.join('?' for _ in values)})")
                params.extend(values)
        if required:
            clauses.append(
                "(SELECT COUNT(DISTINCT rt.tag) FROM rule_tags rt "
                f"WHERE rt.rule_id = r.rule_id AND rt.tag IN ({','.join('?' for _ in required)})) = ?"
            )
            params.extend(required)
            params.append(len(required))
        if required_any:
            clauses.append(
                "EXISTS (SELECT 1 FROM rule_tags rt WHERE rt.rule_id = r.rule_id "
                f"AND rt.tag IN ({','.join('?' for _ in required_any)}))"
            )
            params.extend(required_any)
        if excluded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM rule_tags rt WHERE rt.rule_id = r.rule_id "
                f"AND rt.tag IN ({','.join('?' for _ in excluded)}))"
            )
            params.extend(excluded)
        sql = "SELECT r.rule_id FROM rules r"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY r.rule_id"
        return [str(row["rule_id"]) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def delete_rule_records(
    db_path: str | Path,
    rule_ids: Sequence[str],
) -> dict[str, Any]:
    """Delete rules and every persisted child/index record in one transaction."""

    normalized_ids = list(dict.fromkeys(str(rule_id) for rule_id in rule_ids if str(rule_id)))
    if not normalized_ids:
        return {"status": "completed", "deleted": 0, "rule_ids": []}
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in normalized_ids)
        existing = [
            str(row["rule_id"])
            for row in conn.execute(
                f"SELECT rule_id FROM rules WHERE rule_id IN ({placeholders})",
                normalized_ids,
            ).fetchall()
        ]
        for rule_id in existing:
            delete_lsh_record(conn, index_name=RULE_ANN_INDEX_NAME, record_id=rule_id)
        conn.execute(f"DELETE FROM rules WHERE rule_id IN ({placeholders})", normalized_ids)
        conn.commit()
        return {"status": "completed", "deleted": len(existing), "rule_ids": existing}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _normalized_rule_tags(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    raw = values if isinstance(values, (list, tuple, set)) else [values]
    return list(
        dict.fromkeys(str(value).strip().lower() for value in raw if str(value or "").strip())
    )


def _collect_search_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, Mapping):
        for key in (
            "rule_id",
            "description",
            "debug_text",
            "core",
            "relation",
            "nature",
            "condition_text",
            "conclusion_text",
            "produced_type",
            "confidence_policy",
            "source_preference",
            "operation",
        ):
            text = str(value.get(key) or "").strip()
            if text:
                texts.append(text)
        for key in ("required", "produced", "condition", "conclusion", "relation_patterns", "metadata", "scope_tags"):
            if key in value:
                texts.extend(_collect_search_texts(value.get(key)))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_collect_search_texts(item))
    elif value is not None:
        text = str(value).strip()
        if text:
            texts.append(text)
    return texts


def _collect_search_vectors(value: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    if isinstance(value, Mapping):
        for key in ("rule_vector", "core_vector", "relation_vector", "nature_vector"):
            vector = _vector_to_list(value.get(key))
            if vector:
                vectors.append(vector)
        for key in ("required", "produced", "condition", "conclusion", "relation_patterns", "metadata"):
            if key in value:
                vectors.extend(_collect_search_vectors(value.get(key)))
    elif isinstance(value, list):
        for item in value:
            vectors.extend(_collect_search_vectors(item))
    return vectors


def search_rule_records(
    db_path: str | Path,
    *,
    query: str = "",
    keys: Sequence[str] | None = None,
    query_vector: Any = None,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    rule_kind: str | None = None,
    runtime_active_only: bool = True,
    top_k: int = 256,
) -> list[dict[str, Any]]:
    """Run an exact vector maintenance query.

    Runtime reasoning uses ``search_rule_ann``. Text and keys are accepted only
    for source compatibility and never participate in model retrieval.
    """

    del query, keys
    qvec = _vector_to_list(query_vector)
    if not qvec:
        raise ValueError("rule search requires query_vector")
    path = Path(db_path).expanduser()
    existed = path.exists()
    conn = _connect_readonly_or_create(path)
    try:
        if not existed:
            _init_schema(conn)
        payloads = _load_rule_payloads(
            conn,
            rule_kind=rule_kind,
            runtime_active_only=runtime_active_only,
            limit=None,
        )
        query_dimension = len(qvec or [])
        query_model = str(query_embedding_model or "").strip()
        query_version = str(query_vector_version or "").strip()
        ranked: list[tuple[float, dict[str, Any]]] = []
        for payload in payloads:
            vector_candidates = _collect_search_vectors(payload)
            vector_contract_ok = True
            if qvec:
                stored_dimension = int(payload.get("vector_dimension") or 0)
                stored_model = str(payload.get("embedding_model") or "").strip()
                stored_version = str(payload.get("vector_version") or "").strip()
                if stored_dimension and stored_dimension != query_dimension:
                    vector_contract_ok = False
                if query_model and stored_model and stored_model != query_model:
                    vector_contract_ok = False
                if query_version and stored_version and stored_version != query_version:
                    vector_contract_ok = False
            vector_score = (
                max((_cosine(qvec, vector) for vector in vector_candidates if vector is not None), default=0.0)
                if vector_contract_ok
                else 0.0
            )
            payload["vector_score"] = vector_score
            if not vector_contract_ok:
                payload["vector_status"] = "vector_contract_mismatch"
                continue
            else:
                payload["vector_status"] = "vector_matched" if vector_score > 0.0 else "vector_unmatched"
            if vector_score <= 0.0:
                continue
            ranked.append((vector_score, payload))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in ranked[: max(int(top_k), 1)]]
    finally:
        conn.close()


def build_rule_ann_index(
    db_path: str | Path,
    *,
    rule_kind: str | None = None,
    embedding_model: str = "",
    vector_version: str = "",
    bits: int = 12,
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        clauses = ["rule_vector IS NOT NULL", _runtime_active_rule_sql()]
        params: list[Any] = []
        if rule_kind:
            clauses.append("rule_kind = ?")
            params.append(str(rule_kind))
        rows = conn.execute(f"SELECT * FROM rules WHERE {' AND '.join(clauses)}", params).fetchall()
        ann_rows = []
        for row in rows:
            vector = _blob_to_vector(row["rule_vector"])
            if not vector:
                continue
            ann_rows.append(
                {
                    "record_id": str(row["rule_id"]),
                    "vector_field": "rule_vector",
                    "vector": vector,
                    "embedding_model": str(row["embedding_model"] or embedding_model or ""),
                    "vector_version": str(row["vector_version"] or vector_version or ""),
                }
            )
        summary = rebuild_lsh_index(
            conn,
            index_name=RULE_ANN_INDEX_NAME,
            rows=ann_rows,
            embedding_model=embedding_model,
            vector_version=vector_version,
            now=_now_iso(),
            bits=bits,
        )
        conn.commit()
        return {**summary, "record_count": len(rows), "db_path": str(Path(db_path).expanduser())}
    finally:
        conn.close()


def ensure_rule_ann_index(db_path: str | Path, *, bits: int = 12) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        expected = int(
            conn.execute(
                f"SELECT COUNT(*) AS count FROM rules WHERE rule_vector IS NOT NULL AND {_runtime_active_rule_sql()}"
            ).fetchone()["count"]
        )
        current = ann_index_summary(conn, index_name=RULE_ANN_INDEX_NAME)
    finally:
        conn.close()
    if int(current["indexed_vectors"]) != expected:
        return {**build_rule_ann_index(db_path, bits=bits), "rebuilt": True}
    return {**current, "record_count": expected, "db_path": str(Path(db_path).expanduser()), "rebuilt": False}


def repair_rule_vector_contract(
    db_path: str | Path,
    *,
    embedding_model: str,
    vector_version: str,
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        expected_model = canonical_embedding_model(embedding_model)
        rows = conn.execute(
            "SELECT rule_id, rule_vector, embedding_model, vector_version, payload_json FROM rules WHERE rule_vector IS NOT NULL"
        ).fetchall()
        model_mismatches: list[str] = []
        version_mismatches: list[str] = []
        updates: list[tuple[str, int, str, str, str]] = []
        for row in rows:
            vector = _blob_to_vector(row["rule_vector"])
            if not vector:
                continue
            stored_model = canonical_embedding_model(row["embedding_model"])
            if stored_model and expected_model and stored_model != expected_model:
                model_mismatches.append(str(row["rule_id"]))
                continue
            stored_version = str(row["vector_version"] or "").strip()
            expected_version = str(vector_version or "").strip()
            if stored_version and expected_version and stored_version != expected_version:
                version_mismatches.append(str(row["rule_id"]))
                continue
            model = stored_model or expected_model
            version = stored_version or expected_version
            payload = dict(_json_loads(row["payload_json"], {}))
            payload["embedding_model"] = model
            payload["vector_dimension"] = len(vector)
            payload["vector_version"] = version
            updates.append((model, len(vector), version, _json_dumps(payload), str(row["rule_id"])))
        if model_mismatches or version_mismatches:
            return {
                "ok": False,
                "status": "vector_contract_mismatch",
                "embedding_model_mismatch_rule_ids": model_mismatches,
                "vector_version_mismatch_rule_ids": version_mismatches,
            }
        conn.executemany(
            """
            UPDATE rules
            SET embedding_model = ?, vector_dimension = ?, vector_version = ?, payload_json = ?
            WHERE rule_id = ?
            """,
            updates,
        )
        conn.commit()
    finally:
        conn.close()
    ann = build_rule_ann_index(db_path)
    return {"ok": True, "status": "completed", "repaired_records": len(updates), "ann": ann}


def search_rule_ann(
    db_path: str | Path,
    *,
    query_vector: Any,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    top_k: int = 64,
    candidate_limit: int = 256,
    allowed_rule_ids: Sequence[str] | None = None,
    runtime_active_only: bool = True,
) -> list[dict[str, Any]]:
    path = Path(db_path).expanduser()
    existed = path.exists()
    conn = _connect_readonly_or_create(path)
    try:
        if not existed:
            _init_schema(conn)
        ann = query_lsh_index(
            conn,
            index_name=RULE_ANN_INDEX_NAME,
            query_vector=query_vector,
            embedding_model=query_embedding_model,
            vector_version=query_vector_version,
            candidate_limit=candidate_limit,
            allowed_record_ids=allowed_rule_ids,
        )
        if not ann["ok"] or not ann["record_ids"]:
            return []
        payloads = _load_rule_payloads(
            conn,
            rule_ids=ann["record_ids"],
            runtime_active_only=runtime_active_only,
            limit=None,
        )
        by_id = {str(payload.get("rule_id")): payload for payload in payloads}
        ranked = [by_id[rule_id] for rule_id in ann["record_ids"] if rule_id in by_id]
        for payload in ranked:
            payload["vector_score"] = float(ann["scores"].get(str(payload.get("rule_id")), 0.0))
            payload["vector_status"] = "ann_bucket_match"
        return ranked[: max(1, int(top_k))]
    finally:
        conn.close()


def summarize_database(db_path: str | Path) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        rules = int(conn.execute("SELECT COUNT(*) AS count FROM rules").fetchone()["count"])
        required = int(conn.execute("SELECT COUNT(*) AS count FROM required").fetchone()["count"])
        produced = int(conn.execute("SELECT COUNT(*) AS count FROM produced").fetchone()["count"])
        family_states = int(conn.execute("SELECT COUNT(*) AS count FROM rule_family_state").fetchone()["count"])
        evolution_events = int(conn.execute("SELECT COUNT(*) AS count FROM rule_evolution_events").fetchone()["count"])
        shadow_stats = int(conn.execute("SELECT COUNT(*) AS count FROM rule_shadow_stats").fetchone()["count"])
        vectors = int(
            conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM rules WHERE rule_vector IS NOT NULL) +
                  (SELECT COUNT(*) FROM required WHERE condition_vector IS NOT NULL) +
                  (SELECT COUNT(*) FROM required WHERE relation_vector IS NOT NULL) +
                  (SELECT COUNT(*) FROM required WHERE vector_query_vector IS NOT NULL) +
                  (SELECT COUNT(*) FROM produced WHERE conclusion_vector IS NOT NULL)
                  AS count
                """
            ).fetchone()["count"]
        )
        return {
            "db_path": str(Path(db_path).expanduser()),
            "schema_version": SCHEMA_VERSION,
            "schema": SCHEMA_NAME,
            "rules": rules,
            "required": required,
            "produced": produced,
            "rule_family_states": family_states,
            "rule_evolution_events": evolution_events,
            "rule_shadow_stats": shadow_stats,
            "vectors": vectors,
            "ann": ann_index_summary(conn, index_name=RULE_ANN_INDEX_NAME),
        }
    finally:
        conn.close()
