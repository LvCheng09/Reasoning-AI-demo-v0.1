"""Vector-first memory DB with rule-use traces."""

from __future__ import annotations
from . import base as shared_db
from ..common.math_utils import cosine as shared_cosine
from ..common.normalize import as_list as shared_as_list

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from ..identity import stable_text_id as _stable_id
from ..memory_anchors import score_structural_memory_anchors
from ..vector_contracts import canonical_embedding_model, coerce_vector, cosine_similarity_strict
from .ann_index import ann_index_summary, query_lsh_index, rebuild_lsh_index, upsert_lsh_vector


SCHEMA_VERSION = 6
SCHEMA_NAME = "structural_anchor_memory_stage_rule_trace_v6"
MEMORY_ANN_INDEX_NAME = "memories"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    return shared_db.connect(db_path, foreign_keys=True)


def _json_dumps(value: Any) -> str:
    return shared_db.json_dumps(value)


def _json_loads(value: str | None, default: Any) -> Any:
    return shared_db.json_loads(value, default)


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
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            memory_vector BLOB,
            embedding_model TEXT NOT NULL DEFAULT '',
            vector_dimension INTEGER NOT NULL DEFAULT 0,
            vector_version TEXT NOT NULL DEFAULT '',
            memory_time TEXT NOT NULL DEFAULT '',
            memory_debug_text TEXT NOT NULL DEFAULT '',
            memory_json TEXT NOT NULL DEFAULT '{}',
            memory_index_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _ensure_memory_json_column(conn)
    _ensure_memory_index_json_column(conn)
    _ensure_column(conn, "memories", "embedding_model", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "memories", "vector_dimension", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "memories", "vector_version", "TEXT NOT NULL DEFAULT ''")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_retrieval_anchors (
            anchor_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            anchor_index INTEGER NOT NULL DEFAULT 0,
            anchor_kind TEXT NOT NULL,
            structural_role TEXT NOT NULL DEFAULT '',
            span_id TEXT NOT NULL DEFAULT '',
            relation_id TEXT NOT NULL DEFAULT '',
            source_span_id TEXT NOT NULL DEFAULT '',
            target_span_id TEXT NOT NULL DEFAULT '',
            core_vector BLOB,
            relation_vector BLOB,
            nature_vector BLOB,
            embedding_model TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            vector_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_stage (
            stage_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            stage_index INTEGER NOT NULL DEFAULT 0,
            condition_source_id TEXT NOT NULL DEFAULT '',
            condition_source TEXT NOT NULL DEFAULT '',
            condition_source_confidence REAL NOT NULL DEFAULT 0.5,
            stage_confidence REAL NOT NULL DEFAULT 0.5,
            logical_strength REAL NOT NULL DEFAULT 0.5,
            empirical_support REAL NOT NULL DEFAULT 0.5,
            contradiction_rate REAL NOT NULL DEFAULT 0.0,
            failure_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
        )
        """
    )
    _migrate_rule_uses_to_stage_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_rule_uses (
            usage_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            condition_id TEXT NOT NULL DEFAULT '',
            condition TEXT NOT NULL DEFAULT '',
            relation TEXT NOT NULL DEFAULT '',
            rule_id TEXT NOT NULL DEFAULT '',
            rule_confidence REAL NOT NULL DEFAULT 0.5,
            logical_strength REAL NOT NULL DEFAULT 0.5,
            empirical_support REAL NOT NULL DEFAULT 0.5,
            contradiction_rate REAL NOT NULL DEFAULT 0.0,
            failure_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
            FOREIGN KEY(condition_id) REFERENCES memory_stage(stage_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_success_reasons (
            reason_id TEXT PRIMARY KEY,
            usage_id TEXT NOT NULL,
            success_reason TEXT NOT NULL DEFAULT '',
            success_reason_confidence REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(usage_id) REFERENCES memory_rule_uses(usage_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(memory_time DESC, updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_vector_contract ON memories(embedding_model, vector_dimension, vector_version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_anchors_memory ON memory_retrieval_anchors(memory_id, anchor_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_anchors_contract ON memory_retrieval_anchors(embedding_model, vector_dimension, vector_version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_stage_memory ON memory_stage(memory_id, stage_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_stage_condition_source ON memory_stage(condition_source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_uses_memory ON memory_rule_uses(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_uses_condition ON memory_rule_uses(condition_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_uses_rule ON memory_rule_uses(rule_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_uses_relation ON memory_rule_uses(relation)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_success_usage ON memory_success_reasons(usage_id)")
    _migrate_old_reasoning_memories(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_name', ?)",
        (SCHEMA_NAME,),
    )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return shared_db.table_columns(conn, table)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if not _table_exists(conn, table):
        return
    if column in _table_columns(conn, table):
        return
    shared_db.ensure_column(conn, table, column, definition)
    conn.commit()


def _ensure_memory_json_column(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "memories"):
        return
    if "memory_json" in _table_columns(conn, "memories"):
        return
    conn.execute("ALTER TABLE memories ADD COLUMN memory_json TEXT NOT NULL DEFAULT '{}'")
    conn.commit()


def _ensure_memory_index_json_column(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "memories"):
        return
    if "memory_index_json" not in _table_columns(conn, "memories"):
        conn.execute("ALTER TABLE memories ADD COLUMN memory_index_json TEXT NOT NULL DEFAULT '{}'")
        conn.commit()
    rows = conn.execute(
        "SELECT memory_id, memory_json FROM memories WHERE memory_index_json IS NULL OR memory_index_json = '' OR memory_index_json = '{}'"
    ).fetchall()
    if not rows:
        return
    conn.executemany(
        "UPDATE memories SET memory_index_json = ? WHERE memory_id = ?",
        [
            (_json_dumps(_memory_index_payload(_json_loads(row["memory_json"], {}))), str(row["memory_id"]))
            for row in rows
        ],
    )
    conn.commit()


def _migrate_rule_uses_to_stage_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "memory_rule_uses"):
        return
    if "condition_id" in _table_columns(conn, "memory_rule_uses"):
        return
    rows = conn.execute("SELECT * FROM memory_rule_uses ORDER BY memory_id, usage_id").fetchall()
    if not rows:
        conn.execute("DROP TABLE memory_rule_uses")
        return
    now = _now_iso()
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            CREATE TABLE memory_rule_uses_new (
                usage_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                condition_id TEXT NOT NULL DEFAULT '',
                condition TEXT NOT NULL DEFAULT '',
                relation TEXT NOT NULL DEFAULT '',
                rule_id TEXT NOT NULL DEFAULT '',
                rule_confidence REAL NOT NULL DEFAULT 0.5,
                logical_strength REAL NOT NULL DEFAULT 0.5,
                empirical_support REAL NOT NULL DEFAULT 0.5,
                contradiction_rate REAL NOT NULL DEFAULT 0.0,
                failure_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
                FOREIGN KEY(condition_id) REFERENCES memory_stage(stage_id) ON DELETE CASCADE
            )
            """
        )
        stage_indexes: dict[str, int] = {}
        for row in rows:
            memory_id = str(row["memory_id"])
            source = str(row["condition_source"] or row["condition"] or "").strip()
            condition = str(row["condition"] or source).strip()
            stage_index = stage_indexes.get(memory_id, 0)
            stage_indexes[memory_id] = stage_index + 1
            stage_id = _stable_id("stage", memory_id, stage_index, source or condition)
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_stage(
                    stage_id, memory_id, stage_index, condition_source_id, condition_source,
                    condition_source_confidence, stage_confidence, logical_strength,
                    empirical_support, contradiction_rate, failure_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage_id,
                    memory_id,
                    stage_index,
                    _stable_id("condsrc", memory_id, source or condition),
                    source or condition,
                    float(row["condition_source_confidence"]) if "condition_source_confidence" in row.keys() else 0.5,
                    float(row["rule_confidence"]),
                    float(row["logical_strength"]),
                    float(row["empirical_support"]),
                    float(row["contradiction_rate"]),
                    str(row["failure_reason"] or ""),
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO memory_rule_uses_new(
                    usage_id, memory_id, condition_id, condition, relation, rule_id,
                    rule_confidence, logical_strength, empirical_support,
                    contradiction_rate, failure_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["usage_id"]),
                    memory_id,
                    stage_id,
                    condition,
                    str(row["relation"] or ""),
                    str(row["rule_id"] or ""),
                    float(row["rule_confidence"]),
                    float(row["logical_strength"]),
                    float(row["empirical_support"]),
                    float(row["contradiction_rate"]),
                    str(row["failure_reason"] or ""),
                    str(row["created_at"] or now),
                    str(row["updated_at"] or now),
                ),
            )
        conn.execute("DROP TABLE memory_rule_uses")
        conn.execute("ALTER TABLE memory_rule_uses_new RENAME TO memory_rule_uses")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def _migrate_old_reasoning_memories(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "reasoning_memories"):
        return
    new_count = int(conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"])
    if new_count > 0:
        return
    try:
        rows = conn.execute(
            "SELECT memory_id, memory_json, created_at FROM reasoning_memories ORDER BY updated_at DESC, memory_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return
    now = _now_iso()
    for row in rows:
        payload = _json_loads(row["memory_json"], {})
        if not payload:
            continue
        memory_id = str(payload.get("memory_id") or row["memory_id"] or "").strip()
        if not memory_id:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO memories(
                memory_id, memory_vector, memory_time, memory_debug_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                _vector_to_blob(payload.get("memory_vector", payload.get("vector"))),
                str(payload.get("memory_time") or payload.get("created_at") or row["created_at"] or now),
                _memory_debug_text(payload),
                now,
                now,
            ),
        )
        for index, item in enumerate(_coerce_rule_uses(payload)):
            stage = _coerce_stages(payload, [item])[0]
            stage_id = _stable_id("stage", memory_id, 0, stage["condition_source_id"], stage["condition_source"])
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_stage(
                    stage_id, memory_id, stage_index, condition_source_id, condition_source,
                    condition_source_confidence, stage_confidence, logical_strength,
                    empirical_support, contradiction_rate, failure_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage_id,
                    memory_id,
                    0,
                    stage["condition_source_id"],
                    stage["condition_source"],
                    stage["condition_source_confidence"],
                    stage["stage_confidence"],
                    stage["logical_strength"],
                    stage["empirical_support"],
                    stage["contradiction_rate"],
                    stage["failure_reason"],
                    now,
                    now,
                ),
            )
            usage_id = _stable_id("memuse", memory_id, index, item["rule_id"], item["condition"], item["relation"])
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_rule_uses(
                    usage_id, memory_id, condition_id, condition, relation, rule_id, rule_confidence, logical_strength,
                    empirical_support, contradiction_rate, failure_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage_id,
                    memory_id,
                    stage_id,
                    stage["condition_source"],
                    item["relation"],
                    item["rule_id"],
                    item["rule_confidence"],
                    item["logical_strength"],
                    item["empirical_support"],
                    item["contradiction_rate"],
                    item["failure_reason"],
                    now,
                    now,
                ),
            )
            for reason_index, reason in enumerate(item["success_reasons"]):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_success_reasons(
                        reason_id, usage_id, success_reason, success_reason_confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("memreason", usage_id, reason_index, reason["success_reason"]),
                        usage_id,
                        reason["success_reason"],
                        reason["success_reason_confidence"],
                        now,
                        now,
                    ),
                )


def _payload(memory: Any) -> dict[str, Any]:
    if hasattr(memory, "to_dict"):
        return dict(memory.to_dict())
    return dict(memory or {})


def _as_list(value: Any) -> list[Any]:
    return shared_as_list(value, expand_sets=True)


def _vector_to_list(value: Any) -> list[float] | None:
    return coerce_vector(value)


def _vector_to_blob(value: Any) -> bytes | None:
    return shared_db.vector_to_blob(value, reject_zero=False, inclusive_zero=False)


def _blob_to_vector(value: bytes | None) -> list[float] | None:
    return shared_db.blob_to_vector(value)


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    return shared_cosine(left, right, positive_only=False)


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            chunks = [str(item).strip() for item in value if str(item).strip()]
            if chunks:
                return " ".join(chunks)
    return ""


def _coerce_success_reasons(value: Any, fallback: str = "") -> list[dict[str, Any]]:
    raw_items = _as_list(value)
    if not raw_items and fallback:
        raw_items = [fallback]
    items: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            reason = str(
                item.get("success_reason")
                or item.get("reason")
                or item.get("conclusion")
                or item.get("condition")
                or ""
            ).strip()
            confidence = float(
                item.get("success_reason_confidence")
                or item.get("confidence")
                or item.get("condition_confidence")
                or 0.5
            )
        elif isinstance(item, (list, tuple)) and item:
            reason = str(item[0]).strip()
            confidence = float(item[1]) if len(item) > 1 else 0.5
        else:
            reason = str(item).strip()
            confidence = 0.5
        if reason:
            items.append(
                {
                    "success_reason": reason,
                    "success_reason_confidence": confidence,
                }
            )
    return items


def _coerce_stages(payload: dict[str, Any], rule_uses: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = (
        payload.get("memory_stage")
        or payload.get("memory_stages")
        or payload.get("stages")
        or []
    )
    if not raw_items:
        stage_keys = sorted(
            key
            for key in payload
            if str(key).startswith("stage_") and str(payload.get(key)).strip()
        )
        raw_items = [
            {"condition_source_id": key, "condition_source": payload[key]}
            for key in stage_keys
        ]
    stages: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(raw_items)):
        if isinstance(item, dict):
            source = str(
                item.get("condition_source")
                or item.get("condition")
                or item.get("stage")
                or item.get("text")
                or ""
            ).strip()
            source_id = str(
                item.get("condition_source_id")
                or item.get("stage_id")
                or item.get("id")
                or ""
            ).strip()
            source_confidence = float(item.get("condition_source_confidence", item.get("confidence", 0.5)))
            stage_confidence = float(item.get("stage_confidence", item.get("confidence", source_confidence)))
            logical_strength = float(item.get("logical_strength", item.get("Logical_strength", 0.5)))
            empirical_support = float(item.get("empirical_support", 0.5))
            contradiction_rate = float(item.get("contradiction_rate", 0.0))
            failure_reason = str(item.get("failure_reason") or "").strip()
        else:
            source = str(item).strip()
            source_id = ""
            source_confidence = 0.5
            stage_confidence = 0.5
            logical_strength = 0.5
            empirical_support = 0.5
            contradiction_rate = 0.0
            failure_reason = ""
        if not source:
            continue
        stages.append(
            {
                "condition_source_id": source_id,
                "condition_source": source,
                "condition_source_confidence": source_confidence,
                "stage_confidence": stage_confidence,
                "logical_strength": logical_strength,
                "empirical_support": empirical_support,
                "contradiction_rate": contradiction_rate,
                "failure_reason": failure_reason,
            }
        )
    if stages:
        return stages

    for item in rule_uses:
        source = str(item.get("condition_source") or item.get("condition") or "").strip()
        if not source:
            continue
        stages.append(
            {
                "condition_source_id": str(item.get("condition_source_id") or item.get("condition_id") or "").strip(),
                "condition_source": source,
                "condition_source_confidence": float(item.get("condition_source_confidence", 0.5)),
                "stage_confidence": float(item.get("rule_confidence", 0.5)),
                "logical_strength": float(item.get("logical_strength", 0.5)),
                "empirical_support": float(item.get("empirical_support", 0.5)),
                "contradiction_rate": float(item.get("contradiction_rate", 0.0)),
                "failure_reason": str(item.get("failure_reason") or "").strip(),
            }
        )
    if stages:
        return stages

    source = _memory_debug_text(payload)
    return [
        {
            "condition_source_id": "",
            "condition_source": source,
            "condition_source_confidence": float(payload.get("confidence", 0.5)),
            "stage_confidence": float(payload.get("confidence", 0.5)),
            "logical_strength": float(payload.get("logical_strength", 0.5)),
            "empirical_support": float(payload.get("empirical_support", 0.5)),
            "contradiction_rate": float(payload.get("contradiction_rate", 0.0)),
            "failure_reason": str(payload.get("failure_reason") or "").strip(),
        }
    ]


def _coerce_rule_uses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "memory_rule_uses" in payload:
        raw_items = payload.get("memory_rule_uses")
    else:
        raw_items = payload.get("rule_uses") or payload.get("used_rules")
    if raw_items is not None:
        items: list[dict[str, Any]] = []
        for item in _as_list(raw_items):
            if not isinstance(item, dict):
                item = {"rule_id": str(item)}
            condition_source = str(item.get("condition_source") or payload.get("text") or "").strip()
            rule_id = str(item.get("rule_id") or "").strip()
            if not condition_source and not rule_id:
                continue
            items.append(
                {
                    "condition_source": condition_source,
                    "condition_source_id": str(item.get("condition_source_id") or item.get("condition_id") or "").strip(),
                    "condition_source_confidence": float(item.get("condition_source_confidence", payload.get("confidence", 0.5))),
                    "condition": str(item.get("condition") or item.get("condition_1") or "").strip(),
                    "relation": str(item.get("relation") or item.get("relation_i_j") or "").strip(),
                    "rule_id": rule_id,
                    "rule_confidence": float(item.get("rule_confidence", payload.get("confidence", 0.5))),
                    "logical_strength": float(item.get("logical_strength", item.get("Logical_strength", 0.5))),
                    "empirical_support": float(item.get("empirical_support", 0.5)),
                    "contradiction_rate": float(item.get("contradiction_rate", 0.0)),
                    "success_reasons": _coerce_success_reasons(item.get("success_reason"), str(payload.get("outcome") or "")),
                    "failure_reason": str(item.get("failure_reason") or payload.get("failure_reason") or "no").strip(),
                }
            )
        return items

    selected_rule_ids = [str(item).strip() for item in _as_list(payload.get("selected_rule_ids")) if str(item).strip()]
    heads = [str(item).strip() for item in _as_list(payload.get("heads")) if str(item).strip()]
    context_keys = [str(item).strip() for item in _as_list(payload.get("context_keys")) if str(item).strip()]
    relations = [str(item).strip() for item in _as_list(payload.get("relations")) if str(item).strip()]
    condition = _first_text(heads, context_keys)
    relation = relations[0] if relations else "case_support"
    if not selected_rule_ids:
        selected_rule_ids = [""]
    items = []
    for rule_id in selected_rule_ids:
        items.append(
            {
                "condition_source": str(payload.get("text") or payload.get("input_summary") or "").strip(),
                "condition_source_id": "",
                "condition_source_confidence": float(payload.get("confidence", 0.5)),
                "condition": condition,
                "relation": relation,
                "rule_id": rule_id,
                "rule_confidence": float(payload.get("confidence", 0.5)),
                "logical_strength": float(payload.get("logical_strength", 0.5)),
                "empirical_support": float(payload.get("empirical_support", 0.5)),
                "contradiction_rate": float(payload.get("contradiction_rate", 0.0)),
                "success_reasons": _coerce_success_reasons(payload.get("success_reason"), str(payload.get("outcome") or "")),
                "failure_reason": str(payload.get("failure_reason") or "no").strip(),
            }
        )
    return items


def _memory_debug_text(payload: dict[str, Any]) -> str:
    return _first_text(
        payload.get("memory_debug_text"),
        payload.get("text"),
        payload.get("input_summary"),
        payload.get("outcome"),
        payload.get("heads"),
        payload.get("context_keys"),
    )


def _coerce_retrieval_anchors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    contracts: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(payload.get("retrieval_anchors", []) or []):
        if not isinstance(raw, dict):
            raise ValueError(f"retrieval_anchors[{index}] must be a mapping")
        anchor_kind = str(raw.get("anchor_kind") or "").strip().lower()
        if anchor_kind not in {"span", "relation"}:
            raise ValueError(f"retrieval_anchors[{index}] has invalid anchor_kind")
        embedding_model = canonical_embedding_model(raw.get("embedding_model"))
        vector_version = str(raw.get("vector_version") or "").strip()
        try:
            vector_dimension = int(raw.get("vector_dimension") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"retrieval_anchors[{index}] has invalid vector_dimension"
            ) from exc
        if not embedding_model or not vector_version or vector_dimension <= 0:
            raise ValueError(f"retrieval_anchors[{index}] has incomplete vector contract")
        vectors = {
            field: _vector_to_list(raw.get(field))
            for field in ("core_vector", "relation_vector", "nature_vector")
        }
        required_fields = (
            ("core_vector", "relation_vector", "nature_vector")
            if anchor_kind == "span"
            else ("relation_vector",)
        )
        if any(not vectors[field] for field in required_fields):
            raise ValueError(f"retrieval_anchors[{index}] is missing structural vectors")
        if any(len(vectors[field] or []) != vector_dimension for field in required_fields):
            raise ValueError(f"retrieval_anchors[{index}] vector dimension mismatch")
        contract = (embedding_model, vector_dimension, vector_version)
        contracts.add(contract)
        anchors.append(
            {
                "anchor_id": str(raw.get("anchor_id") or f"anchor_{index}"),
                "anchor_kind": anchor_kind,
                "structural_role": str(raw.get("structural_role") or ""),
                "span_id": str(raw.get("span_id") or ""),
                "relation_id": str(raw.get("relation_id") or ""),
                "source_span_id": str(raw.get("source_span_id") or ""),
                "target_span_id": str(raw.get("target_span_id") or ""),
                **{field: vector for field, vector in vectors.items() if vector},
                "embedding_model": embedding_model,
                "vector_dimension": vector_dimension,
                "vector_version": vector_version,
            }
        )
    if len(contracts) > 1:
        raise ValueError("retrieval anchors use inconsistent vector contracts")
    return anchors


def database_is_empty(db_path: str | Path) -> bool:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        row = conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()
        return int(row["count"]) == 0
    finally:
        conn.close()


def upsert_memory_records(
    db_path: str | Path,
    memories: Sequence[Any],
    *,
    replace: bool = False,
    embedding_model: str = "",
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        if replace:
            conn.execute("DELETE FROM memory_retrieval_anchors")
            conn.execute("DELETE FROM memory_success_reasons")
            conn.execute("DELETE FROM memory_rule_uses")
            conn.execute("DELETE FROM memory_stage")
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM ann_vector_index WHERE index_name = ?", (MEMORY_ANN_INDEX_NAME,))
        saved = 0
        stages_count = 0
        uses_count = 0
        reasons_count = 0
        vector_count = 0
        anchor_count = 0
        now = _now_iso()
        for memory in memories:
            payload = _payload(memory)
            memory_id = str(payload.get("memory_id") or payload.get("id") or "").strip()
            if not memory_id:
                continue
            memory_vector = payload.get("memory_vector", payload.get("vector"))
            memory_vector_list = _vector_to_list(memory_vector)
            vector_blob = _vector_to_blob(memory_vector)
            effective_embedding_model = canonical_embedding_model(payload.get("embedding_model") or embedding_model or "")
            vector_dimension = len(memory_vector_list or [])
            vector_version = str(payload.get("vector_version") or payload.get("embedding_version") or "")
            if vector_blob:
                vector_count += 1
            storage_payload = _memory_storage_payload(payload)
            conn.execute(
                """
                INSERT INTO memories(
                    memory_id, memory_vector, embedding_model, vector_dimension, vector_version,
                    memory_time, memory_debug_text, memory_json, memory_index_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    memory_vector=COALESCE(excluded.memory_vector, memories.memory_vector),
                    embedding_model=COALESCE(NULLIF(excluded.embedding_model, ''), memories.embedding_model),
                    vector_dimension=CASE
                        WHEN excluded.vector_dimension > 0 THEN excluded.vector_dimension
                        ELSE memories.vector_dimension
                    END,
                    vector_version=COALESCE(NULLIF(excluded.vector_version, ''), memories.vector_version),
                    memory_time=excluded.memory_time,
                    memory_debug_text=excluded.memory_debug_text,
                    memory_json=excluded.memory_json,
                    memory_index_json=excluded.memory_index_json,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    vector_blob,
                    effective_embedding_model,
                    vector_dimension,
                    vector_version,
                    str(payload.get("memory_time") or payload.get("created_at") or now),
                    _memory_debug_text(payload),
                    _json_dumps(storage_payload),
                    _json_dumps(_memory_index_payload(storage_payload)),
                    now,
                    now,
                ),
            )
            stored_vector = conn.execute(
                "SELECT memory_vector, embedding_model, vector_version FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            upsert_lsh_vector(
                conn,
                index_name=MEMORY_ANN_INDEX_NAME,
                record_id=memory_id,
                vector_field="memory_vector",
                vector=stored_vector["memory_vector"] if stored_vector is not None else None,
                embedding_model=str(stored_vector["embedding_model"] or "") if stored_vector is not None else "",
                vector_version=str(stored_vector["vector_version"] or "") if stored_vector is not None else "",
                now=now,
            )
            conn.execute(
                "DELETE FROM memory_retrieval_anchors WHERE memory_id = ?",
                (memory_id,),
            )
            for anchor_index, anchor in enumerate(_coerce_retrieval_anchors(payload)):
                conn.execute(
                    """
                    INSERT INTO memory_retrieval_anchors(
                        anchor_id, memory_id, anchor_index, anchor_kind,
                        structural_role, span_id, relation_id, source_span_id,
                        target_span_id, core_vector, relation_vector, nature_vector,
                        embedding_model, vector_dimension, vector_version,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id(
                            "memory_anchor",
                            memory_id,
                            anchor.get("anchor_id") or anchor_index,
                        ),
                        memory_id,
                        anchor_index,
                        anchor["anchor_kind"],
                        anchor["structural_role"],
                        anchor["span_id"],
                        anchor["relation_id"],
                        anchor["source_span_id"],
                        anchor["target_span_id"],
                        _vector_to_blob(anchor.get("core_vector")),
                        _vector_to_blob(anchor.get("relation_vector")),
                        _vector_to_blob(anchor.get("nature_vector")),
                        anchor["embedding_model"],
                        anchor["vector_dimension"],
                        anchor["vector_version"],
                        now,
                        now,
                    ),
                )
                anchor_count += 1
            conn.execute("DELETE FROM memory_rule_uses WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memory_stage WHERE memory_id = ?", (memory_id,))
            rule_items = _coerce_rule_uses(payload)
            stage_items = _coerce_stages(payload, rule_items)
            stage_by_key: dict[str, str] = {}
            first_stage_id = ""
            for stage_index, stage in enumerate(stage_items):
                stage_source_id = str(stage["condition_source_id"] or "").strip()
                stage_source = str(stage["condition_source"] or "").strip()
                stage_id = _stable_id("stage", memory_id, stage_index, stage_source_id, stage_source)
                if not first_stage_id:
                    first_stage_id = stage_id
                for key in (stage_source_id, stage_source):
                    if key:
                        stage_by_key[key] = stage_id
                conn.execute(
                    """
                    INSERT INTO memory_stage(
                        stage_id, memory_id, stage_index, condition_source_id, condition_source,
                        condition_source_confidence, stage_confidence, logical_strength,
                        empirical_support, contradiction_rate, failure_reason, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stage_id,
                        memory_id,
                        stage_index,
                        stage_source_id or _stable_id("condsrc", memory_id, stage_index, stage_source),
                        stage_source,
                        stage["condition_source_confidence"],
                        stage["stage_confidence"],
                        stage["logical_strength"],
                        stage["empirical_support"],
                        stage["contradiction_rate"],
                        stage["failure_reason"],
                        now,
                        now,
                    ),
                )
                stages_count += 1
            for index, item in enumerate(rule_items):
                condition_key = str(
                    item.get("condition_id")
                    or item.get("condition_source_id")
                    or item.get("condition_source")
                    or item.get("condition")
                    or ""
                ).strip()
                condition_id = stage_by_key.get(condition_key) or first_stage_id
                condition_text = next(
                    (
                        stage["condition_source"]
                        for stage_index, stage in enumerate(stage_items)
                        if _stable_id("stage", memory_id, stage_index, stage["condition_source_id"], stage["condition_source"]) == condition_id
                    ),
                    str(item.get("condition") or item.get("condition_source") or "").strip(),
                )
                usage_id = _stable_id("memuse", memory_id, index, item["rule_id"], condition_id, item["relation"])
                conn.execute(
                    """
                    INSERT INTO memory_rule_uses(
                        usage_id, memory_id, condition_id, condition, relation, rule_id, rule_confidence, logical_strength,
                        empirical_support, contradiction_rate, failure_reason, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        usage_id,
                        memory_id,
                        condition_id,
                        condition_text,
                        item["relation"],
                        item["rule_id"],
                        item["rule_confidence"],
                        item["logical_strength"],
                        item["empirical_support"],
                        item["contradiction_rate"],
                        item["failure_reason"],
                        now,
                        now,
                    ),
                )
                uses_count += 1
                for reason_index, reason in enumerate(item["success_reasons"]):
                    conn.execute(
                        """
                        INSERT INTO memory_success_reasons(
                            reason_id, usage_id, success_reason, success_reason_confidence, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _stable_id("memreason", usage_id, reason_index, reason["success_reason"]),
                            usage_id,
                            reason["success_reason"],
                            reason["success_reason_confidence"],
                            now,
                            now,
                        ),
                    )
                    reasons_count += 1
            saved += 1
        conn.commit()
        return {
            "enabled": True,
            "status": "completed",
            "db_path": str(Path(db_path).expanduser()),
            "saved": saved,
            "saved_memories": saved,
            "saved_memory_stages": stages_count,
            "saved_memory_rule_uses": uses_count,
            "saved_success_reasons": reasons_count,
            "saved_vectors": vector_count,
            "saved_retrieval_anchors": anchor_count,
            "schema": SCHEMA_NAME,
        }
    finally:
        conn.close()


def _memory_storage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Store readable history without duplicating large vectors in JSON blobs."""
    return _strip_vector_fields(payload)


def _strip_vector_fields(value: Any) -> Any:
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"memory_vector", "vector"} or key_text.endswith("_vector"):
                continue
            stripped[key] = _strip_vector_fields(item)
        return stripped
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return f"vector[{len(value)}]"
        return [_strip_vector_fields(item) for item in value]
    return value


def _anchor_row_to_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "anchor_id": str(row["anchor_id"]),
        "anchor_index": int(row["anchor_index"]),
        "anchor_kind": str(row["anchor_kind"]),
        "structural_role": str(row["structural_role"] or ""),
        "span_id": str(row["span_id"] or ""),
        "relation_id": str(row["relation_id"] or ""),
        "source_span_id": str(row["source_span_id"] or ""),
        "target_span_id": str(row["target_span_id"] or ""),
        "core_vector": _blob_to_vector(row["core_vector"]),
        "relation_vector": _blob_to_vector(row["relation_vector"]),
        "nature_vector": _blob_to_vector(row["nature_vector"]),
        "embedding_model": str(row["embedding_model"] or ""),
        "vector_dimension": int(row["vector_dimension"] or 0),
        "vector_version": str(row["vector_version"] or ""),
    }


def _row_to_payload(
    memory_row: sqlite3.Row,
    stage_rows: Sequence[sqlite3.Row],
    use_rows: Sequence[sqlite3.Row],
    reasons_by_usage: dict[str, list[sqlite3.Row]],
    anchor_rows: Sequence[sqlite3.Row],
) -> dict[str, Any]:
    memory_id = str(memory_row["memory_id"])
    stage_by_id = {str(item["stage_id"]): item for item in stage_rows}
    stages = [
        {
            "stage_id": str(item["stage_id"]),
            "stage_index": int(item["stage_index"]),
            "condition_source_id": str(item["condition_source_id"] or ""),
            "condition_source": str(item["condition_source"] or ""),
            "condition_source_confidence": float(item["condition_source_confidence"]),
            "stage_confidence": float(item["stage_confidence"]),
            "logical_strength": float(item["logical_strength"]),
            "empirical_support": float(item["empirical_support"]),
            "contradiction_rate": float(item["contradiction_rate"]),
            "failure_reason": str(item["failure_reason"] or ""),
        }
        for item in stage_rows
    ]
    uses: list[dict[str, Any]] = []
    heads: list[str] = []
    relations: list[str] = []
    context_keys: list[str] = []
    selected_rule_ids: list[str] = []
    success_reason_texts: list[str] = []
    failure_reasons: list[str] = []
    confidences: list[float] = []
    for item in use_rows:
        usage_id = str(item["usage_id"])
        success_reasons = [
            {
                "success_reason": str(reason["success_reason"] or ""),
                "success_reason_confidence": float(reason["success_reason_confidence"]),
            }
            for reason in reasons_by_usage.get(usage_id, [])
        ]
        stage = stage_by_id.get(str(item["condition_id"] or ""))
        condition_source = str(stage["condition_source"] or "") if stage is not None else str(item["condition"] or "")
        condition_source_confidence = (
            float(stage["condition_source_confidence"]) if stage is not None else float(item["rule_confidence"])
        )
        uses.append(
            {
                "condition_id": str(item["condition_id"] or ""),
                "condition_source": condition_source,
                "condition_source_confidence": condition_source_confidence,
                "condition": str(item["condition"] or ""),
                "relation": str(item["relation"] or ""),
                "rule_id": str(item["rule_id"] or ""),
                "rule_confidence": float(item["rule_confidence"]),
                "logical_strength": float(item["logical_strength"]),
                "empirical_support": float(item["empirical_support"]),
                "contradiction_rate": float(item["contradiction_rate"]),
                "success_reason": success_reasons,
                "failure_reason": str(item["failure_reason"] or ""),
            }
        )
        if item["condition"]:
            heads.append(str(item["condition"]))
        if item["relation"]:
            relations.append(str(item["relation"]))
        if condition_source:
            context_keys.append(condition_source)
        if item["rule_id"]:
            selected_rule_ids.append(str(item["rule_id"]))
        if item["failure_reason"] and str(item["failure_reason"]).lower() != "no":
            failure_reasons.append(str(item["failure_reason"]))
        confidences.append(float(item["rule_confidence"]))
        success_reason_texts.extend(reason["success_reason"] for reason in success_reasons if reason["success_reason"])
    confidence = sum(confidences) / len(confidences) if confidences else 0.5
    text = str(memory_row["memory_debug_text"] or "").strip()
    outcome = success_reason_texts[0] if success_reason_texts else (failure_reasons[0] if failure_reasons else text)
    base_payload = _json_loads(memory_row["memory_json"] if "memory_json" in memory_row.keys() else "", {})
    if not isinstance(base_payload, dict):
        base_payload = {}
    base_payload.update(
        {
            "memory_id": memory_id,
            "memory_vector": _blob_to_vector(memory_row["memory_vector"]),
            "vector": _blob_to_vector(memory_row["memory_vector"]),
            "retrieval_anchors": [_anchor_row_to_payload(item) for item in anchor_rows],
            "embedding_model": str(memory_row["embedding_model"] or "") if "embedding_model" in memory_row.keys() else "",
            "vector_dimension": int(memory_row["vector_dimension"] or 0) if "vector_dimension" in memory_row.keys() else 0,
            "vector_version": str(memory_row["vector_version"] or "") if "vector_version" in memory_row.keys() else "",
            "memory_time": str(memory_row["memory_time"] or ""),
            "created_at": str(memory_row["memory_time"] or ""),
            "text": text,
            "input_summary": text,
            "proposition_type": base_payload.get("proposition_type", "case_based"),
            "heads": list(dict.fromkeys(heads)),
            "relations": list(dict.fromkeys(relations)),
            "context_keys": list(dict.fromkeys(context_keys)),
            "outcome": outcome,
            "memory_kind": base_payload.get("memory_kind", "archive_case"),
            "task_type": base_payload.get("task_type", "case_based"),
            "selected_rule_ids": list(dict.fromkeys(selected_rule_ids)),
            "reasoning_path": [f"rule:{rule_id}" for rule_id in dict.fromkeys(selected_rule_ids)],
            "failure_reason": "; ".join(dict.fromkeys(failure_reasons)),
            "outcome_status": base_payload.get("outcome_status") or ("failed" if failure_reasons and not success_reason_texts else "resolved"),
            "confidence": confidence,
            "memory_stage": stages,
            "memory_stages": stages,
            "rule_uses": uses,
        }
    )
    _hydrate_event_hypergraph_vectors(base_payload)
    return base_payload


def _hydrate_event_hypergraph_vectors(payload: dict[str, Any]) -> None:
    """Restore graph vectors from normalized retrieval anchors in memory."""

    graph = payload.get("event_hypergraph")
    if not isinstance(graph, dict):
        return
    anchors = [
        item
        for item in payload.get("retrieval_anchors", []) or []
        if isinstance(item, dict)
    ]
    span_anchors = {
        str(item.get("span_id") or ""): item
        for item in anchors
        if str(item.get("anchor_kind") or "") == "span"
        and str(item.get("span_id") or "")
    }
    relation_anchors = {
        str(item.get("relation_id") or ""): item
        for item in anchors
        if str(item.get("anchor_kind") or "") == "relation"
        and str(item.get("relation_id") or "")
    }
    for span in graph.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        anchor = span_anchors.get(str(span.get("span_id") or span.get("id") or ""))
        if anchor is None:
            continue
        for field in ("core_vector", "relation_vector", "nature_vector"):
            if anchor.get(field):
                span[field] = list(anchor[field])
        _copy_vector_contract(span, anchor)
    for relation in graph.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        anchor = relation_anchors.get(
            str(relation.get("relation_id") or relation.get("id") or "")
        )
        if anchor is None:
            continue
        if anchor.get("relation_vector"):
            relation["relation_vector"] = list(anchor["relation_vector"])
        _copy_vector_contract(relation, anchor)


def _copy_vector_contract(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in ("embedding_model", "vector_dimension", "vector_version"):
        if source.get(field) not in (None, "", 0):
            target[field] = source[field]


def _load_memory_payloads(
    conn: sqlite3.Connection,
    *,
    memory_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM memories ORDER BY memory_time DESC, updated_at DESC, memory_id"
    params: list[Any] = []
    normalized_ids = [str(memory_id) for memory_id in (memory_ids or []) if str(memory_id)]
    if normalized_ids:
        sql = f"SELECT * FROM memories WHERE memory_id IN ({','.join('?' for _ in normalized_ids)}) ORDER BY memory_time DESC, updated_at DESC, memory_id"
        params.extend(normalized_ids)
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    selected_ids = [str(row["memory_id"]) for row in rows]
    stages_by_memory: dict[str, list[sqlite3.Row]] = {memory_id: [] for memory_id in selected_ids}
    uses_by_memory: dict[str, list[sqlite3.Row]] = {memory_id: [] for memory_id in selected_ids}
    reasons_by_usage: dict[str, list[sqlite3.Row]] = {}
    anchors_by_memory: dict[str, list[sqlite3.Row]] = {
        memory_id: [] for memory_id in selected_ids
    }
    if selected_ids:
        placeholders = ",".join("?" for _ in selected_ids)
        for anchor in conn.execute(
            f"SELECT * FROM memory_retrieval_anchors WHERE memory_id IN ({placeholders}) ORDER BY memory_id, anchor_index, anchor_id",
            selected_ids,
        ).fetchall():
            anchors_by_memory.setdefault(str(anchor["memory_id"]), []).append(anchor)
        for stage in conn.execute(
            f"SELECT * FROM memory_stage WHERE memory_id IN ({placeholders}) ORDER BY memory_id, stage_index, stage_id",
            selected_ids,
        ).fetchall():
            stages_by_memory.setdefault(str(stage["memory_id"]), []).append(stage)
        use_rows = conn.execute(
            f"SELECT * FROM memory_rule_uses WHERE memory_id IN ({placeholders}) ORDER BY memory_id, rule_confidence DESC, usage_id",
            selected_ids,
        ).fetchall()
        for use in use_rows:
            uses_by_memory.setdefault(str(use["memory_id"]), []).append(use)
        usage_ids = [str(item["usage_id"]) for item in use_rows]
        if usage_ids:
            usage_placeholders = ",".join("?" for _ in usage_ids)
            for reason in conn.execute(
                f"SELECT * FROM memory_success_reasons WHERE usage_id IN ({usage_placeholders}) ORDER BY usage_id, success_reason_confidence DESC, reason_id",
                usage_ids,
            ).fetchall():
                reasons_by_usage.setdefault(str(reason["usage_id"]), []).append(reason)
    payload_by_id = {
        str(row["memory_id"]): _row_to_payload(
            row,
            stages_by_memory.get(str(row["memory_id"]), []),
            uses_by_memory.get(str(row["memory_id"]), []),
            reasons_by_usage,
            anchors_by_memory.get(str(row["memory_id"]), []),
        )
        for row in rows
    }
    if normalized_ids:
        return [payload_by_id[memory_id] for memory_id in normalized_ids if memory_id in payload_by_id]
    return [payload_by_id[str(row["memory_id"])] for row in rows]


def load_memory_records(
    db_path: str | Path,
    *,
    limit: int | None = 512,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        return _load_memory_payloads(conn, limit=limit)
    finally:
        conn.close()


def load_memory_index_records(
    db_path: str | Path,
    *,
    limit: int | None = 512,
) -> list[dict[str, Any]]:
    """Load compact retrieval records while keeping full history in memory_json."""

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        sql = """
            SELECT memory_id, memory_vector, memory_time, memory_debug_text, memory_index_json
            FROM memories
            ORDER BY memory_time DESC, updated_at DESC, memory_id
        """
        params: list[Any] = []
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        records: list[dict[str, Any]] = []
        for row in conn.execute(sql, params).fetchall():
            payload = _json_loads(row["memory_index_json"], {})
            if not isinstance(payload, dict):
                payload = {}
            payload.update(
                {
                    "memory_id": str(row["memory_id"]),
                    "memory_vector": _blob_to_vector(row["memory_vector"]),
                    "vector": _blob_to_vector(row["memory_vector"]),
                    "memory_time": str(row["memory_time"] or ""),
                    "text": str(payload.get("text") or row["memory_debug_text"] or ""),
                }
            )
            records.append(payload)
        return records
    finally:
        conn.close()


def _memory_index_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    event_graph = payload.get("event_hypergraph") if isinstance(payload.get("event_hypergraph"), dict) else {}
    workspace = event_graph.get("workspace") if isinstance(event_graph.get("workspace"), dict) else {}
    execution_trace = payload.get("execution_trace") if isinstance(payload.get("execution_trace"), dict) else {}
    metadata = event_graph.get("metadata") if isinstance(event_graph.get("metadata"), dict) else {}
    rule_trace = (
        execution_trace.get("rule_trace")
        if isinstance(execution_trace.get("rule_trace"), dict)
        else metadata.get("rule_trace")
        if isinstance(metadata.get("rule_trace"), dict)
        else {}
    )
    structural_observations = [
        _compact_structural_residual_observation(item)
        for item in metadata.get("structural_residual_observations", []) or []
        if isinstance(item, dict)
    ][:64]
    candidate_paths = [
        {
            "path_id": item.get("path_id", ""),
            "learning_role": item.get("learning_role", ""),
            "rule_id": item.get("rule_id", ""),
            "strategy_rule_id": item.get("strategy_rule_id", ""),
            "path_rule_trace": list(item.get("path_rule_trace") or [])[:64],
        }
        for item in (event_graph.get("candidate_paths") or workspace.get("candidate_paths") or [])
        if isinstance(item, dict)
    ][:64]
    relations = [
        {
            "relation_id": item.get("relation_id", ""),
            "source_span_id": item.get("source_span_id", ""),
            "relation": item.get("relation", item.get("relation_type", "")),
            "target_span_id": item.get("target_span_id", ""),
        }
        for item in event_graph.get("relations", []) or []
        if isinstance(item, dict)
    ][:128]
    return {
        "text": str(payload.get("text") or payload.get("input_summary") or ""),
        "input_summary": str(payload.get("input_summary") or payload.get("text") or ""),
        "evidence_cluster_id": str(payload.get("evidence_cluster_id") or ""),
        "structural_fingerprint": str(payload.get("structural_fingerprint") or ""),
        "structure_status": str(payload.get("structure_status") or ""),
        "outcome": payload.get("outcome", ""),
        "outcome_status": payload.get("outcome_status", ""),
        "feedback_signal": payload.get("feedback_signal", ""),
        "failure_points": list(payload.get("failure_points") or [])[:32],
        "correction_status": payload.get("correction_status", ""),
        "learning_value": payload.get("learning_value", ""),
        "memory_kind": payload.get("memory_kind", ""),
        "confidence": payload.get("confidence", 0.5),
        "metadata": _compact_feedback_mapping(payload.get("metadata")),
        "state": _compact_feedback_mapping(payload.get("state")),
        "event_hypergraph": {
            "spans": list(event_graph.get("spans") or [])[:128],
            "relations": relations,
            "candidate_paths": candidate_paths,
            "feedback": list(event_graph.get("feedback") or workspace.get("feedback") or [])[:32],
            "metadata": {
                "final_goals": list(metadata.get("final_goals") or [])[:32],
                "rule_trace": {
                    "matched_rule_ids": list(rule_trace.get("matched_rule_ids") or [])[:128],
                    "executed_rule_ids": list(rule_trace.get("executed_rule_ids") or [])[:128],
                    "productive_rule_ids": list(rule_trace.get("productive_rule_ids") or [])[:128],
                },
                "feedback_signal": metadata.get("feedback_signal", ""),
                "structural_residual_observations": structural_observations,
            },
        },
        "execution_trace": {
            "outcome_status": execution_trace.get("outcome_status", payload.get("outcome_status", "")),
            "feedback_signal": execution_trace.get("feedback_signal", ""),
            "failure_points": list(execution_trace.get("failure_points") or payload.get("failure_points") or [])[:32],
            "feedback": list(execution_trace.get("feedback") or [])[:32],
            "rule_trace": {
                "matched_rule_ids": list(rule_trace.get("matched_rule_ids") or [])[:128],
                "executed_rule_ids": list(rule_trace.get("executed_rule_ids") or [])[:128],
                "productive_rule_ids": list(rule_trace.get("productive_rule_ids") or [])[:128],
            },
        },
        "rule_uses": list(payload.get("rule_uses") or payload.get("memory_rule_uses") or [])[:128],
        "memory_stages": list(payload.get("memory_stages") or payload.get("memory_stage") or [])[:128],
    }


def _compact_structural_residual_observation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "observation_id",
            "outcome",
            "feedback_signal",
            "executed_rule_ids",
            "expected_rule_ids",
            "context_vector",
            "equivalent_rule_ids",
            "equivalence_score",
            "outgoing_topology_equal",
            "evidence_provenance_id",
            "independence_verified",
        )
        if key in value
    }


def _compact_feedback_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "feedback_signal",
            "source_confidence",
            "answer_correct",
            "reasoning_clean",
            "status",
            "task_type",
        )
        if key in value
    }


def search_memory_records(
    db_path: str | Path,
    *,
    query: str = "",
    keys: Sequence[str] | None = None,
    query_vector: Any = None,
    top_k: int = 64,
) -> list[dict[str, Any]]:
    """Run an exact vector maintenance query; runtime uses ANN."""

    del query, keys
    query_vec = _vector_to_list(query_vector)
    if not query_vec:
        raise ValueError("memory search requires query_vector")
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        ranked: list[tuple[float, str]] = []
        rows = conn.execute(
            "SELECT memory_id, memory_vector, memory_debug_text FROM memories ORDER BY memory_time DESC, updated_at DESC"
        ).fetchall()
        for row in rows:
            memory_id = str(row["memory_id"])
            vector_score = _cosine(query_vec, _blob_to_vector(row["memory_vector"]))
            if vector_score > 0.0:
                ranked.append((vector_score, memory_id))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected_ids = [memory_id for _, memory_id in ranked[: max(int(top_k), 1)]]
        return _load_memory_payloads(conn, memory_ids=selected_ids)
    finally:
        conn.close()


def build_memory_ann_index(
    db_path: str | Path,
    *,
    embedding_model: str = "",
    vector_version: str = "",
    bits: int = 12,
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        rows = conn.execute("SELECT * FROM memories WHERE memory_vector IS NOT NULL").fetchall()
        ann_rows = []
        for row in rows:
            vector = _blob_to_vector(row["memory_vector"])
            if not vector:
                continue
            ann_rows.append(
                {
                    "record_id": str(row["memory_id"]),
                    "vector_field": "memory_vector",
                    "vector": vector,
                    "embedding_model": str(row["embedding_model"] or embedding_model or ""),
                    "vector_version": str(row["vector_version"] or vector_version or ""),
                }
            )
        summary = rebuild_lsh_index(
            conn,
            index_name=MEMORY_ANN_INDEX_NAME,
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


def ensure_memory_ann_index(db_path: str | Path, *, bits: int = 12) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        expected = int(conn.execute("SELECT COUNT(*) AS count FROM memories WHERE memory_vector IS NOT NULL").fetchone()["count"])
        current = ann_index_summary(conn, index_name=MEMORY_ANN_INDEX_NAME)
    finally:
        conn.close()
    if int(current["indexed_vectors"]) != expected:
        return {**build_memory_ann_index(db_path, bits=bits), "rebuilt": True}
    return {**current, "record_count": expected, "db_path": str(Path(db_path).expanduser()), "rebuilt": False}


def repair_memory_vector_contract(
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
            "SELECT memory_id, memory_vector, embedding_model, vector_version FROM memories WHERE memory_vector IS NOT NULL"
        ).fetchall()
        model_mismatches: list[str] = []
        version_mismatches: list[str] = []
        repaired = 0
        for row in rows:
            vector = _blob_to_vector(row["memory_vector"])
            if not vector:
                continue
            stored_model = canonical_embedding_model(row["embedding_model"])
            if stored_model and expected_model and stored_model != expected_model:
                model_mismatches.append(str(row["memory_id"]))
                continue
            stored_version = str(row["vector_version"] or "").strip()
            expected_version = str(vector_version or "").strip()
            if stored_version and expected_version and stored_version != expected_version:
                version_mismatches.append(str(row["memory_id"]))
                continue
            conn.execute(
                """
                UPDATE memories
                SET embedding_model = ?, vector_dimension = ?, vector_version = ?
                WHERE memory_id = ?
                """,
                (
                    stored_model or expected_model,
                    len(vector),
                    stored_version or expected_version,
                    str(row["memory_id"]),
                ),
            )
            repaired += 1
        if model_mismatches or version_mismatches:
            conn.rollback()
            return {
                "ok": False,
                "status": "vector_contract_mismatch",
                "embedding_model_mismatch_record_ids": model_mismatches,
                "vector_version_mismatch_record_ids": version_mismatches,
            }
        conn.commit()
    finally:
        conn.close()
    ann = build_memory_ann_index(db_path)
    return {"ok": True, "status": "completed", "repaired_records": repaired, "ann": ann}


def search_memory_ann(
    db_path: str | Path,
    *,
    query_vector: Any,
    query_embedding_model: str = "",
    query_vector_version: str = "",
    top_k: int = 64,
    candidate_limit: int = 256,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        ann = query_lsh_index(
            conn,
            index_name=MEMORY_ANN_INDEX_NAME,
            query_vector=query_vector,
            embedding_model=query_embedding_model,
            vector_version=query_vector_version,
            candidate_limit=candidate_limit,
        )
        if not ann["ok"] or not ann["record_ids"]:
            return []
        payloads = _load_memory_payloads(conn, memory_ids=ann["record_ids"])
        by_id = {str(payload.get("memory_id") or payload.get("id")): payload for payload in payloads}
        ranked = [by_id[memory_id] for memory_id in ann["record_ids"] if memory_id in by_id]
        for payload in ranked:
            payload["vector_score"] = float(ann["scores"].get(str(payload.get("memory_id") or payload.get("id")), 0.0))
            payload["vector_status"] = "ann_bucket_match"
        return ranked[: max(1, int(top_k))]
    finally:
        conn.close()


def search_memory_structural(
    db_path: str | Path,
    *,
    query_anchors: Sequence[dict[str, Any]],
    top_k: int = 64,
) -> list[dict[str, Any]]:
    """Scan stored structural anchors without collapsing an event graph."""

    normalized_query = _coerce_retrieval_anchors(
        {"retrieval_anchors": list(query_anchors)}
    )
    if not normalized_query:
        raise ValueError("structural memory search requires query anchors")
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        records = _load_memory_payloads(conn, limit=None)
    finally:
        conn.close()
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for record in records:
        candidate = _coerce_retrieval_anchors(record)
        if not candidate:
            continue
        score = score_structural_memory_anchors(normalized_query, candidate)
        if not score.get("ok") or float(score.get("score") or 0.0) <= 0.0:
            continue
        memory_id = str(record.get("memory_id") or record.get("id") or "")
        ranked.append(
            (
                float(score["score"]),
                float(score["coverage"]),
                memory_id,
                {
                    **record,
                    "structural_score": float(score["score"]),
                    "structural_coverage": float(score["coverage"]),
                    "matched_anchor_count": int(score["matched_anchors"]),
                    "vector_score": float(score["score"]),
                    "vector_status": "structural_anchor_exact_scan",
                },
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked[: max(1, int(top_k))]]


def search_memory_anchor_vectors(
    db_path: str | Path,
    *,
    query_vectors: Sequence[Any],
    top_k: int = 64,
) -> list[dict[str, Any]]:
    """Scan a few structural feature vectors when a complete query graph is unavailable."""

    queries = [vector for value in query_vectors if (vector := _vector_to_list(value))]
    if not queries:
        raise ValueError("anchor vector search requires query vectors")
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        records = _load_memory_payloads(conn, limit=None)
    finally:
        conn.close()
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for record in records:
        anchors = _coerce_retrieval_anchors(record)
        candidate_vectors = [
            vector
            for anchor in anchors
            for field in ("core_vector", "relation_vector", "nature_vector")
            if (vector := _vector_to_list(anchor.get(field)))
        ]
        if not candidate_vectors:
            continue
        query_scores = [
            max((_cosine(query, candidate) for candidate in candidate_vectors), default=0.0)
            for query in queries[:8]
        ]
        if not query_scores:
            continue
        score = sum(query_scores) / len(query_scores)
        memory_id = str(record.get("memory_id") or record.get("id") or "")
        ranked.append(
            (
                score,
                memory_id,
                {
                    **record,
                    "structural_score": score,
                    "structural_coverage": sum(value > 0.0 for value in query_scores)
                    / len(query_scores),
                    "matched_anchor_count": sum(value > 0.0 for value in query_scores),
                    "vector_score": score,
                    "vector_status": "structural_anchor_vector_scan",
                },
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[: max(1, int(top_k))]]


def cleanup_memory_records(
    db_path: str | Path,
    *,
    stale_days: int = 60,
    min_utility_score: float = 0.22,
    prune_ratio: float = 0.10,
    min_records_to_keep: int = 128,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prune low-utility persistent memories.

    Utility is intentionally conservative: useful memories survive if they are
    recent, have confident stages/rule traces, have successful reasons, or are
    repeatedly referenced by rule uses. This mirrors human memory consolidation:
    keep reusable experience, delete stale low-signal noise.
    """

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        rows = conn.execute("SELECT * FROM memories ORDER BY memory_time DESC, updated_at DESC, memory_id").fetchall()
        before_count = len(rows)
        if before_count == 0:
            return {
                "db_path": str(Path(db_path).expanduser()),
                "before_count": 0,
                "after_count": 0,
                "removed_memory_ids": [],
                "kept_memory_ids": [],
                "dry_run": bool(dry_run),
            }

        stale_days = max(1, int(stale_days))
        min_utility_score = max(0.0, min(float(min_utility_score), 1.0))
        prune_ratio = max(0.0, min(float(prune_ratio), 0.95))
        min_records_to_keep = max(1, int(min_records_to_keep))
        now = datetime.now(timezone.utc)

        scored: list[tuple[float, str, int]] = []
        for row in rows:
            memory_id = str(row["memory_id"])
            stage_rows = conn.execute(
                "SELECT * FROM memory_stage WHERE memory_id = ?",
                (memory_id,),
            ).fetchall()
            use_rows = conn.execute(
                "SELECT * FROM memory_rule_uses WHERE memory_id = ?",
                (memory_id,),
            ).fetchall()
            reason_rows = []
            if use_rows:
                usage_ids = [str(item["usage_id"]) for item in use_rows]
                placeholders = ",".join("?" for _ in usage_ids)
                reason_rows = conn.execute(
                    f"SELECT * FROM memory_success_reasons WHERE usage_id IN ({placeholders})",
                    usage_ids,
                ).fetchall()
            score = _persistent_memory_utility(row, stage_rows, use_rows, reason_rows, now=now, stale_days=stale_days)
            age_days = _age_days(str(row["updated_at"] or row["memory_time"] or row["created_at"] or ""), now=now)
            scored.append((score, memory_id, age_days))

        removable: list[str] = []
        for score, memory_id, age_days in sorted(scored, key=lambda item: (item[0], -item[2], item[1])):
            if before_count - len(removable) <= min_records_to_keep:
                break
            if score < min_utility_score and age_days > stale_days:
                removable.append(memory_id)

        keep_count_target = max(min_records_to_keep, int(round(before_count * (1.0 - prune_ratio))))
        if before_count - len(removable) > keep_count_target:
            already = set(removable)
            for score, memory_id, _age_days_value in sorted(scored, key=lambda item: (item[0], item[1])):
                if before_count - len(removable) <= keep_count_target:
                    break
                if memory_id in already:
                    continue
                removable.append(memory_id)
                already.add(memory_id)

        if not dry_run and removable:
            conn.executemany("DELETE FROM memories WHERE memory_id = ?", [(memory_id,) for memory_id in removable])
            conn.commit()
        kept = [memory_id for _score, memory_id, _age in scored if memory_id not in set(removable)]
        projected_after_count = before_count - len(removable)
        return {
            "db_path": str(Path(db_path).expanduser()),
            "before_count": before_count,
            "after_count": before_count if dry_run else projected_after_count,
            "projected_after_count": projected_after_count,
            "removed_memory_ids": removable,
            "kept_memory_ids": kept,
            "removed_count": len(removable),
            "dry_run": bool(dry_run),
            "stale_days": stale_days,
            "min_utility_score": min_utility_score,
            "prune_ratio": prune_ratio,
            "min_records_to_keep": min_records_to_keep,
        }
    finally:
        conn.close()


def summarize_database(db_path: str | Path) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        memories = int(conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"])
        stages = int(conn.execute("SELECT COUNT(*) AS count FROM memory_stage").fetchone()["count"])
        uses = int(conn.execute("SELECT COUNT(*) AS count FROM memory_rule_uses").fetchone()["count"])
        reasons = int(conn.execute("SELECT COUNT(*) AS count FROM memory_success_reasons").fetchone()["count"])
        vectors = int(conn.execute("SELECT COUNT(*) AS count FROM memories WHERE memory_vector IS NOT NULL").fetchone()["count"])
        return {
            "db_path": str(Path(db_path).expanduser()),
            "schema_version": SCHEMA_VERSION,
            "schema": SCHEMA_NAME,
            "reasoning_memories": memories,
            "memory_stages": stages,
            "memory_rule_uses": uses,
            "success_reasons": reasons,
            "vectors": vectors,
            "ann": ann_index_summary(conn, index_name=MEMORY_ANN_INDEX_NAME),
        }
    finally:
        conn.close()


def _persistent_memory_utility(
    memory_row: sqlite3.Row,
    stage_rows: Sequence[sqlite3.Row],
    use_rows: Sequence[sqlite3.Row],
    reason_rows: Sequence[sqlite3.Row],
    *,
    now: datetime,
    stale_days: int,
) -> float:
    age_days = _age_days(str(memory_row["updated_at"] or memory_row["memory_time"] or memory_row["created_at"] or ""), now=now)
    freshness = max(0.0, 1.0 - age_days / max(stale_days * 3, 1))
    stage_confidence = (
        sum(float(row["stage_confidence"]) for row in stage_rows) / max(len(stage_rows), 1)
        if stage_rows
        else 0.35
    )
    rule_confidence = (
        sum(float(row["rule_confidence"]) for row in use_rows) / max(len(use_rows), 1)
        if use_rows
        else 0.35
    )
    logical_strength = (
        sum(float(row["logical_strength"]) for row in use_rows) / max(len(use_rows), 1)
        if use_rows
        else 0.35
    )
    empirical_support = (
        sum(float(row["empirical_support"]) for row in use_rows) / max(len(use_rows), 1)
        if use_rows
        else 0.35
    )
    contradiction_rate = (
        sum(float(row["contradiction_rate"]) for row in use_rows) / max(len(use_rows), 1)
        if use_rows
        else 0.10
    )
    success_signal = min(1.0, len(reason_rows) / 3.0)
    reuse_signal = min(1.0, len(use_rows) / 4.0)
    debug_text = str(memory_row["memory_debug_text"] or "").strip()
    text_signal = 0.0 if len(debug_text) < 4 else min(1.0, len(debug_text) / 160.0)
    return max(
        0.0,
        min(
            1.0,
            0.18 * freshness
            + 0.20 * stage_confidence
            + 0.18 * rule_confidence
            + 0.14 * logical_strength
            + 0.12 * empirical_support
            + 0.08 * success_signal
            + 0.06 * reuse_signal
            + 0.04 * text_signal
            - 0.22 * contradiction_rate,
        ),
    )


def _age_days(value: str, *, now: datetime) -> int:
    if not value:
        return 0
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, int((now - parsed.astimezone(timezone.utc)).total_seconds() // 86400))
