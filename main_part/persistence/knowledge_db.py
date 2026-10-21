"""Vector-native Knowledge graph persistence.

Knowledge is stored only as vector nodes and vector edges. Surface text is
optional provenance for inspection; it is never used for identity, matching or
retrieval. ``knowledge_spans`` is recognized solely to perform a one-time,
atomic migration from pre-v6 databases and is dropped immediately afterwards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from . import base as shared_db
from .ann_index import ann_index_summary, query_lsh_index, rebuild_lsh_index
from ..common.math_utils import cosine as shared_cosine
from ..common.normalize import safe_float as shared_safe_float
from ..identity import stable_content_id
from ..vector_contracts import canonical_embedding_model, coerce_vector


SCHEMA_VERSION = 6
SCHEMA_NAME = "knowledge_vector_graph_ann_v6"
KNOWLEDGE_ANN_INDEX_NAME = "knowledge_vector_edges"
ALLOWED_KNOWLEDGE_TYPES = {
    "common_sense", "domain_fact", "ontology", "relation_prior", "physical_law",
}
_VECTOR_FIELDS = ("core_vector", "relation_vector", "nature_vector")
_NODE_ROLES = ("core", "relation", "nature")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    return shared_db.connect(db_path, foreign_keys=False)


def _json_dumps(value: Any) -> str:
    return shared_db.json_dumps(value, none_as_object=True, stringify=True)


def _json_loads(value: str | None, default: Any) -> Any:
    return shared_db.json_loads(value, default)


def _vector(value: Any) -> list[float] | None:
    result = coerce_vector(value)
    return result if result and any(abs(item) > 1e-12 for item in result) else None


def _blob(value: Any) -> bytes | None:
    return shared_db.vector_to_blob(value, reject_zero=True, inclusive_zero=True)


def _unblob(value: Any) -> list[float] | None:
    return shared_db.blob_to_vector(value)


def _safe_float(value: Any, default: float) -> float:
    return shared_safe_float(value, default)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
    ).fetchone() is not None


def _node_id(vector: list[float], model: str, version: str) -> str:
    """Identity derives solely from a vector and its contract, never a label."""
    return stable_content_id(
        "knode",
        {"embedding_model": canonical_embedding_model(model), "vector_version": str(version), "vector": vector},
        length=32,
    )


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    # v5 graph tables were only a shadow of knowledge_spans.  When that source
    # exists, discard the shadow and rebuild graph-native tables from the source
    # in the same migration transaction.  A detached v5 shadow has no complete
    # edge metrics and is intentionally rejected instead of silently guessing.
    if _table_exists(conn, "knowledge_nodes"):
        node_columns = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_nodes)").fetchall()}
        if "vector_blob" not in node_columns:
            if not _table_exists(conn, "knowledge_spans"):
                raise RuntimeError("cannot migrate detached legacy Knowledge graph without knowledge_spans source")
            conn.execute("DROP TABLE IF EXISTS knowledge_edges")
            conn.execute("DROP TABLE IF EXISTS knowledge_nodes")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS knowledge_nodes (
            node_id TEXT PRIMARY KEY,
            vector_blob BLOB NOT NULL,
            embedding_model TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            vector_version TEXT NOT NULL,
            debug_text TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active', source TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS knowledge_edges (
            knowledge_id TEXT PRIMARY KEY,
            core_node_id TEXT NOT NULL, relation_node_id TEXT NOT NULL, nature_node_id TEXT NOT NULL,
            knowledge_type TEXT NOT NULL DEFAULT 'ontology', confidence REAL NOT NULL DEFAULT 0.5,
            priority REAL NOT NULL DEFAULT 0.5, usage_count INTEGER NOT NULL DEFAULT 0,
            hit_count INTEGER NOT NULL DEFAULT 0, useful_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_used_at TEXT NOT NULL DEFAULT '')"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_edges_status ON knowledge_edges(status, priority DESC, confidence DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_edges_nodes ON knowledge_edges(core_node_id, relation_node_id, nature_node_id)")
    _migrate_legacy_table(conn)
    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_name', ?)", (SCHEMA_NAME,))
    conn.commit()


def _legacy_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM knowledge_spans").fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _migrate_legacy_table(conn: sqlite3.Connection) -> None:
    """Migrate pre-v6 rows once, then remove the old runtime table atomically."""
    if not _table_exists(conn, "knowledge_spans"):
        return
    rows = _legacy_rows(conn)
    try:
        for row in rows:
            _store_record(conn, _coerce_record({
                "id": row.get("id"), "core_vector": _unblob(row.get("core_vector")),
                "relation_vector": _unblob(row.get("relation_vector")), "nature_vector": _unblob(row.get("nature_vector")),
                "embedding_model": row.get("embedding_model"), "vector_version": row.get("vector_version"),
                "knowledge_type": row.get("knowledge_type"), "confidence": row.get("confidence"), "priority": row.get("priority"),
                "usage_count": row.get("usage_count"), "hit_count": row.get("hit_count"), "useful_count": row.get("useful_count"),
                "failure_count": row.get("failure_count"), "status": row.get("status"), "source": row.get("source"),
                "last_used_at": row.get("last_used_at"), "created_at": row.get("created_at"), "updated_at": row.get("updated_at"),
                "metadata": _json_loads(row.get("metadata_json"), {}),
                "debug": {"core": row.get("core", ""), "relation": row.get("relation", ""), "nature": row.get("nature", "")},
            }), now=str(row.get("updated_at") or _now_iso()), preserve_metrics=True)
        conn.execute("DROP TABLE knowledge_spans")
        conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('legacy_knowledge_spans_migrated_at', ?)", (_now_iso(),))
    except Exception:
        conn.rollback()
        raise


def _payload(record: Any) -> dict[str, Any]:
    return dict(record.to_dict()) if hasattr(record, "to_dict") else dict(record or {})


def _debug_value(payload: Mapping[str, Any], role: str) -> str:
    debug = payload.get("debug") if isinstance(payload.get("debug"), Mapping) else {}
    return str(debug.get(role) or payload.get(f"{role}_debug") or payload.get(role) or "").strip()


def _coerce_record(record: Any, *, embedding_model: str = "") -> dict[str, Any]:
    payload = _payload(record)
    vectors = {field: _vector(payload.get(field)) for field in _VECTOR_FIELDS}
    usable = [item for item in vectors.values() if item]
    model = canonical_embedding_model(payload.get("embedding_model") or embedding_model or "")
    version = str(payload.get("vector_version") or payload.get("embedding_version") or "").strip()
    dimensions = {len(item) for item in usable}
    if len(dimensions) > 1:
        raise ValueError("knowledge edge vectors must have one shared dimension")
    if usable and (not model or not version):
        raise ValueError("vectorized knowledge requires embedding_model and vector_version")
    knowledge_id = str(payload.get("knowledge_id") or payload.get("id") or "").strip()
    if not knowledge_id:
        raise ValueError("knowledge_id is required; vector identity is not an authored edge identifier")
    kind = str(payload.get("knowledge_type") or payload.get("type") or "ontology").strip()
    if kind not in ALLOWED_KNOWLEDGE_TYPES:
        raise ValueError(f"unsupported knowledge_type: {kind}")
    status = str(payload.get("status") or "").strip() or "active"
    if not usable and status == "active":
        status = "pending_vectorization"
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {}
    metadata.pop("knowledge_network", None)
    metadata.pop("semantic_alignment_definition", None)
    if metadata.get('span_schema'):
        from ..knowledge_schema_contract import validate_span_schema
        validate_span_schema(metadata['span_schema'])
    return {
        "knowledge_id": knowledge_id, "vectors": vectors, "embedding_model": model, "vector_version": version,
        "vector_dimension": next(iter(dimensions), 0), "knowledge_type": kind,
        "confidence": _clamp(_safe_float(payload.get("confidence", payload.get("confidence_prior", .5)), .5)),
        "priority": _clamp(_safe_float(payload.get("priority", payload.get("confidence", .5)), .5)),
        "usage_count": max(0, int(_safe_float(payload.get("usage_count"), 0))), "hit_count": max(0, int(_safe_float(payload.get("hit_count"), 0))),
        "useful_count": max(0, int(_safe_float(payload.get("useful_count"), 0))), "failure_count": max(0, int(_safe_float(payload.get("failure_count"), 0))),
        "status": status, "source": str(payload.get("source") or "").strip(), "metadata": metadata,
        "debug": {role: _debug_value(payload, role) for role in _NODE_ROLES}, "created_at": str(payload.get("created_at") or ""), "last_used_at": str(payload.get("last_used_at") or ""),
    }


def _store_node(conn: sqlite3.Connection, *, vector: list[float], model: str, version: str, debug_text: str, status: str, source: str, now: str) -> str:
    node_id = _node_id(vector, model, version)
    conn.execute(
        """INSERT INTO knowledge_nodes(node_id, vector_blob, embedding_model, vector_dimension, vector_version, debug_text, status, source, metadata_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
           ON CONFLICT(node_id) DO UPDATE SET debug_text=CASE WHEN excluded.debug_text != '' THEN excluded.debug_text ELSE knowledge_nodes.debug_text END,
             status=excluded.status, source=excluded.source, updated_at=excluded.updated_at""",
        (node_id, _blob(vector), model, len(vector), version, debug_text, status, source, now, now),
    )
    return node_id


def _store_record(conn: sqlite3.Connection, payload: Mapping[str, Any], *, now: str, preserve_metrics: bool = False) -> bool:
    vectors = payload["vectors"]
    if not all(vectors.values()):
        return False
    node_ids = {role: _store_node(conn, vector=vectors[f"{role}_vector"], model=payload["embedding_model"], version=payload["vector_version"], debug_text=payload["debug"][role], status=payload["status"], source=payload["source"], now=now) for role in _NODE_ROLES}
    metrics = (payload["usage_count"], payload["hit_count"], payload["useful_count"], payload["failure_count"])
    conn.execute(
        """INSERT INTO knowledge_edges(knowledge_id, core_node_id, relation_node_id, nature_node_id, knowledge_type, confidence, priority, usage_count, hit_count, useful_count, failure_count, status, source, metadata_json, created_at, updated_at, last_used_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(knowledge_id) DO UPDATE SET core_node_id=excluded.core_node_id, relation_node_id=excluded.relation_node_id, nature_node_id=excluded.nature_node_id, knowledge_type=excluded.knowledge_type, confidence=excluded.confidence, priority=excluded.priority, status=excluded.status, source=excluded.source, metadata_json=excluded.metadata_json,
              usage_count=CASE WHEN ? THEN excluded.usage_count ELSE MAX(knowledge_edges.usage_count, excluded.usage_count) END, hit_count=CASE WHEN ? THEN excluded.hit_count ELSE MAX(knowledge_edges.hit_count, excluded.hit_count) END, useful_count=CASE WHEN ? THEN excluded.useful_count ELSE MAX(knowledge_edges.useful_count, excluded.useful_count) END, failure_count=CASE WHEN ? THEN excluded.failure_count ELSE MAX(knowledge_edges.failure_count, excluded.failure_count) END, updated_at=excluded.updated_at, last_used_at=CASE WHEN excluded.last_used_at != '' THEN excluded.last_used_at ELSE knowledge_edges.last_used_at END""",
        (payload["knowledge_id"], node_ids["core"], node_ids["relation"], node_ids["nature"], payload["knowledge_type"], payload["confidence"], payload["priority"], *metrics, payload["status"], payload["source"], _json_dumps(payload["metadata"]), payload["created_at"] or now, now, payload["last_used_at"], int(preserve_metrics), int(preserve_metrics), int(preserve_metrics), int(preserve_metrics)),
    )
    return True


_EDGE_SELECT = """SELECT e.*, c.vector_blob AS core_vector_blob, r.vector_blob AS relation_vector_blob, n.vector_blob AS nature_vector_blob, c.debug_text AS core_debug, r.debug_text AS relation_debug, n.debug_text AS nature_debug, c.embedding_model AS embedding_model, c.vector_dimension AS vector_dimension, c.vector_version AS vector_version FROM knowledge_edges e JOIN knowledge_nodes c ON c.node_id=e.core_node_id JOIN knowledge_nodes r ON r.node_id=e.relation_node_id JOIN knowledge_nodes n ON n.node_id=e.nature_node_id"""


def _row_payload(row: sqlite3.Row, *, score: float | None = None, include_vectors: bool = True) -> dict[str, Any]:
    payload = {"id": str(row["knowledge_id"]), "knowledge_id": str(row["knowledge_id"]), "edge_id": str(row["knowledge_id"]), "core_node_id": str(row["core_node_id"]), "relation_node_id": str(row["relation_node_id"]), "nature_node_id": str(row["nature_node_id"]), "embedding_model": str(row["embedding_model"]), "vector_dimension": int(row["vector_dimension"]), "vector_version": str(row["vector_version"]), "knowledge_type": str(row["knowledge_type"]), "confidence": float(row["confidence"]), "confidence_prior": float(row["confidence"]), "priority": float(row["priority"]), "usage_count": int(row["usage_count"]), "hit_count": int(row["hit_count"]), "useful_count": int(row["useful_count"]), "failure_count": int(row["failure_count"]), "status": str(row["status"]), "source": str(row["source"]), "created_at": str(row["created_at"]), "updated_at": str(row["updated_at"]), "last_used_at": str(row["last_used_at"]), "metadata": _json_loads(row["metadata_json"], {}), "debug": {"core": str(row["core_debug"]), "relation": str(row["relation_debug"]), "nature": str(row["nature_debug"])}}
    if include_vectors: payload.update({"core_vector": _unblob(row["core_vector_blob"]), "relation_vector": _unblob(row["relation_vector_blob"]), "nature_vector": _unblob(row["nature_vector_blob"])})
    if score is not None: payload["score"] = round(float(score), 6)
    return payload


def _graph_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {"nodes": int(conn.execute("SELECT COUNT(*) FROM knowledge_nodes").fetchone()[0]), "edges": int(conn.execute("SELECT COUNT(*) FROM knowledge_edges").fetchone()[0])}


def database_is_empty(db_path: str | Path) -> bool:
    return not load_knowledge_records(db_path, limit=1)


def upsert_knowledge_records(db_path: str | Path, records: Sequence[Any], *, replace: bool = False, embedding_model: str = "") -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        if replace:
            conn.execute("DELETE FROM knowledge_edges"); conn.execute("DELETE FROM knowledge_nodes"); conn.execute("DELETE FROM ann_vector_index WHERE index_name=?", (KNOWLEDGE_ANN_INDEX_NAME,))
        saved = pending = vectors = 0; now = _now_iso()
        for record in records:
            payload = _coerce_record(record, embedding_model=embedding_model); vectors += sum(bool(value) for value in payload["vectors"].values())
            if _store_record(conn, payload, now=now): saved += 1
            else: pending += 1
        conn.commit()
        return {"enabled": True, "status": "completed", "db_path": str(Path(db_path).expanduser()), "saved": saved, "saved_knowledge_records": saved, "pending_vectorization": pending, "saved_vectors": vectors, "knowledge_graph": _graph_counts(conn), "schema": SCHEMA_NAME}
    finally:
        conn.close()


def load_knowledge_records(db_path: str | Path, *, limit: int | None = 512, knowledge_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn); sql, params = _EDGE_SELECT + " WHERE e.status != 'deleted'", []
        if knowledge_ids is not None:
            if not knowledge_ids: return []
            sql += " AND e.knowledge_id IN (" + ",".join("?" for _ in knowledge_ids) + ")"; params.extend(knowledge_ids)
        sql += " ORDER BY e.priority DESC, e.confidence DESC, e.useful_count DESC, e.knowledge_id"
        if limit is not None and limit > 0: sql += " LIMIT ?"; params.append(int(limit))
        return [_row_payload(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _rank(rows: Sequence[sqlite3.Row], query_vector: Any, top_k: int, min_score: float) -> list[tuple[float, sqlite3.Row]]:
    query = _vector(query_vector)
    if not query: raise ValueError("knowledge search requires query_vector")
    ranked = []
    for row in rows:
        score = max(shared_cosine(query, row[field], positive_only=True) for field in ("core_vector_blob", "relation_vector_blob", "nature_vector_blob"))
        if score > 0 and score >= float(min_score): ranked.append((score, row))
    return sorted(ranked, key=lambda item: (item[0], float(item[1]["priority"]), str(item[1]["knowledge_id"])), reverse=True)[:max(1, int(top_k))]


def _update_usage(conn: sqlite3.Connection, selected: Sequence[tuple[float, sqlite3.Row]]) -> None:
    now = _now_iso(); conn.executemany("UPDATE knowledge_edges SET usage_count=usage_count+1, hit_count=hit_count+1, last_used_at=?, updated_at=? WHERE knowledge_id=?", [(now, now, str(row["knowledge_id"])) for _, row in selected])


def search_knowledge_records(db_path: str | Path, *, query: str = "", keys: Sequence[str] | None = None, query_vector: Any = None, top_k: int = 128, min_score: float = 0.0, update_usage: bool = True) -> list[dict[str, Any]]:
    del query, keys
    conn = _connect(db_path)
    try:
        _init_schema(conn); selected = _rank(conn.execute(_EDGE_SELECT + " WHERE e.status='active'").fetchall(), query_vector, top_k, min_score)
        if update_usage and selected: _update_usage(conn, selected); conn.commit()
        return [_row_payload(row, score=score) for score, row in selected]
    finally:
        conn.close()


def search_knowledge_records_batch(db_path: str | Path, queries: Sequence[Mapping[str, Any]], *, update_usage: bool = True, include_vectors: bool = True) -> list[list[dict[str, Any]]]:
    conn = _connect(db_path)
    try:
        _init_schema(conn); rows = conn.execute(_EDGE_SELECT + " WHERE e.status='active'").fetchall(); batches = [_rank(rows, item.get("query_vector"), int(item.get("top_k", 128) or 128), float(item.get("min_score", 0) or 0)) for item in queries]
        if update_usage: _update_usage(conn, [item for batch in batches for item in batch]); conn.commit()
        return [[_row_payload(row, score=score, include_vectors=include_vectors) for score, row in batch] for batch in batches]
    finally:
        conn.close()


def build_knowledge_ann_index(db_path: str | Path, *, embedding_model: str = "", vector_version: str = "", bits: int = 12) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn); rows = conn.execute(_EDGE_SELECT + " WHERE e.status='active'").fetchall(); ann_rows = []
        for row in rows:
            for field in _VECTOR_FIELDS:
                vector = _unblob(row[field.replace("_vector", "_vector_blob")])
                if vector: ann_rows.append({"record_id": row["knowledge_id"], "vector_field": field, "vector": vector, "embedding_model": row["embedding_model"] or embedding_model, "vector_version": row["vector_version"] or vector_version})
        summary = rebuild_lsh_index(conn, index_name=KNOWLEDGE_ANN_INDEX_NAME, rows=ann_rows, embedding_model=embedding_model, vector_version=vector_version, now=_now_iso(), bits=bits); conn.commit()
        return {**summary, "record_count": len(rows), "db_path": str(Path(db_path).expanduser())}
    finally:
        conn.close()


def ensure_knowledge_ann_index(db_path: str | Path, *, bits: int = 12) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn); expected = 3 * int(conn.execute("SELECT COUNT(*) FROM knowledge_edges WHERE status='active'").fetchone()[0]); current = ann_index_summary(conn, index_name=KNOWLEDGE_ANN_INDEX_NAME)
    finally:
        conn.close()
    return ({**build_knowledge_ann_index(db_path, bits=bits), "rebuilt": True} if int(current["indexed_vectors"]) != expected else {**current, "record_count": expected // 3, "db_path": str(Path(db_path).expanduser()), "rebuilt": False})


def search_knowledge_ann(db_path: str | Path, *, query: dict[str, Any], top_k: int = 10, limit: int | None = None, weights: tuple[float, float, float] = (.42, .3, .28), update_usage: bool = True) -> list[dict[str, Any]]:
    del weights
    vector = query.get("core_vector") or query.get("relation_vector") or query.get("nature_vector") or query.get("vector")
    if not _vector(vector): return []
    conn = _connect(db_path)
    try:
        _init_schema(conn); ann = query_lsh_index(conn, index_name=KNOWLEDGE_ANN_INDEX_NAME, query_vector=vector, embedding_model=str(query.get("embedding_model") or ""), vector_version=str(query.get("vector_version") or ""), candidate_limit=limit or max(64, top_k * 16))
        if not ann["ok"] or not ann["record_ids"]: return []
        placeholders = ",".join("?" for _ in ann["record_ids"]); rows = conn.execute(_EDGE_SELECT + f" WHERE e.status='active' AND e.knowledge_id IN ({placeholders})", ann["record_ids"]).fetchall(); by_id = {str(row["knowledge_id"]): row for row in rows}
        selected = [(float(ann["scores"].get(item, 0)), by_id[item]) for item in ann["record_ids"] if item in by_id]; selected.sort(key=lambda item: (item[0], float(item[1]["priority"]), str(item[1]["knowledge_id"])), reverse=True); selected = selected[:top_k]
        if update_usage and selected: _update_usage(conn, selected); conn.commit()
        result = [_row_payload(row, score=score) for score, row in selected]
        for item in result: item["vector_status"] = "ann_bucket_match"
        return result
    finally:
        conn.close()


def record_knowledge_usage(db_path: str | Path, knowledge_ids: Sequence[str], *, useful: bool = False, failed: bool = False) -> dict[str, Any]:
    ids = [str(item).strip() for item in knowledge_ids if str(item).strip()]
    if not ids: return {"updated": 0}
    conn = _connect(db_path)
    try:
        _init_schema(conn); now = _now_iso(); conn.executemany("UPDATE knowledge_edges SET useful_count=useful_count+?, failure_count=failure_count+?, last_used_at=?, updated_at=? WHERE knowledge_id=?", [(int(useful), int(failed), now, now, item) for item in ids]); conn.commit(); return {"updated": len(ids), "useful": bool(useful), "failed": bool(failed)}
    finally:
        conn.close()


def remove_knowledge_records(db_path: str | Path, knowledge_ids: Sequence[str]) -> dict[str, int]:
    """Remove retired authored edges and their now-unreferenced vector nodes."""
    ids = [str(item).strip() for item in knowledge_ids if str(item).strip()]
    if not ids:
        return {"removed_edges": 0, "removed_orphan_nodes": 0}
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        placeholders = ",".join("?" for _ in ids)
        before = int(conn.execute(f"SELECT COUNT(*) FROM knowledge_edges WHERE knowledge_id IN ({placeholders})", ids).fetchone()[0])
        conn.execute(f"DELETE FROM knowledge_edges WHERE knowledge_id IN ({placeholders})", ids)
        cursor = conn.execute("DELETE FROM knowledge_nodes WHERE node_id NOT IN (SELECT core_node_id FROM knowledge_edges UNION SELECT relation_node_id FROM knowledge_edges UNION SELECT nature_node_id FROM knowledge_edges)")
        conn.execute(f"DELETE FROM ann_vector_index WHERE index_name=? AND record_id IN ({placeholders})", [KNOWLEDGE_ANN_INDEX_NAME, *ids])
        conn.commit()
        return {"removed_edges": before, "removed_orphan_nodes": max(0, int(cursor.rowcount or 0))}
    finally:
        conn.close()


def reprioritize_knowledge_records(db_path: str | Path, *, retire_failure_rate: float = .60, low_usage_threshold: int = 0, delete_retired: bool = False) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn); rows = conn.execute("SELECT * FROM knowledge_edges WHERE status != 'deleted'").fetchall(); updated = retired = deleted = 0; now = _now_iso()
        for row in rows:
            useful, failure, usage, confidence = int(row["useful_count"]), int(row["failure_count"]), int(row["usage_count"]), float(row["confidence"]); rate = failure / max(useful + failure, 1); priority = _clamp(.45 * confidence + .25 * min(1, usage / 20) + .35 * ((useful + .5) / max(useful + failure + 1, 1)) - min(.45, failure / 10)); status = str(row["status"])
            if usage <= low_usage_threshold and row["knowledge_type"] not in {"physical_law", "common_sense"}: priority = min(priority, .2)
            if failure >= 2 and rate >= retire_failure_rate and row["knowledge_type"] not in {"physical_law", "common_sense"}: status, confidence, retired = "retired", max(.05, confidence - .1), retired + 1
            if delete_retired and status == "retired": status, deleted = "deleted", deleted + 1
            conn.execute("UPDATE knowledge_edges SET status=?, priority=?, confidence=?, updated_at=? WHERE knowledge_id=?", (status, priority, confidence, now, row["knowledge_id"])); updated += 1
        conn.commit(); return {"updated": updated, "retired": retired, "deleted": deleted}
    finally:
        conn.close()


def repair_knowledge_vector_contract(db_path: str | Path, *, embedding_model: str, vector_version: str) -> dict[str, Any]:
    expected = canonical_embedding_model(embedding_model); version = str(vector_version or ""); rows = load_knowledge_records(db_path, limit=None); bad_model = [item["knowledge_id"] for item in rows if expected and item["embedding_model"] and item["embedding_model"] != expected]; bad_version = [item["knowledge_id"] for item in rows if version and item["vector_version"] and item["vector_version"] != version]
    if bad_model or bad_version: return {"ok": False, "status": "vector_contract_mismatch", "embedding_model_mismatch_record_ids": bad_model, "vector_version_mismatch_record_ids": bad_version}
    return {"ok": True, "status": "completed", "repaired_records": len(rows), "ann": build_knowledge_ann_index(db_path)}


def load_knowledge_network(db_path: str | Path, *, knowledge_ids: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    records = load_knowledge_records(db_path, limit=None, knowledge_ids=knowledge_ids); nodes: dict[str, dict[str, Any]] = {}; edges = []
    for item in records:
        for role in _NODE_ROLES:
            node_id = item[f"{role}_node_id"]; nodes.setdefault(node_id, {"node_id": node_id, "vector": item[f"{role}_vector"], "embedding_model": item["embedding_model"], "vector_version": item["vector_version"], "debug_text": item["debug"][role]})
        edges.append({"edge_id": item["knowledge_id"], "knowledge_id": item["knowledge_id"], "core_node_id": item["core_node_id"], "relation_node_id": item["relation_node_id"], "nature_node_id": item["nature_node_id"], "metadata": item["metadata"]})
    return {"nodes": sorted(nodes.values(), key=lambda item: item["node_id"]), "edges": edges}


def summarize_database(db_path: str | Path) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn); row = conn.execute("SELECT COUNT(*) records, SUM(usage_count) usage_count, SUM(useful_count) useful_count, SUM(failure_count) failure_count FROM knowledge_edges WHERE status != 'deleted'").fetchone(); by_type = {str(item["knowledge_type"]): int(item["count"]) for item in conn.execute("SELECT knowledge_type, COUNT(*) count FROM knowledge_edges WHERE status != 'deleted' GROUP BY knowledge_type")}; by_status = {str(item["status"]): int(item["count"]) for item in conn.execute("SELECT status, COUNT(*) count FROM knowledge_edges GROUP BY status")}
        return {"db_path": str(Path(db_path).expanduser()), "schema_version": SCHEMA_VERSION, "schema": SCHEMA_NAME, "knowledge_records": int(row["records"] or 0), "vectors": 3 * int(row["records"] or 0), "usage_count": int(row["usage_count"] or 0), "useful_count": int(row["useful_count"] or 0), "failure_count": int(row["failure_count"] or 0), "by_type": by_type, "by_status": by_status, "knowledge_graph": _graph_counts(conn), "ann": ann_index_summary(conn, index_name=KNOWLEDGE_ANN_INDEX_NAME)}
    finally:
        conn.close()


__all__ = ["ALLOWED_KNOWLEDGE_TYPES", "SCHEMA_NAME", "SCHEMA_VERSION", "database_is_empty", "upsert_knowledge_records", "load_knowledge_records", "search_knowledge_records", "search_knowledge_records_batch", "search_knowledge_ann", "build_knowledge_ann_index", "ensure_knowledge_ann_index", "record_knowledge_usage", "remove_knowledge_records", "reprioritize_knowledge_records", "repair_knowledge_vector_contract", "load_knowledge_network", "summarize_database"]
