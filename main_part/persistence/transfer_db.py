"""Reviewed knowledge-transfer batches and controlled model-DB imports."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..rule_layers import normalize_rule_layer
from ..vector_contracts import canonical_embedding_model, coerce_vector
from . import knowledge_db, rule_db


SCHEMA_NAME = "knowledge_transfer_batches_v1"
SCHEMA_VERSION = 1
REVIEWED_STATUS = "reviewed"
SUPPORTED_RECORD_TYPES = frozenset({"knowledge", "rule"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transfer_batches (
            batch_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            source_manifest_json TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            vector_version TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            reviewed_at TEXT NOT NULL DEFAULT '',
            reviewed_by TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS transfer_knowledge (
            batch_id TEXT NOT NULL,
            knowledge_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY(batch_id, knowledge_id),
            FOREIGN KEY(batch_id) REFERENCES transfer_batches(batch_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS transfer_rules (
            batch_id TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            PRIMARY KEY(batch_id, rule_id),
            FOREIGN KEY(batch_id) REFERENCES transfer_batches(batch_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS transfer_import_history (
            import_id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL,
            batch_hash TEXT NOT NULL,
            knowledge_db_path TEXT NOT NULL,
            rule_db_path TEXT NOT NULL,
            status TEXT NOT NULL,
            backup_knowledge_path TEXT NOT NULL DEFAULT '',
            backup_rule_path TEXT NOT NULL DEFAULT '',
            summary_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(batch_id) REFERENCES transfer_batches(batch_id)
        );
        CREATE INDEX IF NOT EXISTS idx_transfer_import_identity
        ON transfer_import_history(batch_id, batch_hash, knowledge_db_path, rule_db_path, status);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_name', ?)",
        (SCHEMA_NAME,),
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def create_transfer_batch(
    db_path: str | Path,
    *,
    batch_id: str,
    source_manifest: Mapping[str, Any],
    embedding_model: str,
    vector_dimension: int,
    vector_version: str,
) -> dict[str, Any]:
    normalized_batch_id = str(batch_id or "").strip()
    if not normalized_batch_id:
        raise ValueError("transfer batch_id is required")
    model = canonical_embedding_model(embedding_model)
    dimension = int(vector_dimension)
    version = str(vector_version or "").strip()
    if not model or dimension <= 0 or not version:
        raise ValueError("transfer batch requires embedding_model, vector_dimension > 0, and vector_version")
    if not source_manifest:
        raise ValueError("transfer batch requires a source manifest")
    _record_types_from_manifest(source_manifest)

    conn = _connect(db_path)
    try:
        _init_schema(conn)
        conn.execute(
            """
            INSERT INTO transfer_batches(
                batch_id, status, source_manifest_json, embedding_model,
                vector_dimension, vector_version, created_at
            ) VALUES (?, 'draft', ?, ?, ?, ?, ?)
            """,
            (
                normalized_batch_id,
                _canonical_json(dict(source_manifest)),
                model,
                dimension,
                version,
                _now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return inspect_transfer_batch(db_path, batch_id=normalized_batch_id)


def stage_transfer_records(
    db_path: str | Path,
    *,
    batch_id: str,
    knowledge_records: Sequence[Mapping[str, Any]],
    rule_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        batch = _require_batch(conn, batch_id)
        if batch["status"] != "draft":
            raise RuntimeError("reviewed transfer batches are immutable")
        for record in knowledge_records:
            payload = dict(record)
            knowledge_id = str(payload.get("knowledge_id") or payload.get("id") or "").strip()
            if not knowledge_id:
                raise ValueError("transfer knowledge record requires knowledge_id")
            payload["knowledge_id"] = knowledge_id
            conn.execute(
                """
                INSERT INTO transfer_knowledge(batch_id, knowledge_id, payload_json, payload_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_id, knowledge_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    payload_hash=excluded.payload_hash
                """,
                (batch_id, knowledge_id, _canonical_json(payload), _content_hash(payload)),
            )
        for record in rule_records:
            payload = dict(record)
            rule_id = str(payload.get("rule_id") or payload.get("id") or "").strip()
            if not rule_id:
                raise ValueError("transfer rule record requires rule_id")
            payload["rule_id"] = rule_id
            conn.execute(
                """
                INSERT INTO transfer_rules(batch_id, rule_id, payload_json, payload_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_id, rule_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    payload_hash=excluded.payload_hash
                """,
                (batch_id, rule_id, _canonical_json(payload), _content_hash(payload)),
            )
        conn.commit()
    finally:
        conn.close()
    return inspect_transfer_batch(db_path, batch_id=batch_id)


def review_transfer_batch(
    db_path: str | Path,
    *,
    batch_id: str,
    reviewed_by: str,
) -> dict[str, Any]:
    reviewer = str(reviewed_by or "").strip()
    if not reviewer:
        raise ValueError("reviewed_by is required")
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        batch = _require_batch(conn, batch_id)
        if batch["status"] == REVIEWED_STATUS:
            return _inspect_transfer_batch_conn(conn, batch_id)
        knowledge_records, rule_records = _load_batch_records(conn, batch_id)
        validation = _validate_batch_records(batch, knowledge_records, rule_records)
        if not validation["ok"]:
            raise ValueError("transfer batch validation failed: " + "; ".join(validation["errors"]))
        batch_hash = _content_hash(
            {
                "batch_id": batch_id,
                "source_manifest": json.loads(batch["source_manifest_json"]),
                "knowledge": knowledge_records,
                "rules": rule_records,
            }
        )
        conn.execute(
            """
            UPDATE transfer_batches
            SET status = ?, content_hash = ?, reviewed_at = ?, reviewed_by = ?
            WHERE batch_id = ?
            """,
            (REVIEWED_STATUS, batch_hash, _now_iso(), reviewer, batch_id),
        )
        conn.commit()
        return _inspect_transfer_batch_conn(conn, batch_id)
    finally:
        conn.close()


def inspect_transfer_batch(db_path: str | Path, *, batch_id: str) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        _init_schema(conn)
        return _inspect_transfer_batch_conn(conn, batch_id)
    finally:
        conn.close()


def _inspect_transfer_batch_conn(conn: sqlite3.Connection, batch_id: str) -> dict[str, Any]:
    batch = _require_batch(conn, batch_id)
    knowledge_records, rule_records = _load_batch_records(conn, batch_id)
    validation = _validate_batch_records(batch, knowledge_records, rule_records)
    record_types = _batch_record_types(batch)
    return {
        "batch_id": str(batch["batch_id"]),
        "status": str(batch["status"]),
        "content_hash": str(batch["content_hash"]),
        "embedding_model": str(batch["embedding_model"]),
        "vector_dimension": int(batch["vector_dimension"]),
        "vector_version": str(batch["vector_version"]),
        "record_types": list(record_types),
        "knowledge_count": len(knowledge_records),
        "rule_count": len(rule_records),
        "validation": validation,
    }


def import_transfer_batch(
    transfer_db_path: str | Path,
    *,
    batch_id: str,
    backup_dir: str | Path,
    knowledge_db_path: str | Path | None = None,
    rule_db_path: str | Path | None = None,
) -> dict[str, Any]:
    transfer_path = Path(transfer_db_path).expanduser().resolve()
    backup_root = Path(backup_dir).expanduser().resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    lock_path = backup_root / ".transfer_import.lock"
    lock_fd = _acquire_import_lock(lock_path)
    import_id = f"transfer-import-{uuid4()}"
    backup_knowledge = backup_root / f"{import_id}.knowledge.sqlite"
    backup_rule = backup_root / f"{import_id}.rule.sqlite"
    knowledge_path: Path | None = None
    rule_path: Path | None = None

    try:
        conn = _connect(transfer_path)
        try:
            _init_schema(conn)
            batch = _require_batch(conn, batch_id)
            if batch["status"] != REVIEWED_STATUS or not batch["content_hash"]:
                raise RuntimeError("only reviewed transfer batches can be imported")
            record_types = _batch_record_types(batch)
            knowledge_path = _resolve_selected_db_path(
                knowledge_db_path,
                record_type="knowledge",
                selected="knowledge" in record_types,
            )
            rule_path = _resolve_selected_db_path(
                rule_db_path,
                record_type="rule",
                selected="rule" in record_types,
            )
            knowledge_path_text = str(knowledge_path) if knowledge_path is not None else ""
            rule_path_text = str(rule_path) if rule_path is not None else ""
            existing = conn.execute(
                """
                SELECT import_id, summary_json FROM transfer_import_history
                WHERE batch_id = ? AND batch_hash = ? AND knowledge_db_path = ?
                  AND rule_db_path = ? AND status = 'completed'
                ORDER BY created_at DESC LIMIT 1
                """,
                (batch_id, batch["content_hash"], knowledge_path_text, rule_path_text),
            ).fetchone()
            if existing is not None:
                return {
                    "status": "already_imported",
                    "import_id": str(existing["import_id"]),
                    "batch_id": batch_id,
                    "summary": json.loads(existing["summary_json"]),
                }
            knowledge_records, rule_records = _load_batch_records(conn, batch_id)
            validation = _validate_batch_records(batch, knowledge_records, rule_records)
            if not validation["ok"]:
                raise RuntimeError("reviewed transfer batch no longer validates")
            conn.execute(
                """
                INSERT INTO transfer_import_history(
                    import_id, batch_id, batch_hash, knowledge_db_path, rule_db_path,
                    status, backup_knowledge_path, backup_rule_path, created_at
                ) VALUES (?, ?, ?, ?, ?, 'processing', ?, ?, ?)
                """,
                (
                    import_id,
                    batch_id,
                    str(batch["content_hash"]),
                    knowledge_path_text,
                    rule_path_text,
                    str(backup_knowledge) if knowledge_path is not None else "",
                    str(backup_rule) if rule_path is not None else "",
                    _now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        _ensure_model_databases(knowledge_path, rule_path)
        if knowledge_path is not None:
            _sqlite_backup(knowledge_path, backup_knowledge)
        if rule_path is not None:
            _sqlite_backup(rule_path, backup_rule)

        try:
            annotated_knowledge = [
                _annotate_transfer_record(record, batch_id=batch_id, batch_hash=str(batch["content_hash"]))
                for record in knowledge_records
            ]
            annotated_rules = [
                _annotate_transfer_record(record, batch_id=batch_id, batch_hash=str(batch["content_hash"]))
                for record in rule_records
            ]
            knowledge_summary = (
                knowledge_db.upsert_knowledge_records(
                    knowledge_path,
                    annotated_knowledge,
                    replace=False,
                    embedding_model=str(batch["embedding_model"]),
                )
                if knowledge_path is not None
                else {"status": "not_selected", "count": 0}
            )
            rule_summary = (
                rule_db.upsert_rule_content_records(
                    rule_path,
                    annotated_rules,
                    rule_kind="content",
                    replace_kind=False,
                    embedding_model=str(batch["embedding_model"]),
                )
                if rule_path is not None
                else {"status": "not_selected", "count": 0}
            )
            knowledge_ann = (
                knowledge_db.ensure_knowledge_ann_index(knowledge_path)
                if knowledge_path is not None
                else {"status": "not_selected", "indexed_vectors": 0}
            )
            rule_ann = (
                rule_db.ensure_rule_ann_index(rule_path)
                if rule_path is not None
                else {"status": "not_selected", "indexed_vectors": 0}
            )
            verification = _verify_import(
                knowledge_path=knowledge_path,
                rule_path=rule_path,
                knowledge_ids=[str(item["knowledge_id"]) for item in knowledge_records],
                rule_ids=[str(item["rule_id"]) for item in rule_records],
                vector_dimension=int(batch["vector_dimension"]),
            )
            if not verification["ok"]:
                raise RuntimeError("transfer import verification failed: " + "; ".join(verification["errors"]))
            summary = {
                "knowledge": knowledge_summary,
                "rules": rule_summary,
                "knowledge_ann": knowledge_ann,
                "rule_ann": rule_ann,
                "verification": verification,
            }
        except Exception:
            if knowledge_path is not None and backup_knowledge.exists():
                _sqlite_restore(backup_knowledge, knowledge_path)
            if rule_path is not None and backup_rule.exists():
                _sqlite_restore(backup_rule, rule_path)
            raise

        _finish_import_history(transfer_path, import_id, status="completed", summary=summary)
        return {
            "status": "completed",
            "import_id": import_id,
            "batch_id": batch_id,
            "batch_hash": str(batch["content_hash"]),
            "summary": summary,
        }
    except Exception as exc:
        _finish_import_history(transfer_path, import_id, status="failed", error=repr(exc))
        raise
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _require_batch(conn: sqlite3.Connection, batch_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM transfer_batches WHERE batch_id = ?", (str(batch_id),)).fetchone()
    if row is None:
        raise KeyError(f"unknown transfer batch: {batch_id}")
    return row


def _load_batch_records(
    conn: sqlite3.Connection,
    batch_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    knowledge = [
        dict(json.loads(row["payload_json"]))
        for row in conn.execute(
            "SELECT payload_json FROM transfer_knowledge WHERE batch_id = ? ORDER BY knowledge_id",
            (batch_id,),
        ).fetchall()
    ]
    rules = [
        dict(json.loads(row["payload_json"]))
        for row in conn.execute(
            "SELECT payload_json FROM transfer_rules WHERE batch_id = ? ORDER BY rule_id",
            (batch_id,),
        ).fetchall()
    ]
    return knowledge, rules


def _validate_batch_records(
    batch: Mapping[str, Any],
    knowledge_records: Sequence[Mapping[str, Any]],
    rule_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    record_types = _batch_record_types(batch)
    if "knowledge" in record_types and not knowledge_records:
        errors.append("batch_has_no_knowledge")
    if "knowledge" not in record_types and knowledge_records:
        errors.append("batch_contains_unselected_knowledge")
    if "rule" in record_types and not rule_records:
        errors.append("batch_has_no_rules")
    if "rule" not in record_types and rule_records:
        errors.append("batch_contains_unselected_rules")
    expected_model = canonical_embedding_model(batch["embedding_model"])
    expected_dimension = int(batch["vector_dimension"])
    expected_version = str(batch["vector_version"])

    for record in knowledge_records:
        record_id = str(record.get("knowledge_id") or record.get("id") or "")
        _validate_common_record(
            record,
            record_id=record_id,
            record_type="knowledge",
            expected_model=expected_model,
            expected_dimension=expected_dimension,
            expected_version=expected_version,
            vector_fields=("core_vector", "relation_vector", "nature_vector"),
            errors=errors,
        )
        if not any(str(record.get(key) or "").strip() for key in ("core", "relation", "nature")):
            errors.append(f"knowledge:{record_id}:missing_content")

    for record in rule_records:
        record_id = str(record.get("rule_id") or record.get("id") or "")
        _validate_common_record(
            record,
            record_id=record_id,
            record_type="rule",
            expected_model=expected_model,
            expected_dimension=expected_dimension,
            expected_version=expected_version,
            vector_fields=("rule_vector",),
            errors=errors,
        )
        try:
            layer = normalize_rule_layer(record.get("rule_layer") or "domain")
        except ValueError:
            layer = ""
        if layer not in {"domain", "common_sense"}:
            errors.append(f"rule:{record_id}:transfer_cannot_modify_system_layer")
        if not record.get("required"):
            errors.append(f"rule:{record_id}:missing_required")
        if not record.get("produced"):
            errors.append(f"rule:{record_id}:missing_produced")
    return {"ok": not errors, "errors": errors}


def _validate_common_record(
    record: Mapping[str, Any],
    *,
    record_id: str,
    record_type: str,
    expected_model: str,
    expected_dimension: int,
    expected_version: str,
    vector_fields: Sequence[str],
    errors: list[str],
) -> None:
    prefix = f"{record_type}:{record_id or '<missing>'}"
    if not record_id:
        errors.append(f"{prefix}:missing_id")
    if not str(record.get("source") or "").strip():
        errors.append(f"{prefix}:missing_source")
    model = canonical_embedding_model(record.get("embedding_model") or "")
    version = str(record.get("vector_version") or "").strip()
    vectors = [coerce_vector(record.get(field)) for field in vector_fields]
    vectors = [vector for vector in vectors if vector]
    if not vectors:
        errors.append(f"{prefix}:missing_vector")
    for vector in vectors:
        if len(vector) != expected_dimension:
            errors.append(f"{prefix}:vector_dimension_mismatch")
            break
    if model != expected_model:
        errors.append(f"{prefix}:embedding_model_mismatch")
    if version != expected_version:
        errors.append(f"{prefix}:vector_version_mismatch")


def _annotate_transfer_record(
    record: Mapping[str, Any],
    *,
    batch_id: str,
    batch_hash: str,
) -> dict[str, Any]:
    payload = dict(record)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "transfer_batch_id": batch_id,
            "transfer_batch_hash": batch_hash,
            "transfer_payload_hash": _content_hash(record),
        }
    )
    payload["metadata"] = metadata
    return payload


def _record_types_from_manifest(source_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = source_manifest.get("record_types")
    if raw is None:
        return ("knowledge", "rule")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        raise ValueError("source_manifest.record_types must be a string or sequence")
    normalized = tuple(dict.fromkeys(str(value or "").strip().lower() for value in values if str(value or "").strip()))
    if not normalized:
        raise ValueError("source_manifest.record_types must not be empty")
    unsupported = sorted(set(normalized) - SUPPORTED_RECORD_TYPES)
    if unsupported:
        raise ValueError(f"unsupported transfer record types: {unsupported}")
    return tuple(record_type for record_type in ("knowledge", "rule") if record_type in normalized)


def _batch_record_types(batch: Mapping[str, Any]) -> tuple[str, ...]:
    manifest = json.loads(str(batch["source_manifest_json"]))
    if not isinstance(manifest, Mapping):
        raise ValueError("transfer source manifest must be a mapping")
    return _record_types_from_manifest(manifest)


def _resolve_selected_db_path(
    value: str | Path | None,
    *,
    record_type: str,
    selected: bool,
) -> Path | None:
    if not selected:
        if value is not None:
            raise ValueError(f"{record_type}_db_path was provided but {record_type} is not selected")
        return None
    if value is None:
        raise ValueError(f"{record_type}_db_path is required for this transfer batch")
    return Path(value).expanduser().resolve()


def _ensure_model_databases(knowledge_path: Path | None, rule_path: Path | None) -> None:
    if knowledge_path is not None:
        knowledge_db.load_knowledge_records(knowledge_path, limit=1)
    if rule_path is not None:
        rule_db.load_rule_records(rule_path, limit=1)


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with sqlite3.connect(str(source)) as source_conn, sqlite3.connect(str(destination)) as destination_conn:
        source_conn.backup(destination_conn)


def _sqlite_restore(source: Path, destination: Path) -> None:
    with sqlite3.connect(str(source)) as source_conn, sqlite3.connect(str(destination)) as destination_conn:
        source_conn.backup(destination_conn)


def _verify_import(
    *,
    knowledge_path: Path | None,
    rule_path: Path | None,
    knowledge_ids: Sequence[str],
    rule_ids: Sequence[str],
    vector_dimension: int,
) -> dict[str, Any]:
    knowledge_by_id = (
        {
            str(record.get("knowledge_id") or record.get("id")): record
            for record in knowledge_db.load_knowledge_records(knowledge_path, limit=None)
        }
        if knowledge_path is not None
        else {}
    )
    rules_by_id = (
        {
            str(record.get("rule_id") or record.get("id")): record
            for record in rule_db.load_rule_records(rule_path, rule_ids=rule_ids, limit=None)
        }
        if rule_path is not None
        else {}
    )
    errors: list[str] = []
    for record_id in knowledge_ids:
        record = knowledge_by_id.get(record_id)
        if record is None:
            errors.append(f"knowledge_not_imported:{record_id}")
        elif int(record.get("vector_dimension") or 0) != vector_dimension:
            errors.append(f"knowledge_vector_contract_failed:{record_id}")
    for record_id in rule_ids:
        record = rules_by_id.get(record_id)
        if record is None:
            errors.append(f"rule_not_imported:{record_id}")
        elif int(record.get("vector_dimension") or 0) != vector_dimension:
            errors.append(f"rule_vector_contract_failed:{record_id}")
    return {
        "ok": not errors,
        "errors": errors,
        "knowledge_ids": list(knowledge_ids),
        "rule_ids": list(rule_ids),
    }


def _finish_import_history(
    transfer_path: Path,
    import_id: str,
    *,
    status: str,
    summary: Mapping[str, Any] | None = None,
    error: str = "",
) -> None:
    if not transfer_path.exists():
        return
    conn = _connect(transfer_path)
    try:
        _init_schema(conn)
        conn.execute(
            """
            UPDATE transfer_import_history
            SET status = ?, summary_json = ?, error = ?, finished_at = ?
            WHERE import_id = ?
            """,
            (status, _canonical_json(dict(summary or {})), str(error), _now_iso(), import_id),
        )
        conn.commit()
    finally:
        conn.close()


def _acquire_import_lock(lock_path: Path) -> int:
    try:
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"another transfer import is active: {lock_path}") from exc


__all__ = [
    "create_transfer_batch",
    "import_transfer_batch",
    "inspect_transfer_batch",
    "review_transfer_batch",
    "stage_transfer_records",
]
