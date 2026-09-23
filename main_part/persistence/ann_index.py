"""SQLite-backed approximate vector index.

This is a deterministic LSH index. It is intentionally small and dependency-light:
SQLite stores bucket membership, while Python only reranks the candidate bucket.
"""

from __future__ import annotations

import hashlib
from array import array
from itertools import combinations
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..vector_contracts import canonical_embedding_model, coerce_vector, cosine_similarity_strict


ANN_SCHEMA_NAME = "sqlite_lsh_ann_v1"


def ensure_ann_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ann_vector_index (
            index_name TEXT NOT NULL,
            record_id TEXT NOT NULL,
            vector_field TEXT NOT NULL,
            embedding_model TEXT NOT NULL DEFAULT '',
            vector_dimension INTEGER NOT NULL,
            vector_version TEXT NOT NULL DEFAULT '',
            bucket_key TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            updated_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(index_name, record_id, vector_field)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ann_bucket
        ON ann_vector_index(index_name, embedding_model, vector_dimension, vector_version, bucket_key)
        """
    )


def vector_to_blob(vector: Any) -> bytes | None:
    values = coerce_vector(vector)
    if not values:
        return None
    return array("f", values).tobytes()


def blob_to_vector(value: bytes | bytearray | memoryview | None) -> list[float] | None:
    return coerce_vector(value)


def rebuild_lsh_index(
    conn: sqlite3.Connection,
    *,
    index_name: str,
    rows: Sequence[Mapping[str, Any]],
    embedding_model: str = "",
    vector_version: str = "",
    now: str = "",
    bits: int = 12,
) -> dict[str, Any]:
    ensure_ann_schema(conn)
    conn.execute("DELETE FROM ann_vector_index WHERE index_name = ?", (index_name,))
    inserted = 0
    skipped = 0
    dimensions: set[int] = set()
    for row in rows:
        record_id = str(row.get("record_id") or row.get("id") or "").strip()
        vector_field = str(row.get("vector_field") or "vector").strip()
        vector = coerce_vector(row.get("vector"))
        if not record_id or not vector:
            skipped += 1
            continue
        dimension = len(vector)
        dimensions.add(dimension)
        bucket_key = lsh_bucket(vector, index_name=index_name, bits=bits)
        blob = vector_to_blob(vector)
        if blob is None:
            skipped += 1
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO ann_vector_index(
                index_name, record_id, vector_field, embedding_model,
                vector_dimension, vector_version, bucket_key, vector_blob, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                index_name,
                record_id,
                vector_field,
                canonical_embedding_model(row.get("embedding_model") or embedding_model or ""),
                dimension,
                str(row.get("vector_version") or vector_version or ""),
                bucket_key,
                blob,
                now,
            ),
        )
        inserted += 1
    return {
        "backend": "sqlite_lsh",
        "schema": ANN_SCHEMA_NAME,
        "index_name": index_name,
        "indexed_vectors": inserted,
        "skipped_vectors": skipped,
        "dimensions": sorted(dimensions),
        "bits": int(bits),
    }


def upsert_lsh_vector(
    conn: sqlite3.Connection,
    *,
    index_name: str,
    record_id: str,
    vector_field: str,
    vector: Any,
    embedding_model: str = "",
    vector_version: str = "",
    now: str = "",
    bits: int = 12,
) -> bool:
    """Keep one persisted vector and its ANN membership in one transaction."""

    ensure_ann_schema(conn)
    normalized_record_id = str(record_id or "").strip()
    normalized_vector_field = str(vector_field or "vector").strip()
    values = coerce_vector(vector)
    if not normalized_record_id or not values:
        if normalized_record_id:
            conn.execute(
                "DELETE FROM ann_vector_index WHERE index_name = ? AND record_id = ? AND vector_field = ?",
                (index_name, normalized_record_id, normalized_vector_field),
            )
        return False
    blob = vector_to_blob(values)
    if blob is None:
        raise ValueError(f"invalid ANN vector for {index_name}:{normalized_record_id}:{normalized_vector_field}")
    conn.execute(
        """
        INSERT OR REPLACE INTO ann_vector_index(
            index_name, record_id, vector_field, embedding_model,
            vector_dimension, vector_version, bucket_key, vector_blob, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            index_name,
            normalized_record_id,
            normalized_vector_field,
            canonical_embedding_model(embedding_model),
            len(values),
            str(vector_version or ""),
            lsh_bucket(values, index_name=index_name, bits=bits),
            blob,
            str(now or ""),
        ),
    )
    return True


def delete_lsh_record(
    conn: sqlite3.Connection,
    *,
    index_name: str,
    record_id: str,
) -> None:
    ensure_ann_schema(conn)
    conn.execute(
        "DELETE FROM ann_vector_index WHERE index_name = ? AND record_id = ?",
        (index_name, str(record_id)),
    )


def query_lsh_index(
    conn: sqlite3.Connection,
    *,
    index_name: str,
    query_vector: Any,
    embedding_model: str = "",
    vector_version: str = "",
    candidate_limit: int = 256,
    bits: int = 12,
    probe_radius: int = 3,
    allowed_record_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    ensure_ann_schema(conn)
    vector = coerce_vector(query_vector)
    if not vector:
        return {"ok": False, "status": "missing_query_vector", "record_ids": [], "scores": {}}
    dimension = len(vector)
    bucket_key = lsh_bucket(vector, index_name=index_name, bits=bits)
    bucket_keys = lsh_probe_buckets(bucket_key, radius=probe_radius)
    clauses = [
        "index_name = ?",
        "vector_dimension = ?",
        f"bucket_key IN ({','.join('?' for _ in bucket_keys)})",
    ]
    params: list[Any] = [index_name, dimension, *bucket_keys]
    normalized_allowed_ids = list(
        dict.fromkeys(str(record_id) for record_id in (allowed_record_ids or []) if str(record_id))
    )
    if allowed_record_ids is not None and not normalized_allowed_ids:
        return {
            "ok": True,
            "status": "no_allowed_records",
            "record_ids": [],
            "scores": {},
            "bucket_key": bucket_key,
            "probe_radius": min(max(0, int(probe_radius)), len(bucket_key)),
            "probed_bucket_count": len(bucket_keys),
            "candidate_count": 0,
        }
    if normalized_allowed_ids:
        clauses.append(f"record_id IN ({','.join('?' for _ in normalized_allowed_ids)})")
        params.extend(normalized_allowed_ids)
    if embedding_model:
        clauses.append("embedding_model = ?")
        params.append(canonical_embedding_model(embedding_model))
    if vector_version:
        clauses.append("vector_version = ?")
        params.append(str(vector_version))
    rows = conn.execute(
        f"""
        SELECT record_id, vector_field, vector_blob
        FROM ann_vector_index
        WHERE {' AND '.join(clauses)}
        ORDER BY CASE WHEN bucket_key = ? THEN 0 ELSE 1 END,
                 bucket_key, record_id, vector_field
        LIMIT ?
        """,
        [*params, bucket_key, max(1, int(candidate_limit))],
    ).fetchall()
    ranked: dict[str, float] = {}
    for row in rows:
        candidate_vector = blob_to_vector(row["vector_blob"])
        result = cosine_similarity_strict(vector, candidate_vector)
        if not result["ok"]:
            continue
        record_id = str(row["record_id"])
        ranked[record_id] = max(ranked.get(record_id, 0.0), float(result["score"]))
    ordered = sorted(ranked, key=lambda item: ranked[item], reverse=True)
    return {
        "ok": True,
        "status": "ok" if ordered else "no_bucket_candidates",
        "record_ids": ordered,
        "scores": ranked,
        "bucket_key": bucket_key,
        "probe_radius": min(max(0, int(probe_radius)), len(bucket_key)),
        "probed_bucket_count": len(bucket_keys),
        "candidate_count": len(rows),
    }


def ann_index_summary(conn: sqlite3.Connection, *, index_name: str) -> dict[str, Any]:
    ensure_ann_schema(conn)
    row = conn.execute(
        """
        SELECT COUNT(*) AS vectors, COUNT(DISTINCT bucket_key) AS buckets
        FROM ann_vector_index
        WHERE index_name = ?
        """,
        (index_name,),
    ).fetchone()
    return {
        "backend": "sqlite_lsh",
        "schema": ANN_SCHEMA_NAME,
        "index_name": index_name,
        "indexed_vectors": int(row["vectors"] or 0),
        "bucket_count": int(row["buckets"] or 0),
    }


def lsh_bucket(vector: Sequence[float], *, index_name: str, bits: int = 12) -> str:
    values = np.asarray([float(item) for item in vector], dtype=np.float32)
    planes = _hyperplanes(len(values), bits=max(1, int(bits)), seed_text=index_name)
    projections = planes @ values
    return "".join("1" if value >= 0.0 else "0" for value in projections)


def lsh_probe_buckets(bucket_key: str, *, radius: int = 3) -> list[str]:
    """Return deterministic nearby LSH buckets ordered by Hamming distance."""

    key = str(bucket_key)
    max_radius = min(max(0, int(radius)), len(key))
    buckets = [key]
    for distance in range(1, max_radius + 1):
        for positions in combinations(range(len(key)), distance):
            chars = list(key)
            for position in positions:
                chars[position] = "0" if chars[position] == "1" else "1"
            buckets.append("".join(chars))
    return buckets


def _hyperplanes(dimension: int, *, bits: int, seed_text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(f"{seed_text}:{dimension}:{bits}".encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    planes = rng.standard_normal((bits, dimension)).astype(np.float32)
    norms = np.linalg.norm(planes, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return planes / norms


__all__ = [
    "ANN_SCHEMA_NAME",
    "ann_index_summary",
    "blob_to_vector",
    "delete_lsh_record",
    "ensure_ann_schema",
    "lsh_probe_buckets",
    "query_lsh_index",
    "rebuild_lsh_index",
    "upsert_lsh_vector",
    "vector_to_blob",
]
