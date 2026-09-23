"""Immutable small-model tensors in the run DB; backup() carries all dependencies."""
import hashlib
import json
from contextlib import closing
from pathlib import Path

from .base import connect


def _schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS function_versions (
        function_id TEXT NOT NULL, version TEXT NOT NULL, weights BLOB NOT NULL,
        sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
        PRIMARY KEY(function_id, version));
    CREATE TABLE IF NOT EXISTS function_active (
        function_id TEXT PRIMARY KEY, version TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS function_publications (
        id INTEGER PRIMARY KEY, function_id TEXT NOT NULL, version TEXT NOT NULL,
        previous_version TEXT, reason TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS function_acceptance (
        function_id TEXT NOT NULL, version TEXT NOT NULL, sha256 TEXT NOT NULL,
        report TEXT NOT NULL, PRIMARY KEY(function_id, version));
    """)


def store(db_path, function_id, version, weights: bytes, metadata: dict):
    digest = hashlib.sha256(weights).hexdigest()
    with closing(connect(db_path)) as conn:
        _schema(conn)
        with conn:
            existing = conn.execute("SELECT sha256,metadata FROM function_versions WHERE function_id=? AND version=?",
                                    (function_id, version)).fetchone()
            encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
            if existing:
                if existing["sha256"] != digest or existing["metadata"] != encoded:
                    raise ValueError("immutable function version already exists with different content")
            else:
                conn.execute("INSERT INTO function_versions VALUES(?,?,?,?,?)",
                             (function_id, version, weights, digest, encoded))
    return digest


def load(db_path, function_id, version):
    if not db_path or not version or version == "active":
        raise ValueError("model execution requires a DB and an exact parameter version")
    with closing(connect(db_path, readonly=True)) as conn:
        row = conn.execute("SELECT * FROM function_versions WHERE function_id=? AND version=?",
                           (function_id, version)).fetchone()
    if row is None:
        raise ValueError(f"unknown function version: {function_id}/{version}")
    if hashlib.sha256(row["weights"]).hexdigest() != row["sha256"]:
        raise ValueError("function artifact checksum mismatch")
    return bytes(row["weights"]), json.loads(row["metadata"]), row["sha256"]


def version_metadata(db_path, function_id, version):
    if not Path(db_path).exists():
        return None
    with closing(connect(db_path, readonly=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='function_versions'").fetchone():
            return None
        row = conn.execute("SELECT metadata FROM function_versions WHERE function_id=? AND version=?",
                           (function_id, version)).fetchone()
    return json.loads(row[0]) if row else None


def publish(db_path, function_id, version, *, reason="validated"):
    _, metadata, _ = load(db_path, function_id, version)
    with closing(connect(db_path)) as conn:
        report = conn.execute("SELECT sha256,report FROM function_acceptance WHERE function_id=? AND version=?",
                              (function_id, version)).fetchone()
        if not report or metadata.get("local_validation_passed") is not True:
            raise ValueError("cannot publish without local and pipeline validation")
        if report["sha256"] != load(db_path, function_id, version)[2]:
            raise ValueError("acceptance artifact mismatch")
        rule = json.loads(report["report"]).get("rule_template")
        if rule:
            from .rule_db import _init_schema, upsert_rule_records
            _init_schema(conn)
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT version FROM function_active WHERE function_id=?", (function_id,)).fetchone()
            if row and row[0] == version:
                return False
            if rule:
                upsert_rule_records(db_path, [rule], rule_kind="domain", _connection=conn)
            conn.execute("INSERT OR REPLACE INTO function_active VALUES(?,?)", (function_id, version))
            conn.execute("INSERT INTO function_publications(function_id,version,previous_version,reason) VALUES(?,?,?,?)",
                         (function_id, version, row[0] if row else None, reason))
    return True


def active_version(db_path, function_id):
    if not db_path:
        raise ValueError("active parameters require a configured run DB")
    with closing(connect(db_path, readonly=True)) as conn:
        row = conn.execute("SELECT version FROM function_active WHERE function_id=?", (function_id,)).fetchone()
    if row is None:
        raise ValueError("function has no published version")
    return row[0]


def record_acceptance(db_path, function_id, version, report):
    """Called by the real pipeline evaluator after all assertions pass."""
    _, _, digest = load(db_path, function_id, version)
    if report.get("status") != "PASS" or report.get("case_count", 0) < 1:
        raise ValueError("pipeline acceptance failed or empty")
    with closing(connect(db_path)) as conn:
        with conn:
            conn.execute("INSERT OR REPLACE INTO function_acceptance VALUES(?,?,?,?)",
                         (function_id, version, digest, json.dumps(report, allow_nan=False)))
