"""SQLite and serialization mechanics shared by the persistence adapters."""

from array import array
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

from ..vector_contracts import coerce_vector


def connect(db_path: str | Path, *, foreign_keys: bool = True, readonly: bool = False) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if str(db_path) == ":memory:":
        if readonly:
            raise ValueError("an in-memory database cannot be reopened read-only")
        conn = sqlite3.connect(":memory:")
    elif readonly:
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if not readonly:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        conn.close()
        raise
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    quoted = '"' + table.replace('"', '""') + '"'
    return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({quoted})")]


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if column not in table_columns(conn, table):
        table_sql = '"' + table.replace('"', '""') + '"'
        column_sql = '"' + column.replace('"', '""') + '"'
        conn.execute(f"ALTER TABLE {table_sql} ADD COLUMN {column_sql} {definition}")


def json_dumps(value: Any, *, none_as_object: bool = False, stringify: bool = False) -> str:
    if value is None and none_as_object:
        value = {}
    options = {"default": str} if stringify else {}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), **options)


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logging.getLogger(__name__).warning("Malformed persisted JSON; using the caller's explicit default")
        return default


def vector_to_blob(value: Any, *, reject_zero: bool = False, inclusive_zero: bool = False) -> bytes | None:
    vector = coerce_vector(value)
    if not vector:
        return None
    if reject_zero and all(abs(item) <= 1e-12 if inclusive_zero else abs(item) < 1e-12 for item in vector):
        return None
    return array("f", vector).tobytes()


def blob_to_vector(value: Any) -> list[float] | None:
    return coerce_vector(value)
