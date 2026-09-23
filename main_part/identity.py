"""Stable identifiers for persisted content and runtime records."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from typing import Any


_RUNTIME_FIELDS = {
    "application_count",
    "created_at",
    "executed_count",
    "failure_count",
    "hit_count",
    "last_used_at",
    "retrieved_count",
    "success_count",
    "support_count",
    "updated_at",
    "usage_count",
    "workspace_entered_count",
}


def stable_json_dumps(value: Any) -> str:
    """Serialize content deterministically for hashing."""

    return json.dumps(
        _stable_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_content_hash(value: Any, *, length: int = 24) -> str:
    digest = hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
    return digest[: max(1, int(length))]


def stable_content_id(prefix: str, value: Any, *, length: int = 24) -> str:
    return f"{_clean_prefix(prefix)}_{stable_content_hash(value, length=length)}"


def stable_text_id(prefix: str, *parts: Any, length: int = 24) -> str:
    return stable_content_id(
        prefix,
        ["" if part is None else str(part) for part in parts],
        length=length,
    )


def new_runtime_id(prefix: str) -> str:
    return f"{_clean_prefix(prefix)}_{uuid.uuid4().hex}"


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _RUNTIME_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_stable_json_value(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _clean_prefix(prefix: str) -> str:
    text = str(prefix or "id").strip().lower()
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() or char == "_" else "_")
    return "".join(cleaned).strip("_") or "id"

