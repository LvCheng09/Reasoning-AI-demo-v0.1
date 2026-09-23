"""Redis-backed cache for memory DB ANN recall results."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence


class RedisMemoryRecallCache:
    def __init__(self, client: Any, *, prefix: str = "event_reasoning:memory_recall", ttl_seconds: int = 900) -> None:
        self.client = client
        self.prefix = str(prefix).rstrip(":")
        self.ttl_seconds = max(1, int(ttl_seconds))

    def get_records(self, cache_key: str) -> list[dict[str, Any]] | None:
        raw = self.client.get(self._key(cache_key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw))
        if not isinstance(payload, list):
            raise ValueError("Redis memory recall cache payload must be a list")
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def set_records(self, cache_key: str, records: Sequence[Mapping[str, Any]]) -> None:
        payload = json.dumps([dict(record) for record in records], ensure_ascii=False, sort_keys=True)
        self.client.setex(self._key(cache_key), self.ttl_seconds, payload)

    def delete(self, cache_key: str) -> None:
        self.client.delete(self._key(cache_key))

    def _key(self, cache_key: str) -> str:
        return f"{self.prefix}:{cache_key}"


def build_memory_redis_cache_from_env() -> dict[str, Any]:
    url = os.getenv("MAIN_PART_MEMORY_REDIS_URL", "").strip()
    if not url:
        return {"ok": True, "status": "not_configured", "cache": None}
    try:
        import redis
    except ImportError as exc:
        return {
            "ok": False,
            "status": "redis_dependency_missing",
            "cache": None,
            "error": repr(exc),
        }
    try:
        client = redis.Redis.from_url(url, decode_responses=False)
        client.ping()
    except Exception as exc:
        return {
            "ok": False,
            "status": "redis_connection_failed",
            "cache": None,
            "error": repr(exc),
        }
    return {
        "ok": True,
        "status": "ready",
        "cache": RedisMemoryRecallCache(
            client,
            prefix=os.getenv("MAIN_PART_MEMORY_REDIS_PREFIX", "event_reasoning:memory_recall"),
            ttl_seconds=int(os.getenv("MAIN_PART_MEMORY_REDIS_TTL_SECONDS", "900")),
        ),
    }


__all__ = ["RedisMemoryRecallCache", "build_memory_redis_cache_from_env"]
