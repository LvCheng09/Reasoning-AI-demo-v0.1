from __future__ import annotations

from pathlib import Path
from typing import Any

from ..persistence import knowledge_db


def maintain_knowledge_db(
    db_path: str | Path,
    *,
    retire_failure_rate: float = 0.60,
    low_usage_threshold: int = 0,
    delete_retired: bool = False,
) -> dict[str, Any]:
    """Re-rank or retire surface knowledge according to usage feedback."""

    return knowledge_db.reprioritize_knowledge_records(
        db_path,
        retire_failure_rate=retire_failure_rate,
        low_usage_threshold=low_usage_threshold,
        delete_retired=delete_retired,
    )


__all__ = ["maintain_knowledge_db"]
