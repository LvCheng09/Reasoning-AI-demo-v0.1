"""Cosine policies share validation but keep callers' score conventions."""

from typing import Any
from ..vector_contracts import cosine_similarity_strict


def cosine(left: Any, right: Any, *, invalid: float | None = 0.0, positive_only: bool = False) -> float | None:
    result = cosine_similarity_strict(left, right)
    if not result["ok"]:
        return invalid
    score = float(result["score"])
    return max(0.0, min(1.0, score)) if positive_only else score
