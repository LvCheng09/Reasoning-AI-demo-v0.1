"""Small normalization helpers with explicit failure/default policies."""

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_confidence(value: Any, default: float = 0.7) -> float:
    number = safe_float(value, default)
    return max(0.0, min(1.0, number)) if math.isfinite(number) else default


def as_list(value: Any, *, expand_sets: bool = True) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or (expand_sets and isinstance(value, set)):
        return list(value)
    return [value]
