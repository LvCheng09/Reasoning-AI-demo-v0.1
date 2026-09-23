"""Vector coercion and strict similarity contracts."""

from __future__ import annotations

from array import array
from math import sqrt
from typing import Any


def canonical_embedding_model(value: Any) -> str:
    """Normalize a model repository id and its local Hugging Face snapshot path."""

    text = str(value or "").strip().replace("\\", "/")
    marker = "/models--"
    if marker in text:
        encoded = text.split(marker, 1)[1].split("/snapshots/", 1)[0].strip("/")
        if encoded:
            return encoded.replace("--", "/")
    if text.startswith("models--"):
        encoded = text.split("/snapshots/", 1)[0][len("models--") :]
        if encoded:
            return encoded.replace("--", "/")
    return text.rstrip("/")


def coerce_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        vector = array("f")
        vector.frombytes(bytes(value))
        return [float(item) for item in vector] or None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        value = value.tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, array)) or not value:
        return None
    vector: list[float] = []
    for item in value:
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            return None
    return vector or None


def validate_vector_pair(left: Any, right: Any) -> dict[str, Any]:
    left_vec = coerce_vector(left)
    right_vec = coerce_vector(right)
    left_dimension = len(left_vec) if left_vec else None
    right_dimension = len(right_vec) if right_vec else None
    if not left_vec or not right_vec:
        return {
            "ok": False,
            "status": "missing_vector",
            "left_dimension": left_dimension,
            "right_dimension": right_dimension,
            "dimension": None,
        }
    if left_dimension != right_dimension:
        return {
            "ok": False,
            "status": "dimension_mismatch",
            "left_dimension": left_dimension,
            "right_dimension": right_dimension,
            "dimension": None,
        }
    if all(abs(item) <= 1e-12 for item in left_vec) or all(abs(item) <= 1e-12 for item in right_vec):
        return {
            "ok": False,
            "status": "zero_vector",
            "left_dimension": left_dimension,
            "right_dimension": right_dimension,
            "dimension": left_dimension,
        }
    return {
        "ok": True,
        "status": "ok",
        "left_dimension": left_dimension,
        "right_dimension": right_dimension,
        "dimension": left_dimension,
        "left_vector": left_vec,
        "right_vector": right_vec,
    }


def cosine_similarity_strict(left: Any, right: Any) -> dict[str, Any]:
    validation = validate_vector_pair(left, right)
    if not validation["ok"]:
        return {**validation, "score": 0.0}
    left_vec = validation["left_vector"]
    right_vec = validation["right_vector"]
    left_norm = sqrt(sum(item * item for item in left_vec))
    right_norm = sqrt(sum(item * item for item in right_vec))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return {
            **{key: value for key, value in validation.items() if key not in {"left_vector", "right_vector"}},
            "ok": False,
            "status": "zero_vector",
            "score": 0.0,
        }
    dot = sum(left_item * right_item for left_item, right_item in zip(left_vec, right_vec))
    return {
        **{key: value for key, value in validation.items() if key not in {"left_vector", "right_vector"}},
        "score": dot / (left_norm * right_norm),
    }


def rescale_cosine_strict(left: Any, right: Any) -> dict[str, Any]:
    result = cosine_similarity_strict(left, right)
    if not result["ok"]:
        return result
    return {
        **result,
        "score": max(0.0, min(1.0, (float(result["score"]) + 1.0) / 2.0)),
    }
