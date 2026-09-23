"""Execute matcher rules that prepare candidates before hard matching.

The rules are control rules scoped to matching. This module applies their
selection output; it does not decide whether a candidate rule matches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..vector_contracts import cosine_similarity_strict


def prepare_pre_match_candidates(
    rules: list[dict[str, Any]],
    *,
    args: dict[str, Any],
    target_spec: dict[str, Any],
    target_rule_layer: str,
    matcher_rules: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply pre-match matcher rules and the selector produced by control."""

    selected, metadata = select_rule_candidates(
        rules,
        args=args,
        target_spec=target_spec,
        target_rule_layer=target_rule_layer,
        matcher_rules=matcher_rules or [],
    )
    metadata = {
        **metadata,
        "matcher_rule_phase": "pre_match",
        "candidate_decision_owner": "control_rule",
    }
    return selected, metadata


def select_rule_candidates(
    rules: list[dict[str, Any]],
    *,
    args: dict[str, Any],
    target_spec: dict[str, Any],
    target_rule_layer: str,
    matcher_rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .routing import (
        collect_vectors,
        rule_layer,
        safe_float,
    )

    del target_spec
    scope_tags = {str(item).strip().lower() for item in args.get("scope_tags", []) if str(item).strip()}
    preferred_tags = {str(item).strip().lower() for item in args.get("preferred_rule_tags", []) if str(item).strip()}
    graph_vectors = [
        vector
        for vector in [*(args.get("graph_vectors") or []), *(args.get("_source_vectors") or [])]
        if isinstance(vector, list)
    ]
    graph_vector_index = build_normalized_vector_index(graph_vectors)
    activated = activate_pre_match_rules(
        matcher_rules,
        graph_vectors=graph_vectors,
        scope_tags=scope_tags,
        target_rule_layer=target_rule_layer,
        collect_vectors_fn=collect_vectors,
        rule_layer_fn=rule_layer,
    )
    candidate_k = max(1, int(safe_float(args.get("candidate_k"), 256)))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for rule in rules:
        layer = rule_layer(rule)
        score = 0.0
        if target_rule_layer and layer == target_rule_layer:
            score += 3.0
        score += vector_support_score(
            rule,
            graph_vectors,
            collect_vectors,
            graph_vector_index=graph_vector_index,
        )
        score += matcher_rule_support_score(activated)
        score += 0.05 * max(0.0, min(float(rule.get("confidence", 0.5) or 0.5), 1.0))
        routed = dict(rule)
        routed["_routing_score"] = round(score, 4)
        ranked.append((score, routed))
    ranked.sort(
        key=lambda item: (
            item[0],
            float(item[1].get("confidence", 0.5) or 0.5),
            str(item[1].get("rule_id") or ""),
        ),
        reverse=True,
    )
    selected = [rule for _, rule in ranked[:candidate_k]]
    return selected, {
        "candidate_policy": "pre_match_matcher_rule_then_hard_match",
        "graph_vector_count": len(graph_vectors),
        "scope_tags": sorted(scope_tags),
        "preferred_rule_tags": sorted(preferred_tags),
        "target_rule_layer": target_rule_layer,
        "candidate_count": len(rules),
        "routed_count": len(selected),
        "activated_matcher_rule_ids": [str(item.get("rule_id") or "") for item in activated[:12]],
        "top_rule_ids": [str(rule.get("rule_id") or "") for rule in selected[:12]],
    }


def activate_pre_match_rules(
    matcher_rules: list[dict[str, Any]],
    *,
    graph_vectors: list[list[float]],
    scope_tags: set[str],
    target_rule_layer: str,
    collect_vectors_fn: Any,
    rule_layer_fn: Any,
) -> list[dict[str, Any]]:
    activated: list[dict[str, Any]] = []
    for rule in matcher_rules:
        vector_score = 0.0
        for required in rule.get("required") or []:
            if not isinstance(required, dict):
                continue
            for vector in collect_vectors_fn(required):
                for graph_vector in graph_vectors:
                    vector_score = max(vector_score, cosine_floor(graph_vector, vector))
        activation = (
            (3.5 * vector_score)
            + (0.6 if rule_layer_fn(rule) == "matcher_rule" else 0.0)
            + (0.3 if target_rule_layer and rule_layer_fn(rule) == "matcher_rule" else 0.0)
        )
        if activation < 1.1:
            continue
        activated.append(
            {
                "rule_id": str(rule.get("rule_id") or ""),
                "rule_layer": rule_layer_fn(rule),
                "activation_score": round(activation, 4),
                "preferred_tags": [],
                "rule": rule,
            }
        )
    activated.sort(key=lambda item: item["activation_score"], reverse=True)
    return activated[:24]


def matcher_rule_support_score(
    activated_matcher_rules: list[dict[str, Any]],
) -> float:
    if not activated_matcher_rules:
        return 0.0
    support = 0.0
    for item in activated_matcher_rules:
        activation = float(item.get("activation_score") or 0.0)
        support += 0.05 * activation
    return min(6.0, support)


def vector_support_score(
    rule: dict[str, Any],
    graph_vectors: list[list[float]],
    collect_vectors_fn: Any,
    *,
    graph_vector_index: dict[int, np.ndarray] | None = None,
) -> float:
    if not graph_vectors:
        return 0.0
    graph_index = graph_vector_index or build_normalized_vector_index(graph_vectors)
    rule_index = build_normalized_vector_index(collect_vectors_fn(rule))
    best_cosine: float | None = None
    for dimension, rule_matrix in rule_index.items():
        graph_matrix = graph_index.get(dimension)
        if graph_matrix is None:
            continue
        score = float(np.max(graph_matrix @ rule_matrix.T))
        best_cosine = score if best_cosine is None else max(best_cosine, score)
    if best_cosine is None:
        return 0.0
    return 3.0 * max(0.0, min(1.0, (best_cosine + 1.0) / 2.0))


def build_normalized_vector_index(vectors: list[list[float]]) -> dict[int, np.ndarray]:
    """Group nonzero vectors by dimension for strict batched cosine scoring."""

    grouped: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[int, bytes]] = set()
    for vector in vectors:
        try:
            values = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if values.ndim != 1 or values.size == 0:
            continue
        norm = float(np.linalg.norm(values))
        if norm <= 1e-12:
            continue
        normalized = values / norm
        identity = (int(values.size), normalized.tobytes())
        if identity in seen:
            continue
        seen.add(identity)
        grouped.setdefault(int(values.size), []).append(normalized)
    return {
        dimension: np.stack(items)
        for dimension, items in grouped.items()
        if items
    }


def cosine_floor(left: list[float], right: list[float]) -> float:
    result = cosine_similarity_strict(left, right)
    if not result["ok"]:
        return 0.0
    cosine = float(result["score"])
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def load_pre_match_rules(
    args: dict[str, Any],
    state: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Load only control rules explicitly assigned to matcher pre-match."""

    from .routing import as_dict_list

    inline = as_dict_list(args.get("matcher_rule_records") or args.get("pre_match_rule_records"))
    records = inline or load_cached_matcher_rules(args, state, limit=limit)
    return [rule for rule in records if is_matcher_rule(rule, phase="pre_match")]


def load_cached_matcher_rules(
    args: dict[str, Any],
    state: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Load the static matcher policy once per unchanged rule DB snapshot."""

    from ..persistence.rule_db import load_rule_records
    from .routing import rule_is_executable_candidate, state_rule_db_path

    rule_db_path = str(args.get("rule_db_path") or state_rule_db_path(state) or "").strip()
    if not rule_db_path:
        return []
    cache_key = _rule_db_version(rule_db_path)
    cache = getattr(state, "matcher_rule_cache", None)
    if isinstance(cache, dict) and cache_key in cache:
        return [dict(rule) for rule in cache[cache_key]]

    del limit
    records = load_rule_records(rule_db_path, limit=None)
    matcher_rules = [
        rule
        for rule in records
        if is_matcher_rule(rule) and rule_is_executable_candidate(rule)
    ]
    if isinstance(cache, dict):
        cache.clear()
        cache[_rule_db_version(rule_db_path)] = [dict(rule) for rule in matcher_rules]
    return matcher_rules


def _rule_db_version(rule_db_path: str) -> tuple[Any, ...]:
    path = Path(rule_db_path).expanduser().resolve()
    version: list[Any] = [str(path)]
    for candidate in (path, Path(f"{path}-wal")):
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            version.extend((0, 0))
        else:
            version.extend((stat.st_mtime_ns, stat.st_size))
    return tuple(version)


def is_matcher_rule(rule: dict[str, Any], *, phase: str | None = None) -> bool:
    layer = str(rule.get("rule_layer") or "").strip().lower()
    kind = str(rule.get("rule_kind") or "").strip().lower()
    tags = {str(tag).strip().lower() for tag in (rule.get("scope_tags") or []) if str(tag).strip()}
    assigned = layer == "matcher_rule" or kind == "matcher_rule" or "matcher_rule" in tags
    if not assigned:
        return False
    if not phase:
        return True
    phase_tags = {phase, phase.replace("_", "-")}
    return bool(tags.intersection(phase_tags))


__all__ = [
    "activate_pre_match_rules",
    "is_matcher_rule",
    "load_cached_matcher_rules",
    "load_pre_match_rules",
    "prepare_pre_match_candidates",
    "select_rule_candidates",
]
