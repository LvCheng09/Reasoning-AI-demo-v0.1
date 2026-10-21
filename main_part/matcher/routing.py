"""Resolve matcher endpoints and convert working-memory graphs."""

from __future__ import annotations

from typing import Any

from ..data_structure import WorkingMemoryState
from ..rule_layers import normalize_rule_layer
from ..rule_selector import selector_attempted_rule_ids


def normalize_endpoint(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"type": value.strip()}
    return {}


def resolve_endpoint(
    spec: dict[str, Any],
    args: dict[str, Any],
    state: Any,
    working_memory: WorkingMemoryState,
    *,
    role: str,
) -> dict[str, Any]:
    endpoint_type = str(spec.get("type") or "").strip().lower()
    if endpoint_type in {"working_memory", "event_workspace", "event_graph", "graph"} or looks_like_graph(spec):
        return graph_bundle_from_spec(spec, state, working_memory)
    if endpoint_type == "rule":
        if role == "source":
            return required_bundle_from_spec(spec, args)
        return rules_bundle_from_spec({"type": "rules", "rule_records": [spec]}, args, state)
    if endpoint_type in {"required", "rule_required"} or spec.get("required") is not None:
        return required_bundle_from_spec(spec, args)
    if endpoint_type in {"rule_db", "rules", "rule_records"} or spec.get("rule_records") is not None or spec.get("rules") is not None:
        return rules_bundle_from_spec(spec, args, state)
    return {"kind": "empty", "spec": spec}


def looks_like_graph(spec: dict[str, Any]) -> bool:
    return any(key in spec for key in ("event_workspaces", "spans", "relations", "workspace"))


def graph_bundle_from_spec(spec: dict[str, Any], state: Any, working_memory: WorkingMemoryState) -> dict[str, Any]:
    endpoint_type = str(spec.get("type") or "").strip().lower()
    if endpoint_type in {"working_memory", "event_workspace", "event_graph", "graph"} and not looks_like_graph(spec):
        event_ids = [str(value) for value in spec.get("event_ids", []) or getattr(state, "active_event_ids", []) or [] if str(value)]
        zones = _source_zones(spec)
        workspaces: list[dict[str, Any]] = []
        if "event_workspace" in zones:
            workspaces.extend(
                iter_match_workspaces(
                    working_memory,
                    event_ids,
                    include_feedback_events=bool(spec.get("include_feedback_events")),
                )
            )
        for zone in zones:
            if zone == "event_workspace":
                continue
            workspaces.extend(iter_zone_entry_workspaces(working_memory, zone))
        return {
            "kind": "graph",
            "spec": spec or default_event_graph_source(),
            "workspaces": workspaces,
            "vectors": vectors_from_workspaces(workspaces),
        }
    workspace = spec.get("workspace") if isinstance(spec.get("workspace"), dict) else spec
    workspaces = [workspace_to_match_bundle(str(workspace.get("event_id") or "inline_graph"), workspace)]
    return {
        "kind": "graph",
        "spec": spec,
        "workspaces": workspaces,
        "vectors": vectors_from_workspaces(workspaces),
    }


def required_bundle_from_spec(spec: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    bundle = {
        "kind": "required",
        "spec": spec,
        "rule_id": str(spec.get("rule_id") or args.get("rule_id") or ""),
        "rule_layer": str(spec.get("rule_layer") or args.get("rule_layer") or ""),
        "required": as_dict_list(spec.get("required") if spec.get("required") is not None else args.get("required")),
    }
    from ..rule_execution_contract import CONTRACT_FIELDS
    if "execution_contract_version" in spec or any(spec.get(k) for k in CONTRACT_FIELDS[1:]):
        bundle["execution_rule"] = spec
    return bundle


def required_bundle_from_rule(rule: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "required",
        "spec": spec,
        "rule_id": str(rule.get("rule_id") or ""),
        "rule_layer": str(rule.get("rule_layer") or ""),
        "required": [dict(required) for required in as_dict_list(rule.get("required"))],
        "execution_rule": rule,
    }


def rules_bundle_from_spec(spec: dict[str, Any], args: dict[str, Any], state: Any) -> dict[str, Any]:
    raw_target_rule_layer = spec.get("rule_layer") or spec.get("target_rule_layer") or args.get("target_rule_layer") or ""
    target_rule_layer = normalize_rule_layer(raw_target_rule_layer, default="") if raw_target_rule_layer else ""
    limit = max(1, int(safe_float(args.get("candidate_k") or spec.get("candidate_k"), 256)))
    merged_args = {
        **args,
        "rule_records": spec.get("rule_records") or spec.get("rules") or args.get("rule_records") or [],
        "rule_db_path": spec.get("rule_db_path") or args.get("rule_db_path") or "",
    }
    rules = load_rule_candidates(merged_args, state, target_rule_layer=target_rule_layer, limit=limit)
    from .pre_match import load_pre_match_rules, prepare_pre_match_candidates

    matcher_rules = load_pre_match_rules(merged_args, state, limit=max(limit, 128))
    routed_rules, routing_metadata = prepare_pre_match_candidates(
        rules,
        args=merged_args,
        target_spec=spec,
        target_rule_layer=target_rule_layer,
        matcher_rules=matcher_rules,
    )
    return {
        "kind": "rules",
        "spec": spec,
        "rules": routed_rules,
        "routing_metadata": routing_metadata,
    }


def graph_from_required_bundle(source: dict[str, Any]) -> dict[str, Any]:
    workspace = {
        "event_id": "required_source_graph",
        "match_patterns": [
            normalize_required_pattern(required)
            for required in source["required"]
            if normalize_required_pattern(required)
        ],
        "spans": [],
        "relations": [],
    }
    workspaces = [workspace_to_match_bundle("required_source_graph", workspace)]
    return {
        "kind": "graph",
        "spec": {"type": "required_as_graph", "source": source["spec"]},
        "workspaces": workspaces,
        "vectors": vectors_from_workspaces(workspaces),
    }


def workspace_to_match_bundle(event_id: str, workspace: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "workspace": workspace,
        "span_index": build_span_index(workspace),
        "patterns": collect_event_match_patterns(workspace),
        "relations": collect_event_relations(workspace),
    }


def vectors_from_workspaces(workspaces: list[dict[str, Any]]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for workspace in workspaces:
        vectors.extend(collect_vectors(workspace.get("workspace", {})))
    return vectors


def default_event_graph_source() -> dict[str, Any]:
    return {
        "type": "working_memory",
        "zone": "event_workspace",
        "fields": [
            "event_workspaces.*.spans",
            "event_workspaces.*.relations",
        ],
    }


def _source_zones(spec: dict[str, Any]) -> list[str]:
    raw = spec.get("zones") if spec.get("zones") is not None else spec.get("zone")
    if raw is None:
        return ["event_workspace"]
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw]
    else:
        values = ["event_workspace"]
    zones: list[str] = []
    for value in values:
        zone = _normalize_source_zone(value)
        if zone and zone not in zones:
            zones.append(zone)
    return zones or ["event_workspace"]


def _normalize_source_zone(value: Any) -> str:
    zone = str(value or "").strip()
    return "event_workspace" if zone == "event_workspace" else ""


def iter_zone_entry_workspaces(working_memory: WorkingMemoryState, zone: str) -> list[dict[str, Any]]:
    zone = _normalize_source_zone(zone)
    if not zone:
        return []
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(working_memory.zone_entries.get(zone, []) or []):
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        text = _entry_match_text(payload)
        if not text:
            continue
        workspace = {
            "event_id": f"{zone}_{index}",
            "spans": [
                {
                    "span_id": f"{zone}_{index}_state",
                    "core": str(payload.get("status") or payload.get("phase") or zone),
                    "relation": str(payload.get("reason") or payload.get("stage") or "state"),
                    "nature": text,
                    "text": text,
                    "source": zone,
                }
            ],
            "relations": [],
        }
        results.append(workspace_to_match_bundle(workspace["event_id"], workspace))
    return results


def _entry_match_text(value: Any) -> str:
    parts: list[str] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "vector" or key.endswith("_vector"):
                    continue
                if key in {"status", "reason", "phase", "stage", "payload_type", "source", "missing_span_query", "operation"}:
                    if isinstance(item, (str, int, float, bool)):
                        parts.append(str(item))
                walk(item, depth + 1)
        elif isinstance(node, list):
            for item in node[:24]:
                walk(item, depth + 1)
        elif isinstance(node, (str, int, float, bool)):
            text = str(node).strip()
            if text and len(text) <= 240:
                parts.append(text)

    walk(value)
    return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))[:4000]


def load_rule_candidates(
    args: dict[str, Any],
    state: Any,
    *,
    target_rule_layer: str,
    limit: int,
) -> list[dict[str, Any]]:
    inline_rules = as_dict_list(args.get("rule_records"))
    selector = args.get("rule_selector") if isinstance(args.get("rule_selector"), dict) else {}
    selector_with_context = dict(selector)
    selector_with_context.setdefault("target_rule_layer", target_rule_layer)
    if inline_rules:
        filtered_inline = [
            rule for rule in inline_rules
            if rule_layer(rule) == target_rule_layer or not target_rule_layer
            if rule_is_executable_candidate(rule, selector_with_context)
            if rule_matches_selector_numeric_bounds(rule, selector_with_context)
        ]
        return _include_memory_preferred_candidates(
            rank_loaded_rule_candidates(filtered_inline, selector_with_context)[: max(1, int(limit))],
            filtered_inline,
            selector_with_context,
        )
    rule_db_path = str(args.get("rule_db_path") or state_rule_db_path(state) or "").strip()
    if not rule_db_path:
        return []
    selector_vectors = [
        vector for vector in (args.get("graph_vectors") or []) if isinstance(vector, list)
    ]
    source_vectors = [
        vector for vector in (args.get("_source_vectors") or []) if isinstance(vector, list)
    ]
    graph_vectors = selector_vectors or source_vectors
    from ..persistence.rule_db import load_rule_ids_by_tags, load_rule_records, search_rule_ann

    hard_tag_fields = (
        "required_rule_tags",
        "required_any_rule_tags",
        "excluded_rule_tags",
        "activation_policies",
        "repair_target_rule_ids",
        "missing_structure_signatures",
        "repair_family_keys",
    )
    allowed_rule_ids = None
    if any(selector_with_context.get(field) for field in hard_tag_fields):
        allowed_rule_ids = load_rule_ids_by_tags(
            rule_db_path,
            required_tags=selector_with_context.get("required_rule_tags"),
            required_any_tags=selector_with_context.get("required_any_rule_tags"),
            excluded_tags=selector_with_context.get("excluded_rule_tags"),
            rule_layer=target_rule_layer or None,
            activation_policies=selector_with_context.get("activation_policies"),
            repair_target_rule_ids=selector_with_context.get("repair_target_rule_ids"),
            missing_structure_signatures=selector_with_context.get("missing_structure_signatures"),
            repair_family_keys=selector_with_context.get("repair_family_keys"),
            runtime_active_only=True,
        )
        if not allowed_rule_ids:
            return []

    use_ann = bool(args.get("use_ann", True))
    if use_ann and not graph_vectors:
        # Only a positive, layer-scoped Control selector can bypass ANN.
        # Exclusion-only selectors must never expand into an unrestricted scan.
        if target_rule_layer == "control" and allowed_rule_ids is not None and (
            selector_with_context.get("required_rule_tags") or selector_with_context.get("required_any_rule_tags")
        ):
            use_ann = False
        else:
            return []
    if use_ann:
        ann_records: list[dict[str, Any]] = []
        seen_rule_ids: set[str] = set()
        query_vectors = select_ann_query_vectors(
            graph_vectors,
            max_queries=max(1, int(safe_float(args.get("max_ann_query_vectors"), 64))),
        )
        for query_vector in query_vectors:
            for record in search_rule_ann(
                rule_db_path,
                query_vector=query_vector,
                query_embedding_model=str(args.get("query_embedding_model") or args.get("embedding_model") or ""),
                query_vector_version=str(args.get("query_vector_version") or args.get("vector_version") or ""),
                top_k=max(limit, 1),
                candidate_limit=max(limit * 8, 64),
                allowed_rule_ids=allowed_rule_ids,
                runtime_active_only=True,
            ):
                rule_id = str(record.get("rule_id") or "")
                if not rule_id or rule_id in seen_rule_ids:
                    continue
                seen_rule_ids.add(rule_id)
                ann_records.append(record)
        filtered_ann = [
            rule for rule in ann_records
            if rule_layer(rule) == target_rule_layer or not target_rule_layer
            if rule_is_executable_candidate(rule, selector_with_context)
            if rule_matches_selector_numeric_bounds(rule, selector_with_context)
        ]
        regular = rank_loaded_rule_candidates(filtered_ann, selector_with_context)[: max(1, int(limit))]
        # ANN is an approximate ordering aid, not the authority that may erase
        # Rules from a small Control-authorized scope.  When positive hard
        # constraints have already identified a bounded set, supplement ANN
        # misses from that exact set and let pre-match plus the strict Matcher
        # decide.  Exclusion-only selectors deliberately do not get this path:
        # they could otherwise expand into an almost unrestricted DB scan.
        positive_scope = any(
            selector_with_context.get(field)
            for field in (
                "required_rule_tags",
                "required_any_rule_tags",
                "activation_policies",
                "repair_target_rule_ids",
                "missing_structure_signatures",
                "repair_family_keys",
            )
        )
        if allowed_rule_ids is not None and positive_scope and len(allowed_rule_ids) <= max(limit, 256):
            loaded_ids = {str(rule.get("rule_id") or "") for rule in regular}
            missing_ids = [rule_id for rule_id in allowed_rule_ids if rule_id not in loaded_ids]
            if missing_ids:
                supplemental_records = load_rule_records(
                    rule_db_path,
                    rule_layer=target_rule_layer or None,
                    rule_ids=missing_ids,
                    runtime_active_only=True,
                    limit=None,
                )
                regular.extend(
                    rule
                    for rule in supplemental_records
                    if rule_layer(rule) == target_rule_layer or not target_rule_layer
                    if rule_is_executable_candidate(rule, selector_with_context)
                    if rule_matches_selector_numeric_bounds(rule, selector_with_context)
                )
        preferred_ids = _memory_preferred_rule_ids(selector_with_context)
        preferred_records = load_rule_records(
            rule_db_path,
            rule_layer=target_rule_layer or None,
            rule_ids=preferred_ids,
            runtime_active_only=True,
            limit=None,
        ) if preferred_ids else []
        preferred = [
            rule for rule in preferred_records
            if rule_layer(rule) == target_rule_layer or not target_rule_layer
            if rule_is_executable_candidate(rule, selector_with_context)
            if rule_matches_selector_numeric_bounds(rule, selector_with_context)
        ]
        return _include_memory_preferred_candidates(regular, preferred, selector_with_context)
    records = load_rule_records(
        rule_db_path,
        rule_layer=target_rule_layer or None,
        rule_ids=allowed_rule_ids,
        min_priority=selector.get("min_priority"),
        min_confidence=selector.get("min_confidence"),
        max_contradiction_rate=selector.get("max_contradiction_rate"),
        runtime_active_only=True,
        limit=None if allowed_rule_ids is not None else max(limit * 4, limit) if target_rule_layer else None,
    )
    filtered = [
        rule for rule in records
        if rule_layer(rule) == target_rule_layer or not target_rule_layer
        if rule_is_executable_candidate(rule, selector_with_context)
        if rule_matches_selector_numeric_bounds(rule, selector_with_context)
    ]
    regular = rank_loaded_rule_candidates(filtered, selector_with_context)[: max(1, int(limit))]
    return _include_memory_preferred_candidates(regular, filtered, selector_with_context)


def _memory_preferred_rule_ids(selector: dict[str, Any]) -> list[str]:
    """Return Control-declared historical paths; they are candidates, not evidence."""
    return list(dict.fromkeys(
        str(rule_id).strip()
        for rule_id in selector.get("memory_preferred_rule_ids", [])
        if str(rule_id).strip()
    ))


def _include_memory_preferred_candidates(
    regular: list[dict[str, Any]],
    available: list[dict[str, Any]],
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep ordinary candidates and append viable historical paths for hard matching.

    A successful Memory path must never replace the normal ANN candidate pool
    before strict matching.  It is added to the pool only by its explicit Rule
    ID; Control may prefer it later if it matches and can execute.
    """
    preferred_order = _memory_preferred_rule_ids(selector)
    by_id = {str(rule.get("rule_id") or ""): rule for rule in available}
    merged = list(regular)
    known = {str(rule.get("rule_id") or "") for rule in merged}
    for rule_id in preferred_order:
        rule = by_id.get(rule_id)
        if rule is not None and rule_id not in known:
            merged.append(rule)
            known.add(rule_id)
    return merged


def select_ann_query_vectors(
    graph_vectors: list[list[float]],
    *,
    max_queries: int,
) -> list[list[float]]:
    """Select ANN queries without making graph field order a retrieval boundary."""

    unique_vectors: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for vector in graph_vectors:
        if not isinstance(vector, list) or not vector:
            continue
        identity = tuple(float(value) for value in vector)
        if identity in seen:
            continue
        seen.add(identity)
        unique_vectors.append(vector)
    if len(unique_vectors) <= max_queries:
        return unique_vectors
    if max_queries == 1:
        return [unique_vectors[0]]
    last_index = len(unique_vectors) - 1
    indices = {
        round(position * last_index / (max_queries - 1))
        for position in range(max_queries)
    }
    return [unique_vectors[index] for index in sorted(indices)]


def rule_is_executable_candidate(rule: dict[str, Any], selector: Any = None) -> bool:
    if not _rule_satisfies_selector_tags(rule, selector):
        return False
    if not _rule_satisfies_runtime_activation(rule, selector):
        return False
    if _auto_generated_learning_candidate_blocked(rule, selector):
        return False
    status = str(
        rule.get("status")
        or rule.get("revision_status")
        or rule.get("abstraction_level")
        or ""
    ).strip().lower()
    if not status:
        return True
    return status not in {
        "candidate",
        "candidates",
        "instance_seed",
        "recurring_pattern_candidate",
        "causal_pattern_candidate",
        "modified_candidate",
        "copy_ready",
        "draft",
        "inactive",
        "deprecated",
        "superseded",
        "archived",
        "deleted",
        "rejected",
        "blocked",
    }


def _rule_satisfies_selector_tags(rule: dict[str, Any], selector: Any) -> bool:
    """Apply explicit structural selector constraints, never debug-text hints."""

    if not isinstance(selector, dict):
        return True
    rule_tags: set[str] = set()
    for field in ("scope_tags", "tags"):
        raw = rule.get(field)
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        rule_tags.update(str(tag).strip().lower() for tag in values if str(tag or "").strip())
    required = _selector_tag_values(selector.get("required_rule_tags"))
    required_any = _selector_tag_values(selector.get("required_any_rule_tags"))
    excluded = _selector_tag_values(selector.get("excluded_rule_tags"))
    if required and not required.issubset(rule_tags):
        return False
    if required_any and rule_tags.isdisjoint(required_any):
        return False
    return rule_tags.isdisjoint(excluded)


def _rule_satisfies_runtime_activation(rule: dict[str, Any], selector: Any) -> bool:
    """Apply Control-authored Rewire identity constraints without reading debug text."""

    if not isinstance(selector, dict):
        return True
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    activation = rule.get("runtime_activation") if isinstance(rule.get("runtime_activation"), dict) else {}
    rule_values = {
        "activation_policies": str(
            activation.get("policy") or metadata.get("repair_activation_policy") or ""
        ).strip().lower(),
        "repair_target_rule_ids": str(
            activation.get("target_rule_id") or metadata.get("repair_target_rule_id") or ""
        ).strip(),
        "missing_structure_signatures": str(
            activation.get("missing_structure_signature")
            or metadata.get("missing_structure_signature")
            or ""
        ).strip(),
        "repair_family_keys": str(
            activation.get("repair_family_key") or metadata.get("repair_family_key") or ""
        ).strip(),
    }
    for field, actual in rule_values.items():
        requested = _selector_activation_values(selector.get(field), lower=(field == "activation_policies"))
        if requested and actual not in requested:
            return False
    return True


def _selector_activation_values(value: Any, *, lower: bool) -> set[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {
        (str(item).strip().lower() if lower else str(item).strip())
        for item in values
        if str(item or "").strip()
    }


def _selector_tag_values(value: Any) -> set[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {
        str(tag).strip().lower()
        for tag in values
        if str(tag or "").strip()
    }


def _auto_generated_learning_candidate_blocked(rule: dict[str, Any], selector: Any = None) -> bool:
    """Keep learning samples out of normal reasoning until explicitly promoted."""

    selector_map = selector if isinstance(selector, dict) else {}
    rid = str(rule.get("rule_id") or "").strip()
    layer = str(rule.get("rule_layer") or "").strip().lower()
    kind = str(rule.get("rule_kind") or "").strip().lower()
    target_layer = str(selector_map.get("target_rule_layer") or "").strip().lower()
    tags = {
        str(tag).strip().lower()
        for tag in (rule.get("scope_tags") or rule.get("tags") or [])
        if str(tag).strip()
    }
    status = str(rule.get("status") or rule.get("revision_status") or "").strip().lower()
    promotion_status = str(
        rule.get("promotion_status")
        or rule.get("validation_status")
        or rule.get("promoted_status")
        or ""
    ).strip().lower()
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}

    if promotion_status == "shadow" and not selector_map.get("isolated_shadow_evaluation"):
        return True
    if selector_map.get("include_learning_candidates") or selector_map.get("include_auto_generated_candidates"):
        return False
    if target_layer == "auto_learning" and layer == "auto_learning":
        return False
    if promotion_status in {"active", "validated", "verified", "promoted", "passed"} and status not in {"candidate", "instance_seed"}:
        return False

    generated_id = rid.startswith("auto_family_") or rid.startswith("auto_path_supervision_")
    candidate_status = status in {
        "candidate",
        "candidates",
        "instance_seed",
        "recurring_pattern_candidate",
        "causal_pattern_candidate",
        "modified_candidate",
        "copy_ready",
        "draft",
    }
    if kind == "auto_learning_event" and layer == target_layer and not candidate_status:
        return False

    learning_sample = kind == "auto_learning_event" or bool(
        tags.intersection(
            {
                "auto_learning",
                "event_to_rule",
                "path_supervision",
                "instance_seed",
                "causal_pattern_candidate",
                "recurring_pattern_candidate",
            }
        )
    )
    return generated_id or learning_sample


def rank_loaded_rule_candidates(rules: list[dict[str, Any]], selector: Any) -> list[dict[str, Any]]:
    if not isinstance(selector, dict) or not selector:
        return list(rules)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for rule in rules:
        score = 0.0
        score += 3.0 * max(0.0, min(safe_float(rule.get("vector_score"), 0.0), 1.0))
        score += 0.05 * max(0.0, min(safe_float(rule.get("priority"), 1.0), 1.0))
        score += 0.05 * max(0.0, min(safe_float(rule.get("confidence"), 0.5), 1.0))
        ranked.append((score, rule))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [rule for _, rule in ranked]


def rule_matches_selector_numeric_bounds(rule: dict[str, Any], selector: Any) -> bool:
    if not isinstance(selector, dict) or not selector:
        return True
    visited = set(selector_attempted_rule_ids(selector))
    if str(rule.get("rule_id") or "") in visited:
        return False
    min_priority = selector.get("min_priority")
    if min_priority is not None and safe_float(rule.get("priority"), 0.0) < safe_float(min_priority, 0.0):
        return False
    min_confidence = selector.get("min_confidence")
    if min_confidence is not None and safe_float(rule.get("confidence"), 0.0) < safe_float(min_confidence, 0.0):
        return False
    max_contradiction = selector.get("max_contradiction_rate")
    if max_contradiction is not None and safe_float(rule.get("contradiction_rate"), 0.0) > safe_float(max_contradiction, 1.0):
        return False
    return True


def state_rule_db_path(state: Any) -> str:
    db_paths = getattr(state, "db_paths", {}) or {}
    if isinstance(db_paths, dict):
        return str(db_paths.get("rule") or db_paths.get("rule_db") or "")
    return ""


def rule_layer(rule: dict[str, Any]) -> str:
    raw = rule.get("rule_layer") or rule.get("layer") or ""
    return normalize_rule_layer(raw, default="") if raw else ""


def iter_match_workspaces(
    working_memory: WorkingMemoryState,
    active_event_ids: list[str],
    *,
    include_feedback_events: bool = False,
) -> list[dict[str, Any]]:
    selected = set(active_event_ids)
    results: list[dict[str, Any]] = []
    for event_id, workspace in (working_memory.event_workspaces or {}).items():
        normalized_event_id = str(event_id or workspace.get("event_id") or "")
        if selected and normalized_event_id not in selected:
            continue
        if not include_feedback_events and _is_auto_learn_feedback_workspace(workspace):
            continue
        results.append(
            {
                "event_id": normalized_event_id,
                "workspace": workspace,
                "span_index": build_span_index(workspace),
                "patterns": collect_event_match_patterns(workspace),
                "relations": collect_event_relations(workspace),
            }
        )
    return results


def _is_auto_learn_feedback_workspace(workspace: dict[str, Any]) -> bool:
    metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), dict) else {}
    source = str(metadata.get("source") or workspace.get("source") or "").strip().lower()
    source_type = str(metadata.get("source_type") or workspace.get("source_type") or "").strip().lower()
    learning_route = str(metadata.get("learning_route") or workspace.get("learning_route") or "").strip().lower()
    event_type = str(workspace.get("event_type") or "").strip().lower()
    if learning_route == "auto_learn":
        return True
    if source in {"runtime_feedback", "diagnostic_harness", "expert_feedback"}:
        return True
    if source_type in {"runtime_evaluation", "expert", "diagnostic_feedback", "authority_feedback"}:
        return True
    return event_type in {"runtime_task_feedback", "expert_feedback_learning"}


def normalize_required_pattern(required: dict[str, Any]) -> dict[str, Any]:
    condition_spans = as_dict_list(required.get("condition"))
    if condition_spans:
        return {
            "span_slots": [_span_slot_from_condition_span(span, index) for index, span in enumerate(condition_spans)],
            "relation_patterns": [_relation_pattern_from_rule(relation) for relation in as_dict_list(required.get("relation_patterns"))],
            "shared_semantic_relation_constraints": [
                _shared_semantic_relation_constraint(item)
                for item in as_dict_list(required.get("shared_semantic_relation_constraints"))
            ],
            "predicates": as_dict_list(required.get("predicates")),
        }
    return {}


def _span_slot_from_condition_span(span: dict[str, Any], index: int) -> dict[str, Any]:
    slot = {
        "slot_id": str(span.get("span_id") or span.get("slot_id") or f"condition_span_{index + 1}"),
        "span_id": str(span.get("span_id") or span.get("source_span_id") or ""),
        "source_span_id": str(span.get("source_span_id") or ""),
        "debug_text": " ".join(
            str(span.get(key) or "").strip()
            for key in ("core", "relation", "nature", "text", "debug_text")
            if str(span.get(key) or "").strip()
        ),
        "core": str(span.get("core") or ""),
        "relation": str(span.get("relation") or ""),
        "nature": str(span.get("nature") or ""),
        "text": str(span.get("text") or ""),
        "match_threshold": safe_float(span.get("match_threshold"), safe_float(span.get("threshold"), 0.5)),
        "condition_priority": safe_float(span.get("condition_priority"), 1.0),
        "missing_effect": str(span.get("missing_effect") or "").strip(),
        "allow_shared_binding": bool(span.get("allow_shared_binding", False)),
        "feature_type": str(span.get("feature_type") or span.get("feature_schema_id") or "").strip(),
    }
    # These are Rule-declared evidence requirements, not routing hints.  They
    # must reach ``structured_slot_match`` unchanged; otherwise a Rule can
    # declare authoritative evidence but be matched as if it only required a
    # semantically similar span.
    if span.get("min_source_reliability") is not None:
        slot["min_source_reliability"] = safe_float(span.get("min_source_reliability"), 0.0)
    if bool(span.get("require_authority_verified")):
        slot["require_authority_verified"] = True
    for key in ("core_vector", "relation_vector", "nature_vector"):
        vector = as_number_list(span.get(key))
        if vector:
            slot[key] = vector
    return {key: value for key, value in slot.items() if value not in ("", [], None)}


def _relation_pattern_from_rule(relation: dict[str, Any]) -> dict[str, Any]:
    pattern = {
        "relation_id": str(relation.get("relation_id") or relation.get("id") or ""),
        "source_span_id": str(relation.get("source_span_id") or relation.get("source_slot") or ""),
        "source_slot": str(relation.get("source_span_id") or relation.get("source_slot") or ""),
        "relation": str(relation.get("relation") or relation.get("relation_type") or ""),
        "target_span_id": str(relation.get("target_span_id") or relation.get("target_slot") or ""),
        "target_slot": str(relation.get("target_span_id") or relation.get("target_slot") or ""),
        "debug_relation": str(relation.get("debug_relation") or relation.get("relation") or relation.get("relation_type") or ""),
        "required_for_output": bool(relation.get("required_for_output", False)),
        "missing_effect": str(relation.get("missing_effect") or "").strip(),
        "match_threshold": safe_float(relation.get("match_threshold"), 0.5),
        "relation_confidence": safe_float(relation.get("relation_confidence"), 0.5),
    }
    vector = as_number_list(relation.get("relation_vector"))
    if vector:
        pattern["relation_vector"] = vector
    return {key: value for key, value in pattern.items() if value not in ("", [], None)}


def _shared_semantic_relation_constraint(value: dict[str, Any]) -> dict[str, Any]:
    """Keep vector-only cross-span context constraints in the match contract."""
    slot_ids = [str(item).strip() for item in value.get("slot_ids", []) if str(item).strip()]
    constraint = {
        "slot_ids": slot_ids,
        "match_threshold": safe_float(value.get("match_threshold"), 0.995),
    }
    for field in ("core_vector", "relation_vector"):
        vector = as_number_list(value.get(field))
        if vector:
            constraint[field] = vector
    # Text stays only for trace/debug. Matcher accepts this constraint only
    # when its vector fields are present.
    constraint["debug"] = {
        field: str(value.get(field) or "").strip()
        for field in ("core", "relation")
        if str(value.get(field) or "").strip()
    }
    return constraint


def collect_event_match_patterns(node: Any) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    synthetic = {
        "pattern_id": "workspace_synthetic_pattern",
        "span_slots": [{"span_id": span_id(span)} for span in as_dict_list(node.get("spans") if isinstance(node, dict) else []) if span_id(span)],
        "relation_patterns": collect_event_relations(node if isinstance(node, dict) else {}),
    }
    if synthetic["span_slots"] or synthetic["relation_patterns"]:
        patterns.append(synthetic)
    return patterns


def collect_event_relations(workspace: dict[str, Any]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = as_dict_list(workspace.get("relations"))
    for context in as_dict_list(workspace.get("context_items")):
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        context_id = str(context.get("context_id") or "")
        context_spans = _context_spans(payload)
        endpoint_ids = {
            original_id: _context_span_id(context_id, original_id)
            for position, span in enumerate(context_spans, start=1)
            if (original_id := span_id(span) or str(position))
        }
        for relation in _context_relations(payload):
            item = dict(relation)
            source_id = str(item.get("source_span_id") or "")
            target_id = str(item.get("target_span_id") or "")
            if source_id in endpoint_ids:
                item["source_span_id"] = endpoint_ids[source_id]
            if target_id in endpoint_ids:
                item["target_span_id"] = endpoint_ids[target_id]
            item.setdefault("source", "event_context")
            item.setdefault("context_id", context_id)
            relations.append(item)
    return relations


def build_span_index(workspace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for span in as_dict_list(workspace.get("spans")):
        sid = span_id(span)
        if sid:
            index[sid] = span
    for context in as_dict_list(workspace.get("context_items")):
        payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
        context_id = str(context.get("context_id") or "")
        for position, span in enumerate(_context_spans(payload), start=1):
            item = dict(span)
            original_id = span_id(item) or str(position)
            item["span_id"] = _context_span_id(context_id, original_id)
            item.setdefault("source", "event_context")
            item["context_id"] = context_id
            index[item["span_id"]] = item
    return index


def _context_span_id(context_id: str, original_id: str) -> str:
    return f"context_{context_id}_{original_id}"


def _context_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    # Memory is a historical precedent, never fresh evidence for this task.
    # Its old Event Graph therefore cannot satisfy a current Rule require.
    if str(payload.get("retrieval_source") or "").strip() in {
        "memory_db", "memory_redis", "memory_workspace",
    }:
        return []
    spans = as_dict_list(payload.get("spans"))
    event_hypergraph = payload.get("event_hypergraph") if isinstance(payload.get("event_hypergraph"), dict) else {}
    spans.extend(as_dict_list(event_hypergraph.get("spans")))
    event_workspaces = event_hypergraph.get("event_workspaces") if isinstance(event_hypergraph.get("event_workspaces"), dict) else {}
    for workspace in event_workspaces.values():
        if isinstance(workspace, dict):
            spans.extend(as_dict_list(workspace.get("spans")))
    if spans:
        return spans
    if str(payload.get("retrieval_source") or "").strip() == "knowledge_db" or payload.get("knowledge_id"):
        return []
    text = str(payload.get("text") or payload.get("memory_debug_text") or payload.get("input_summary") or "").strip()
    core = str(payload.get("core") or payload.get("topic") or text[:160]).strip()
    relation = str(payload.get("relation") or "provides_context").strip()
    nature = str(payload.get("nature") or payload.get("knowledge_type") or payload.get("memory_kind") or text).strip()
    if not any((core, nature)):
        return []
    return [
        {
            "span_id": str(payload.get("id") or payload.get("knowledge_id") or payload.get("memory_id") or "record"),
            "core": core,
            "relation": relation,
            "nature": nature,
            "text": text,
            "core_vector": payload.get("core_vector") or payload.get("memory_vector"),
            "relation_vector": payload.get("relation_vector"),
            "nature_vector": payload.get("nature_vector"),
            "vector": payload.get("vector") or payload.get("memory_vector"),
            "embedding_model": payload.get("embedding_model"),
            "vector_version": payload.get("vector_version"),
        }
    ]


def _context_relations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    relations = as_dict_list(payload.get("relations"))
    event_hypergraph = payload.get("event_hypergraph") if isinstance(payload.get("event_hypergraph"), dict) else {}
    relations.extend(as_dict_list(event_hypergraph.get("relations")))
    event_workspaces = event_hypergraph.get("event_workspaces") if isinstance(event_hypergraph.get("event_workspaces"), dict) else {}
    for workspace in event_workspaces.values():
        if isinstance(workspace, dict):
            relations.extend(as_dict_list(workspace.get("relations")))
    return relations


def span_id(span: dict[str, Any]) -> str:
    return str(span.get("span_id") or span.get("mention_id") or span.get("id") or "")


def collect_vectors(node: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    if isinstance(node, dict):
        for key in (
            "core_vector",
            "span_core_vector",
            "nature_vector",
            "relation_vector",
            "rule_vector",
        ):
            vector = as_number_list(node.get(key))
            if vector:
                vectors.append(vector)
        for value in node.values():
            vectors.extend(collect_vectors(value))
    elif isinstance(node, list):
        for item in node:
            vectors.extend(collect_vectors(item))
    return vectors


def as_number_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                nested = as_number_list(item)
                if nested:
                    result.extend(nested)
                continue
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                return None
        return result or None
    return None


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []
