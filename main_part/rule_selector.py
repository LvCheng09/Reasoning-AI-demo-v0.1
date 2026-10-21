"""Canonical runtime contract for rule-search requests."""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_HOPS = 8


def normalize_rule_selectors(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    return [normalize_rule_selector(item) for item in values if isinstance(item, dict)]


def selector_attempted_rule_ids(selector: Any) -> list[str]:
    if not isinstance(selector, dict):
        return []
    trace = selector.get("search_trace") if isinstance(selector.get("search_trace"), dict) else {}
    values = trace.get("attempted_rule_ids")
    if values is None:
        values = selector.get("visited_rule_ids")
    return _unique_strings(values)


def selector_control_rule_ids(selector: Any) -> list[str]:
    if not isinstance(selector, dict):
        return []
    trace = selector.get("search_trace") if isinstance(selector.get("search_trace"), dict) else {}
    return _unique_strings(trace.get("control_rule_ids"))


def selector_execution_context(selector: Any) -> dict[str, Any]:
    normalized = normalize_rule_selector(selector)
    return {
        "search_trace": dict(normalized["search_trace"]),
        "authorization": dict(normalized.get("authorization") or {}),
        "rule_layer": str(normalized.get("rule_layer") or normalized.get("target_rule_layer") or ""),
        "operator_family": _unique_strings(normalized.get("operator_family")),
        "search_mode": str(normalized.get("search_mode") or ""),
        "required_rule_tags": list(normalized.get("required_rule_tags") or []),
        "required_any_rule_tags": list(normalized.get("required_any_rule_tags") or []),
        "excluded_rule_tags": list(normalized.get("excluded_rule_tags") or []),
        "activation_policies": list(normalized.get("activation_policies") or []),
        "activation_signal": normalized.get("activation_signal"),
        "activation_weight": normalized.get("activation_weight", 1.0),
        "repair_target_rule_ids": list(normalized.get("repair_target_rule_ids") or []),
        "missing_structure_signatures": list(
            normalized.get("missing_structure_signatures") or []
        ),
        "repair_family_keys": list(normalized.get("repair_family_keys") or []),
    }


def child_rule_selectors(
    produced: dict[str, Any],
    *,
    rule_match: dict[str, Any],
) -> list[dict[str, Any]]:
    templates = _dict_list(produced.get("rule_selectors"))
    if isinstance(produced.get("rule_selector"), dict):
        templates.append(dict(produced["rule_selector"]))
    if not templates:
        return []

    rule = rule_match.get("rule") if isinstance(rule_match.get("rule"), dict) else rule_match
    rule_id = str(rule.get("rule_id") or "").strip()
    rule_layer = str(rule.get("rule_layer") or "").strip().lower()
    parent = normalize_rule_selector(rule_match.get("selector_context"))
    parent_trace = parent["search_trace"]

    children: list[dict[str, Any]] = []
    for template in templates:
        template_trace = (
            template.get("search_trace")
            if isinstance(template.get("search_trace"), dict)
            else {}
        )
        explicit_max_hops = (
            template_trace.get("max_hops")
            if "max_hops" in template_trace
            else template.get("max_depth")
            if "max_depth" in template
            else None
        )
        child = normalize_rule_selector(template)
        child_trace = child["search_trace"]
        parent_max_hops = _integer(parent_trace.get("max_hops"), DEFAULT_MAX_HOPS)
        max_hops = (
            min(parent_max_hops, _integer(explicit_max_hops, parent_max_hops))
            if explicit_max_hops is not None
            else parent_max_hops
        )
        hop = _integer(parent_trace.get("hop"), 0) + 1
        if hop > max_hops:
            continue
        attempted = _unique_strings(
            [
                *selector_attempted_rule_ids(parent),
                *selector_attempted_rule_ids(child),
                *([rule_id] if rule_id else []),
            ]
        )
        control_lineage = _unique_strings(
            [
                *selector_control_rule_ids(parent),
                *selector_control_rule_ids(child),
                *([rule_id] if rule_layer == "control" and rule_id else []),
            ]
        )
        child["search_trace"] = {
            "hop": hop,
            "max_hops": max_hops,
            "attempted_rule_ids": attempted,
            "control_rule_ids": control_lineage,
            "parent_rule_id": rule_id,
        }
        if rule_layer == "control" and rule_id:
            child["authorization"] = {"kind": "control_rule", "rule_id": rule_id}
        children.append(child)
    return children


def normalize_rule_selector(selector: Any) -> dict[str, Any]:
    raw = dict(selector) if isinstance(selector, dict) else {}
    trace = dict(raw.get("search_trace") or {}) if isinstance(raw.get("search_trace"), dict) else {}
    legacy_hop = raw.pop("depth", 0)
    legacy_max_hops = raw.pop("max_depth", DEFAULT_MAX_HOPS)
    legacy_attempted = raw.pop("visited_rule_ids", [])
    legacy_parent = raw.pop("parent_rule_id", "")
    legacy_kind = str(raw.pop("selection_authority", "") or "")
    legacy_rule_id = str(raw.pop("control_rule_id", "") or "")
    trace.setdefault("hop", _integer(legacy_hop, 0))
    trace.setdefault("max_hops", _integer(legacy_max_hops, DEFAULT_MAX_HOPS))
    trace.setdefault("attempted_rule_ids", _unique_strings(legacy_attempted))
    trace.setdefault("control_rule_ids", [])
    trace.setdefault("parent_rule_id", str(legacy_parent or ""))
    raw["search_trace"] = {
        "hop": _integer(trace.get("hop"), 0),
        "max_hops": _integer(trace.get("max_hops"), DEFAULT_MAX_HOPS),
        "attempted_rule_ids": _unique_strings(trace.get("attempted_rule_ids")),
        "control_rule_ids": _unique_strings(trace.get("control_rule_ids")),
        "parent_rule_id": str(trace.get("parent_rule_id") or ""),
    }
    if not isinstance(raw.get("authorization"), dict):
        raw["authorization"] = (
            {"kind": legacy_kind, "rule_id": legacy_rule_id}
            if legacy_kind or legacy_rule_id
            else {}
        )
    else:
        raw["authorization"] = {
            "kind": str(raw["authorization"].get("kind") or ""),
            "rule_id": str(raw["authorization"].get("rule_id") or ""),
        }
    for field in (
        "required_rule_tags",
        "required_any_rule_tags",
        "excluded_rule_tags",
    ):
        raw[field] = _normalized_tags(raw.get(field))
    activation_aliases = {
        "activation_policies": "activation_policy",
        "repair_target_rule_ids": "repair_target_rule_id",
        "missing_structure_signatures": "missing_structure_signature",
        "repair_family_keys": "repair_family_key",
    }
    for field, singular in activation_aliases.items():
        value = raw.get(field, raw.get(singular))
        raw[field] = _unique_strings(value)
    raw["activation_policies"] = [
        policy.strip().lower()
        for policy in raw["activation_policies"]
        if policy.strip()
    ]
    return raw


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _unique_strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item) for item in values if str(item)))


def _normalized_tags(value: Any) -> list[str]:
    return list(
        dict.fromkeys(
            str(item).strip().lower()
            for item in _list_value(value)
            if str(item).strip()
        )
    )


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
