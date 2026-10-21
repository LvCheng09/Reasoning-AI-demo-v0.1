"""Build internally compatible runtime rule-path candidates from matcher results."""

from __future__ import annotations

from typing import Any
from copy import deepcopy

from .identity import stable_content_id
from .rule_execution_contract import candidate_can_execute, is_typed_rule, span_output_items
from .activation_semantics import ActivationError, number as activation_number


def attach_candidate_rule_paths(
    operation: dict[str, Any],
    *,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Split conflicting matches into alternative paths without choosing one."""

    normalized = dict(operation)
    matched_rules = _dict_list(operation.get("matched_rules"))
    if len(matched_rules) < 2:
        return normalized

    conflict_pairs = _conflict_pairs(matched_rules, graph=graph)
    if not conflict_pairs:
        return normalized

    paths, truncated = _maximal_compatible_paths(matched_rules, conflict_pairs)
    normalized["matched_rules"] = []
    normalized["candidate_rule_paths"] = [
        _candidate_path(items, index=index, graph=graph)
        for index, items in enumerate(paths, start=1)
    ]
    normalized["requires_control_selection"] = True
    normalized["path_conflicts"] = [
        {
            "left_rule_id": _rule_id(matched_rules[left]),
            "right_rule_id": _rule_id(matched_rules[right]),
            "reasons": reasons,
        }
        for (left, right), reasons in sorted(conflict_pairs.items())
    ]
    normalized["candidate_path_generation"] = {
        "policy": "bounded_maximal_structurally_compatible_paths",
        "candidate_path_count": len(paths),
        "truncated": truncated,
    }
    return normalized


def rule_path_claims(
    rule_match: dict[str, Any],
    *,
    graph: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return explicit one-use/capacity claims bound by a successful match."""

    outcome = rule_match.get("match") if isinstance(rule_match.get("match"), dict) else {}
    metadata = outcome.get("metadata") if isinstance(outcome.get("metadata"), dict) else {}
    claims: list[dict[str, Any]] = []
    for required_outcome in _dict_list(metadata.get("per_required")):
        required = required_outcome.get("required") if isinstance(required_outcome.get("required"), dict) else {}
        details = required_outcome.get("details") if isinstance(required_outcome.get("details"), dict) else {}
        bindings = details.get("slot_bindings") if isinstance(details.get("slot_bindings"), dict) else {}
        event_id = str(required_outcome.get("event_id") or details.get("event_id") or "").strip()
        for condition in _dict_list(required.get("condition")):
            usage_policy = str(condition.get("usage_policy") or "reusable").strip().lower()
            if usage_policy not in {"consumable", "capacity"}:
                continue
            slot_id = str(condition.get("slot_id") or condition.get("span_id") or condition.get("id") or "").strip()
            span_id = str(bindings.get(slot_id) or "").strip()
            evaluation = rule_match.get("require_evaluation")
            if evaluation is not None and evaluation["bindings"].get(slot_id) != span_id:
                continue
            if not event_id or not span_id:
                continue
            claims.append(
                {
                    "event_id": event_id,
                    "span_id": span_id,
                    "usage_policy": usage_policy,
                    "amount": max(0.0, _number(condition.get("capacity_cost"), 1.0)),
                    "capacity": _claim_capacity(
                        graph,
                        event_id=event_id,
                        span_id=span_id,
                        usage_policy=usage_policy,
                        declared_capacity=condition.get("capacity"),
                    ),
                    "required_slot_id": slot_id,
                    "rule_id": _rule_id(rule_match),
                }
            )
    return claims


def choose_candidate_rule_path(
    candidate_paths: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
    attempted_rule_ids: list[str] | None = None,
    memory_preferred_rule_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Apply a Control rule's declared scoring policy to compatible paths."""

    weights = {
        "match_quality": 0.45,
        "routing_quality": 0.20,
        "confidence": 0.20,
        "priority": 0.10,
        "risk": -0.20,
        "claim_pressure": -0.05,
        "attempted_rule": -0.50,
        "activation": 0.0,
    }
    supplied = (policy or {}).get("weights") if isinstance((policy or {}).get("weights"), dict) else {}
    for key in weights:
        if key in supplied:
            weights[key] = _number(supplied[key], weights[key])
    attempted = {str(value) for value in attempted_rule_ids or [] if str(value)}
    preferred = set(memory_preferred_rule_ids or [])
    ranked = []
    for path in _dict_list(candidate_paths):
        features = path.get("selection_features") if isinstance(path.get("selection_features"), dict) else {}
        path_rule_ids = {str(value) for value in path.get("rule_ids", []) if str(value)}
        matched = _dict_list(path.get('matched_rules'))
        if not matched or not all(candidate_can_execute(m) for m in matched):
            continue
        if weights["activation"]:
            signal = (policy or {}).get("activation_signal")
            if not signal:
                raise ActivationError("missing_path_activation_signal", "activation weight requires a named signal")
            features = {**features, "activation": _average([
                activation_number(m["activation"]["signals"][signal]["value"], signal) for m in matched])}
        score = sum(
            weights[name] * _number(features.get(name), 0.0)
            for name in ("match_quality", "routing_quality", "confidence", "priority", "risk", "claim_pressure", "activation")
        )
        if attempted.intersection(path_rule_ids):
            score += weights["attempted_rule"]
        ranked.append((bool(preferred.intersection(path_rule_ids)), score, str(path.get("candidate_path_id") or ""), path))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:3], reverse=True)
    selected = deepcopy(ranked[0][3])
    selected["control_selection_score"] = ranked[0][1]
    selected['control_memory_preference_applied'] = ranked[0][0]
    selected['control_selection_policy'] = deepcopy(policy or {})
    selected['normal_fallback_candidates'] = [deepcopy(item[3]) for item in ranked if not item[0]] if ranked[0][0] else []
    return selected


def _conflict_pairs(
    matched_rules: list[dict[str, Any]],
    *,
    graph: dict[str, Any] | None,
) -> dict[tuple[int, int], list[str]]:
    pairs: dict[tuple[int, int], list[str]] = {}
    claims = [rule_path_claims(item, graph=graph) for item in matched_rules]
    declarations = [_rule_branch_contract(item) for item in matched_rules]
    mutation_scopes = [_rule_mutation_scopes(item) for item in matched_rules]
    for left in range(len(matched_rules)):
        for right in range(left + 1, len(matched_rules)):
            reasons: list[str] = []
            left_id = _rule_id(matched_rules[left])
            right_id = _rule_id(matched_rules[right])
            if right_id in declarations[left]["incompatible_rule_ids"] or left_id in declarations[right]["incompatible_rule_ids"]:
                reasons.append("explicit_rule_incompatibility")
            left_group = declarations[left]["exclusive_group"]
            right_group = declarations[right]["exclusive_group"]
            if left_group and left_group == right_group and declarations[left]["exclusive_value"] != declarations[right]["exclusive_value"]:
                reasons.append(f"exclusive_branch:{left_group}")
            if _claims_exceed_capacity(claims[left], claims[right]):
                reasons.append("shared_consumable_or_capacity_claim")
            if mutation_scopes[left].intersection(mutation_scopes[right]) - _shared_fusion_writes(matched_rules[left], matched_rules[right]):
                reasons.append("competing_graph_mutation")
            if reasons:
                pairs[(left, right)] = reasons
    return pairs


def _shared_fusion_writes(left: dict, right: dict) -> set[str]:
    """Declared reducers may share their output; other graph writes conflict."""
    def declarations(item):
        rule = _rule(item)
        q = item.get("resolved_q", rule.get("q_semantics", {}))
        weights = {entry["target"]["span_id"]: (entry["group"], entry["normalization"])
                   for entry in q.get("transform", []) if entry["type"] == "weight"}
        return {f"{'write' if is_typed_rule(rule) else 'add'}:{p.get('event_id') or '*'}:span:{s['span_id']}": weights[s["span_id"]]
                for p in rule.get("produced", []) if p.get("operation") in ({"add", "update"} if is_typed_rule(rule) else {"add"})
                for s in (span_output_items(p) if is_typed_rule(rule) else p.get("conclusion", [])) if s.get("span_id") in weights}
    a, b = declarations(left), declarations(right)
    return {scope for scope in a.keys() & b.keys() if a[scope] == b[scope]}


def _maximal_compatible_paths(
    matched_rules: list[dict[str, Any]],
    conflict_pairs: dict[tuple[int, int], list[str]],
    *,
    limit: int = 32,
) -> tuple[list[list[dict[str, Any]]], bool]:
    """Build bounded alternatives without enumerating all independent sets.

    Matcher order already represents retrieval quality. Each seed preserves
    one branch and greedily adds every compatible rule in that order. This
    exposes conflicting alternatives to Control while keeping generation
    bounded for large matcher batches.
    """

    adjacency = {index: set() for index in range(len(matched_rules))}
    for left, right in conflict_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)

    results: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    truncated = False
    for seed in range(len(matched_rules)):
        chosen = [seed]
        for candidate in range(len(matched_rules)):
            if candidate == seed:
                continue
            if all(candidate not in adjacency[selected] for selected in chosen):
                chosen.append(candidate)
        path = frozenset(chosen)
        if path in seen:
            continue
        seen.add(path)
        if len(results) >= limit:
            truncated = True
            continue
        results.append(path)

    selected = sorted(results, key=lambda value: (-len(value), tuple(value)))
    return [
        [matched_rules[index] for index in sorted(indices)]
        for indices in selected
    ], truncated


def _candidate_path(
    items: list[dict[str, Any]],
    *,
    index: int,
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    rule_ids = [_rule_id(item) for item in items if _rule_id(item)]
    claims = [claim for item in items for claim in rule_path_claims(item, graph=graph)]
    match_values = [_match_quality(item) for item in items]
    routing_values = [_number(item.get("routing_score"), 0.0) for item in items]
    confidence_values = [_number((_rule(item)).get("confidence"), 0.5) for item in items]
    priority_values = [min(1.0, max(0.0, _number((_rule(item)).get("priority"), 0.0) / 10.0)) for item in items]
    risk_values = [_number((_rule(item)).get("risk"), 0.0) for item in items]
    signature = {"rule_ids": rule_ids, "claims": claims}
    event_ids = sorted(
        {
            event_id
            for item in items
            for event_id in _matched_event_ids(item)
            if event_id
        }
    )
    if not event_ids:
        event_ids = _graph_event_ids(graph)
    return {
        "candidate_path_id": stable_content_id("runtime_candidate_rule_path", signature),
        "candidate_index": index,
        "rule_ids": rule_ids,
        "matched_rules": items,
        "claims": claims,
        "event_ids": event_ids,
        "selection_features": {
            "match_quality": _average(match_values),
            "routing_quality": _average(routing_values),
            "confidence": _average(confidence_values),
            "priority": _average(priority_values),
            "risk": _average(risk_values),
            "claim_pressure": min(1.0, len(claims) / 4.0),
        },
    }


def _matched_event_ids(item: dict[str, Any]) -> list[str]:
    outcome = item.get("match") if isinstance(item.get("match"), dict) else {}
    event_ids = {
        str(required.get("event_id") or "").strip()
        for required in _dict_list(
            (outcome.get("metadata") or {}).get("per_required")
            if isinstance(outcome.get("metadata"), dict)
            else []
        )
        if str(required.get("event_id") or "").strip()
    }
    for produced in _dict_list(outcome.get("produced")):
        for required in _dict_list(produced.get("matched_required")):
            event_id = str(required.get("event_id") or "").strip()
            if event_id:
                event_ids.add(event_id)
    return sorted(event_ids)


def _graph_event_ids(graph: dict[str, Any] | None) -> list[str]:
    if not isinstance(graph, dict):
        return []
    event_ids: set[str] = set()
    direct_event_id = str(graph.get("event_id") or "").strip()
    if direct_event_id:
        event_ids.add(direct_event_id)
    for event in _dict_list(graph.get("events")):
        event_id = str(event.get("event_id") or "").strip()
        if event_id:
            event_ids.add(event_id)
    return sorted(event_ids)


def _rule_branch_contract(rule_match: dict[str, Any]) -> dict[str, Any]:
    rule = _rule(rule_match)
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    branch = rule.get("branch_contract") if isinstance(rule.get("branch_contract"), dict) else {}
    group = str(branch.get("exclusive_group") or rule.get("exclusive_group") or metadata.get("exclusive_group") or "").strip()
    value = str(branch.get("exclusive_value") or rule.get("exclusive_value") or metadata.get("exclusive_value") or "").strip()
    if group and not value:
        value = _rule_id(rule_match)
    incompatible = _strings(branch.get("incompatible_rule_ids"))
    incompatible.extend(_strings(rule.get("incompatible_rule_ids")))
    incompatible.extend(_strings(metadata.get("incompatible_rule_ids")))
    return {
        "exclusive_group": group,
        "exclusive_value": value,
        "incompatible_rule_ids": set(incompatible),
    }


def _rule_mutation_scopes(rule_match: dict[str, Any]) -> set[str]:
    """Identify graph locations that cannot be mutated by sibling paths together."""

    rule = _rule(rule_match)
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    scopes: set[str] = set()
    predecessor_id = str(metadata.get("repair_predecessor_rule_id") or "").strip()
    target_rule_id = str(metadata.get("repair_target_rule_id") or "").strip()
    if predecessor_id or target_rule_id:
        scopes.add(f"rule_edge:{predecessor_id}:{target_rule_id}")
    for produced in _dict_list(rule.get("produced")):
        operation = str(produced.get("operation") or produced.get("op") or "").strip().lower()
        if operation not in {"add", "update", "delete", "merge", "split"}:
            continue
        event_id = str(produced.get("event_id") or "*").strip() or "*"
        if is_typed_rule(rule) and produced.get("target", "spans") == "spans":
            ids = [s["span_id"] for s in span_output_items(produced)]
            if operation == "delete" and produced.get("id"):
                ids.append(produced["id"])
            scopes.update(f"write:{event_id}:span:{sid}" for sid in ids)
            continue
        for key in (
            "target_span_id",
            "span_id",
            "target_relation_id",
            "relation_id",
            "target_event_id",
        ):
            target_id = str(produced.get(key) or "").strip()
            if target_id:
                scopes.add(f"{operation}:{event_id}:{key}:{target_id}")
        for conclusion in _dict_list(produced.get("conclusion")):
            span_id = str(conclusion.get("span_id") or conclusion.get("id") or "").strip()
            if span_id:
                scopes.add(f"{operation}:{event_id}:span:{span_id}")
        for relation in _dict_list(produced.get("relations")):
            relation_id = str(relation.get("relation_id") or relation.get("id") or "").strip()
            if relation_id:
                scopes.add(f"{operation}:{event_id}:relation:{relation_id}")
    return scopes


def _claims_exceed_capacity(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    left_totals: dict[tuple[str, str], float] = {}
    right_totals: dict[tuple[str, str], float] = {}
    capacities: dict[tuple[str, str], float] = {}
    policies: dict[tuple[str, str], set[str]] = {}
    for claim in left:
        key = (str(claim.get("event_id") or ""), str(claim.get("span_id") or ""))
        left_totals[key] = left_totals.get(key, 0.0) + _number(claim.get("amount"), 1.0)
        capacities[key] = _number(claim.get("capacity"), capacities.get(key, 0.0))
        policies.setdefault(key, set()).add(str(claim.get("usage_policy") or ""))
    for claim in right:
        key = (str(claim.get("event_id") or ""), str(claim.get("span_id") or ""))
        right_totals[key] = right_totals.get(key, 0.0) + _number(claim.get("amount"), 1.0)
        capacities[key] = _number(claim.get("capacity"), capacities.get(key, 0.0))
        policies.setdefault(key, set()).add(str(claim.get("usage_policy") or ""))
    for key in left_totals.keys() & right_totals.keys():
        if "consumable" in policies.get(key, set()):
            return True
        capacity = capacities.get(key, 0.0)
        if capacity <= 0.0 or left_totals[key] + right_totals[key] > capacity + 1e-12:
            return True
    return False


def _claim_capacity(
    graph: dict[str, Any] | None,
    *,
    event_id: str,
    span_id: str,
    usage_policy: str,
    declared_capacity: Any,
) -> float:
    if usage_policy == "consumable":
        return 1.0
    declared = _number(declared_capacity, 0.0)
    if declared > 0.0:
        return declared
    for event in _graph_events(graph):
        current_event_id = str(event.get("event_id") or event.get("id") or "").strip()
        if current_event_id != event_id:
            continue
        for span in _dict_list(event.get("spans")):
            current_span_id = str(span.get("span_id") or span.get("id") or "").strip()
            if current_span_id != span_id:
                continue
            return _number(span.get("capacity", span.get("available_capacity")), 0.0)
    return 0.0


def _graph_events(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(graph, dict):
        return []
    events = _dict_list(graph.get("events"))
    if events:
        return events
    workspace = graph.get("workspace")
    if isinstance(workspace, dict):
        events = _dict_list(workspace.get("events"))
        if events:
            return events
        return [workspace]
    return [graph]


def _match_quality(item: dict[str, Any]) -> float:
    outcome = item.get("match") if isinstance(item.get("match"), dict) else {}
    metadata = outcome.get("metadata") if isinstance(outcome.get("metadata"), dict) else {}
    return _number(metadata.get("bundle_score", item.get("rank_score")), 0.0)


def _rule(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("rule") if isinstance(item.get("rule"), dict) else {}


def _rule_id(item: dict[str, Any]) -> str:
    return str(_rule(item).get("rule_id") or "").strip()


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _strings(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item) for item in values if str(item)]


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
