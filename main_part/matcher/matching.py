"""Matcher exact matching.

This module assumes routing already narrowed the candidate rule pool.
Its job is only to answer whether required clauses actually match the graph.
"""

from __future__ import annotations

from typing import Any

from ..data_structure import WorkingMemoryState
from ..semantic_subgraph import normalize_semantic_relations
from ..structured_arithmetic import StructuredComputationError, evaluate_structured_predicate
from ..vector_contracts import rescale_cosine_strict
from .routing import (
    as_dict_list,
    as_number_list,
    graph_from_required_bundle,
    normalize_required_pattern,
    required_bundle_from_rule,
)


def match_resolved_endpoints(
    source: dict[str, Any],
    target: dict[str, Any],
    args: dict[str, Any],
    state: Any,
    working_memory: WorkingMemoryState,
) -> dict[str, Any]:
    source_kind = source["kind"]
    target_kind = target["kind"]
    if source_kind == "required" and target_kind == "graph":
        return match_required_bundle_against_graph(source, target, args)
    if source_kind == "graph" and target_kind == "rules":
        return match_rules_against_graph(
            target["rules"],
            source,
            args,
            target_spec=target["spec"],
            routing_metadata=target.get("routing_metadata") or {},
            state=state,
        )
    if source_kind == "required" and target_kind == "rules":
        graph = graph_from_required_bundle(source)
        return match_rules_against_graph(
            target["rules"],
            graph,
            args,
            target_spec=target["spec"],
            routing_metadata=target.get("routing_metadata") or {},
            state=state,
        )
    return {
        "ok": False,
        "produced": [
            {
                "operation": "match",
                "source": source["spec"],
                "target": target["spec"],
                "error": f"unsupported source->target: {source_kind}->{target_kind}",
                "allowed_paths": [
                    "rule->working_memory",
                    "working_memory->rule_db",
                    "rule->rule",
                ],
            }
        ],
        "metadata": {"source_kind": source_kind, "target_kind": target_kind},
    }


def match_required_bundle_against_graph(
    source: dict[str, Any],
    target: dict[str, Any],
    args: dict[str, Any],
) -> dict[str, Any]:
    required_items = source["required"]
    rule_id = str(source.get("rule_id") or "")
    rule_layer = str(source.get("rule_layer") or "").strip().lower()
    if not required_items:
        ok = rule_layer in {"control", "matcher_rule"}
        missing = [] if ok else [{"reason": "empty_required_non_control"}]
        return {
            "ok": ok,
            "produced": [
                {
                    "operation": "match",
                    "match_type": "rule_to_working_memory",
                    "rule_id": rule_id,
                    "source": source["spec"],
                    "target": target["spec"],
                    "matched_required": [] if ok else None,
                    "missing_required": missing,
                }
            ],
            "metadata": {
                "rule_id": rule_id,
                "rule_layer": rule_layer,
                "source": source["spec"],
                "target": target["spec"],
                "match_policy": "empty_required_allowed_for_control_and_matcher_rule",
            },
        }

    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    per_required: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for index, required in enumerate(required_items):
        required = {**required, "_target_rule_id": rule_id}
        outcome = match_required_item(required, target["workspaces"], rule_layer=rule_layer)
        outcome["required_index"] = index
        priority = required_priority(required)
        outcome["condition_priority"] = priority
        per_required.append(outcome)
        weighted_sum += priority * float(outcome.get("score") or 0.0)
        weight_total += priority
        if outcome["matched"]:
            matched.append(outcome)
        else:
            missing.append(outcome)

    bundle_score = weighted_sum / weight_total if weight_total else 0.0
    bundle_threshold = safe_float(args.get("bundle_match_threshold"), safe_float(args.get("match_threshold"), 0.55))
    ok = not missing and bundle_score >= bundle_threshold
    return {
        "ok": ok,
        "produced": [
            {
                "operation": "match",
                "match_type": "rule_to_working_memory",
                "rule_id": rule_id,
                "source": source["spec"],
                "target": target["spec"],
                "matched_required": matched,
                "missing_required": missing,
                "bundle_score": bundle_score,
                "bundle_threshold": bundle_threshold,
            }
        ],
        "metadata": {
            "rule_id": rule_id,
            "rule_layer": rule_layer,
            "source": source["spec"],
            "target": target["spec"],
            "match_policy": "priority_weighted_required_span_graph",
            "per_required": per_required,
            "bundle_score": bundle_score,
            "bundle_threshold": bundle_threshold,
        },
    }


def match_rules_against_graph(
    rules: list[dict[str, Any]],
    graph: dict[str, Any],
    args: dict[str, Any],
    *,
    target_spec: dict[str, Any],
    routing_metadata: dict[str, Any],
    state: Any = None,
) -> dict[str, Any]:
    candidate_rules = list(rules)
    matched_rules: list[dict[str, Any]] = []
    rejected_rules: list[dict[str, Any]] = []
    post_match_enabled = bool(args.get("enable_post_match", True))
    post_match_rules: list[dict[str, Any]] = []
    if post_match_enabled:
        from .post_match import load_post_match_rules

        post_match_rules = load_post_match_rules(
            args,
            state,
            limit=max(32, int(args.get("matcher_rule_k") or 64)),
        )
    for rule in candidate_rules:
        source = required_bundle_from_rule(rule, {"type": "rule", "rule_id": rule.get("rule_id", ""), "field": "required", "location": target_spec})
        outcome = match_required_bundle_against_graph(source, graph, args)
        item = {
            "rule": rule,
            "match": outcome,
            "rank_score": float(rule.get("_candidate_score", 0.0) or 0.0),
            "routing_score": float(rule.get("_routing_score", 0.0) or 0.0),
            "match_mode": "hard_match",
        }
        if outcome.get("ok"):
            matched_rules.append(item)
        else:
            revised_rule = None
            matcher_rule_ids: list[str] = []
            semantic_gap = any(
                slot.get("alignment_candidates") and not slot.get("matched")
                for requirement in outcome.get("metadata", {}).get("per_required", [])
                for slot in requirement.get("details", {}).get("slot_matches", [])
            )
            if post_match_enabled and not semantic_gap:
                from .post_match import propose_temporary_rule_revision

                revised_rule, matcher_rule_ids = propose_temporary_rule_revision(
                    rule,
                    outcome,
                    args=args,
                    state=state,
                    matcher_rules=post_match_rules,
                )
            if revised_rule is not None:
                revised_source = required_bundle_from_rule(
                    revised_rule,
                    {
                        "type": "rule",
                        "rule_id": revised_rule.get("rule_id", ""),
                        "field": "required",
                        "location": target_spec,
                    },
                )
                retried_outcome = match_required_bundle_against_graph(revised_source, graph, args)
                retried_item = {
                    **item,
                    "rule": revised_rule,
                    "match": retried_outcome,
                    "match_mode": "hard_match_after_temporary_require_revision",
                    "matcher_rule_ids": matcher_rule_ids,
                    "original_rule": rule,
                    "original_hard_match": outcome,
                }
                if retried_outcome.get("ok"):
                    matched_rules.append(retried_item)
                    continue
                item["temporary_revision_attempt"] = retried_item
            rejected_rules.append(item)
    metadata = {
        "source": graph["spec"],
        "target": target_spec,
        "candidate_count": len(rules),
        "matched_count": len(matched_rules),
        "selected_count": len(matched_rules),
        "candidate_policy": "control_selected_candidates_then_hard_required_match",
        "routing": dict(routing_metadata or {}),
    }
    # Keep rejected requirements available to Control, without admitting them.
    metadata["alignment_requirements"] = [request for item in rejected_rules
        for request in alignment_requirements_from_metadata(item["match"].get("metadata", {}))]
    from ..rule_path_candidates import attach_candidate_rule_paths

    operation = attach_candidate_rule_paths(
        {
            "operation": "match",
            "match_type": rules_graph_match_type(graph["spec"], target_spec),
            "source": graph["spec"],
            "target": target_spec,
            "matched_rules": matched_rules,
            "rejected_rules": rejected_rules,
        },
        graph=graph,
    )
    if operation.get("requires_control_selection"):
        metadata["selected_count"] = 0
        metadata["candidate_path_count"] = len(operation.get("candidate_rule_paths") or [])
    return {
        "ok": bool(matched_rules),
        "produced": [operation],
        "metadata": metadata,
    }

def required_priority(required: dict[str, Any]) -> float:
    priority = safe_float(required.get("condition_priority", required.get("priority", 1.0)), 1.0)
    return max(0.0, min(priority, 1.0))


def rules_graph_match_type(source_spec: dict[str, Any], target_spec: dict[str, Any]) -> str:
    if str(source_spec.get("type") or "") == "required_as_graph":
        return "rule_to_rule"
    if str(target_spec.get("type") or "") in {"rule_db", "rules", "rule_records", "rule"}:
        return "working_memory_to_rule_db"
    return "match"

def match_required_item(
    required: dict[str, Any],
    workspaces: list[dict[str, Any]],
    *,
    rule_layer: str,
) -> dict[str, Any]:
    structure_errors = validate_required_relation_endpoints(required)
    if structure_errors:
        return {
            "matched": False,
            "score": 0.0,
            "reason": "invalid_required_relation_endpoints",
            "required": required,
            "event_id": "",
            "details": {"structure_errors": structure_errors},
        }
    pattern = normalize_required_pattern(required)
    for slot in pattern.get("span_slots", []):
        slot["_target_rule_id"] = required.get("_target_rule_id", "")
    threshold = safe_float(required.get("match_threshold"), 0.55)
    if empty_match_pattern(pattern):
        control_like = rule_layer in {"control", "matcher_rule"}
        return {
            "matched": control_like,
            "score": 1.0 if control_like else 0.0,
            "reason": "empty_control_required" if control_like else "empty_required_pattern",
            "required": required,
            "event_id": "",
        }

    best: dict[str, Any] = {
        "matched": False,
        "score": 0.0,
        "reason": "no_event_workspace_match",
        "required": required,
        "event_id": "",
    }
    for candidate in workspaces:
        from ..alignment_contract import alignment_is_current
        from ..control_state import control_state_span_is_active
        span_index = {
            sid: span if alignment_is_current(span, candidate["span_index"], candidate["relations"])
            and control_state_span_is_active(span)
            else {
                **span,
                "_invalid_alignment": True,
                "_inactive_control_state": not control_state_span_is_active(span),
            }
            for sid, span in candidate["span_index"].items()
        }
        for candidate_pattern in candidate["patterns"] or [{}]:
            outcome = score_pattern_match(
                pattern,
                candidate_pattern,
                span_index,
                candidate["relations"],
                threshold,
            )
            if outcome["score"] > best["score"] or "details" not in best:
                best = {
                    "matched": outcome["matched"],
                    "score": outcome["score"],
                    "reason": outcome["reason"],
                    "required": required,
                    "event_id": candidate["event_id"],
                    "candidate_pattern": candidate_pattern,
                    "details": outcome,
                }
    return best


def validate_required_relation_endpoints(required: dict[str, Any]) -> list[dict[str, Any]]:
    condition_span_ids = {
        str(span.get("span_id") or span.get("id") or "").strip()
        for span in as_dict_list(required.get("condition"))
        if str(span.get("span_id") or span.get("id") or "").strip()
    }
    errors: list[dict[str, Any]] = []
    for index, relation in enumerate(as_dict_list(required.get("relation_patterns"))):
        source_span_id = str(relation.get("source_span_id") or relation.get("source_slot") or "").strip()
        target_span_id = str(relation.get("target_span_id") or relation.get("target_slot") or "").strip()
        invalid_endpoints = [
            endpoint
            for endpoint in (source_span_id, target_span_id)
            if not endpoint or endpoint not in condition_span_ids
        ]
        if invalid_endpoints:
            errors.append(
                {
                    "relation_index": index,
                    "relation_id": str(relation.get("relation_id") or relation.get("id") or ""),
                    "source_span_id": source_span_id,
                    "target_span_id": target_span_id,
                    "invalid_endpoints": invalid_endpoints,
                    "condition_span_ids": sorted(condition_span_ids),
                }
            )
    return errors


def empty_match_pattern(pattern: dict[str, Any]) -> bool:
    return not (pattern.get("span_slots") or pattern.get("relation_patterns"))


def score_pattern_match(
    required_pattern: dict[str, Any],
    candidate_pattern: dict[str, Any],
    span_index: dict[str, dict[str, Any]],
    relations: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    span_result = bind_span_slots(
        as_dict_list(required_pattern.get("span_slots")),
        as_dict_list(candidate_pattern.get("span_slots")),
        span_index,
        threshold,
    )
    relation_score, relation_ok, relation_reason, relation_details = score_relation_patterns(
        as_dict_list(required_pattern.get("relation_patterns")),
        as_dict_list(candidate_pattern.get("relation_patterns")),
        relations,
        span_result["bindings"],
    )
    required_relations = as_dict_list(required_pattern.get("relation_patterns"))
    if required_relations and not relation_ok:
        required_slots = as_dict_list(required_pattern.get("span_slots"))
        candidate_relations = [
            *as_dict_list(candidate_pattern.get("relation_patterns")),
            *relations,
        ]
        relation_slot_ids = {
            str(value).strip()
            for relation in required_relations
            for value in (relation.get("source_span_id"), relation.get("target_span_id"))
            if str(value).strip()
        }
        endpoint_preferences = relation_endpoint_binding_preferences(
            required_slots,
            required_relations,
            candidate_relations,
            span_index,
            threshold,
        )
        structural_span_result = bind_span_slots(
            required_slots,
            as_dict_list(candidate_pattern.get("span_slots")),
            span_index,
            threshold,
            distinct_slot_ids=relation_slot_ids,
            preferred_candidate_ids=endpoint_preferences,
        )
        structural_relation_score, structural_relation_ok, structural_relation_reason, structural_relation_details = score_relation_patterns(
            required_relations,
            as_dict_list(candidate_pattern.get("relation_patterns")),
            relations,
            structural_span_result["bindings"],
        )
        if structural_relation_ok or structural_relation_score > relation_score:
            span_result = structural_span_result
            relation_score = structural_relation_score
            relation_ok = structural_relation_ok
            relation_reason = structural_relation_reason
            relation_details = structural_relation_details
    semantic_constraint_details, semantic_constraints_ok = score_shared_semantic_relation_constraints(
        as_dict_list(required_pattern.get("shared_semantic_relation_constraints")),
        span_result["bindings"],
        span_index,
    )
    predicate_details: list[dict[str, Any]] = []
    predicate_ok = True
    for predicate in as_dict_list(required_pattern.get("predicates")):
        try:
            predicate_result = evaluate_structured_predicate(
                predicate,
                spans=list(span_index.values()),
                slot_bindings=span_result["bindings"],
            )
        except StructuredComputationError as exc:
            predicate_result = {
                "status": "error",
                "matched": False,
                "error_code": exc.code,
                "details": exc.details,
            }
        predicate_details.append(predicate_result)
        predicate_ok = predicate_ok and bool(predicate_result.get("matched"))
    scores = []
    if required_pattern.get("span_slots"):
        scores.append(span_result["score"])
    if required_pattern.get("relation_patterns"):
        scores.append(relation_score)
    score = sum(scores) / len(scores) if scores else 0.0
    matched = (
        span_result["ok"]
        and relation_ok
        and semantic_constraints_ok
        and predicate_ok
        and score >= threshold
    )
    reasons = [span_result["reason"], relation_reason]
    if not semantic_constraints_ok:
        reasons.append("shared_semantic_relation_constraint_not_satisfied")
    if not predicate_ok:
        reasons.append("structured_predicate_not_satisfied")
    return {
        "matched": matched,
        "score": score,
        "span_score": span_result["score"],
        "relation_score": relation_score,
        "reason": "matched" if matched else ";".join(item for item in reasons if item).strip(";"),
        "slot_bindings": span_result["bindings"],
        "slot_matches": span_result["slot_matches"],
        "reuse_count": span_result["reuse_count"],
        "reuse_penalty": span_result["reuse_penalty"],
        "vector_contract_errors": span_result.get("vector_contract_errors", []),
        "relation_matches": relation_details,
        "shared_semantic_relation_constraint_matches": semantic_constraint_details,
        "predicate_matches": predicate_details,
    }


def score_shared_semantic_relation_constraints(
    constraints: list[dict[str, Any]],
    slot_bindings: dict[str, str],
    span_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Prove that selected spans share an explicitly declared semantic context.

    The Rule identifies the semantic relation with vectors.  The Event Graph
    supplies each span's vectorized core-relation-nature triples.  Debug labels
    never decide this check.
    """
    details: list[dict[str, Any]] = []
    all_ok = True
    for index, constraint in enumerate(constraints):
        slots = [str(slot_id) for slot_id in constraint.get("slot_ids", []) if str(slot_id)]
        threshold = safe_float(constraint.get("match_threshold"), 0.995)
        required_core = as_number_list(constraint.get("core_vector"))
        required_relation = as_number_list(constraint.get("relation_vector"))
        if len(slots) < 2 or not required_core or not required_relation:
            details.append({
                "constraint_index": index,
                "matched": False,
                "status": "invalid_shared_semantic_relation_constraint",
            })
            all_ok = False
            continue
        candidates_by_slot: dict[str, list[dict[str, Any]]] = {}
        for slot_id in slots:
            span_id = str(slot_bindings.get(slot_id) or "")
            span = span_index.get(span_id)
            if not span:
                candidates_by_slot[slot_id] = []
                continue
            try:
                relations = normalize_semantic_relations(span)
            except ValueError:
                relations = []
            candidates: list[dict[str, Any]] = []
            for relation in relations:
                core_match = rescale_cosine_strict(required_core, relation.get("core_vector"))
                relation_match = rescale_cosine_strict(required_relation, relation.get("relation_vector"))
                if (
                    core_match.get("ok")
                    and relation_match.get("ok")
                    and float(core_match["score"]) >= threshold
                    and float(relation_match["score"]) >= threshold
                ):
                    candidates.append(relation)
            candidates_by_slot[slot_id] = candidates
        first_slot = slots[0]
        shared: dict[str, dict[str, Any]] | None = None
        for reference in candidates_by_slot.get(first_slot, []):
            selection = {first_slot: reference}
            compatible = True
            for slot_id in slots[1:]:
                matched_relation = next(
                    (
                        relation
                        for relation in candidates_by_slot.get(slot_id, [])
                        if relation.get("embedding_model") == reference.get("embedding_model")
                        and relation.get("vector_version") == reference.get("vector_version")
                        and all(
                            result.get("ok") and float(result["score"]) >= threshold
                            for result in (
                                rescale_cosine_strict(reference.get("core_vector"), relation.get("core_vector")),
                                rescale_cosine_strict(reference.get("nature_vector"), relation.get("nature_vector")),
                            )
                        )
                    ),
                    None,
                )
                if matched_relation is None:
                    compatible = False
                    break
                selection[slot_id] = matched_relation
            if compatible:
                shared = selection
                break
        matched = shared is not None
        details.append({
            "constraint_index": index,
            "slot_ids": slots,
            "matched": matched,
            "status": "matched" if matched else "shared_semantic_relation_not_proven",
            "candidate_counts": {slot_id: len(candidates_by_slot.get(slot_id, [])) for slot_id in slots},
            "semantic_node_ids": {
                slot_id: {
                    "core_node_id": relation.get("core_node_id", ""),
                    "relation_node_id": relation.get("relation_node_id", ""),
                    "nature_node_id": relation.get("nature_node_id", ""),
                }
                for slot_id, relation in (shared or {}).items()
            },
        })
        all_ok = all_ok and matched
    return details, all_ok


def score_span_slots(
    required_slots: list[dict[str, Any]],
    candidate_slots: list[dict[str, Any]],
    span_index: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[float, bool, str]:
    result = bind_span_slots(required_slots, candidate_slots, span_index, threshold)
    return result["score"], result["ok"], result["reason"]


def bind_span_slots(
    required_slots: list[dict[str, Any]],
    candidate_slots: list[dict[str, Any]],
    span_index: dict[str, dict[str, Any]],
    threshold: float,
    *,
    distinct_slot_ids: set[str] | None = None,
    preferred_candidate_ids: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    if not required_slots:
        return {
            "score": 1.0,
            "ok": True,
            "reason": "",
            "bindings": {},
            "slot_matches": [],
            "reuse_count": 0,
            "reuse_penalty": 0.0,
            "vector_contract_errors": [],
        }
    if not candidate_slots and not span_index:
        return {
            "score": 0.0,
            "ok": False,
            "reason": "no_candidate_spans",
            "bindings": {},
            "slot_matches": [],
            "reuse_count": 0,
            "reuse_penalty": 0.0,
            "vector_contract_errors": [],
        }

    weighted_sum = 0.0
    weight_total = 0.0
    missing: list[str] = []
    candidates_by_id: dict[str, dict[str, Any]] = {}
    anonymous_index = 0
    for candidate in candidate_slots:
        candidate_span_id = str(candidate.get("span_id") or "")
        candidate_span = span_index.get(candidate_span_id) or candidate.get("span") or candidate
        if not candidate_span_id:
            candidate_span_id = _span_id(candidate_span)
        if not candidate_span_id:
            anonymous_index += 1
            candidate_span_id = f"anonymous_candidate_{anonymous_index}"
        candidates_by_id[candidate_span_id] = {"span_id": candidate_span_id, "span": candidate_span}
    for span in span_index.values():
        candidate_span_id = _span_id(span)
        if candidate_span_id:
            candidates_by_id[candidate_span_id] = {"span_id": candidate_span_id, "span": span}
    candidates = list(candidates_by_id.values())
    bindings: dict[str, str] = {}
    slot_matches: list[dict[str, Any]] = []
    vector_contract_errors: list[dict[str, Any]] = []
    required_by_slot: dict[str, dict[str, Any]] = {}
    candidate_span_by_id: dict[str, dict[str, Any]] = {
        str(candidate.get("span_id") or ""): (
            span_index.get(str(candidate.get("span_id") or ""))
            or candidate.get("span")
            or candidate
        )
        for candidate in candidates
        if str(candidate.get("span_id") or "")
    }
    usage_count: dict[str, int] = {}
    distinct_ids = {str(value) for value in (distinct_slot_ids or set()) if str(value)}
    preferred_ids = {
        str(slot_id): {str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)}
        for slot_id, candidate_ids in (preferred_candidate_ids or {}).items()
        if str(slot_id)
    }
    slot_evaluations: list[dict[str, Any]] = []
    for slot_index, required in enumerate(required_slots):
        slot_id = str(required.get("slot_id") or required.get("span_id") or "span_slot")
        requires_distinct_binding = (
            slot_id in distinct_ids
            or not bool(required.get("allow_shared_binding", False))
        )
        local_threshold = safe_float(required.get("match_threshold"), threshold)
        priority = max(0.0, min(safe_float(required.get("condition_priority"), 1.0), 1.0))
        missing_effect = str(required.get("missing_effect") or "block_output").strip().lower()
        required_by_slot[slot_id] = required
        candidate_matches = [
            best_span_slot_match(required, [candidate], span_index)
            for candidate in candidates
        ]
        candidate_matches.sort(
            key=lambda item: (
                str(item.get("candidate_span_id") or "") in preferred_ids.get(slot_id, set()),
                float(item.get("evidence_quality") or 0.0),
                float(item.get("score") or 0.0),
                str(item.get("candidate_span_id") or ""),
            ),
            reverse=True,
        )
        slot_evaluations.append(
            {
                "index": slot_index,
                "slot_id": slot_id,
                "requires_distinct_binding": requires_distinct_binding,
                "threshold": local_threshold,
                "priority": priority,
                "missing_effect": missing_effect,
                "candidate_matches": candidate_matches,
                "eligible_matches": [
                    item
                    for item in candidate_matches
                    if float(item.get("score") or 0.0) >= local_threshold
                    and str(item.get("candidate_span_id") or "")
                ],
            }
        )

    assigned_candidate_to_slot: dict[str, int] = {}
    assigned_match_by_slot: dict[int, dict[str, Any]] = {}

    def assign_distinct_slot(slot_index: int, seen_candidate_ids: set[str]) -> bool:
        evaluation = slot_evaluations[slot_index]
        eligible_matches = list(evaluation["eligible_matches"])
        eligible_matches.sort(
            key=lambda item: str(item.get("candidate_span_id") or "") in assigned_candidate_to_slot
        )
        for candidate_match in eligible_matches:
            candidate_span_id = str(candidate_match.get("candidate_span_id") or "")
            if candidate_span_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_span_id)
            incumbent = assigned_candidate_to_slot.get(candidate_span_id)
            if incumbent is not None and not assign_distinct_slot(incumbent, seen_candidate_ids):
                continue
            assigned_candidate_to_slot[candidate_span_id] = slot_index
            assigned_match_by_slot[slot_index] = candidate_match
            return True
        return False

    distinct_evaluations = [
        item for item in slot_evaluations if item["requires_distinct_binding"]
    ]
    distinct_evaluations.sort(
        key=lambda item: (
            len(item["eligible_matches"]),
            -float(item["priority"]),
            int(item["index"]),
        )
    )
    for evaluation in distinct_evaluations:
        assign_distinct_slot(int(evaluation["index"]), set())

    reserved_candidate_ids = set(assigned_candidate_to_slot)
    for evaluation in slot_evaluations:
        if evaluation["requires_distinct_binding"]:
            continue
        available_matches = [
            item
            for item in evaluation["eligible_matches"]
            if str(item.get("candidate_span_id") or "") not in reserved_candidate_ids
        ]
        if available_matches:
            assigned_match_by_slot[int(evaluation["index"])] = available_matches[0]

    for evaluation in slot_evaluations:
        slot_index = int(evaluation["index"])
        slot_id = str(evaluation["slot_id"])
        assigned_match = assigned_match_by_slot.get(slot_index)
        best_match = assigned_match or next(iter(evaluation["candidate_matches"]), {
            "score": 0.0,
            "candidate_span_id": "",
            "field_scores": {},
            "vector_status": "missing_candidate_vector",
            "vector_error": {},
        })
        best = float(best_match.get("score") or 0.0)
        local_threshold = float(evaluation["threshold"])
        priority = float(evaluation["priority"])
        missing_effect = str(evaluation["missing_effect"])
        candidate_span_id = str(best_match.get("candidate_span_id") or "")
        weighted_sum += priority * best
        weight_total += priority
        slot_matched = assigned_match is not None
        if slot_matched and candidate_span_id:
            bindings[slot_id] = candidate_span_id
            usage_count[candidate_span_id] = usage_count.get(candidate_span_id, 0) + 1
        if not slot_matched:
            missing.append(slot_id)
        if best_match.get("vector_status") not in {"ok", "empty_required_slot"}:
            vector_contract_errors.append(
                {
                    "slot_id": slot_id,
                    "candidate_span_id": candidate_span_id,
                    **dict(best_match.get("vector_error") or {}),
                }
            )
        slot_matches.append(
            {
                "slot_id": slot_id,
                "candidate_span_id": candidate_span_id,
                "score": best,
                "match_threshold": local_threshold,
                "condition_priority": priority,
                "missing_effect": missing_effect,
                "matched": slot_matched,
                "field_scores": best_match.get("field_scores") or {},
                "evidence_quality": float(best_match.get("evidence_quality") or 0.0),
                "vector_status": best_match.get("vector_status") or "",
                "vector_error": best_match.get("vector_error") or {},
                "alignment_required": required_by_slot[slot_id] if not slot_matched else {},
                "alignment_candidates": [
                    {"span_id": match["candidate_span_id"],
                     "core_score": structured_slot_match(
                         {k: v for k, v in required_by_slot[slot_id].items() if k in {"core", "core_vector"}},
                         candidate_span_by_id.get(match["candidate_span_id"], {}))["score"]}
                    for match in evaluation["candidate_matches"]
                    if match.get("vector_status") in {"feature_type_mismatch", "ok"}
                ] if not slot_matched else [],
            }
        )
    reuse_conflicts = repeated_binding_semantic_conflicts(
        bindings,
        required_by_slot,
        candidate_span_by_id,
    )
    for conflict in reuse_conflicts:
        slot_id = str(conflict.get("slot_id") or "")
        if slot_id and slot_id not in missing:
            missing.append(slot_id)
        for slot_match in slot_matches:
            if str(slot_match.get("slot_id") or "") == slot_id:
                slot_match["matched"] = False
                slot_match["reuse_semantic_conflict"] = conflict
    score = weighted_sum / weight_total if weight_total else 1.0
    reuse_count = sum(max(0, count - 1) for count in usage_count.values())
    reuse_penalty = 0.0
    return {
        "score": score,
        "ok": not missing,
        "reason": "" if not missing else f"missing_span_slots={missing}",
        "bindings": bindings,
        "slot_matches": slot_matches,
        "reuse_count": reuse_count,
        "reuse_penalty": reuse_penalty,
        "reuse_conflicts": reuse_conflicts,
        "vector_contract_errors": vector_contract_errors,
    }


def relation_endpoint_binding_preferences(
    required_slots: list[dict[str, Any]],
    required_relations: list[dict[str, Any]],
    candidate_relations: list[dict[str, Any]],
    span_index: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, set[str]]:
    """Prefer semantically valid relation endpoints when span scores are tied."""
    slots_by_id = {
        str(slot.get("slot_id") or slot.get("span_id") or ""): slot
        for slot in required_slots
        if str(slot.get("slot_id") or slot.get("span_id") or "")
    }
    preferences: dict[str, set[str]] = {}
    for required_relation in required_relations:
        required_vector = as_number_list(required_relation.get("relation_vector"))
        if not required_vector:
            continue
        relation_threshold = safe_float(required_relation.get("match_threshold"), threshold)
        source_slot_id = str(
            required_relation.get("source_span_id")
            or required_relation.get("source_slot")
            or ""
        )
        target_slot_id = str(
            required_relation.get("target_span_id")
            or required_relation.get("target_slot")
            or ""
        )
        source_required = slots_by_id.get(source_slot_id)
        target_required = slots_by_id.get(target_slot_id)
        if not source_required or not target_required:
            continue
        for candidate_relation in candidate_relations:
            candidate_vector = as_number_list(candidate_relation.get("relation_vector"))
            if not candidate_vector:
                continue
            relation_result = rescale_cosine_strict(required_vector, candidate_vector)
            if not relation_result["ok"] or float(relation_result["score"]) < relation_threshold:
                continue
            candidate_source_id = str(
                candidate_relation.get("source_span_id")
                or candidate_relation.get("source_slot")
                or ""
            )
            candidate_target_id = str(
                candidate_relation.get("target_span_id")
                or candidate_relation.get("target_slot")
                or ""
            )
            candidate_source = span_index.get(candidate_source_id)
            candidate_target = span_index.get(candidate_target_id)
            if not candidate_source or not candidate_target:
                continue
            source_match = structured_slot_match(source_required, candidate_source)
            target_match = structured_slot_match(target_required, candidate_target)
            source_threshold = safe_float(source_required.get("match_threshold"), threshold)
            target_threshold = safe_float(target_required.get("match_threshold"), threshold)
            if (
                float(source_match.get("score") or 0.0) < source_threshold
                or float(target_match.get("score") or 0.0) < target_threshold
            ):
                continue
            preferences.setdefault(source_slot_id, set()).add(candidate_source_id)
            preferences.setdefault(target_slot_id, set()).add(candidate_target_id)
    return preferences


def repeated_binding_semantic_conflicts(
    bindings: dict[str, str],
    required_by_slot: dict[str, dict[str, Any]],
    candidate_span_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    slots_by_candidate: dict[str, list[str]] = {}
    for slot_id, candidate_span_id in bindings.items():
        proof = candidate_span_by_id.get(candidate_span_id, {}).get("metadata", {}).get("semantic_alignment", {})
        identity = proof.get("source_span_id", candidate_span_id)
        slots_by_candidate.setdefault(str(identity), []).append(str(slot_id))

    conflicts: list[dict[str, Any]] = []
    for candidate_span_id, slot_ids in slots_by_candidate.items():
        if len(slot_ids) < 2:
            continue
        protected_slots = [
            slot_id
            for slot_id in slot_ids
            if not bool((required_by_slot.get(slot_id) or {}).get("allow_shared_binding", False))
        ]
        if not protected_slots:
            continue
        for slot_id in slot_ids[1:]:
            conflicts.append(
                {
                    "slot_id": slot_id,
                    "candidate_span_id": candidate_span_id,
                    "conflicting_slot_ids": protected_slots,
                    "candidate_exists": candidate_span_id in candidate_span_by_id,
                    "reason": "required_slots_must_bind_distinct_spans",
                }
            )
    return conflicts


def best_span_slot_score(
    required: dict[str, Any],
    candidates: list[dict[str, Any]],
    span_index: dict[str, dict[str, Any]],
) -> float:
    return float(best_span_slot_match(required, candidates, span_index)["score"])


def best_span_slot_match(
    required: dict[str, Any],
    candidates: list[dict[str, Any]],
    span_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    best = 0.0
    best_candidate_span_id = ""
    best_field_scores: dict[str, float] = {}
    best_evidence_quality = 0.0
    best_vector_status = "missing_candidate_vector"
    best_vector_error: dict[str, Any] = {}
    for candidate in candidates:
        candidate_span_id = str(candidate.get("span_id") or "")
        candidate_span = span_index.get(candidate_span_id) or candidate.get("span") or candidate
        from ..alignment_contract import alignment_is_current
        from ..control_state import control_state_span_is_active
        if (
            not alignment_is_current(candidate_span, span_index)
            or not control_state_span_is_active(candidate_span)
        ):
            continue
        result = structured_slot_match(required, candidate_span)
        score = float(result["score"])
        evidence_quality = _span_evidence_quality(candidate_span)
        field_scores = dict(result.get("field_scores") or {})
        if (score, evidence_quality) > (best, best_evidence_quality):
            best = score
            best_evidence_quality = evidence_quality
            best_candidate_span_id = candidate_span_id or _span_id(candidate_span)
            best_field_scores = field_scores
            best_vector_status = str(result.get("vector_status") or "")
            best_vector_error = dict(result.get("vector_error") or {})
        elif not best_candidate_span_id and result.get("vector_status") != "ok":
            best_candidate_span_id = candidate_span_id or _span_id(candidate_span)
            best_vector_status = str(result.get("vector_status") or "")
            best_vector_error = dict(result.get("vector_error") or {})
    return {
        "score": best,
        "candidate_span_id": best_candidate_span_id,
        "field_scores": best_field_scores,
        "evidence_quality": best_evidence_quality,
        "vector_status": best_vector_status,
        "vector_error": best_vector_error,
    }


def _span_evidence_quality(span: dict[str, Any]) -> float:
    """Rank structurally eligible evidence without inspecting debug text."""

    metadata = span.get("metadata") if isinstance(span.get("metadata"), dict) else {}
    confidence = max(0.0, min(safe_float(span.get("confidence"), 0.5), 1.0))
    reliability = max(
        0.0,
        min(
            safe_float(
                metadata.get("source_reliability", span.get("source_reliability")),
                confidence,
            ),
            1.0,
        ),
    )
    authority = bool(metadata.get("authority_verified", span.get("authority_verified", False)))
    return 0.55 * reliability + 0.35 * confidence + 0.10 * float(authority)


def alignment_requirements_from_metadata(metadata):
    items = metadata.get("per_required", [])
    slots = [s for r in items for s in r.get("details", {}).get("slot_matches", [])]
    matched = sum(bool(s.get("matched")) for s in slots)
    return [{"rule_id": metadata.get("rule_id"), "event_id": r.get("event_id"), "slot": s,
             "matched_slot_count": matched, "total_slot_count": len(slots)}
            for r in items for s in r.get("details", {}).get("slot_matches", [])
            if not s.get("matched") and s.get("alignment_candidates")]


def structured_slot_match(
    required: dict[str, Any],
    candidate_span: dict[str, Any],
) -> dict[str, Any]:
    alignment = candidate_span.get("metadata", {}).get("semantic_alignment", {})
    if alignment and (alignment.get("target_rule_id") != required.get("_target_rule_id")
                      or alignment.get("target_slot_id") != required.get("slot_id", required.get("span_id"))):
        return {"score": 0.0, "field_scores": {}, "vector_status": "alignment_scope_mismatch", "vector_error": {}}
    required_feature_type = str(required.get("feature_type") or required.get("feature_schema_id") or "").strip()
    candidate_feature_type = str(
        candidate_span.get("feature_type") or candidate_span.get("feature_schema_id") or ""
    ).strip()
    if required_feature_type and not candidate_feature_type:
        return {
            "score": 0.0,
            "field_scores": {},
            "vector_status": "missing_candidate_feature_type",
            "vector_error": {"required_feature_type": required_feature_type},
        }
    if required_feature_type and candidate_feature_type != required_feature_type:
        return {
            "score": 0.0,
            "field_scores": {},
            "vector_status": "feature_type_mismatch",
            "vector_error": {
                "required_feature_type": required_feature_type,
                "candidate_feature_type": candidate_feature_type,
            },
        }
    # Evidence quality may rank otherwise valid candidates, but some Rules
    # explicitly require authoritative evidence.  Keep that condition in the
    # Rule contract rather than silently allowing a weak caveat to authorize a
    # path.  These fields are declarative constraints; they do not use debug
    # text or change semantic vector matching.
    metadata = candidate_span.get("metadata") if isinstance(candidate_span.get("metadata"), dict) else {}
    minimum_reliability = required.get("min_source_reliability")
    if minimum_reliability is not None:
        reliability = safe_float(
            metadata.get("source_reliability", candidate_span.get("source_reliability")),
            safe_float(candidate_span.get("confidence"), 0.5),
        )
        if reliability < safe_float(minimum_reliability, 0.0):
            return {
                "score": 0.0,
                "field_scores": {},
                "vector_status": "source_reliability_below_required_minimum",
                "vector_error": {
                    "minimum_source_reliability": safe_float(minimum_reliability, 0.0),
                    "source_reliability": reliability,
                },
            }
    if bool(required.get("require_authority_verified")) and not bool(
        metadata.get("authority_verified", candidate_span.get("authority_verified", False))
    ):
        return {
            "score": 0.0,
            "field_scores": {},
            "vector_status": "authority_verification_required",
            "vector_error": {"require_authority_verified": True},
        }
    field_specs = {
        "core": ("core_vector", ("core_vector", "span_core_vector")),
        "relation": ("relation_vector", ("relation_vector",)),
        "nature": ("nature_vector", ("nature_vector",)),
    }
    scores: dict[str, float] = {}
    for field, (required_vector_key, candidate_vector_keys) in field_specs.items():
        required_vector = as_number_list(required.get(required_vector_key))
        required_text = str(required.get(field) or "").strip()
        if not required_text and not required_vector:
            continue
        if not required_vector:
            return {
                "score": 0.0,
                "field_scores": scores,
                "vector_status": "missing_required_vector",
                "vector_error": {"field": field, "status": "missing_required_vector"},
            }
        candidate_vector: list[float] | None = None
        for key in candidate_vector_keys:
            candidate_vector = as_number_list(candidate_span.get(key))
            if candidate_vector:
                break
        if not candidate_vector:
            return {
                "score": 0.0,
                "field_scores": scores,
                "vector_status": "missing_candidate_vector",
                "vector_error": {"field": field, "status": "missing_candidate_vector"},
            }
        vector_result = rescale_cosine_strict(required_vector, candidate_vector)
        if not vector_result["ok"]:
            return {
                "score": 0.0,
                "field_scores": scores,
                "vector_status": "vector_contract_mismatch",
                "vector_error": {"field": field, **dict(vector_result)},
            }
        scores[field] = float(vector_result["score"])
    if not scores:
        return {
            "score": 0.0,
            "field_scores": {},
            "vector_status": "empty_required_slot",
            "vector_error": {"status": "empty_required_slot"},
        }
    return {
        "score": min(scores.values()),
        "field_scores": scores,
        "vector_status": "ok",
        "vector_error": {},
    }


def is_blocking_effect(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("block")


def field_vector_score(required: dict[str, Any], candidate_span: dict[str, Any]) -> float | None:
    pairs = {
        "core_vector": ("core_vector", "span_core_vector"),
        "relation_vector": ("relation_vector",),
        "nature_vector": ("nature_vector",),
    }
    scores: list[float] = []
    for required_key, candidate_keys in pairs.items():
        required_vector = as_number_list(required.get(required_key))
        if not required_vector:
            continue
        candidate_vector: list[float] | None = None
        for candidate_key in candidate_keys:
            candidate_vector = as_number_list(candidate_span.get(candidate_key))
            if candidate_vector:
                break
        if not candidate_vector:
            return 0.0
        result = rescale_cosine_strict(required_vector, candidate_vector)
        if not result["ok"]:
            return 0.0
        scores.append(float(result["score"]))
    if not scores:
        return None
    return min(scores)


def score_relation_patterns(
    required_patterns: list[dict[str, Any]],
    candidate_patterns: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    bindings: dict[str, str] | None = None,
) -> tuple[float, bool, str, list[dict[str, Any]]]:
    if not required_patterns:
        return 1.0, True, "", []
    candidates = [*candidate_patterns, *relations]
    candidate_tags = {relation_tag(pattern) for pattern in candidates if relation_tag(pattern)}
    if not candidate_tags and not candidates:
        return 0.0, False, "no_candidate_relations", []
    scores: list[float] = []
    missing: list[str] = []
    details: list[dict[str, Any]] = []
    for required in required_patterns:
        tag = relation_tag(required)
        best_match = best_relation_pattern_match(required, candidates, bindings or {})
        best = float(best_match["score"])
        threshold = safe_float(required.get("match_threshold"), 0.5)
        scores.append(best)
        if best < threshold:
            missing.append(tag or str(required.get("relation_id") or "relation_pattern"))
        details.append(
            {
                "required_relation": tag,
                "required_relation_id": str(required.get("relation_id") or required.get("id") or ""),
                "required_source_slot": str(required.get("source_span_id") or required.get("source_slot") or ""),
                "required_target_slot": str(required.get("target_span_id") or required.get("target_slot") or ""),
                "bound_source_span_id": best_match.get("bound_source_span_id") or "",
                "bound_target_span_id": best_match.get("bound_target_span_id") or "",
                "candidate_source_span_id": best_match.get("candidate_source_span_id") or "",
                "candidate_target_span_id": best_match.get("candidate_target_span_id") or "",
                "score": best,
                "match_threshold": threshold,
                "matched": best >= threshold,
            }
        )
    score = sum(scores) / len(scores) if scores else 1.0
    return score, not missing, "" if not missing else f"missing_relation_patterns={missing}", details


def best_relation_pattern_score(required: dict[str, Any], candidates: list[dict[str, Any]]) -> float:
    return float(best_relation_pattern_match(required, candidates, {})["score"])


def best_relation_pattern_match(
    required: dict[str, Any],
    candidates: list[dict[str, Any]],
    bindings: dict[str, str],
) -> dict[str, Any]:
    required_tag = relation_tag(required)
    required_vector = as_number_list(required.get("relation_vector"))
    source_slot = str(required.get("source_span_id") or required.get("source_slot") or "")
    target_slot = str(required.get("target_span_id") or required.get("target_slot") or "")
    bound_source = str(bindings.get(source_slot) or "")
    bound_target = str(bindings.get(target_slot) or "")
    best = 0.0
    best_candidate: dict[str, Any] = {}
    for candidate in candidates:
        candidate_source = str(candidate.get("source_span_id") or candidate.get("source_slot") or "")
        candidate_target = str(candidate.get("target_span_id") or candidate.get("target_slot") or "")
        if bound_source and candidate_source != bound_source:
            continue
        if bound_target and candidate_target != bound_target:
            continue
        candidate_tag = relation_tag(candidate)
        score = 0.0
        candidate_vector = as_number_list(candidate.get("relation_vector"))
        if required_vector and candidate_vector:
            vector_result = rescale_cosine_strict(required_vector, candidate_vector)
            if vector_result["ok"]:
                score = max(score, float(vector_result["score"]))
        if score > best:
            best = score
            best_candidate = candidate
    return {
        "score": best,
        "bound_source_span_id": bound_source,
        "bound_target_span_id": bound_target,
        "candidate_source_span_id": str(best_candidate.get("source_span_id") or best_candidate.get("source_slot") or ""),
        "candidate_target_span_id": str(best_candidate.get("target_span_id") or best_candidate.get("target_slot") or ""),
    }


def relation_tag(relation: dict[str, Any]) -> str:
    return str(
        relation.get("debug_relation")
        or relation.get("relation")
        or relation.get("relation_type")
        or relation.get("edge_type")
        or ""
    ).strip()


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


def first_vector(node: Any) -> list[float] | None:
    vectors = collect_vectors(node)
    return vectors[0] if vectors else None


def rescale_cosine(left: list[float], right: list[float]) -> float:
    return float(rescale_cosine_strict(left, right)["score"])


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    result = rescale_cosine_strict(left, right)
    if not result["ok"]:
        return None
    return 2.0 * float(result["score"]) - 1.0


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _span_id(span: dict[str, Any]) -> str:
    return str(span.get("span_id") or span.get("mention_id") or span.get("id") or "")
