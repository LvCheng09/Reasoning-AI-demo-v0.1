"""Execute matcher rules after a hard-match failure.

Post-match rules never perform a second kind of matching. They may produce a
temporary revision of a failed rule's ``required`` boundary, after which the
same hard matcher is called again.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .pre_match import is_matcher_rule, load_cached_matcher_rules


def build_match_failure_workspace(
    rule: dict[str, Any],
    hard_match: dict[str, Any],
) -> dict[str, Any]:
    metadata = hard_match.get("metadata") if isinstance(hard_match.get("metadata"), dict) else {}
    per_required = metadata.get("per_required") if isinstance(metadata.get("per_required"), list) else []
    spans: list[dict[str, Any]] = [
        {
            "span_id": "matcher_failure_rule",
            "core": str(rule.get("rule_id") or "candidate rule"),
            "relation": "failed_hard_match",
            "nature": "candidate rule did not satisfy its current required boundary",
            "text": str(rule.get("description") or ""),
        }
    ]
    relations: list[dict[str, Any]] = []
    for required_index, outcome in enumerate(per_required, start=1):
        details = outcome.get("details") if isinstance(outcome.get("details"), dict) else {}
        for slot_index, slot in enumerate(details.get("slot_matches") or [], start=1):
            span_id = f"matcher_failure_required_{required_index}_{slot_index}"
            spans.append(
                {
                    "span_id": span_id,
                    "core": str(slot.get("slot_id") or "required span"),
                    "relation": "matched" if slot.get("matched") else "missing_or_below_threshold",
                    "nature": (
                        f"score={slot.get('score', 0.0)} "
                        f"threshold={slot.get('match_threshold', 0.0)} "
                        f"candidate={slot.get('candidate_span_id', '')}"
                    ),
                }
            )
            relations.append(
                {
                    "source_span_id": span_id,
                    "relation": "diagnoses",
                    "target_span_id": "matcher_failure_rule",
                }
            )
        for relation_index, relation in enumerate(details.get("relation_matches") or [], start=1):
            span_id = f"matcher_failure_relation_{required_index}_{relation_index}"
            spans.append(
                {
                    "span_id": span_id,
                    "core": str(relation.get("required_relation") or "required relation"),
                    "relation": "matched" if relation.get("matched") else "wrong_or_missing_endpoint_relation",
                    "nature": (
                        f"score={relation.get('score', 0.0)} "
                        f"threshold={relation.get('match_threshold', 0.0)} "
                        f"source={relation.get('candidate_source_span_id', '')} "
                        f"target={relation.get('candidate_target_span_id', '')}"
                    ),
                }
            )
            relations.append(
                {
                    "source_span_id": span_id,
                    "relation": "diagnoses",
                    "target_span_id": "matcher_failure_rule",
                }
            )
    return {
        "event_id": f"matcher_failure_{rule.get('rule_id', 'unknown')}",
        "spans": spans,
        "relations": relations,
    }


def load_post_match_rules(
    args: dict[str, Any],
    state: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    from .routing import as_dict_list

    inline = as_dict_list(args.get("matcher_rule_records") or args.get("post_match_rule_records"))
    records = inline or load_cached_matcher_rules(args, state, limit=limit)
    configured = [rule for rule in records if is_matcher_rule(rule, phase="post_match")]
    rule_ids = {str(rule.get("rule_id") or "") for rule in configured}
    startup = _startup_post_match_policy_rules()
    return [
        *configured,
        *[rule for rule in startup if str(rule.get("rule_id") or "") not in rule_ids],
    ]


def propose_temporary_rule_revision(
    rule: dict[str, Any],
    hard_match: dict[str, Any],
    *,
    args: dict[str, Any],
    state: Any,
    matcher_rules: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Match post-match control rules and apply their temporary revisions."""

    explicit_operations = _as_dict_list(args.get("post_match_require_revisions"))
    applied_rule_ids: list[str] = []
    operations = list(explicit_operations)
    if matcher_rules is None:
        matcher_rules = load_post_match_rules(
            args,
            state,
            limit=max(32, int(args.get("matcher_rule_k") or 64)),
        )
    if matcher_rules:
        from .matching import match_required_bundle_against_graph
        from .routing import required_bundle_from_rule, workspace_to_match_bundle

        workspace = build_match_failure_workspace(rule, hard_match)
        target = {
            "kind": "graph",
            "spec": {"type": "matcher_failure_graph", "rule_id": rule.get("rule_id")},
            "workspaces": [workspace_to_match_bundle(workspace["event_id"], workspace)],
        }
        for matcher_rule in matcher_rules:
            source = required_bundle_from_rule(
                matcher_rule,
                {"type": "rule", "rule_id": matcher_rule.get("rule_id"), "field": "required"},
            )
            outcome = match_required_bundle_against_graph(source, target, args)
            if not outcome.get("ok"):
                continue
            applied_rule_ids.append(str(matcher_rule.get("rule_id") or ""))
            operations.extend(_as_dict_list(matcher_rule.get("produced")))
    if any(
        str(operation.get("operation") or "").strip().lower()
        in {"stop_path", "reject_require_revision"}
        for operation in operations
    ):
        return None, [rule_id for rule_id in applied_rule_ids if rule_id]
    revised = apply_require_revisions(
        rule,
        resolve_failure_scoped_operations(operations, hard_match),
    )
    return revised, [rule_id for rule_id in applied_rule_ids if rule_id]


def apply_require_revisions(
    rule: dict[str, Any],
    operations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    revised = deepcopy(rule)
    changed = False
    for operation in operations:
        operation_name = str(operation.get("operation") or "").strip().lower()
        if operation_name not in {"revise_require", "retry_hard_match"}:
            continue
        target_rule_id = str(operation.get("target_rule_id") or "")
        if target_rule_id and target_rule_id != str(rule.get("rule_id") or ""):
            continue
        updates = _as_dict_list(operation.get("condition_updates"))
        relation_updates = _as_dict_list(operation.get("relation_updates"))
        required_items = revised.get("required") if isinstance(revised.get("required"), list) else []
        for required in required_items:
            if not isinstance(required, dict):
                continue
            conditions = required.get("condition") if isinstance(required.get("condition"), list) else []
            for update in updates:
                slot_id = str(update.get("span_id") or update.get("slot_id") or "")
                if update.get("remove_from_required"):
                    before = len(conditions)
                    conditions[:] = [
                        condition
                        for condition in conditions
                        if str(condition.get("span_id") or condition.get("slot_id") or "") != slot_id
                    ]
                    if len(conditions) != before:
                        changed = True
                        relations = required.get("relation_patterns") if isinstance(required.get("relation_patterns"), list) else []
                        required["relation_patterns"] = [
                            relation
                            for relation in relations
                            if slot_id not in {
                                str(relation.get("source_span_id") or ""),
                                str(relation.get("target_span_id") or ""),
                            }
                        ]
                    continue
                for condition in conditions:
                    if not isinstance(condition, dict):
                        continue
                    current_id = str(condition.get("span_id") or condition.get("slot_id") or "")
                    if slot_id and slot_id != current_id:
                        continue
                    changed |= _apply_boundary_update(condition, update)
            relations = required.get("relation_patterns") if isinstance(required.get("relation_patterns"), list) else []
            for update in relation_updates:
                relation_id = str(update.get("relation_id") or "")
                for relation in relations:
                    if not isinstance(relation, dict):
                        continue
                    current_id = str(relation.get("relation_id") or relation.get("id") or "")
                    if relation_id and relation_id != current_id:
                        continue
                    changed |= _apply_boundary_update(relation, update)
        revised["required"] = [
            required
            for required in required_items
            if isinstance(required, dict)
            and (
                bool(required.get("condition"))
                or bool(required.get("relation_patterns"))
            )
        ]
        revised.setdefault("metadata", {})
        revised["metadata"]["temporary_match_revision"] = True
    if changed and not revised.get("required"):
        return None
    return revised if changed else None


def resolve_failure_scoped_operations(
    operations: list[dict[str, Any]],
    hard_match: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn matcher-rule selectors into explicit temporary boundary updates."""

    metadata = hard_match.get("metadata") if isinstance(hard_match.get("metadata"), dict) else {}
    per_required = metadata.get("per_required") if isinstance(metadata.get("per_required"), list) else []
    failed_slots: list[dict[str, Any]] = []
    failed_relations: list[dict[str, Any]] = []
    for outcome in per_required:
        details = outcome.get("details") if isinstance(outcome.get("details"), dict) else {}
        failed_slots.extend(
            slot for slot in (details.get("slot_matches") or [])
            if isinstance(slot, dict) and not slot.get("matched")
        )
        failed_relations.extend(
            relation for relation in (details.get("relation_matches") or [])
            if isinstance(relation, dict) and not relation.get("matched")
        )

    resolved: list[dict[str, Any]] = []
    for operation in operations:
        cloned = deepcopy(operation)
        condition_updates: list[dict[str, Any]] = []
        for update in _as_dict_list(cloned.get("condition_updates")):
            apply_to = str(update.get("apply_to") or "")
            if apply_to == "failed_soft_conditions":
                for slot in failed_slots:
                    missing_effect = str(slot.get("missing_effect") or "").strip().lower()
                    if not _soft_missing_effect(missing_effect):
                        continue
                    explicit = {key: deepcopy(value) for key, value in update.items() if key != "apply_to"}
                    explicit["span_id"] = str(slot.get("slot_id") or "")
                    condition_updates.append(explicit)
                continue
            if apply_to != "failed_near_threshold_conditions":
                condition_updates.append(update)
                continue
            max_gap = float(update.get("max_threshold_gap", 0.10) or 0.10)
            for slot in failed_slots:
                score = float(slot.get("score", 0.0) or 0.0)
                threshold = float(slot.get("match_threshold", 0.5) or 0.5)
                if threshold - score > max_gap:
                    continue
                explicit = {key: deepcopy(value) for key, value in update.items() if key != "apply_to"}
                explicit["span_id"] = str(slot.get("slot_id") or "")
                condition_updates.append(explicit)
        relation_updates: list[dict[str, Any]] = []
        for update in _as_dict_list(cloned.get("relation_updates")):
            apply_to = str(update.get("apply_to") or "")
            if apply_to != "failed_relations_with_correct_endpoints":
                relation_updates.append(update)
                continue
            for relation in failed_relations:
                bound_source = str(relation.get("bound_source_span_id") or "")
                bound_target = str(relation.get("bound_target_span_id") or "")
                candidate_source = str(relation.get("candidate_source_span_id") or "")
                candidate_target = str(relation.get("candidate_target_span_id") or "")
                if not bound_source or not bound_target:
                    continue
                if candidate_source != bound_source or candidate_target != bound_target:
                    continue
                explicit = {key: deepcopy(value) for key, value in update.items() if key != "apply_to"}
                explicit["relation_id"] = str(relation.get("required_relation_id") or "")
                relation_updates.append(explicit)
        cloned["condition_updates"] = condition_updates
        cloned["relation_updates"] = relation_updates
        resolved.append(cloned)
    return resolved


def _apply_boundary_update(target: dict[str, Any], update: dict[str, Any]) -> bool:
    changed = False
    if update.get("match_threshold") is not None:
        target["match_threshold"] = max(0.0, min(1.0, float(update["match_threshold"])))
        changed = True
    if update.get("threshold_delta") is not None:
        current = float(target.get("match_threshold", 0.5) or 0.5)
        target["match_threshold"] = max(0.0, min(1.0, current + float(update["threshold_delta"])))
        changed = True
    for key in ("core", "relation", "nature", "core_vector", "relation_vector", "nature_vector"):
        if key in update:
            target[key] = deepcopy(update[key])
            changed = True
    return changed


def _soft_missing_effect(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return any(
        token in normalized
        for token in ("lower_confidence", "deprioritize", "weaken", "support_only", "soft")
    ) and not normalized.startswith("block")


def _startup_post_match_policy_rules() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "matcher_post_soft_condition_temporary_revision",
            "rule_layer": "matcher_rule",
            "rule_kind": "matcher_rule",
            "status": "bootstrap",
            "description": "post-match control policy temporarily removes failed soft conditions before retrying the same hard matcher",
            "scope_tags": ["matcher_rule", "post_match", "soft_condition", "temporary_revision"],
            "required": [],
            "produced": [
                {
                    "operation": "revise_require",
                    "condition_updates": [
                        {
                            "apply_to": "failed_soft_conditions",
                            "remove_from_required": True,
                        }
                    ],
                }
            ],
            "priority": 1.0,
            "confidence": 1.0,
            "metadata": {
                "startup_policy_rule": True,
                "temporary_only": True,
                "hard_matcher_reused": True,
            },
        }
    ]


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


__all__ = [
    "apply_require_revisions",
    "build_match_failure_workspace",
    "load_post_match_rules",
    "propose_temporary_rule_revision",
    "resolve_failure_scoped_operations",
]
