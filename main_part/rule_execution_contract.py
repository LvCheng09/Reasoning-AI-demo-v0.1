"""One execution contract for eligibility, partial modes and rule effects."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from .activation_semantics import ActivationError, number, q_entries, validate_activation


CONTRACT_FIELDS = ("execution_contract_version", "require_policy", "activation_spec", "q_semantics",
                   "io_contract", "effect_contract", "composition_contract")

CONTROL_EFFECT_OPERATIONS = frozenset({
    "run_rule", "output", "stop_path", "request_frontend_patch", "control_decide", "update_control_state",
    "decide_rule_activation", "select_candidate_path", "select_alignment_targets", "align_event_spans",
    "retrieve_memory", "query_memory", "retrieve_workspace_memory", "query_memory_workspace",
    "retrieve_knowledge", "query_knowledge", "enrich_knowledge", "inspect_goal_requirements",
    "ground_input_semantics", "review_event_evidence", "inspect_rule_frontier", "inspect_require_gaps", "bind_knowledge_context", "mark_context_used",
})


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def is_typed_rule(rule: Mapping) -> bool:
    return rule.get("execution_contract_version", 1) == 2


def contract_digest(rule: Mapping) -> str:
    return digest({k: rule.get(k) for k in (*CONTRACT_FIELDS, "rule_id", "rule_layer", "required", "produced", "hard_constraints")})


def span_output_items(produced: Mapping) -> list[dict]:
    """Normalize V2 span writes before validation and before execution.

    Both public payload forms use the same computation, Q and provenance path.
    Graph surgery needs a separate return/provenance contract; reject it rather
    than sending typed results through the legacy raw mutation path.
    """
    op = produced.get("operation", "add")
    target = produced.get("target", "spans")
    conclusions, item = produced.get("conclusion"), produced.get("item")
    if conclusions and target != "spans":
        raise ActivationError("invalid_conclusion_target", "conclusions write spans only")
    if op not in {"add", "update", "merge", "split", "delete"} or target != "spans":
        return []
    if op in {"merge", "split"}:
        raise ActivationError("unsupported_typed_graph_surgery", "V2 span merge/split need an explicit provenance contract")
    if op == "delete":
        if conclusions or item or produced.get("items"):
            raise ActivationError("unused_graph_output", "delete cannot declare a result payload")
        return []
    if conclusions:
        if op != "add":
            raise ActivationError("unsupported_typed_conclusion", "computed conclusions use add; update uses item")
        if item is not None or produced.get("items"):
            raise ActivationError("conflicting_graph_outputs", "use conclusion or item, not both")
        if not isinstance(conclusions, list) or any(not isinstance(s, Mapping) for s in conclusions):
            raise ActivationError("invalid_graph_output", "conclusion must be a list of span objects")
        outputs = deepcopy(conclusions)
    else:
        if not isinstance(item, Mapping) or not item or produced.get("items"):
            raise ActivationError("invalid_graph_output", "add/update needs a span item or added conclusions")
        outputs = [deepcopy(dict(item))]
        if op == "update":
            sid = produced.get("id") or item.get("span_id")
            if item.get("span_id", sid) != sid:
                raise ActivationError("conflicting_graph_output_id", "update id must equal item.span_id")
            outputs[0].setdefault("span_id", sid)
    ids = [s.get("span_id") for s in outputs]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ActivationError("invalid_graph_output_id", "V2 results need unique non-empty span_id values")
    for output in outputs:
        computations = [key for key in ("calculation", "expression", "value_policy") if key in output]
        if len(computations) > 1:
            raise ActivationError("conflicting_calculation", "a result must select one computation mechanism")
        if computations and not isinstance(output[computations[0]], dict):
            raise ActivationError("invalid_calculation", "a declared computation must be an object")
    return outputs


def validate_rule_contract(rule: Mapping) -> None:
    version = rule.get("execution_contract_version", 1)
    if isinstance(version, bool) or version not in {1, 2}:
        raise ActivationError("unsupported_execution_contract", str(version))
    if version == 1:
        if any(rule.get(k) for k in CONTRACT_FIELDS[1:]):
            raise ActivationError("missing_execution_contract_version", "new rule fields require version 2")
        return
    validate_activation(rule.get("activation_spec"), rule.get("q_semantics", {}))
    if rule.get("hard_constraints"):
        raise ActivationError("unmigrated_hard_constraints", "express executable constraints in Require predicates")
    policy = rule.get("require_policy", {})
    if not isinstance(policy, Mapping) or set(policy) - {"partial_mode", "min_degree", "execution_mode", "bundle_match_threshold", "q_selection", "execution_selection"}:
        raise ActivationError("invalid_require_policy", "unknown Require policy field")
    if "execution_selection" in policy and policy["execution_selection"] != "control":
        raise ActivationError("invalid_execution_selection", "execution_selection must be control")
    profiles = rule.get("q_semantics", {}).get("profiles", {"default": rule.get("q_semantics", {})})
    for profile in profiles.values():
        selections = profile.get("selection", [])
        if selections and policy.get("execution_selection") != "control":
            raise ActivationError("unused_selection_scores", "selection scores require execution_selection=control")
        if policy.get("execution_selection") == "control":
            keys = [s["comparison_key"] for s in selections]
            if not keys or len(keys) != len(set(keys)):
                raise ActivationError("invalid_selection_scores", "each profile needs unique comparable score directions")
    if "by_mode" in rule["activation_spec"]:
        names = {"full", *rule.get("io_contract", {}).get("partial_modes", {})}
        if policy.get("execution_selection") != "control" or not set(rule["activation_spec"]["by_mode"]).issubset(names):
            raise ActivationError("invalid_mode_activation", "mode graphs need Control selection and declared modes")
    if policy.get("partial_mode", "disabled") not in {"disabled", "control", "declared"}:
        raise ActivationError("invalid_partial_mode", str(policy))
    if "execution_mode" in policy and policy.get("partial_mode") != "declared":
        raise ActivationError("unused_partial_execution_mode", "Control selects the mode in its own action")
    if policy.get("q_selection", "declared") not in {"declared", "control"}:
        raise ActivationError("invalid_q_selection", str(policy))
    for key in ("min_degree", "bundle_match_threshold"):
        if key in policy and not 0 <= number(policy[key], key) <= 1:
            raise ActivationError("invalid_require_threshold", key)
    ids = []
    for required in rule.get("required", []):
        rid = required.get("required_id")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ActivationError("invalid_requirement_id", "typed Require clauses need unique required_id")
        ids.append(rid)
        if not 0 < number(required.get("condition_priority", 1.0), rid) <= 1:
            raise ActivationError("invalid_requirement_weight", rid)
    io = rule.get("io_contract")
    if not isinstance(io, Mapping) or set(io) - {"required_slots", "outputs", "accepts_partial_inputs", "partial_modes"}:
        raise ActivationError("invalid_io_contract", "typed rules need an io_contract")
    if not isinstance(io.get("accepts_partial_inputs", False), bool):
        raise ActivationError("invalid_io_contract", "accepts_partial_inputs must be boolean")
    modes = io.get("partial_modes", {})
    if not isinstance(modes, Mapping):
        raise ActivationError("invalid_partial_contract", "partial_modes must be an object")
    for name, mode in modes.items():
        if not name or name == "full" or not isinstance(mode, Mapping):
            raise ActivationError("invalid_partial_contract", str(name))
        if set(mode) - {"required_ids", "required_slots", "produced_indices", "outputs", "result_scope"}:
            raise ActivationError("unknown_partial_contract_field", name)
        if not isinstance(mode.get("required_ids"), list) or not set(mode["required_ids"]).issubset(ids):
            raise ActivationError("invalid_partial_requirements", name)
        indices = mode.get("produced_indices")
        if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices) or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(rule.get("produced", [])) for i in indices):
            raise ActivationError("invalid_partial_produced", name)
        if mode.get("result_scope") != "partial":
            raise ActivationError("invalid_partial_scope", "partial modes must declare partial results")
        for index in indices:
            produced = rule["produced"][index]
            if produced.get("operation", "add") not in {"add", "update"} or not span_output_items(produced):
                raise ActivationError("unsupported_partial_effect", "partial execution supports declared span results")
            if "task_result" in produced:
                raise ActivationError("partial_task_completion", name)
    if policy.get("partial_mode", "disabled") != "disabled" and not modes:
        raise ActivationError("missing_partial_contract", "partial execution needs actual declared modes")
    if policy.get("partial_mode") == "declared" and policy.get("execution_mode") not in modes:
        raise ActivationError("unknown_partial_execution_mode", str(policy.get("execution_mode")))
    for section in [io, *modes.values()]:
        for key in ("required_slots", "outputs"):
            values = section.get(key, [])
            if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
                raise ActivationError("invalid_io_contract", key)
    effects = rule.get("effect_contract")
    if not isinstance(effects, Mapping) or set(effects) - {"kind", "write_targets", "child_effects"}:
        raise ActivationError("invalid_effect_contract", "effect_contract is required")
    if effects.get("kind") not in {"graph", "control", "composite"}:
        raise ActivationError("unsupported_effect_contract", "rules declare graph/control effects; pure local functions stay in Calculator")
    if effects["kind"] == "composite":
        if (not rule.get("composition_contract") or not isinstance(effects.get("child_effects"), list)
                or not effects["child_effects"] or not set(effects["child_effects"]).issubset({"graph", "control"})):
            raise ActivationError("invalid_composite_effects", "declare the actual graph/control child effects")
    elif effects.get("child_effects"):
        raise ActivationError("unused_child_effects", str(effects["kind"]))
    targets = effects.get("write_targets", [])
    if not isinstance(targets, list) or any(x not in {"spans", "relations", "events", "event"} for x in targets):
        raise ActivationError("invalid_effect_target", str(targets))
    if effects["kind"] == "control" and rule.get("rule_layer") != "control":
        raise ActivationError("invalid_control_effect", "only Control rules may declare control effects")
    all_outputs = set(io.get("outputs", [])) | {s for m in modes.values() for s in m.get("outputs", [])}
    defined_outputs = {s["span_id"] for p in rule.get("produced", []) for s in span_output_items(p)}
    defined_outputs.update(rule.get("composition_contract", {}).get("output_bindings", {}))
    if not all_outputs.issubset(defined_outputs):
        raise ActivationError("unknown_contract_output", str(all_outputs - defined_outputs))
    if not defined_outputs.issubset(all_outputs):
        raise ActivationError("undeclared_rule_output", str(defined_outputs - all_outputs))
    for stage in ("transform", "assessment"):
        for q in q_entries(rule.get("q_semantics", {}), stage):
            if q["target"]["span_id"] not in defined_outputs:
                raise ActivationError("unknown_q_target", q["id"])
    if rule.get("composition_contract"):
        from .rule_composition import validate_composition
        validate_composition(rule)
    for produced in rule.get("produced", []):
        check_effect(rule, produced)
        if produced.get("operation", "add") != "add" and produced.get("conclusion"):
            raise ActivationError("unsupported_typed_conclusion", "computed conclusions use add; other effects use their explicit item contract")
        if produced.get("operation") == "run_rule" and not rule.get("composition_contract") and rule["effect_contract"]["kind"] != "control":
            raise ActivationError("missing_composition_contract", "domain child calls need an explicit return contract")
    if rule.get("composition_contract", {}).get("output_bindings") and (
            (effects["kind"] != "graph" and "graph" not in effects.get("child_effects", [])) or "spans" not in targets):
        raise ActivationError("undeclared_composition_return_effect", "parent return values write graph spans")


def require_evaluation(rule: Mapping, match: Mapping) -> dict:
    """Wrap actual matcher outcomes; similarity is kept separate from coverage."""
    validate_rule_contract(rule)
    metadata = match.get("metadata", {})
    outcomes = metadata.get("per_required", [])
    clauses = rule.get("required", [])
    bindings, units, missing, blocked, events = {}, [], [], [], set()
    for index, outcome in enumerate(outcomes):
        clause = clauses[index]
        rid = clause["required_id"]
        details = outcome.get("details", {})
        checks = [bool(s.get("matched")) for s in details.get("slot_matches", [])]
        checks.extend(bool(r.get("matched")) for r in details.get("relation_matches", []))
        checks.extend(bool(r.get("matched")) for r in details.get("predicate_matches", []))
        checks.extend(bool(r.get("matched")) for r in details.get("shared_semantic_relation_constraint_matches", []))
        # Keep the actual clause result as a unit: thresholds and reuse rules
        # cannot disappear just because all visible slots have a binding.
        checks.append(outcome.get("matched") is True)
        weight = number(clause.get("condition_priority", 1.0), rid)
        units.extend({"required_id": rid, "satisfied": v, "weight": weight / len(checks)} for v in checks)
        if not outcome.get("matched"):
            missing.append(rid)
        fatal_slot_statuses = {"missing_required_vector", "vector_contract_mismatch",
                               "source_reliability_below_required_minimum", "authority_verification_required"}
        if (outcome.get("reason") == "invalid_required_relation_endpoints"
                or any(s.get("vector_status") in fatal_slot_statuses for s in details.get("slot_matches", []))
                or any(p.get("status") == "error" for p in details.get("predicate_matches", []))):
            blocked.append(rid)
        successful_slots = {s["slot_id"] for s in details.get("slot_matches", []) if s.get("matched")}
        for slot, sid in details.get("slot_bindings", {}).items():
            if slot not in successful_slots:
                continue
            if slot in bindings and bindings[slot] != sid:
                raise ActivationError("conflicting_slot_bindings", slot)
            bindings[slot] = sid
        if outcome.get("event_id"):
            events.add(outcome["event_id"])
    if clauses and len(outcomes) != len(clauses):
        raise ActivationError("incomplete_require_evaluation", "all original Require clauses must be checked")
    if clauses:
        # This gate is part of the old ok formula, not a new execution threshold.
        units.append({"required_id": "$bundle", "satisfied": metadata.get("bundle_score", 0) >= metadata.get("bundle_threshold", 1), "weight": 1.0})
        degree = sum(u["weight"] for u in units if u["satisfied"]) / sum(u["weight"] for u in units)
    else:
        degree = 1.0 if match.get("ok") is True else 0.0
    full = match.get("ok") is True
    if (degree == 1) != full:
        raise ActivationError("inconsistent_require_evaluation", "coverage and full Require result disagree")
    return {"full_match": full, "require_degree": degree, "missing_ids": missing, "blocking_ids": blocked,
            "bindings": bindings, "event_ids": sorted(events), "coverage_basis": units,
            "contract_digest": contract_digest(rule), "evaluator_version": "require_coverage.v2"}


def partial_modes(rule: Mapping, evaluation: Mapping) -> list[str]:
    policy = rule.get("require_policy", {})
    degree = evaluation["require_degree"]
    if policy.get("partial_mode", "disabled") == "disabled" or evaluation["blocking_ids"] or not (
            0 < degree <= 1) or (degree == 1 and policy.get("execution_selection") != "control") or degree < policy.get("min_degree", 0):
        return []
    missing = set(evaluation["missing_ids"])
    return [name for name, mode in rule["io_contract"].get("partial_modes", {}).items()
            if not missing.intersection(mode["required_ids"])
            and set(mode.get("required_slots", [])).issubset(evaluation["bindings"])]


def candidate_can_execute(item: Mapping) -> bool:
    if item.get("match", {}).get("ok") is True:
        return True
    rule = item.get("rule", {})
    return bool(is_typed_rule(rule) and item.get("allowed_partial_modes") and
                (rule.get("require_policy", {}).get("partial_mode") == "declared" or item.get("execution_authorization")))


def execution_mode(rule: Mapping, evaluation: Mapping, mode: str) -> dict:
    if mode == "full":
        if not evaluation["full_match"]:
            raise ActivationError("require_not_satisfied", str(rule.get("rule_id")))
        io = rule["io_contract"]
        return {**deepcopy(io), "produced_indices": list(range(len(rule.get("produced", [])))), "result_scope": "full"}
    if mode not in partial_modes(rule, evaluation):
        raise ActivationError("partial_mode_not_eligible", mode)
    return deepcopy(rule["io_contract"]["partial_modes"][mode])


def check_inputs(rule: Mapping, evaluation: Mapping, workspace: Mapping, mode: Mapping) -> None:
    from .evidence_lifecycle import evidence_validity
    index = {str(s.get("span_id") or s.get("id")): s for s in workspace.get("spans", [])}
    validity = evidence_validity(index, workspace.get("relations", []))
    for slot in mode.get("required_slots", []):
        if slot not in evaluation["bindings"]:
            raise ActivationError("missing_execution_input", slot)
    for slot, sid in evaluation["bindings"].items():
        span = index.get(sid)
        if span is None or not validity.get(sid):
            raise ActivationError("stale_execution_input", slot)
        scope = span.get("execution_scope", "full")
        if scope == "partial" and not rule["io_contract"].get("accepts_partial_inputs", False):
            raise ActivationError("partial_input_not_accepted", slot)


def execution_workspace(workspace: Mapping) -> dict:
    """The same read-only evidence view used by Matcher, including Knowledge.

    Context values remain context, rather than being silently promoted into
    live graph facts. Historical Memory exclusions stay in matcher.routing.
    """
    from .matcher.routing import build_span_index, collect_event_relations
    return {**workspace, "spans": list(build_span_index(workspace).values()),
            "relations": collect_event_relations(workspace)}


def check_effect(rule: Mapping, produced: Mapping) -> None:
    op = produced.get("operation", "add")
    effect = rule["effect_contract"]
    if op == "decide_rule_activation" and produced.get("action") == "select_execution":
        from .execution_selection import validate_selection_policy
        if set(produced) - {"operation", "action", "selection_policy", "rule_id"}:
            raise ActivationError("unused_selection_action_field", "selection_policy determines mode and profile")
        validate_selection_policy(produced.get("selection_policy"))
    if op in {"add", "update", "delete", "merge", "split"}:
        if effect["kind"] != "graph" or produced.get("target_workspace", "event_workspace") != "event_workspace":
            raise ActivationError("undeclared_rule_effect", str(op))
        if produced.get("target", "spans") not in effect.get("write_targets", []):
            raise ActivationError("undeclared_write_target", str(produced.get("target", "spans")))
    elif op not in CONTROL_EFFECT_OPERATIONS:
        raise ActivationError("undeclared_rule_effect", str(op))
    elif op != "run_rule" and effect["kind"] != "control":
        raise ActivationError("undeclared_control_effect", str(op))
    # Decisions are containers, not authority to bypass the effect contract.
    if op == "control_decide":
        branches = produced.get("branches", [])
        if not isinstance(branches, list) or any(not isinstance(b, Mapping) for b in branches):
            raise ActivationError("invalid_control_branches", "branches must be objects")
        lists = [produced.get("before_decide", []), produced.get("default", []), *[b.get("then", []) for b in branches]]
        for actions in lists:
            if not isinstance(actions, list) or any(not isinstance(a, Mapping) for a in actions):
                raise ActivationError("invalid_control_actions", "Control actions must be a list of objects")
            for action in actions:
                check_effect(rule, action)
    for field in ("after_frontend", "on_unresolved"):
        if field in produced:
            if not isinstance(produced[field], list) or any(not isinstance(a, Mapping) for a in produced[field]):
                raise ActivationError("invalid_control_actions", field)
            for action in produced[field]:
                check_effect(rule, action)


def result_metadata(item: dict, decision: Mapping) -> None:
    item["execution_scope"] = decision["result_scope"]
    item["rule_execution"] = {k: deepcopy(decision[k]) for k in
                              ("decision_id", "rule_id", "contract_digest", "execution_mode", "require_degree", "missing_ids")}


def check_output(rule: Mapping, item: Mapping, mode: Mapping) -> None:
    if item.get("span_id") not in mode.get("outputs", []):
        raise ActivationError("undeclared_rule_output", str(item.get("span_id")))
    if mode.get("result_scope") == "partial" and item.get("task_result"):
        raise ActivationError("partial_task_completion", str(item.get("span_id")))
