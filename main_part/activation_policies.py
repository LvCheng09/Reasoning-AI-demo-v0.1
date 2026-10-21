"""Ordinary, serializable Control rules for activation decisions."""
from __future__ import annotations

from .rule_execution_contract import validate_rule_contract
from .execution_selection import validate_selection_policy


def selection_scope_tag(comparison_key: str) -> str:
    return 'activation_score:' + comparison_key


def execution_selection_control_rule(rule_id: str, *, policy: dict) -> dict:
    """A real Control rule; selection is dispatched through normal control_decide."""
    validate_selection_policy(policy)
    rule = activation_control_rule(rule_id, action="execute_full")
    rule['scope_tags'].append(selection_scope_tag(policy['comparison_key']))
    rule["produced"][0]["branches"][0]["then"] = [{
        "operation": "decide_rule_activation", "action": "select_execution", "selection_policy": policy}]
    validate_rule_contract(rule)
    return rule


def activation_control_rule(rule_id: str, *, action: str, execution_mode: str | None = None,
                            rank_signal: str | None = None, q_profile: str | None = None) -> dict:
    if not rule_id or action not in {"execute_partial", "execute_full", "defer", "retry", "reject"}:
        raise ValueError("a rule ID and a supported activation action are required")
    decision = {"operation": "decide_rule_activation", "action": action}
    if execution_mode is not None:
        decision["execution_mode"] = execution_mode
    if rank_signal is not None:
        decision["rank_signal"] = rank_signal
    if q_profile is not None:
        decision["q_profile"] = q_profile
    rule = {"rule_id": rule_id, "rule_layer": "control", "status": "active",
            "scope_tags": ["rule_activation_decision"], "execution_contract_version": 2,
            "required": [], "require_policy": {"partial_mode": "disabled"},
            "activation_spec": {"primary": "ready", "signals": {"ready": {
                "domain": "boolean", "source": {"kind": "context", "path": "trigger.has_activation_candidates"}}}},
            "q_semantics": {"dispatch": [{"id": "candidate_present", "type": "boolean", "source": {"signal": "ready"}}]},
            "io_contract": {"required_slots": [], "outputs": []}, "effect_contract": {"kind": "control"},
            "produced": [{"operation": "control_decide", "branches": [{
                "when": {"path": "trigger.kind", "op": "eq", "value": "rule_activation_decision_required"},
                "then": [decision]}], "default": [{"operation": "stop_path", "reason": "activation trigger is absent"}]}]}
    validate_rule_contract(rule)
    return rule
