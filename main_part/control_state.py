"""Data contract for Control-authored workflow state, without strategy logic."""

from copy import deepcopy
from typing import Any


def control_state_span_is_active(span: Any) -> bool:
    """Return whether a Control-state span may participate in a new match.

    A strategy decision is historical evidence after Control switches to a
    replacement route.  Keep that evidence in the Event Graph for audit, but
    do not let it select another executable branch.  Ordinary Control spans
    have no lifecycle flag and remain usable.
    """
    if not isinstance(span, dict) or str(span.get("span_role") or "") != "control_state":
        return True
    metadata = span.get("metadata") if isinstance(span.get("metadata"), dict) else {}
    return metadata.get("control_state_active") is not False


def control_state_update(rule: dict[str, Any], values: Any) -> dict[str, Any]:
    if rule.get("rule_layer") != "control" or not rule.get("rule_id"):
        raise ValueError("workflow state may only be written by a Control rule")
    if not isinstance(values, dict) or not values:
        raise ValueError("update_control_state requires nonempty values")
    if "strategy_rule_id" in values and not isinstance(values["strategy_rule_id"], str):
        raise ValueError("strategy_rule_id must be a string")
    if "active_strategy_state" in values and not isinstance(values["active_strategy_state"], str):
        raise ValueError("active_strategy_state must be a string")
    if "task_result" in values:
        validate_task_result(values["task_result"])
    return {"rule_id": str(rule["rule_id"]), "values": deepcopy(values)}


def validate_task_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") not in {
        "unknown", "pending", "solved", "partial", "blocked", "failed",
    }:
        raise ValueError("task_result requires a supported status")
    return deepcopy(value)
