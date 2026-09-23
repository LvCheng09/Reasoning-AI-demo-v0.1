"""Value-of-continuation state for controller decisions."""

from __future__ import annotations
from .common.normalize import safe_float as shared_safe_float
from typing import Any


DEFAULT_THRESHOLDS = {
    "risk_requirement": 0.28,
    "meta_loop_ratio": 0.85,
    "meta_error_count": 3,
}

TASK_THRESHOLDS = {
    "financial_analysis": {
        "risk_requirement": 0.18,
        "meta_loop_ratio": 0.80,
        "meta_error_count": 2,
    },
}


def evaluate_control_voc(state: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    task_type = str(state.get("task_type") or "general").strip().lower() or "general"
    thresholds = _task_thresholds(task_type, state.get("control_voc_thresholds"))
    pending_work = _pending_work_count(state)
    output_count = len(state.get("outputs") or [])
    candidate_count = len(state.get("pending_output_candidates") or [])
    error_count = len(state.get("errors") or [])
    evidence_count = _memory_evidence_count(memory)
    loop_index = _safe_float(state.get("loop_index"), 0.0)
    max_loops = max(1.0, _safe_float(state.get("max_loops"), 32.0))

    remaining_uncertainty = _clamp(
        0.35
        + (0.14 if pending_work else 0.0)
        + (0.18 if not output_count else -0.12)
        + (0.10 if candidate_count else 0.0)
        + min(0.18, error_count * 0.06)
        - min(0.20, evidence_count * 0.015)
    )
    expected_information_gain = _clamp(
        0.10
        + min(0.45, pending_work * 0.08)
        + (0.16 if candidate_count else 0.0)
        + (0.12 if not output_count else 0.0)
        - min(0.18, evidence_count * 0.01)
    )
    continuation_cost = _clamp(0.08 + (loop_index / max_loops) * 0.32 + min(0.18, pending_work * 0.025))
    risk_requirement = _risk_requirement(state, thresholds)
    voc_positive = expected_information_gain - continuation_cost
    voc_negative = continuation_cost - expected_information_gain
    if pending_work:
        status = "continue_recommended"
    elif output_count and voc_positive <= 0.05 and remaining_uncertainty <= risk_requirement:
        status = "output_ready"
    elif voc_positive <= -0.05:
        status = "stop_recommended"
    else:
        status = "continue_recommended"

    loop_ratio = loop_index / max_loops
    meta_triggered = (
        error_count >= int(thresholds["meta_error_count"])
        and loop_ratio >= float(thresholds["meta_loop_ratio"])
    )
    return {
        "task_type": task_type,
        "thresholds": dict(thresholds),
        "remaining_uncertainty": round(remaining_uncertainty, 4),
        "expected_information_gain": round(expected_information_gain, 4),
        "continuation_cost": round(continuation_cost, 4),
        "voc_positive": round(max(0.0, voc_positive), 4),
        "voc_negative": round(max(0.0, voc_negative), 4),
        "risk_requirement": round(risk_requirement, 4),
        "evaluation_checkpoint": {
            "pending_work_count": pending_work,
            "output_count": output_count,
            "candidate_count": candidate_count,
            "error_count": error_count,
            "evidence_count": evidence_count,
            "loop_index": int(loop_index),
            "max_loops": int(max_loops),
        },
        "control_status": status,
        "meta_trigger": {
            "triggered": meta_triggered,
            "learning_mode": "meta" if meta_triggered else "study",
            "target_scope": "model_self" if meta_triggered else "knowledge",
            "reason": (
                "repeated runtime errors near the loop budget require model-level review"
                if meta_triggered
                else "model-level review threshold not reached"
            ),
        },
    }


def _pending_work_count(state: dict[str, Any]) -> int:
    keys = (
        "pending_inputs",
        "pending_operations",
        "pending_frontend_patches",
        "pending_rule_ids",
        "pending_rule_selectors",
        "pending_output_candidates",
    )
    return sum(len(state.get(key) or []) for key in keys)


def _memory_evidence_count(memory: dict[str, Any]) -> int:
    count = 0
    for zone in ("event_workspaces", "input_workspace", "memory_workspace"):
        value = memory.get(zone)
        if isinstance(value, dict):
            count += len(value)
        elif isinstance(value, list):
            count += len(value)
    return count


def _task_thresholds(task_type: str, overrides: Any) -> dict[str, float]:
    values = {
        **DEFAULT_THRESHOLDS,
        **TASK_THRESHOLDS.get(task_type, {}),
    }
    if isinstance(overrides, dict):
        for key in DEFAULT_THRESHOLDS:
            if key in overrides:
                values[key] = _safe_float(overrides[key], values[key])
    return values


def _risk_requirement(state: dict[str, Any], thresholds: dict[str, float]) -> float:
    outputs = state.get("outputs") or []
    if any(str(item.get("operation") or "") == "stop_path" for item in outputs if isinstance(item, dict)):
        return min(float(thresholds["risk_requirement"]), 0.20)
    if state.get("pending_output_candidates"):
        return min(float(thresholds["risk_requirement"]) + 0.04, 0.24)
    return float(thresholds["risk_requirement"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    return shared_safe_float(value, default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
