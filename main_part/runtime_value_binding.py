"""Runtime binding/derivation for parameterized rule conclusions.

Rules may learn structural conclusions from supervised graph deltas without
freezing an instance literal from the training G*.  A conclusion can declare a
``value_policy`` and the executor resolves it against the current event graph.

The registry is intentionally closed: rules select a named algorithm, but they
cannot execute arbitrary code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RuntimeValueResolution:
    ok: bool
    value: Any = None
    unit: str = ""
    algorithm: str = ""
    reason: str = ""
    evidence_span_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "unit": self.unit,
            "algorithm": self.algorithm,
            "reason": self.reason,
            "evidence_span_ids": list(self.evidence_span_ids),
        }


ValueResolver = Callable[[Mapping[str, Any], Mapping[str, Any]], RuntimeValueResolution]
_REGISTRY: dict[str, ValueResolver] = {}


def register_runtime_value_resolver(name: str) -> Callable[[ValueResolver], ValueResolver]:
    key = str(name or "").strip()
    if not key:
        raise ValueError("runtime value resolver requires a non-empty name")

    def decorator(fn: ValueResolver) -> ValueResolver:
        if key in _REGISTRY and _REGISTRY[key] is not fn:
            raise ValueError(f"runtime value resolver already registered: {key}")
        _REGISTRY[key] = fn
        return fn

    return decorator


def registered_runtime_value_resolvers() -> list[str]:
    return sorted(_REGISTRY)


def resolve_runtime_value(
    value_policy: Mapping[str, Any],
    *,
    workspace: Mapping[str, Any],
) -> RuntimeValueResolution:
    algorithm = str(value_policy.get("algorithm") or "").strip()
    if not algorithm:
        return RuntimeValueResolution(
            ok=False,
            algorithm="",
            reason="value_policy_missing_algorithm",
        )
    resolver = _REGISTRY.get(algorithm)
    if resolver is None:
        return RuntimeValueResolution(
            ok=False,
            algorithm=algorithm,
            reason="value_policy_algorithm_not_registered",
        )
    result = resolver(value_policy, workspace)
    if result.algorithm:
        return result
    return RuntimeValueResolution(
        ok=result.ok,
        value=result.value,
        unit=result.unit,
        algorithm=algorithm,
        reason=result.reason,
        evidence_span_ids=result.evidence_span_ids,
    )


def _spans(workspace: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in workspace.get("spans", []) or [] if isinstance(item, Mapping)]


def _feature_type(span: Mapping[str, Any]) -> str:
    return str(span.get("feature_type") or span.get("feature_schema_id") or "").strip().lower()


def _span_id(span: Mapping[str, Any]) -> str:
    return str(span.get("span_id") or span.get("id") or "").strip()


def _allowed_feature_types(policy: Mapping[str, Any]) -> set[str]:
    raw = policy.get("source_feature_types") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).strip().lower() for item in raw if str(item).strip()}


_SCALE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("billions", re.compile(r"\bbillions?\b", re.I)),
    ("millions", re.compile(r"\bmillions?\b", re.I)),
    ("thousands", re.compile(r"\bthousands?\b|\bk\b", re.I)),
    ("units", re.compile(r"\bunits?\b|\bones?\b", re.I)),
)


def _scale_from_text(text: str) -> str:
    normalized = str(text or "").strip()
    for value, pattern in _SCALE_PATTERNS:
        if pattern.search(normalized):
            return value
    return ""


@register_runtime_value_resolver("finance_amount_scale_from_units_v1")
def _finance_amount_scale_from_units_v1(
    policy: Mapping[str, Any],
    workspace: Mapping[str, Any],
) -> RuntimeValueResolution:
    """Derive an amount scale from current runtime evidence units.

    This deliberately ignores the supervised training literal.  It considers
    only current event spans and succeeds only when the runtime evidence gives a
    single unambiguous scale.
    """

    allowed = _allowed_feature_types(policy) or {
        "avg_ar",
        "begin_ar",
        "end_ar",
        "sales",
        "sales_proxy",
        "revenue",
        "revenue_proxy",
        "inventory",
        "avg_inventory",
        "begin_inventory",
        "end_inventory",
        "ap",
        "avg_ap",
        "begin_ap",
        "end_ap",
    }
    evidence: list[tuple[str, str]] = []
    for span in _spans(workspace):
        if _feature_type(span) not in allowed:
            continue
        scale = _scale_from_text(str(span.get("unit") or ""))
        if not scale:
            metadata = span.get("metadata") if isinstance(span.get("metadata"), Mapping) else {}
            scale = _scale_from_text(str(metadata.get("unit") or metadata.get("scale") or ""))
        if scale:
            evidence.append((_span_id(span), scale))

    unique_scales = sorted({scale for _, scale in evidence})
    if len(unique_scales) == 1:
        return RuntimeValueResolution(
            ok=True,
            value=unique_scales[0],
            algorithm="finance_amount_scale_from_units_v1",
            reason="single_runtime_amount_scale",
            evidence_span_ids=tuple(span_id for span_id, _ in evidence if span_id),
        )
    if not unique_scales:
        return RuntimeValueResolution(
            ok=False,
            algorithm="finance_amount_scale_from_units_v1",
            reason="no_runtime_scale_evidence",
        )
    return RuntimeValueResolution(
        ok=False,
        algorithm="finance_amount_scale_from_units_v1",
        reason=f"conflicting_runtime_scale_evidence:{','.join(unique_scales)}",
        evidence_span_ids=tuple(span_id for span_id, _ in evidence if span_id),
    )


@register_runtime_value_resolver("finance_period_from_days_v1")
def _finance_period_from_days_v1(
    policy: Mapping[str, Any],
    workspace: Mapping[str, Any],
) -> RuntimeValueResolution:
    """Derive the DSO/DPO/DIO reporting-period label from current day evidence."""

    tolerance = float(policy.get("days_tolerance", 0.51) or 0.51)
    mapping_raw = policy.get("days_mapping")
    mapping: dict[float, str]
    if isinstance(mapping_raw, Mapping):
        mapping = {float(key): str(value) for key, value in mapping_raw.items()}
    else:
        mapping = {
            30.0: "monthly period",
            91.0: "quarterly period",
            360.0: "banking annual convention",
            365.0: "annual period",
        }

    candidates: list[tuple[str, float]] = []
    for span in _spans(workspace):
        if _feature_type(span) != "days":
            continue
        try:
            candidates.append((_span_id(span), float(span.get("value"))))
        except (TypeError, ValueError):
            continue

    if len(candidates) != 1:
        return RuntimeValueResolution(
            ok=False,
            algorithm="finance_period_from_days_v1",
            reason=(
                "missing_days_evidence"
                if not candidates
                else "ambiguous_days_evidence"
            ),
            evidence_span_ids=tuple(span_id for span_id, _ in candidates if span_id),
        )

    span_id, days = candidates[0]
    matches = [label for expected_days, label in mapping.items() if abs(days - expected_days) <= tolerance]
    if len(matches) != 1:
        return RuntimeValueResolution(
            ok=False,
            algorithm="finance_period_from_days_v1",
            reason=f"unsupported_days_value:{days}",
            evidence_span_ids=(span_id,) if span_id else (),
        )
    return RuntimeValueResolution(
        ok=True,
        value=matches[0],
        algorithm="finance_period_from_days_v1",
        reason="period_derived_from_runtime_days",
        evidence_span_ids=(span_id,) if span_id else (),
    )
