"""Strict evaluation of rule-owned structured arithmetic expressions.

The evaluator operates on matched event spans.  It never parses rule text and
never executes source code.  Rules provide a small expression AST; matcher
slot bindings identify the event spans that supply operand values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SUPPORTED_OPERATORS = frozenset({"mean", "weighted_mean", "add", "subtract", "multiply", "divide"})
SUPPORTED_PREDICATES = frozenset({"eq", "ne", "gt", "gte", "lt", "lte", "and", "or", "not"})


class StructuredComputationError(ValueError):
    """A deterministic, user-visible computation failure."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(message)


@dataclass(frozen=True)
class _Quantity:
    value: float
    dimensions: tuple[tuple[str, int], ...] = ()

    @property
    def dimension_map(self) -> dict[str, int]:
        return dict(self.dimensions)


def evaluate_structured_expression(
    expression: Mapping[str, Any],
    *,
    spans: Sequence[Mapping[str, Any]],
    slot_bindings: Mapping[str, str],
    result_unit: str = "",
) -> dict[str, Any]:
    """Evaluate one expression AST against matcher-bound event spans."""

    if not isinstance(expression, Mapping):
        raise StructuredComputationError("invalid_expression", "expression must be an object")
    span_index = _build_span_index(spans)
    bindings = {str(key): str(value) for key, value in slot_bindings.items() if str(key) and str(value)}
    trace: list[dict[str, Any]] = []
    used_spans: dict[str, dict[str, Any]] = {}
    quantity = _evaluate_node(expression, span_index=span_index, bindings=bindings, trace=trace, used_spans=used_spans)
    declared_unit = _resolve_result_unit(result_unit, span_index=span_index, bindings=bindings)
    if declared_unit:
        expected_dimensions = _unit_dimensions(declared_unit)
        if expected_dimensions != quantity.dimension_map:
            raise StructuredComputationError(
                "result_unit_mismatch",
                f"computed dimensions do not match declared result unit {declared_unit!r}",
                details={
                    "computed_dimensions": quantity.dimension_map,
                    "declared_dimensions": expected_dimensions,
                    "declared_unit": declared_unit,
                },
            )
        unit = declared_unit
    else:
        unit = _render_unit(quantity.dimension_map)
    if not math.isfinite(quantity.value):
        raise StructuredComputationError("non_finite_result", "structured expression produced a non-finite result")
    return {
        "status": "computed",
        "value": quantity.value,
        "unit": unit,
        "operator": str(expression.get("operator") or "").strip().lower(),
        "slot_bindings": bindings,
        "source_span_ids": sorted({item["span_id"] for item in used_spans.values()}),
        "bound_operands": used_spans,
        "trace": trace,
    }


def evaluate_structured_predicate(
    predicate: Mapping[str, Any],
    *,
    spans: Sequence[Mapping[str, Any]],
    slot_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate a rule predicate against bound event values without parsing text."""

    if not isinstance(predicate, Mapping):
        raise StructuredComputationError("invalid_predicate", "predicate must be an object")
    span_index = _build_span_index(spans)
    bindings = {str(key): str(value) for key, value in slot_bindings.items() if str(key) and str(value)}
    trace: list[dict[str, Any]] = []
    matched = _evaluate_predicate_node(predicate, span_index=span_index, bindings=bindings, trace=trace)
    return {
        "status": "matched" if matched else "not_matched",
        "matched": matched,
        "slot_bindings": bindings,
        "trace": trace,
    }


def _evaluate_predicate_node(
    node: Mapping[str, Any],
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
    trace: list[dict[str, Any]],
) -> bool:
    operator = str(node.get("operator") or "").strip().lower()
    if operator not in SUPPORTED_PREDICATES:
        raise StructuredComputationError(
            "unsupported_predicate",
            f"unsupported structured predicate operator: {operator or '<missing>'}",
            details={"operator": operator},
        )
    if operator in {"and", "or"}:
        operands = node.get("operands")
        if not isinstance(operands, list) or not operands:
            raise StructuredComputationError(
                "missing_predicate_operands",
                f"{operator} requires a non-empty operands list",
            )
        values = [
            _evaluate_predicate_node(item, span_index=span_index, bindings=bindings, trace=trace)
            if isinstance(item, Mapping)
            else _invalid_predicate_operand(item)
            for item in operands
        ]
        result = all(values) if operator == "and" else any(values)
        trace.append({"operator": operator, "operands": values, "result": result})
        return result
    if operator == "not":
        operand = node.get("operand")
        if not isinstance(operand, Mapping):
            raise StructuredComputationError("missing_predicate_operand", "not requires one predicate operand")
        result = not _evaluate_predicate_node(operand, span_index=span_index, bindings=bindings, trace=trace)
        trace.append({"operator": operator, "result": result})
        return result

    if "left" not in node or "right" not in node:
        raise StructuredComputationError(
            "missing_predicate_operands",
            f"{operator} requires left and right operands",
        )
    left = _resolve_predicate_operand(node["left"], span_index=span_index, bindings=bindings)
    right = _resolve_predicate_operand(node["right"], span_index=span_index, bindings=bindings)
    if left[1] != right[1]:
        raise StructuredComputationError(
            "incompatible_units",
            "predicate operands must have compatible units",
            details={"operator": operator, "left_dimensions": left[1], "right_dimensions": right[1]},
        )
    comparisons = {
        "eq": left[0] == right[0],
        "ne": left[0] != right[0],
        "gt": left[0] > right[0],
        "gte": left[0] >= right[0],
        "lt": left[0] < right[0],
        "lte": left[0] <= right[0],
    }
    result = comparisons[operator]
    trace.append({"operator": operator, "left": left[0], "right": right[0], "result": result})
    return result


def _resolve_predicate_operand(
    operand: Any,
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
) -> tuple[float, dict[str, int]]:
    if isinstance(operand, bool):
        return float(operand), {}
    if isinstance(operand, (int, float)):
        return _finite_number(operand, operand="predicate_constant"), {}
    if isinstance(operand, str):
        span_id, span = _bound_span(operand, span_index=span_index, bindings=bindings)
        value = span.get("value")
        if isinstance(value, bool):
            return float(value), {}
        return _finite_number(value, operand=span_id), _unit_dimensions(str(span.get("unit") or ""))
    if isinstance(operand, Mapping) and "value" in operand:
        value = operand.get("value")
        unit = str(operand.get("unit") or "").strip()
        if unit.startswith("inherit:"):
            slot_id = unit.split(":", 1)[1].strip()
            _, inherited_span = _bound_span(slot_id, span_index=span_index, bindings=bindings)
            unit = str(inherited_span.get("unit") or "").strip()
            if not unit:
                raise StructuredComputationError(
                    "missing_inherited_unit",
                    f"cannot inherit predicate unit from slot {slot_id!r}",
                    details={"slot_id": slot_id},
                )
        if isinstance(value, bool):
            return float(value), _unit_dimensions(unit)
        return (
            _finite_number(value, operand="predicate_constant"),
            _unit_dimensions(unit),
        )
    raise StructuredComputationError(
        "invalid_predicate_operand",
        "predicate operands must be slot ids or typed constants",
        details={"operand_type": type(operand).__name__},
    )


def _invalid_predicate_operand(value: Any) -> bool:
    raise StructuredComputationError(
        "invalid_predicate_operand",
        "logical predicate operands must be predicate objects",
        details={"operand_type": type(value).__name__},
    )


def _evaluate_node(
    node: Any,
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
    trace: list[dict[str, Any]],
    used_spans: dict[str, dict[str, Any]],
) -> _Quantity:
    if isinstance(node, bool):
        raise StructuredComputationError("invalid_numeric_value", "boolean values are not numeric operands")
    if isinstance(node, (int, float)):
        return _Quantity(_finite_number(node, operand="constant"))
    if isinstance(node, str):
        return _resolve_slot(node, span_index=span_index, bindings=bindings, used_spans=used_spans)
    if not isinstance(node, Mapping):
        raise StructuredComputationError(
            "invalid_expression_node",
            "expression nodes must be numbers, slot ids, or expression objects",
            details={"node_type": type(node).__name__},
        )
    if "value" in node and "operator" not in node:
        value = _finite_number(node.get("value"), operand="constant")
        unit = str(node.get("unit") or "").strip()
        return _Quantity(value, _freeze_dimensions(_unit_dimensions(unit)))

    operator = str(node.get("operator") or "").strip().lower()
    if operator not in SUPPORTED_OPERATORS:
        raise StructuredComputationError(
            "unsupported_operator",
            f"unsupported structured arithmetic operator: {operator or '<missing>'}",
            details={"operator": operator},
        )

    if operator == "weighted_mean":
        result = _weighted_mean(node, span_index=span_index, bindings=bindings, used_spans=used_spans)
        trace.append({"operator": operator, "result": result.value, "dimensions": result.dimension_map})
        return result

    if operator in {"mean", "add", "multiply"}:
        raw_operands = node.get("operands")
        if not isinstance(raw_operands, list) or not raw_operands:
            raise StructuredComputationError(
                "missing_operands",
                f"{operator} requires a non-empty operands list",
                details={"operator": operator},
            )
        operands = [
            _evaluate_node(item, span_index=span_index, bindings=bindings, trace=trace, used_spans=used_spans)
            for item in raw_operands
        ]
        if operator in {"mean", "add"}:
            _require_same_dimensions(operands, operator)
            total = sum(item.value for item in operands)
            value = total / len(operands) if operator == "mean" else total
            result = _Quantity(value, operands[0].dimensions)
        else:
            value = math.prod(item.value for item in operands)
            result = _Quantity(value, _combine_dimensions(operands, divide=False))
        trace.append({"operator": operator, "operands": [item.value for item in operands], "result": result.value})
        return result

    left_key, right_key = ("left", "right") if operator == "subtract" else ("numerator", "denominator")
    if left_key not in node or right_key not in node:
        raise StructuredComputationError(
            "missing_operands",
            f"{operator} requires {left_key} and {right_key}",
            details={"operator": operator},
        )
    left = _evaluate_node(node[left_key], span_index=span_index, bindings=bindings, trace=trace, used_spans=used_spans)
    right = _evaluate_node(node[right_key], span_index=span_index, bindings=bindings, trace=trace, used_spans=used_spans)
    if operator == "subtract":
        _require_same_dimensions([left, right], operator)
        result = _Quantity(left.value - right.value, left.dimensions)
    else:
        if right.value == 0.0:
            raise StructuredComputationError("division_by_zero", "structured expression denominator is zero")
        result = _Quantity(left.value / right.value, _combine_dimensions([left, right], divide=True))
    trace.append({"operator": operator, left_key: left.value, right_key: right.value, "result": result.value})
    return result


def _weighted_mean(
    node: Mapping[str, Any],
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
    used_spans: dict[str, dict[str, Any]],
) -> _Quantity:
    operands = node.get("operands")
    if not isinstance(operands, list) or len(operands) != 1 or not isinstance(operands[0], str):
        raise StructuredComputationError(
            "invalid_weighted_mean",
            "weighted_mean requires exactly one slot operand containing weighted observations",
        )
    slot_id = operands[0]
    span_id, span = _bound_span(slot_id, span_index=span_index, bindings=bindings)
    observations = span.get("value")
    weight_key = str(node.get("weight") or "weight").strip()
    if not isinstance(observations, list) or not observations:
        raise StructuredComputationError(
            "invalid_weighted_observations",
            f"slot {slot_id!r} must contain a non-empty observation list",
            details={"slot_id": slot_id, "span_id": span_id},
        )
    weighted_total = 0.0
    total_weight = 0.0
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise StructuredComputationError(
                "invalid_weighted_observation",
                "weighted observations must be objects",
                details={"slot_id": slot_id, "index": index},
            )
        value = _finite_number(observation.get("value"), operand=f"{slot_id}[{index}].value")
        weight = _finite_number(observation.get(weight_key), operand=f"{slot_id}[{index}].{weight_key}")
        if weight < 0:
            raise StructuredComputationError("negative_weight", "weighted_mean does not accept negative weights")
        weighted_total += value * weight
        total_weight += weight
    if total_weight == 0.0:
        raise StructuredComputationError("zero_total_weight", "weighted_mean total weight is zero")
    unit = str(span.get("unit") or "").strip()
    used_spans[slot_id] = {"span_id": span_id, "value": observations, "unit": unit}
    return _Quantity(weighted_total / total_weight, _freeze_dimensions(_unit_dimensions(unit)))


def _resolve_slot(
    slot_id: str,
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
    used_spans: dict[str, dict[str, Any]],
) -> _Quantity:
    span_id, span = _bound_span(slot_id, span_index=span_index, bindings=bindings)
    if "value" not in span:
        raise StructuredComputationError(
            "missing_operand_value",
            f"bound span {span_id!r} has no numeric value",
            details={"slot_id": slot_id, "span_id": span_id},
        )
    value = _finite_number(span.get("value"), operand=slot_id)
    unit = str(span.get("unit") or "").strip()
    used_spans[slot_id] = {"span_id": span_id, "value": value, "unit": unit}
    return _Quantity(value, _freeze_dimensions(_unit_dimensions(unit)))


def _bound_span(
    slot_id: str,
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
) -> tuple[str, Mapping[str, Any]]:
    span_id = str(bindings.get(slot_id) or slot_id)
    span = span_index.get(span_id)
    if span is None:
        raise StructuredComputationError(
            "missing_slot_binding",
            f"no event span is bound to expression slot {slot_id!r}",
            details={"slot_id": slot_id, "bound_span_id": span_id},
        )
    return span_id, span


def _build_span_index(spans: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        span_id = str(span.get("span_id") or span.get("id") or "").strip()
        if span_id:
            index[span_id] = span
    return index


def _finite_number(value: Any, *, operand: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredComputationError(
            "invalid_numeric_value",
            f"operand {operand!r} must be an int or float",
            details={"operand": operand, "value_type": type(value).__name__},
        )
    result = float(value)
    if not math.isfinite(result):
        raise StructuredComputationError(
            "non_finite_operand",
            f"operand {operand!r} must be finite",
            details={"operand": operand},
        )
    return result


def _require_same_dimensions(quantities: Sequence[_Quantity], operator: str) -> None:
    dimensions = [item.dimension_map for item in quantities]
    if any(item != dimensions[0] for item in dimensions[1:]):
        raise StructuredComputationError(
            "incompatible_units",
            f"{operator} operands must have compatible units",
            details={"operator": operator, "dimensions": dimensions},
        )


def _combine_dimensions(quantities: Sequence[_Quantity], *, divide: bool) -> tuple[tuple[str, int], ...]:
    dimensions: dict[str, int] = {}
    for index, quantity in enumerate(quantities):
        sign = -1 if divide and index == 1 else 1
        for name, exponent in quantity.dimensions:
            dimensions[name] = dimensions.get(name, 0) + sign * exponent
    return _freeze_dimensions(dimensions)


def _unit_dimensions(unit: str) -> dict[str, int]:
    normalized = " ".join(str(unit or "").strip().lower().split())
    if not normalized or normalized in {"ratio", "times", "state"}:
        return {}
    if normalized == "percent":
        return {}
    return {normalized: 1}


def _freeze_dimensions(dimensions: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((name, int(exponent)) for name, exponent in dimensions.items() if int(exponent) != 0))


def _render_unit(dimensions: Mapping[str, int]) -> str:
    if not dimensions:
        return "ratio"
    if len(dimensions) == 1:
        name, exponent = next(iter(dimensions.items()))
        if exponent == 1:
            return name
    numerator = [name if exponent == 1 else f"{name}^{exponent}" for name, exponent in dimensions.items() if exponent > 0]
    denominator = [name if exponent == -1 else f"{name}^{-exponent}" for name, exponent in dimensions.items() if exponent < 0]
    if not denominator:
        return "*".join(numerator)
    return f"{'*'.join(numerator) or '1'}/{'*'.join(denominator)}"


def _resolve_result_unit(
    result_unit: str,
    *,
    span_index: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, str],
) -> str:
    declared = str(result_unit or "").strip()
    if not declared.startswith("inherit:"):
        return declared
    slot_id = declared.split(":", 1)[1].strip()
    _, span = _bound_span(slot_id, span_index=span_index, bindings=bindings)
    unit = str(span.get("unit") or "").strip()
    if not unit:
        raise StructuredComputationError(
            "missing_inherited_unit",
            f"cannot inherit result unit from slot {slot_id!r}",
            details={"slot_id": slot_id},
        )
    return unit
