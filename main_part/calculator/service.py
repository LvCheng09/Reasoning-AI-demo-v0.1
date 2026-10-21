"""Rule-independent computation. Receives copied values, never writable memory."""
from copy import deepcopy
import math
import hashlib
import json
from typing import Callable

from ..structured_arithmetic import StructuredComputationError, evaluate_structured_expression
from ..runtime_value_binding import resolve_runtime_value


class Calculator:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path
        self._functions: dict[tuple[str, str], Callable] = {}
        self._pinned_versions: dict[str, str] = {}
        self._deterministic_failures: dict[str, dict] = {}

    def bind_failure_state(self, failures: dict) -> None:
        """Task-local, serializable arithmetic failures, shared across retries."""
        self._deterministic_failures = failures

    def _calculate_expression(self, request, workspace, slot_bindings):
        key = hashlib.sha256(json.dumps([request, slot_bindings], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        spans = {str(s.get("span_id") or s.get("id")): s for s in workspace.get("spans", [])}
        previous = self._deterministic_failures.get(key)
        if previous is not None:
            unchanged = all(
                sid in spans and {"value": spans[sid].get("value"), "unit": spans[sid].get("unit", "")} == snapshot
                for sid, snapshot in previous["inputs"].items())
            if unchanged:
                raise StructuredComputationError("unchanged_computation_failure", previous["message"], details={
                    "original_error_code": previous["error_code"], "failure_id": key,
                    "retry_without_input_change": False, "retry_suppressed": True})
        try:
            result = evaluate_structured_expression(request["expression"], spans=workspace.get("spans", []),
                slot_bindings=slot_bindings, result_unit=request.get("result_unit", ""))
        except StructuredComputationError as exc:
            # Only a known deterministic arithmetic failure is retained.
            # Model, network and other potentially transient errors are not.
            if exc.code == "division_by_zero":
                inputs = {v["span_id"]: {"value": deepcopy(spans[v["span_id"]].get("value")),
                          "unit": spans[v["span_id"]].get("unit", "")} for v in exc.details["used_inputs"].values()}
                self._deterministic_failures[key] = {"error_code": exc.code, "message": str(exc), "inputs": inputs}
                exc.details.update(failure_id=key, retry_without_input_change=False, retry_suppressed=False)
            raise
        self._deterministic_failures.pop(key, None)
        return result

    def reset_versions(self):
        """Called only at a new task boundary, never while a task is executing."""
        self._pinned_versions.clear()

    def bind_version_state(self, versions: dict[str, str]) -> None:
        """Use the task's serialized version pins across process restoration."""
        if not isinstance(versions, dict):
            raise ValueError("function version state must be an object")
        self._pinned_versions = versions

    def function_registry_identity(self) -> list[tuple[str, str]]:
        return sorted(self._functions)

    def register(self, function_id: str, version: str, function: Callable) -> None:
        key = (function_id, version)
        if not all(key) or key in self._functions:
            raise ValueError(f"invalid or duplicate function registration: {key}")
        self._functions[key] = function

    def calculate(self, request: dict, *, workspace: dict, slot_bindings: dict) -> dict:
        """Return computed values; invalid contracts raise a typed, visible error."""
        request, workspace, slot_bindings = deepcopy((request, workspace, slot_bindings))
        kind = request.get("kind", "function")
        if kind == "expression":
            return self._calculate_expression(request, workspace, slot_bindings)
        if kind == "value_policy":
            result = resolve_runtime_value(request["value_policy"], workspace=workspace)
            return {**result.to_dict(), "status": "computed" if result.ok else "abstained"}
        if kind != "function" or request.get("schema_version") != 1:
            raise StructuredComputationError("invalid_calculation_contract", "unsupported calculation schema")
        spans = {str(s.get("span_id", s.get("id", ""))): s for s in workspace.get("spans", [])}
        inputs, evidence = {}, []
        for name, binding in request.get("inputs", {}).items():
            if not isinstance(binding, dict) or binding.get("field", "value") != "value":
                raise StructuredComputationError("invalid_input_binding", str(name))
            span_id = slot_bindings.get(binding.get("slot"))
            span = spans.get(str(span_id))
            if span is None or "value" not in span:
                raise StructuredComputationError("missing_calculation_input", str(name))
            value = span["value"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise StructuredComputationError("invalid_calculation_input", str(name))
            inputs[name] = {"value": value, "unit": span.get("unit", "")}
            evidence.append(str(span_id))
        key = (request.get("function_id"), str(request.get("function_version", "")))
        if request.get("parameter_version") == "active":
            from ..persistence.function_db import active_version
            if key[0] not in self._pinned_versions:
                self._pinned_versions[key[0]] = active_version(self.db_path, key[0])
            request["parameter_version"] = self._pinned_versions[key[0]]
        function = self._functions.get(key)
        if function is not None:
            result = function(inputs, deepcopy(request))
        else:
            from .model_runtime import infer
            result = infer(self.db_path, inputs, request)
        if not isinstance(result, dict) or not isinstance(result.get("outputs"), dict):
            raise StructuredComputationError("invalid_function_output", str(key))
        for output in result["outputs"].values():
            value = output.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise StructuredComputationError("invalid_function_output", str(key))
        return {**result, "status": "computed", "source_span_ids": sorted(set(evidence)),
                "function_id": key[0], "function_version": key[1],
                "parameter_version": request.get("parameter_version")}
