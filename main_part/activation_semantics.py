"""Typed activation graphs and output semantics, independent of eligibility.

Signals are a versioned computation graph: inputs, context, existing functions
and vector operations can be combined without interpreting Python or rule text.
Dispatch consumers live in rule_execution_contract; graph effects stay in the
executor. No signal is implicitly a probability or a confidence score.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping

from .structured_arithmetic import StructuredComputationError


class ActivationError(StructuredComputationError):
    """An invalid activation definition or a failed activation computation."""


DOMAINS = frozenset({"boolean", "real", "unit_interval", "vector", "matrix"})
SOURCES = frozenset({"constant", "input", "context", "require_degree", "operation", "function"})
OPERATIONS = frozenset({"add", "subtract", "multiply", "divide", "mean", "min", "max",
                        "dot", "matmul", "concat", "sigmoid", "tanh", "relu", "silu",
                        "softmax", "abs", "negate", "clip", "gt", "gte", "lt", "lte",
                        "eq", "and", "or", "not"})
Q_TYPES = {"dispatch": {"boolean", "probability", "frequency"},
           "selection": {"score"},
           "transform": {"intensity", "signed_effect", "weight"},
           "assessment": {"confidence"}}


def number(value: Any, label: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ActivationError("invalid_activation_number", f"{label} must be a finite number")
    return float(value)


def read_path(value: Any, path: str) -> Any:
    if not isinstance(path, str) or not path:
        raise ActivationError("invalid_activation_path", "a non-empty path is required")
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ActivationError("missing_activation_input", f"missing path: {path}")
    return deepcopy(value)


def _numeric_tree(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            raise ActivationError("empty_activation_vector", "vectors cannot be empty")
        return [_numeric_tree(item) for item in value]
    return number(value)


def _unary(value: Any, fn) -> Any:
    return [_unary(item, fn) for item in value] if isinstance(value, list) else fn(number(value))


def _binary(left: Any, right: Any, fn) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ActivationError("activation_shape_mismatch", "vector shapes must agree")
        return [_binary(a, b, fn) for a, b in zip(left, right)]
    if isinstance(left, list):
        return [_binary(a, right, fn) for a in left]
    if isinstance(right, list):
        return [_binary(left, b, fn) for b in right]
    return fn(number(left), number(right))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value or any(isinstance(x, list) for x in value):
        raise ActivationError("invalid_activation_vector", "expected a non-empty numeric vector")
    return [number(x) for x in value]


def _validate_operands(op: str, args: list[Any]) -> None:
    if op not in OPERATIONS:
        raise ActivationError("unsupported_activation_operation", op)
    fixed = {**dict.fromkeys(('sigmoid', 'tanh', 'relu', 'silu', 'abs', 'negate', 'not', 'softmax'), 1),
             **dict.fromkeys(('subtract', 'divide', 'dot', 'matmul', 'gt', 'gte', 'lt', 'lte', 'eq'), 2), 'clip': 3}
    if not isinstance(args, list) or not args or (op in fixed and len(args) != fixed[op]):
        raise ActivationError('activation_arity', f'invalid operands for {op}')


def _operate(op: str, args: list[Any]) -> Any:
    _validate_operands(op, args)
    unary = {"sigmoid": _sigmoid, "tanh": math.tanh, "relu": lambda x: max(x, 0),
             "silu": lambda x: x * _sigmoid(x), "abs": abs, "negate": lambda x: -x}
    if op in unary:
        if len(args) != 1:
            raise ActivationError("activation_arity", f"{op} needs one operand")
        return _unary(args[0], unary[op])
    if op in {"and", "or", "not"}:
        if not args or any(not isinstance(x, bool) for x in args) or (op == "not" and len(args) != 1):
            raise ActivationError("activation_boolean_operands", op)
        return all(args) if op == "and" else any(args) if op == "or" else not args[0]
    if op in {"gt", "gte", "lt", "lte", "eq"}:
        if len(args) != 2:
            raise ActivationError("activation_arity", f"{op} needs two operands")
        a, b = map(number, args)
        return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b, "eq": a == b}[op]
    if op == "softmax":
        if len(args) != 1:
            raise ActivationError("activation_arity", "softmax needs one vector")
        values = _vector(args[0])
        exps = [math.exp(x - max(values)) for x in values]
        return [x / sum(exps) for x in exps]
    if op == "concat":
        return [x for arg in args for x in _vector(arg)]
    if op == "dot":
        if len(args) != 2:
            raise ActivationError("activation_arity", "dot needs two vectors")
        a, b = map(_vector, args)
        if len(a) != len(b):
            raise ActivationError("activation_shape_mismatch", "dot dimensions differ")
        return sum(x * y for x, y in zip(a, b))
    if op == "matmul":
        if len(args) != 2 or not isinstance(args[0], list):
            raise ActivationError("activation_arity", "matmul needs a matrix and vector")
        vector = _vector(args[1])
        return [_operate("dot", [row, vector]) for row in args[0]]
    if op == "clip":
        if len(args) != 3 or number(args[1]) > number(args[2]):
            raise ActivationError("activation_clip", "clip requires value, lower, upper")
        return _unary(args[0], lambda x: max(args[1], min(args[2], x)))
    binary = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
              "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
              "min": min, "max": max, "mean": lambda a, b: a + b}
    if not args or (op in {"subtract", "divide"} and len(args) != 2):
        raise ActivationError("activation_arity", f"invalid operands for {op}")
    result = args[0]
    for arg in args[1:]:
        result = _binary(result, arg, binary[op])
    return _unary(result, lambda x: x / len(args)) if op == "mean" else result


def validate_activation(spec: Mapping[str, Any], q: Mapping[str, Any]) -> None:
    if isinstance(spec, Mapping) and "by_mode" in spec:
        modes = spec["by_mode"]
        if not isinstance(modes, Mapping) or not modes:
            raise ActivationError("invalid_mode_activation", "by_mode must contain named graphs")
        for name, graph in modes.items():
            if not isinstance(name, str) or not name or not isinstance(graph, Mapping) or "by_mode" in graph:
                raise ActivationError("invalid_mode_activation", str(name))
            validate_activation(graph, q)
        spec = {k: v for k, v in spec.items() if k != "by_mode"}
    if isinstance(q, Mapping) and "profiles" in q:
        profiles = q["profiles"]
        if (set(q) != {"profiles", "default_profile"} or not isinstance(profiles, Mapping) or not profiles
                or q["default_profile"] not in profiles):
            raise ActivationError("invalid_q_profiles", "profiles need a declared default_profile")
        for name, profile in profiles.items():
            if not isinstance(name, str) or not name or not isinstance(profile, Mapping) or "profiles" in profile:
                raise ActivationError("invalid_q_profile", str(name))
            validate_activation(spec, profile)
        return
    if not isinstance(spec, Mapping) or set(spec) - {"signals", "primary"}:
        raise ActivationError("invalid_activation_spec", "activation_spec needs signals and primary")
    signals = spec.get("signals")
    if not isinstance(signals, Mapping) or not signals or spec.get("primary") not in signals:
        raise ActivationError("invalid_activation_signals", "primary must name a declared signal")
    for name, signal in signals.items():
        if not isinstance(name, str) or not isinstance(signal, Mapping):
            raise ActivationError("invalid_activation_signal", str(name))
        if set(signal) - {"domain", "unit", "size", "shape", "source", "normalization"}:
            raise ActivationError("unknown_activation_field", name)
        if signal.get("domain") not in DOMAINS:
            raise ActivationError("invalid_activation_domain", name)
        if not isinstance(signal.get("unit", ""), str):
            raise ActivationError("invalid_activation_unit", name)
        if "size" in signal and (signal["domain"] != "vector" or isinstance(signal["size"], bool)
                                  or not isinstance(signal["size"], int) or signal["size"] <= 0):
            raise ActivationError("invalid_activation_shape", name)
        if "shape" in signal and (signal["domain"] != "matrix" or not isinstance(signal["shape"], list)
                or len(signal["shape"]) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in signal["shape"])):
            raise ActivationError("invalid_activation_shape", name)
        source = signal.get("source")
        if not isinstance(source, Mapping) or source.get("kind") not in SOURCES:
            raise ActivationError("invalid_activation_source", name)
        allowed = {"constant": {"kind", "value"}, "input": {"kind", "slot", "field"},
                   "context": {"kind", "path"}, "require_degree": {"kind"},
                   "operation": {"kind", "op", "inputs"}, "function": {"kind", "request", "output_name"}}[source["kind"]]
        if set(source) - allowed:
            raise ActivationError("unknown_activation_source_field", name)
        if source["kind"] == "operation" and (source.get("op") not in OPERATIONS or not isinstance(source.get("inputs"), list)):
            raise ActivationError("invalid_activation_operation", name)
        if source['kind'] == 'operation':
            _validate_operands(source['op'], source['inputs'])
        required_keys = {"constant": {"value"}, "input": {"slot"}, "context": {"path"},
                         "require_degree": set(), "operation": {"op", "inputs"}, "function": {"request", "output_name"}}[source["kind"]]
        if not required_keys.issubset(source):
            raise ActivationError("incomplete_activation_source", name)
        if source["kind"] == "function":
            request = source["request"]
            if (not isinstance(request, Mapping) or request.get("kind", "function") != "function"
                    or request.get("schema_version") != 1 or not request.get("function_id") or not request.get("function_version")
                    or not isinstance(source["output_name"], str) or not source["output_name"]):
                raise ActivationError("invalid_activation_function_contract", name)
        norm = signal.get("normalization")
        if norm is not None and (not isinstance(norm, Mapping) or set(norm) - {"op", "lower", "upper"}
                                 or norm.get("op") not in {"sigmoid", "tanh", "softmax", "clip"}):
            raise ActivationError("invalid_activation_normalization", name)
        if norm is not None:
            if norm['op'] == 'clip':
                if set(norm) != {'op', 'lower', 'upper'} or number(norm['lower']) > number(norm['upper']):
                    raise ActivationError('invalid_activation_normalization', name)
            elif set(norm) != {'op'}:
                raise ActivationError('invalid_activation_normalization', name)
    visiting, visited = set(), set()
    def visit(name):
        if name in visiting or name not in signals:
            raise ActivationError("activation_dependency_error", f"unknown or cyclic signal: {name}")
        if name in visited:
            return
        visiting.add(name)
        source = signals[name]["source"]
        if source["kind"] == "operation":
            for arg in source["inputs"]:
                if not isinstance(arg, Mapping) or len(arg) != 1 or not set(arg).issubset({"signal", "value"}):
                    raise ActivationError("invalid_activation_operand", name)
                if "signal" in arg:
                    visit(arg["signal"])
        visiting.remove(name)
        visited.add(name)
    for name in signals:
        visit(name)
    if not isinstance(q, Mapping) or set(q) - set(Q_TYPES):
        raise ActivationError("invalid_q_semantics", "unknown Q stage")
    ids = set()
    for stage, definitions in q.items():
        if not isinstance(definitions, list):
            raise ActivationError("invalid_q_stage", stage)
        for entry in definitions:
            if not isinstance(entry, Mapping) or entry.get("type") not in Q_TYPES[stage]:
                raise ActivationError("unsupported_q_type", f"{stage}: {entry}")
            qid = entry.get("id")
            if not isinstance(qid, str) or not qid or qid in ids:
                raise ActivationError("invalid_q_id", "Q ids must be unique and non-empty")
            ids.add(qid)
            allowed = {"id", "type", "source", "mapping"} | {
                "score": {"comparison_key", "score_definition", "cost"},
                "boolean": {"threshold"}, "probability": set(), "frequency": {"unit"},
                "intensity": {"target"}, "signed_effect": {"target"},
                "weight": {"target", "group", "normalization"},
                "confidence": {"target", "calibrated", "calibration_id", "score_definition"}}[entry["type"]]
            if set(entry) - allowed:
                raise ActivationError("unknown_q_field", qid)
            source = entry.get("source")
            if not isinstance(source, Mapping) or len(source) != 1:
                raise ActivationError("invalid_q_source", qid)
            if "signal" in source:
                if source["signal"] not in signals:
                    raise ActivationError("unknown_q_signal", qid)
                domain = signals[source["signal"]]["domain"]
                if (domain == "matrix" or (domain == "vector" and "mapping" not in entry)
                        or (domain == "boolean" and entry["type"] != "boolean")):
                    raise ActivationError("incompatible_q_signal", qid)
                if entry["type"] == "boolean":
                    if domain == "boolean" and "threshold" in entry:
                        raise ActivationError("unused_boolean_threshold", qid)
                    if domain != "boolean" and "threshold" not in entry:
                        raise ActivationError("missing_activation_threshold", qid)
            elif stage != "assessment" or not isinstance(source.get("result_field"), str):
                raise ActivationError("invalid_q_source", qid)
            if "mapping" in entry:
                mapping = entry["mapping"]
                if ("signal" not in source or domain == "boolean" or not isinstance(mapping, Mapping)
                        or set(mapping) != {"weights", "bias", "normalization", "parameter_version"}
                        or mapping["normalization"] not in {"identity", "sigmoid", "tanh"}
                        or not isinstance(mapping["parameter_version"], str) or not mapping["parameter_version"]):
                    raise ActivationError("invalid_q_mapping", qid)
                _vector(mapping["weights"])
                number(mapping["bias"])
                expected_size = signals[source['signal']].get('size') if domain == 'vector' else 1
                if expected_size is not None and len(mapping['weights']) != expected_size:
                    raise ActivationError('activation_shape_mismatch', qid)
            if 'threshold' in entry:
                number(entry['threshold'], 'threshold')
            if entry["type"] == "score":
                if any(not isinstance(entry.get(k), str) or not entry[k] for k in ("comparison_key", "score_definition")):
                    raise ActivationError("invalid_selection_semantics", qid)
                if number(entry.get("cost", 0), "cost") < 0:
                    raise ActivationError("invalid_selection_cost", qid)
            if stage in {"transform", "assessment"}:
                target = entry.get("target")
                expected = "confidence" if stage == "assessment" else "value"
                if not isinstance(target, Mapping) or set(target) != {"span_id", "field"} or target["field"] != expected:
                    raise ActivationError("invalid_q_target", qid)
            if entry["type"] == "frequency" and entry.get("unit") not in {"calls/s", "calls/min", "calls/hour"}:
                raise ActivationError("invalid_frequency_unit", qid)
            if "signal" in source:
                unit = signals[source["signal"]].get("unit", "")
                if unit and (entry["type"] != "frequency" or entry["unit"] != unit):
                    raise ActivationError("incompatible_q_unit", qid)
            if entry["type"] == "weight" and (not isinstance(entry.get("group"), str) or not entry["group"]
                                               or entry.get("normalization") not in {"sum", "softmax", "none"}):
                raise ActivationError("invalid_weight_semantics", qid)
            if entry["type"] == "confidence":
                if not entry.get("score_definition") or not isinstance(entry.get("calibrated", False), bool):
                    raise ActivationError("invalid_confidence_semantics", qid)
                if entry.get("calibrated") and not entry.get("calibration_id"):
                    raise ActivationError("missing_calibration_provenance", qid)


def evaluate_activation(spec: Mapping[str, Any], *, workspace: dict, bindings: dict,
                        context: dict, require_degree: float, calculator) -> dict:
    signals = spec["signals"]
    values: dict[str, dict] = {}
    visiting: set[str] = set()
    function_results: dict[str, dict] = {}
    spans = {str(s.get("span_id") or s.get("id")): s for s in workspace.get("spans", [])}

    def argument(arg):
        if not isinstance(arg, Mapping) or len(arg) != 1:
            raise ActivationError("invalid_activation_operand", "use a signal reference or literal value")
        if "signal" in arg:
            return resolve(arg["signal"])["value"]
        if "value" in arg:
            return deepcopy(arg["value"])
        raise ActivationError("invalid_activation_operand", str(arg))

    def resolve(name):
        if name in values:
            return values[name]
        if name not in signals or name in visiting:
            raise ActivationError("activation_dependency_error", f"unknown or cyclic signal: {name}")
        visiting.add(name)
        definition = signals[name]
        source = definition["source"]
        kind = source["kind"]
        provenance = {}
        try:
            if kind == "constant":
                raw = deepcopy(source["value"])
            elif kind == "input":
                slot = source["slot"]
                sid = bindings.get(slot)
                if sid not in spans:
                    raise ActivationError("missing_activation_input", str(slot))
                raw = read_path(spans[sid], source.get("field", "value"))
                provenance = {"span_id": sid, "field": source.get("field", "value")}
            elif kind == "context":
                raw = read_path(context, source["path"])
            elif kind == "require_degree":
                raw = require_degree
            elif kind == "function":
                request_key = json.dumps(source["request"], sort_keys=True, allow_nan=False)
                if request_key not in function_results:
                    function_results[request_key] = calculator.calculate(source["request"], workspace=workspace, slot_bindings=bindings)
                computation = function_results[request_key]
                raw = read_path(computation, f"outputs.{source['output_name']}.value")
                provenance = {"computation": computation}
            else:
                raw = _operate(source["op"], [argument(arg) for arg in source["inputs"]])
            normalized = deepcopy(raw)
            norm = definition.get("normalization")
            if norm:
                operands = [normalized, norm["lower"], norm["upper"]] if norm["op"] == "clip" else [normalized]
                normalized = _operate(norm["op"], operands)
            domain = definition["domain"]
            if domain == "boolean":
                if not isinstance(normalized, bool):
                    raise ActivationError("activation_domain_mismatch", name)
            elif domain == "vector":
                normalized = _vector(normalized)
                if "size" in definition and len(normalized) != definition["size"]:
                    raise ActivationError("activation_shape_mismatch", name)
            elif domain == "matrix":
                if not isinstance(normalized, list) or not normalized:
                    raise ActivationError("activation_shape_mismatch", name)
                normalized = [_vector(row) for row in normalized]
                shape = [len(normalized), len(normalized[0])]
                if any(len(row) != shape[1] for row in normalized) or ("shape" in definition and definition["shape"] != shape):
                    raise ActivationError("activation_shape_mismatch", name)
            else:
                normalized = number(normalized, name)
                if domain == "unit_interval" and not 0 <= normalized <= 1:
                    raise ActivationError("activation_domain_mismatch", name)
            # Check intermediate raw values too; normalization cannot hide NaN.
            if not isinstance(raw, bool):
                _numeric_tree(raw)
            values[name] = {"raw": raw, "value": normalized, "domain": domain,
                            "unit": definition.get("unit", ""), "source": deepcopy(source), **provenance}
            visiting.remove(name)
            return values[name]
        except (ZeroDivisionError, OverflowError, KeyError, TypeError) as exc:
            raise ActivationError("activation_computation_failed", f"{name}: {exc}") from exc
        except ActivationError:
            raise
        except Exception as exc:
            raise ActivationError("activation_source_failed", f"{name}: {type(exc).__name__}: {exc}") from exc

    for name in signals:
        resolve(name)
    return {"signals": values, "primary": spec["primary"], "value": deepcopy(values[spec["primary"]]["value"])}


def activation_context(spec: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    """Capture only declared reads, not recursively growing Control histories."""
    snapshot: dict = {}
    for definition in spec["signals"].values():
        source = definition["source"]
        if source["kind"] != "context":
            continue
        path = source["path"]
        value = read_path(context, path)
        cursor = snapshot
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return snapshot


def q_entries(q: Mapping[str, Any], stage: str) -> list[dict]:
    if "profiles" in q:
        return [entry for profile in q["profiles"].values() for entry in profile.get(stage, [])]
    return list(q.get(stage, []))


def resolve_q(q: Mapping[str, Any], activation: Mapping[str, Any], profile: str | None = None) -> dict:
    if "profiles" in q:
        name = profile or q["default_profile"]
        if name not in q["profiles"]:
            raise ActivationError("unknown_q_profile", name)
        q = q["profiles"][name]
    elif profile is not None:
        raise ActivationError("unknown_q_profile", profile)
    resolved = deepcopy(q)
    for definitions in resolved.values():
        for entry in definitions:
            if "signal" in entry["source"]:
                entry["value"] = deepcopy(activation["signals"][entry["source"]["signal"]]["value"])
                if "mapping" in entry:
                    mapping = entry["mapping"]
                    raw = entry["value"]
                    vector = raw if isinstance(raw, list) else [raw]
                    value = number(_operate("dot", [mapping["weights"], vector]) + mapping["bias"])
                    entry["raw_response"] = raw
                    entry["value"] = value if mapping["normalization"] == "identity" else number(_operate(mapping["normalization"], [value]))
    return resolved


def apply_output_semantics(item: dict, *, q: dict, decision: dict, workspace: dict) -> dict:
    """Apply semantics only to the explicitly named new numeric result."""
    sid = item.get("span_id")
    applied = []
    for entry in q.get("transform", []):
        if entry["target"]["span_id"] != sid:
            continue
        value, factor = number(item.get("value"), "output"), number(entry["value"], entry["id"])
        if entry["type"] in {"intensity", "signed_effect"}:
            if entry["type"] == "intensity" and factor < 0:
                raise ActivationError("negative_intensity", entry["id"])
            item["value"] = number(value * factor)
        else:
            if entry["normalization"] in {"sum", "none"} and factor < 0:
                raise ActivationError("negative_mixture_weight", entry["id"])
            previous = next((s for s in workspace.get("spans", []) if s.get("span_id") == sid), {})
            fusion = deepcopy(previous.get("activation_fusion", {}))
            identity = {"group": entry["group"], "normalization": entry["normalization"], "unit": item.get("unit", "")}
            if fusion and any(fusion[k] != v for k, v in identity.items()):
                raise ActivationError("incompatible_fusion", entry["id"])
            fusion.update(identity)
            contributors = fusion.setdefault("contributors", {})
            from .evidence_lifecycle import evidence_fingerprint, evidence_validity
            span_index = {s["span_id"]: s for s in workspace.get("spans", [])}
            validity = evidence_validity(span_index, workspace.get("relations", []))
            for cid, contribution in list(contributors.items()):
                fingerprints = contribution["source_fingerprints"]
                if any(not validity.get(sid) or evidence_fingerprint(span_index[sid]) != fp
                       for sid, fp in fingerprints.items()):
                    fusion.setdefault("invalidated_contributors", {})[cid] = "stale_support"
                    del contributors[cid]
            contributors[decision["decision_id"]] = {"value": value, "weight": factor,
                                                      "rule_id": decision["rule_id"],
                                                      "bindings": deepcopy(decision["bindings"]),
                                                      "source_fingerprints": {sid: evidence_fingerprint(span_index[sid])
                                                          for sid in decision["bindings"].values()}}
            entries = list(contributors.values())
            weights = [e["weight"] for e in entries]
            if entry["normalization"] == "softmax":
                weights = _operate("softmax", [weights])
            elif entry["normalization"] == "sum":
                if sum(weights) <= 0:
                    raise ActivationError("zero_fusion_weight", entry["id"])
                weights = [w / sum(weights) for w in weights]
            item["value"] = number(sum(w * e["value"] for w, e in zip(weights, entries)))
            item["activation_fusion"] = fusion
        applied.append(entry["id"])
    for entry in q.get("assessment", []):
        if entry["target"]["span_id"] != sid:
            continue
        score = (entry["value"] if "value" in entry else read_path(item.get("computation", {}), entry["source"]["result_field"]))
        score = number(score, "confidence")
        if not 0 <= score <= 1:
            raise ActivationError("invalid_confidence_score", entry["id"])
        item["confidence"] = score
        item.setdefault("metadata", {})["confidence_semantics"] = {
            "definition": entry["score_definition"], "source": deepcopy(entry["source"]),
            "calibrated": entry.get("calibrated", False), "calibration_id": entry.get("calibration_id")}
        applied.append(entry["id"])
    if applied:
        item["activation_effects"] = {"decision_id": decision["decision_id"], "q_ids": applied}
    return item
