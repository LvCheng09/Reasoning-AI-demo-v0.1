"""Read-only AND/OR requirement inspection requested by a Control Rule.

This does not execute hypothetical rules, relax matching, invent facts, or
choose a task verdict. Business rules declare external inputs and the meaning
of failed predicates; Control rules select the final action from this report.
Unsupported structures and bounded searches remain unproven, never forbidden.
"""
from copy import deepcopy

from .matcher.matching import match_required_bundle_against_graph
from .matcher.routing import graph_bundle_from_spec, required_bundle_from_spec


BLOCKED = {"policy_condition_not_satisfied", "invalid_information"}


def _references(value, slots):
    if isinstance(value, str):
        return {value} if value in slots else set()
    if isinstance(value, dict):
        return set().union(*(_references(v, slots) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_references(v, slots) for v in value))
    return set()


class RequirementInspector:
    def __init__(self, rules, memory, event_id, *, max_depth, max_nodes):
        self.memory, self.event_id = memory, event_id
        self.max_depth, self.max_nodes = max_depth, max_nodes
        self.nodes = 0
        self.producers = {}
        self.external = set()
        self.graph = graph_bundle_from_spec({"type": "working_memory", "zones": ["event_workspace"],
                                           "event_ids": [event_id]}, {}, memory)
        self.workspace = memory.get_event_workspace(event_id)
        self.spans = {s["span_id"]: s for s in self.workspace.get("spans", [])}
        for rule in rules:
            if rule.get("rule_layer") in {"control", "matcher_rule", "auto_learning"} or rule.get("status") != "active":
                continue
            contract = rule.get("metadata", {}).get("terminal_contract", {})
            self.external.update(contract.get("external_inputs", []))
            for action in rule.get("produced", []):
                if action.get("operation") != "add":
                    continue
                for span in action.get("conclusion", []):
                    if span.get("feature_type"):
                        self.producers.setdefault(span["feature_type"], []).append(rule)

    def match(self, rule, pattern):
        # This is a diagnostic projection, not a new executable Rule contract.
        source = required_bundle_from_spec({"rule_id": rule["rule_id"], "rule_layer": rule.get("rule_layer"),
                                            "required": [pattern]}, {})
        result = match_required_bundle_against_graph(source, self.graph, {})
        result["metadata"]["evaluation_scope"] = "require_projection"
        details = result["metadata"]["per_required"][0].get("details", {})
        return result, details

    def project(self, rule, pattern, names, predicates=()):
        contract = rule["metadata"]["terminal_contract"]
        names = set(names)
        context = contract.get("context_slot")
        if context:
            names.add(context)
        projected = {**pattern, "condition": [s for s in pattern["condition"] if s.get("slot_id", s.get("span_id")) in names],
                     "predicates": list(predicates), "relation_patterns": []}
        # Preserve declared context checks for selected slots. A missing input
        # in a different slot cannot turn a proven amount boundary into unknown.
        projected["shared_semantic_relation_constraints"] = []
        for constraint in pattern.get("shared_semantic_relation_constraints", []):
            selected = [s for s in constraint["slot_ids"] if s in names]
            if len(selected) >= 2:
                projected["shared_semantic_relation_constraints"].append({**constraint, "slot_ids": selected})
        return projected

    def evidence(self, rule, details, message, **extra):
        bound = {slot: self.spans[sid] for slot, sid in details.get("slot_bindings", {}).items() if sid in self.spans}
        if "predicate" in extra:
            labels = rule["metadata"]["terminal_contract"].get("slot_labels", {})
            observations = [f"{labels.get(slot, slot)}={span.get('value')} {span.get('unit', '')}" for slot, span in bound.items()
                            if slot != rule["metadata"]["terminal_contract"].get("context_slot")]
            message += "（" + "，".join(observations) + "）"
        return {"rule_id": rule["rule_id"], "event_id": self.event_id, "message": message,
                "bindings": {slot: {"span_id": sid, "feature_type": self.spans[sid].get("feature_type"),
                                    "value": self.spans[sid].get("value"), "unit": self.spans[sid].get("unit", ""),
                                    **{k: self.spans[sid][k] for k in ("entity_id", "period", "policy_version") if k in self.spans[sid]}}
                             for slot, sid in details.get("slot_bindings", {}).items() if sid in self.spans}, **extra}

    @staticmethod
    def combine(children, *, alternatives):
        if not children:
            return {"kind": "no_valid_path", "evidence": [{"message": "No supported producer is available under the current rule network."}]}
        if alternatives:
            # One viable alternative prevents a rejection. An unexamined
            # alternative prevents claiming all paths are prohibited.
            for kind in ("route_ready", "no_valid_path", "insufficient_information", "invalid_information", "policy_condition_not_satisfied"):
                selected = [c for c in children if c["kind"] == kind]
                if selected:
                    return {"kind": kind, "evidence": [e for c in selected for e in c["evidence"]]}
        else:
            for kind in ("policy_condition_not_satisfied", "invalid_information", "no_valid_path", "insufficient_information"):
                selected = [c for c in children if c["kind"] == kind]
                if selected:
                    return {"kind": kind, "evidence": [e for c in selected for e in c["evidence"]]}
            return {"kind": "route_ready", "evidence": []}
        raise ValueError("Unsupported inspection result")

    def rule(self, rule, trail):
        contract = rule.get("metadata", {}).get("terminal_contract", {})
        required = rule.get("required", [])
        if self.nodes >= self.max_nodes or sum(t.startswith("feature:") for t in trail) > self.max_depth:
            return {"kind": "no_valid_path", "evidence": [{"message": "Requirement inspection budget reached; unexamined routes remain possible."}]}
        self.nodes += 1
        if rule.get("rule_layer") != "domain" or contract.get("schema") != "goal_requirements.v1" or len(required) != 1 or required[0].get("relation_patterns"):
            return {"kind": "no_valid_path", "evidence": [{"rule_id": rule["rule_id"], "message": "This rule has no supported terminal proof contract; its failure is not proof of prohibition."}]}
        pattern = required[0]
        match, details = self.match(rule, pattern)
        if match["ok"]:
            return {"kind": "route_ready", "evidence": [{"rule_id": rule["rule_id"], "message": "A strictly executable producer remains; continue reasoning."}]}
        slots = {s.get("slot_id", s.get("span_id")): s for s in pattern["condition"]}
        failures = {p["index"]: p for p in contract.get("predicate_failures", [])}
        for i, predicate in enumerate(pattern.get("predicates", [])):
            clause = failures.get(i)
            if not clause:
                continue
            probe = self.project(rule, pattern, _references(predicate, slots), [predicate])
            _, result = self.match(rule, probe)
            # The predicate itself must be false using strictly bound,
            # context-compatible evidence. Missing/invalid data is not false.
            bound = result.get("slot_matches", [])
            scoped = result.get("shared_semantic_relation_constraint_matches", [])
            checks = result.get("predicate_matches", [])
            if bound and all(s["matched"] for s in bound) and all(s["matched"] for s in scoped) and checks and checks[0].get("status") != "error" and not checks[0].get("matched"):
                kind = clause["kind"]
                if kind not in BLOCKED:
                    raise ValueError("Invalid business terminal failure category")
                return {"kind": kind, "evidence": [self.evidence(rule, result, clause["message"],
                    code=clause["code"], predicate=deepcopy(predicate), predicate_result=deepcopy(checks[0]))]}
        children = []
        for name, slot in slots.items():
            probe = self.project(rule, pattern, [name])
            current, slot_details = self.match(rule, probe)
            if current["ok"]:
                continue
            feature = slot.get("feature_type")
            if not feature:
                children.append({"kind": "no_valid_path", "evidence": [{"rule_id": rule["rule_id"], "message": "An untyped structural condition cannot be reduced to an external-fact request."}]})
            elif feature in contract.get("external_inputs", []):
                children.append({"kind": "insufficient_information", "evidence": [self.evidence(rule, slot_details,
                    contract.get("input_messages", {}).get(feature, "Current, compatible evidence is required for " + feature),
                    code="compatible_evidence_required", feature_type=feature,
                    matching_reason=slot_details.get("reason", "no_current_evidence"))]})
            else:
                children.append(self.feature(feature, (*trail, rule["rule_id"])))
        if not children:
            return {"kind": "no_valid_path", "evidence": [self.evidence(rule, details, "All inputs appear present, but an unproved predicate or structural condition remains; no prohibition is established.")]}
        return self.combine(children, alternatives=False)

    def feature(self, feature, trail=()):
        key = "feature:" + feature
        if key in trail:
            return {"kind": "no_valid_path", "evidence": [{"feature_type": feature, "message": "A dependency cycle remains unresolved."}]}
        producers = self.producers.get(feature, [])
        return self.combine([self.rule(r, (*trail, key)) for r in producers], alternatives=True)


def inspect_goal_requirements(rules, memory, event_ids, *, max_depth=16, max_nodes=128):
    """Return proof categories, not task_result or a scheduling decision."""
    if max_depth < 1 or max_nodes < 1:
        raise ValueError("Inspection budgets must be positive")
    if len(event_ids) != 1:
        return {"kind": "no_valid_path", "explanation": "A single active decision workspace is required by this inspection contract.", "evidence": []}
    inspector = RequirementInspector(rules, memory, event_ids[0], max_depth=max_depth, max_nodes=max_nodes)
    goals = [s for s in inspector.workspace.get("spans", []) if s.get("span_role") == "objective"
             and s.get("metadata", {}).get("target_feature_type")]
    if len(goals) != 1:
        return {"kind": "no_valid_path", "explanation": "No unique declared objective is available; the current search cannot establish a business verdict.", "evidence": []}
    target = goals[0]["metadata"]["target_feature_type"]
    result = inspector.feature(target)
    result.update(target_feature_type=target, event_id=event_ids[0], inspected_nodes=inspector.nodes,
                  max_depth=max_depth, max_nodes=max_nodes,
                  scope="current_evidence_and_declared_rule_network")
    result["explanation"] = "；".join(dict.fromkeys(e["message"] for e in result["evidence"]))
    return result
