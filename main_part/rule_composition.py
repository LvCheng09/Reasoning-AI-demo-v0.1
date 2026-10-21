"""Sequential, resumable rule composition through the existing executor.

Parents retain an explicit call frame and verify actual returned graph values.
Every child uses the same Matcher, admission runtime and Executor as a top-level
rule. A shared ledger bounds the entire nested call, including failed work.
"""
from __future__ import annotations

from copy import deepcopy

from .activation_semantics import ActivationError, apply_output_semantics
from .rule_execution_contract import (check_inputs, check_output, contract_digest, digest,
                                       is_typed_rule, result_metadata)


def validate_composition(rule: dict) -> None:
    contract = rule["composition_contract"]
    if not isinstance(contract, dict) or set(contract) - {"kind", "max_depth", "max_steps", "output_bindings"}:
        raise ActivationError("invalid_composition_contract", "unknown composition field")
    if contract.get("kind") != "sequential":
        raise ActivationError("unsupported_composition_kind", "only sequential composition is supported")
    for key in ("max_depth", "max_steps"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ActivationError("invalid_composition_budget", key)
    if not rule.get("produced"):
        raise ActivationError("empty_composition", "a composition needs child calls")
    for call in rule["produced"]:
        ids = call.get("rule_ids")
        selector = call.get("rule_selector")
        if call.get("operation") != "run_rule" or bool(ids) == bool(selector):
            raise ActivationError("invalid_composition_call", "each step needs one run_rule reference or selector")
        if ids and (not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str)):
            raise ActivationError("invalid_composition_call", "a sequential step invokes one child")
        if selector and not isinstance(selector, dict):
            raise ActivationError("invalid_composition_selector", str(selector))
    outputs = contract.get("output_bindings", {})
    if not isinstance(outputs, dict) or set(outputs) != set(rule["io_contract"].get("outputs", [])):
        raise ActivationError("invalid_composition_outputs", "every parent output needs an explicit child return binding")
    if any(not isinstance(target, str) or not target or target == source for source, target in outputs.items()):
        raise ActivationError("invalid_composition_outputs", "parent and child output IDs must be distinct")
    for required in rule.get("required", []):
        if any(s.get("usage_policy", "reusable") != "reusable" for s in required.get("condition", [])):
            raise ActivationError("unsupported_parent_claim", "declare consumable claims on actual child operations")


class RuleCompositionRuntime:
    def __init__(self, executor):
        self.executor = executor
        self.runtime = executor.rule_runtime
        self.memory = executor.working_memory
        self.frames = self.runtime.state.setdefault("frames", {})

    def execute(self, *, matched_rules, chain_id=None, source_match=None, max_steps=64, resume=True):
        from .executor_new import ExecutorResult
        result = ExecutorResult(ok=True, chain_id=chain_id or digest(["composition", self.executor.session_id, matched_rules]), status="completed")
        budget = {"limit": max_steps, "used": 0}
        for item in matched_rules:
            child = self._run(item, ancestors=[], root_frame=None, budget=budget, depth_limit=None)
            result.steps.extend(child["steps"])
            result.errors.extend(child["errors"])
            for field in ("produced", "next_rule_ids", "next_rule_selectors", "frontend_requests", "outputs",
                          "stopped_paths", "pending_operations", "control_updates", "activation_decisions"):
                getattr(result, field).extend(child.get(field, []))
            if not child["ok"] or child["status"] in {"deferred", "waiting_composition"}:
                result.ok, result.status = child["ok"], child["status"]
                break
            if child["status"] != "completed":
                result.status = child["status"]
        return result

    def _resolve(self, call: dict, parent: dict) -> dict:
        from .persistence.rule_db import load_rule_records
        from .matcher import match_source_to_target
        path = self.executor.calculator.db_path
        if not path:
            raise ActivationError("missing_composition_registry", "composition requires the run rule registry")
        scope = parent.get("match_scope") or {"type": "event_workspace"}
        if call.get("rule_ids"):
            rules = load_rule_records(path, rule_ids=call["rule_ids"], runtime_active_only=True)
            if len(rules) != 1:
                raise ActivationError("missing_composition_child", str(call["rule_ids"]))
            return {"rule": rules[0], "match_scope": deepcopy(scope)}
        selector = deepcopy(call["rule_selector"])
        result = match_source_to_target({"source": scope, "target": {"type": "rule_db", "rule_db_path": path,
            "rule_layer": selector.get("rule_layer", "domain")}, "rule_db_path": path,
            "rule_selector": selector, "use_ann": True, "candidate_k": selector.get("candidate_k", 192)}, None, self.memory)
        candidates = [m for op in result.get("produced", []) for m in [*op.get("matched_rules", []), *op.get("partial_rules", [])]]
        if not candidates:
            raise ActivationError("no_composition_child_match", str(selector))
        candidates = [self.runtime.inspect(candidate, self.executor.control_context) for candidate in candidates]
        # Selection is a parent-declared selector, not a hardcoded child route.
        from .scheduler_new import _control_select_matched_rules
        selected, _ = _control_select_matched_rules([{"operation": "match", "matched_rules": candidates}],
                                                   rule_selector={**selector, "execute_top_k": 1}, explicit_rule_ids=[])
        return selected[0]["matched_rules"][0]

    def _run(self, item, *, ancestors, root_frame, budget, depth_limit):
        from .executor_new import ExecutorResult
        rule = item.get("rule", item)
        if is_typed_rule(rule) and rule.get("require_policy", {}).get("execution_selection") == "control":
            candidates = self.runtime.execution_candidates(item, self.executor.control_context)
            authorized = [c for c in candidates if c["candidate_id"] in self.runtime.state["authorizations"]]
            if len(authorized) > 1:
                raise ActivationError("ambiguous_execution_authorization", rule["rule_id"])
            if authorized:
                item = authorized[0]
            else:
                for candidate in candidates:
                    candidate["composition_parent_frame"] = root_frame
                    self.runtime.state["candidates"][candidate["candidate_id"]] = deepcopy(candidate)
                pending = ExecutorResult(ok=True, chain_id=candidates[0]["candidate_id"], status="waiting_composition")
                pending.pending_operations.append({"operation": "request_rule_activation_control", "candidates": candidates})
                return pending.to_dict()
        if not rule.get("composition_contract"):
            remaining = min([budget["limit"], *[self.frames[a]["end_limit"] for a in ancestors]]) - budget["used"]
            if remaining <= 0:
                raise ActivationError("composition_budget_exhausted", str(budget))
            if is_typed_rule(rule):
                current = self.runtime.inspect(item, self.executor.control_context)
                authorization = self.runtime.state["authorizations"].get(current["candidate_id"])
                needs_control = (rule.get("require_policy", {}).get("q_selection") == "control" or
                    (not current["require_evaluation"]["full_match"] and
                     rule.get("require_policy", {}).get("partial_mode") == "control"))
                if needs_control and authorization is None:
                    current["composition_parent_frame"] = root_frame
                    self.runtime.state["candidates"][current["candidate_id"]] = deepcopy(current)
                    pending = ExecutorResult(ok=True, chain_id=current["candidate_id"], status="waiting_composition")
                    pending.pending_operations.append({"operation": "request_rule_activation_control", "candidates": [current]})
                    return pending.to_dict()
            result = self.executor.execute(matched_rules=[item], max_steps=remaining, source_match={})
            budget["used"] += len(result.steps)
            return result.to_dict()
        result = ExecutorResult(ok=True, chain_id=digest(["frame", self.executor.session_id, rule["rule_id"], ancestors]), status="completed")
        decision = None
        frame = None
        try:
            validate_composition(rule)
            item = self.runtime.inspect(item, item.get("activation_context", self.executor.control_context))
            frame_id = digest([self.executor.session_id, rule["rule_id"], ancestors, item.get("match_scope"), item.get("execution_request_id", "")])
            previous_frame = self.frames.get(frame_id)
            if previous_frame and previous_frame.get("started"):
                decision = deepcopy(self.runtime.state["decisions"][previous_frame["decision_id"]])
                self.runtime.revalidate(item, decision)
                decision["status"] = "ready"
            else:
                item, decision = self.runtime.admit(item, item.get("activation_context", self.executor.control_context))
            if frame_id in ancestors or rule["rule_id"] in [self.frames[a]["rule_id"] for a in ancestors]:
                raise ActivationError("recursive_composition_cycle", rule["rule_id"])
            contract = rule["composition_contract"]
            local_depth_limit = len(ancestors) + contract["max_depth"]
            limit = min(depth_limit, local_depth_limit) if depth_limit is not None else local_depth_limit
            if len(ancestors) >= limit:
                raise ActivationError("composition_depth_exhausted", rule["rule_id"])
            frame = self.frames.get(frame_id)
            if frame is None:
                if not ancestors:
                    budget["limit"] = min(budget["limit"], contract["max_steps"])
                end_limit = min(budget["limit"], budget["used"] + contract["max_steps"],
                                *[self.frames[a]["end_limit"] for a in ancestors])
                if budget["used"] >= end_limit:
                    raise ActivationError("composition_budget_exhausted", str(budget))
                budget["used"] += 1
                frame = {"frame_id": frame_id, "rule_id": rule["rule_id"], "parent_frame_id": ancestors[-1] if ancestors else None,
                         "root_frame_id": root_frame or frame_id, "contract_digest": contract_digest(rule),
                         "decision_id": decision["decision_id"], "item": deepcopy(item), "position": 0,
                         "children": {}, "status": "running", "budget": budget, "end_limit": end_limit, "result": None}
                self.frames[frame_id] = frame
            elif frame["contract_digest"] != contract_digest(rule) or frame["decision_id"] != decision["decision_id"]:
                raise ActivationError("stale_composition_frame", frame_id)
            elif frame["status"] in {"completed", "failed"}:
                return deepcopy(frame["result"])
            else:
                # Reconnect shared budget objects after a JSON restore.
                root = self.frames[frame["root_frame_id"]]
                budget = root["budget"]
                frame["budget"] = budget
            result.chain_id = frame_id
            if not frame.get("started") and decision["status"] == "ready":
                decision = self.runtime.start(decision, item)
                frame["started"] = decision["status"] == "ready"
            result.activation_decisions.append(decision)
            if decision["status"] != "ready":
                if decision["status"] == "deferred":
                    self._wait(frame, result, "deferred")
                else:
                    result.status = decision["status"]
                return result.to_dict()
            frame["status"] = "running"
            for index in range(frame["position"], len(rule["produced"])):
                # Pin the selected child contract on first invocation.
                key = str(index)
                child_item = frame["children"].get(key)
                if child_item is None:
                    child_item = self._resolve(rule["produced"][index], item)
                    if not is_typed_rule(child_item["rule"]):
                        raise ActivationError("untyped_composition_child", "migrate child contracts before composition")
                    frame["children"][key] = deepcopy(child_item)
                parent_effect = rule["effect_contract"]
                child_effect = child_item["rule"]["effect_contract"]
                allowed = set(parent_effect["child_effects"]) if parent_effect["kind"] == "composite" else {parent_effect["kind"]}
                actual = set(child_effect["child_effects"]) if child_effect["kind"] == "composite" else {child_effect["kind"]}
                if (not actual.issubset(allowed)
                        or not set(child_effect.get("write_targets", [])).issubset(parent_effect.get("write_targets", []))):
                    raise ActivationError("undeclared_child_effect", child_item["rule"]["rule_id"])
                child = self._run(child_item, ancestors=[*ancestors, frame_id], root_frame=frame["root_frame_id"],
                                  budget=budget, depth_limit=limit)
                result.steps.extend(child["steps"])
                result.errors.extend(child["errors"])
                result.activation_decisions.extend(child.get("activation_decisions", []))
                for field in ("next_rule_ids", "next_rule_selectors", "frontend_requests", "pending_operations", "control_updates", "produced"):
                    getattr(result, field).extend(child.get(field, []))
                if not child["ok"]:
                    raise ActivationError("composition_child_failed", child["status"])
                if child["status"] in {"deferred", "waiting_composition", "needs_frontend", "needs_rule_match", "operation_pending"}:
                    self._wait(frame, result, "deferred" if child["status"] == "deferred" else "waiting_composition")
                    return result.to_dict()
                if child["status"] in {"stop_path", "output_ready", "activation_disabled", "sampled_out"}:
                    raise ActivationError("composition_child_did_not_return", child["status"])
                frame["position"] = index + 1
            # The original parent premise must still hold at return time.
            self.runtime.revalidate(item, decision)
            self._return_outputs(rule, decision)
            frame["status"] = "completed"
            result.steps.append({"rule_id": rule["rule_id"], "operation": "compose", "status": "completed",
                                 "frame_id": frame_id, "execution_decision": decision, "budget": deepcopy(budget)})
            self.runtime.complete(decision, success=True)
        except ActivationError as exc:
            result.ok, result.status = False, "composition_failed"
            result.errors.append({"rule_id": rule.get("rule_id"), "error_code": exc.code, "error": str(exc)})
            if decision and decision.get("decision_id") in self.runtime.state["decisions"]:
                self.runtime.complete(decision, success=False)
            if frame is not None:
                frame["status"] = "failed"
        if frame is not None:
            frame["result"] = result.to_dict()
        self.executor._remember(stage="rule_composition", payload_type="rule_composition",
                                payload={"frame_id": result.chain_id, "status": result.status, "errors": result.errors})
        return result.to_dict()

    def _wait(self, frame: dict, result, status: str) -> None:
        frame["status"] = "waiting"
        frame["wait_state_digest"] = digest(self.memory.event_workspaces)
        result.status = status
        decision_ids = {d.get("decision_id") for d in result.activation_decisions}
        for cid, pending in self.runtime.state["waiting"].items():
            if cid in decision_ids:
                pending["resume_operation"] = {"operation": "resume_rule_composition", "frame_id": frame["root_frame_id"]}
        frame["result"] = result.to_dict()
        self.executor._remember(stage="rule_composition", payload_type="rule_composition",
                                payload={"frame_id": frame["frame_id"], "status": "waiting", "position": frame["position"]})

    def _return_outputs(self, rule, decision) -> None:
        from .data_structure import WorkingMemoryState
        from .evidence_lifecycle import attach_derivation
        eid = decision["event_id"]
        original = self.memory.get_event_workspace(eid)
        staged = WorkingMemoryState(event_workspaces={eid: deepcopy(original)})
        for output, source in rule["composition_contract"].get("output_bindings", {}).items():
            workspace = staged.get_event_workspace(eid)
            value = next((s for s in workspace.get("spans", []) if s.get("span_id") == source), None)
            if value is None:
                raise ActivationError("missing_composition_return", source)
            returned_ev = {"bindings": {source: source}}
            check_inputs(rule, returned_ev, workspace, {"required_slots": [source]})
            item = deepcopy(value)
            item["span_id"] = output
            check_output(rule, item, decision["mode_contract"])
            item = apply_output_semantics(item, q=decision["q"], decision=decision, workspace=workspace)
            result_metadata(item, decision)
            attach_derivation(item, workspace=workspace, bindings={source: source}, rule_id=rule["rule_id"], step_id=decision["decision_id"])
            staged.apply_event_workspace_operation(eid, {"operation": "update", "target": "spans", "id": output, "item": item})
        if rule["composition_contract"].get("output_bindings"):
            self.memory.commit_calculated_workspace(eid, staged.get_event_workspace(eid), expected_version=original.get("version", 0))
