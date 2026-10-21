"""Shared rule admission, activation decisions and durable dispatch state.

All Scheduler and direct Executor callers use this runtime. Its state is plain
JSON in WorkingMemory, so frequency deadlines and probability decisions survive
serialization. It never runs Produced or mutates an event graph.
"""
from __future__ import annotations

from copy import deepcopy
import secrets
import time

from .activation_semantics import ActivationError, activation_context, evaluate_activation, number, resolve_q
from .rule_execution_contract import (check_inputs, contract_digest, digest, execution_mode,
                                       execution_workspace, is_typed_rule, partial_modes, require_evaluation,
                                       validate_rule_contract)


class RuleExecutionRuntime:
    def __init__(self, memory, calculator, *, session_id: str, clock=None):
        self.memory, self.calculator = memory, calculator
        self.session_id = session_id
        self.clock = clock or time.time
        memory.rule_runtime.setdefault(session_id, {})
        self.state.setdefault("seed", secrets.token_hex(16))
        self.state.setdefault("generation", 0)
        for key in ("decisions", "rates", "waiting", "authorizations", "candidates"):
            self.state.setdefault(key, {})
        self.calculator.bind_version_state(self.state.setdefault("function_versions", {}))
        self.calculator.bind_failure_state(self.state.setdefault("computation_failures", {}))
        self._match_cache: dict[str, dict] = {}
        self._activation_cache: dict[str, dict] = {}

    @property
    def state(self) -> dict:
        return self.memory.rule_runtime[self.session_id]

    def reset_for_task(self) -> None:
        seed, generation = self.state["seed"], self.state["generation"] + 1
        self.state.clear()
        self.state.update(seed=seed, generation=generation,
                          **{k: {} for k in ("decisions", "rates", "waiting", "authorizations", "candidates", "function_versions")})
        self.calculator.bind_version_state(self.state["function_versions"])
        self.calculator.bind_failure_state(self.state.setdefault("computation_failures", {}))
        self._match_cache.clear()
        self._activation_cache.clear()

    def revalidate(self, item: dict, decision: dict) -> None:
        current = self.inspect(item, item.get("activation_context", {}))
        evaluation = current["require_evaluation"]
        mode = execution_mode(current["rule"], evaluation, decision["execution_mode"])
        workspace = execution_workspace(self.memory.get_event_workspace(evaluation["event_id"])) if evaluation["event_id"] else {"spans": []}
        check_inputs(current["rule"], evaluation, workspace, mode)
        if (evaluation["bindings"] != decision["bindings"] or current["activation"] != decision["activation"]
                or current["resolved_q"] != decision["q"]):
            raise ActivationError("execution_inputs_changed", "reselect the rule after its bound inputs or activation change")

    def _fresh_match(self, item: dict) -> dict:
        from .matcher.matching import match_required_bundle_against_graph
        from .matcher.routing import graph_bundle_from_spec, required_bundle_from_rule

        rule = item.get("rule", item)
        scope = deepcopy(item.get("match_scope") or {"type": "event_workspace"})
        # Only graph references are accepted here, never a caller-supplied graph.
        event_ids = scope.get("event_ids")
        if not event_ids:
            event_ids = list(self.memory.event_workspaces)
        if any(eid not in self.memory.event_workspaces for eid in event_ids):
            raise ActivationError("missing_execution_event", str(event_ids))
        scope = {"type": "event_workspace", "event_ids": event_ids}
        graph = graph_bundle_from_spec(scope, None, self.memory)
        args = {"bundle_match_threshold": rule.get("require_policy", {}).get("bundle_match_threshold", 0.55)}
        key = digest({"contract": contract_digest(rule), "graph": graph, "args": args})
        if key not in self._match_cache:
            match = match_required_bundle_against_graph(required_bundle_from_rule(rule, {}), graph, args)
            if len(self._match_cache) >= 128:
                self._match_cache.clear()
            self._match_cache[key] = deepcopy(match)
        return deepcopy(self._match_cache[key])

    def inspect(self, item: dict, context: dict | None = None) -> dict:
        rule = item.get("rule", item)
        validate_rule_contract(rule)
        if not is_typed_rule(rule):
            return deepcopy(item)
        from .matcher.routing import rule_is_executable_candidate
        if not rule_is_executable_candidate(rule):
            raise ActivationError("inactive_execution_rule", str(rule.get("rule_id")))
        if self.calculator.db_path:
            from .persistence.rule_db import load_rule_records
            registered = load_rule_records(self.calculator.db_path, rule_ids=[rule["rule_id"]], limit=1)
            if registered and (not rule_is_executable_candidate(registered[0])
                               or contract_digest(registered[0]) != contract_digest(rule)):
                raise ActivationError("stale_rule_contract", rule["rule_id"])
        match = self._fresh_match(item)
        evaluation = require_evaluation(rule, match)
        event_ids = evaluation["event_ids"]
        if len(event_ids) > 1:
            raise ActivationError("ambiguous_execution_scope", "one rule invocation must bind one event workspace")
        if not event_ids:
            event_ids = list(item.get("match_scope", {}).get("event_ids", []))
        event_id = event_ids[0] if len(event_ids) == 1 else ""
        workspace = execution_workspace(self.memory.get_event_workspace(event_id)) if event_id else {"spans": [], "relations": []}
        self.calculator.bind_version_state(self.state.setdefault("function_versions", {}))
        control_inputs = {}
        if rule.get("rule_layer") == "control":
            if context and any(k in context for k in ("trigger", "workflow", "output")):
                control_inputs = {k: deepcopy(context[k]) for k in ("trigger", "workflow", "output") if k in context}
            else:
                control_inputs = deepcopy(item.get("control_inputs", {}))
        selected_mode = item.get("candidate_execution_mode")
        selection_enabled = rule.get("require_policy", {}).get("execution_selection") == "control"
        if selection_enabled:
            selected_mode = selected_mode or ("full" if evaluation["full_match"] else next(iter(partial_modes(rule, evaluation)), None))
            if selected_mode is None:
                raise ActivationError("no_feasible_execution_mode", rule["rule_id"])
            mode_contract = execution_mode(rule, evaluation, selected_mode)
            check_inputs(rule, evaluation, workspace, mode_contract)
        elif selected_mode is not None or "candidate_q_profile" in item:
            raise ActivationError("unexpected_execution_candidate", rule["rule_id"])
        spec = rule["activation_spec"].get("by_mode", {}).get(selected_mode, rule["activation_spec"])
        context = activation_context(spec, context or {})
        bound = {s["span_id"]: s for s in workspace.get("spans", []) if s.get("span_id") in evaluation["bindings"].values()}
        def activation_key():
            return digest([evaluation["contract_digest"], bound, context, evaluation["require_degree"],
                           selected_mode, self.state["function_versions"], self.calculator.function_registry_identity()])
        cache_key = activation_key()
        if cache_key not in self._activation_cache:
            activation = evaluate_activation(spec, workspace=workspace, bindings=evaluation["bindings"],
                                             context=context, require_degree=evaluation["require_degree"], calculator=self.calculator)
            if len(self._activation_cache) >= 128:
                self._activation_cache.clear()
            self._activation_cache[activation_key()] = deepcopy(activation)
        else:
            activation = deepcopy(self._activation_cache[cache_key])
        profile = item.get("candidate_q_profile")
        q = resolve_q(rule.get("q_semantics", {}), activation, profile)
        identity = {"contract": evaluation["contract_digest"], "event_id": event_id, "inputs": bound,
                    "bindings": evaluation["bindings"], "full": evaluation["full_match"],
                    "missing": evaluation["missing_ids"], "activation": activation, "q": q,
                    "execution_mode": selected_mode, "q_profile": profile,
                    "control_inputs": control_inputs,
                    "request_id": item.get("execution_request_id", self.state["generation"])}
        cid = digest(identity)
        authorization = self.state["authorizations"].get(cid)
        if authorization is not None:
            self._validate_authorization(authorization)
            q = resolve_q(rule.get("q_semantics", {}), activation, authorization.get("q_profile"))
        evaluation.update(evaluation_id=cid, state_revision=workspace.get("version", 0), event_id=event_id)
        result = {**deepcopy(item), "rule": deepcopy(rule), "match": match,
                  "require_evaluation": evaluation, "activation": activation, "resolved_q": q,
                  "candidate_id": cid, "allowed_partial_modes": partial_modes(rule, evaluation)}
        # This context is required to repeat context-dependent activation after
        # a timer or serialization boundary. Executor still recomputes inputs.
        if selection_enabled:
            result["candidate_execution_mode"] = selected_mode
            result["candidate_q_profile"] = profile
            outputs = set(mode_contract["outputs"])
            if any(entry["target"]["span_id"] not in outputs for stage in ("transform", "assessment") for entry in q.get(stage, [])):
                raise ActivationError("q_target_not_in_execution_mode", selected_mode)
        result["activation_context"] = deepcopy(context or {})
        if control_inputs:
            result["control_inputs"] = control_inputs
        result["match_scope"] = {"type": "event_workspace", "event_ids": [event_id]} if event_id else item.get("match_scope", {})
        self.state["candidates"][cid] = deepcopy(result)
        return result

    def execution_candidates(self, item: dict, context: dict | None = None) -> list[dict]:
        """Enumerate feasible mode/profile pairs before computing their A/Q."""
        rule = item.get("rule", item)
        validate_rule_contract(rule)
        evaluation = require_evaluation(rule, self._fresh_match(item))
        modes = (["full"] if evaluation["full_match"] else []) + partial_modes(rule, evaluation)
        profiles = list(rule.get("q_semantics", {}).get("profiles", {})) or [None]
        candidates = []
        for mode in modes:
            outputs = set(execution_mode(rule, evaluation, mode)["outputs"])
            for profile in profiles:
                q = rule.get("q_semantics", {})
                q = q["profiles"][profile] if profile is not None else q
                if any(entry["target"]["span_id"] not in outputs for stage in ("transform", "assessment")
                       for entry in q.get(stage, [])):
                    continue
                candidate = self.inspect({**deepcopy(item), "candidate_execution_mode": mode,
                                          "candidate_q_profile": profile}, context)
                candidates.append(candidate)
        if not candidates:
            raise ActivationError("no_feasible_execution_mode", rule["rule_id"])
        return candidates

    def authorize(self, candidate_id: str, *, mode: str, control_rule: dict, q_profile: str | None = None) -> dict:
        if control_rule.get("rule_layer") != "control":
            raise ActivationError("invalid_activation_authority", "a Control rule is required")
        owner_match = self._fresh_match({"rule": control_rule})
        if owner_match.get("ok") is not True:
            raise ActivationError("control_require_not_satisfied", control_rule["rule_id"])
        candidate = self.state["candidates"].get(candidate_id)
        if candidate is None:
            raise ActivationError("unknown_activation_candidate", candidate_id)
        current = self.inspect(candidate, candidate["activation_context"])
        if current["candidate_id"] != candidate_id:
            raise ActivationError("stale_activation_candidate", candidate_id)
        if current["rule"].get("require_policy", {}).get("execution_selection") == "control":
            if mode != current["candidate_execution_mode"] or q_profile != current.get("candidate_q_profile"):
                raise ActivationError("candidate_mode_mismatch", candidate_id)
        mode_contract = execution_mode(current["rule"], current["require_evaluation"], mode)
        check_inputs(current["rule"], current["require_evaluation"],
                     execution_workspace(self.memory.get_event_workspace(current["require_evaluation"]["event_id"])), mode_contract)
        selected_q = resolve_q(current["rule"].get("q_semantics", {}), current["activation"], q_profile)
        auth = {"candidate_id": candidate_id, "execution_mode": mode,
                "control_rule_id": control_rule["rule_id"], "control_contract_digest": contract_digest(control_rule),
                "q_profile": q_profile}
        self._validate_authorization(auth)
        self.state["authorizations"][candidate_id] = auth
        return {**current, "execution_authorization": deepcopy(auth), "resolved_q": selected_q}

    def _validate_authorization(self, authorization: dict) -> None:
        """A retained decision cannot outlive its stored Control contract."""
        from .matcher.routing import rule_is_executable_candidate
        from .persistence.rule_db import load_rule_records
        if not self.calculator.db_path:
            raise ActivationError("missing_activation_registry", "Control authorization needs a stored rule")
        owners = load_rule_records(self.calculator.db_path, rule_ids=[authorization["control_rule_id"]], limit=1)
        if (len(owners) != 1 or owners[0].get("rule_layer") != "control"
                or not rule_is_executable_candidate(owners[0])
                or contract_digest(owners[0]) != authorization["control_contract_digest"]):
            raise ActivationError("stale_activation_authority", authorization["control_rule_id"])
        if self._fresh_match({"rule": owners[0]}).get("ok") is not True:
            raise ActivationError("control_require_not_satisfied", authorization["control_rule_id"])

    def admit(self, item: dict, context: dict | None = None) -> tuple[dict, dict]:
        """Recheck state and return one auditable dispatch decision, never F."""
        rule = item.get("rule", item)
        self.calculator.bind_version_state(self.state.setdefault("function_versions", {}))
        if not is_typed_rule(rule):
            return self._admit_legacy(item)
        item = self.inspect(item, context if context is not None else item.get("activation_context", {}))
        ev, cid = item["require_evaluation"], item["candidate_id"]
        policy = rule.get("require_policy", {})
        auth = self.state["authorizations"].get(cid)
        if (policy.get("q_selection") == "control" or policy.get("execution_selection") == "control") and auth is None:
            raise ActivationError("q_control_decision_required", cid)
        if policy.get("execution_selection") == "control":
            name, source = auth["execution_mode"], {"kind": "control_rule", **auth}
        elif ev["full_match"]:
            name, source = "full", {"kind": "full_require"}
        elif policy.get("partial_mode") == "declared":
            name, source = policy["execution_mode"], {"kind": "rule_declaration"}
        else:
            if auth is None:
                raise ActivationError("partial_control_decision_required", cid)
            name, source = auth["execution_mode"], {"kind": "control_rule", **auth}
        if auth is not None:
            source = {"kind": "control_rule", **auth}
        profile = (auth or {}).get("q_profile") or rule.get("q_semantics", {}).get("default_profile")
        if profile is not None:
            cid = digest([cid, profile])
            self.state["candidates"][cid] = deepcopy(item)
        mode = execution_mode(rule, ev, name)
        selected_outputs = set(mode.get("outputs", []))
        for stage in ("transform", "assessment"):
            if any(q["target"]["span_id"] not in selected_outputs for q in item["resolved_q"].get(stage, [])):
                raise ActivationError("q_target_not_in_execution_mode", name)
        workspace = execution_workspace(self.memory.get_event_workspace(ev["event_id"])) if ev["event_id"] else {"spans": []}
        check_inputs(rule, ev, workspace, mode)
        decision = {"decision_id": cid, "rule_id": rule["rule_id"], "contract_digest": ev["contract_digest"],
                    "execution_mode": name, "result_scope": mode["result_scope"], "produced_indices": mode["produced_indices"],
                    "require_degree": ev["require_degree"], "missing_ids": ev["missing_ids"], "bindings": ev["bindings"],
                    "event_id": ev["event_id"], "state_revision": ev["state_revision"], "source": source,
                    "q": item["resolved_q"], "activation": item["activation"], "status": "ready", "mode_contract": mode}
        decision["q_profile"] = profile
        previous = self.state["decisions"].get(cid)
        if previous and previous["status"] == "completed":
            return item, {**deepcopy(previous), "status": "already_completed"}
        now = number(self.clock(), "clock")
        rate_keys = []
        for entry in item["resolved_q"].get("dispatch", []):
            value, kind = entry.get("value"), entry["type"]
            if kind == "boolean":
                if isinstance(value, bool):
                    execute = value
                elif "threshold" in entry:
                    execute = number(value) >= number(entry["threshold"])
                else:
                    raise ActivationError("missing_activation_threshold", entry["id"])
                if not execute:
                    decision["status"] = "activation_disabled"
            elif kind == "probability":
                value = number(value)
                if not 0 <= value <= 1:
                    raise ActivationError("invalid_execution_probability", entry["id"])
                # Counter-based randomness is stable across retry and restore;
                # the persisted run seed is independent of any model parameter.
                sample = int(digest([self.state["seed"], cid, entry["id"]])[:13], 16) / (16 ** 13)
                decision.setdefault("samples", {})[entry["id"]] = sample
                if sample >= value:
                    decision["status"] = "sampled_out"
            else:
                rate = number(value)
                if rate < 0:
                    raise ActivationError("invalid_execution_frequency", entry["id"])
                if rate == 0:
                    decision["status"] = "activation_disabled"
                    continue
                key = digest([rule["rule_id"], ev["contract_digest"], ev["event_id"], entry["id"]])
                rate_keys.append(key)
                last = self.state["rates"].get(key)
                seconds = {"calls/s": 1, "calls/min": 60, "calls/hour": 3600}[entry["unit"]]
                decision.setdefault("frequency_periods", {})[key] = seconds / rate
                if last is not None:
                    due = last + seconds / rate
                    if now < due:
                        decision["not_before"] = max(decision.get("not_before", due), due)
        decision["rate_keys"] = rate_keys
        if decision["status"] == "ready" and "not_before" in decision:
            decision["status"] = "deferred"
        self.state["decisions"][cid] = deepcopy(decision)
        if decision["status"] == "deferred":
            self.state["waiting"][cid] = {"not_before": decision["not_before"], "rule_match": deepcopy(item)}
        return item, decision

    def start(self, decision: dict, item: dict) -> dict:
        """Reserve actual call frequency at start, including failed calls."""
        if not decision.get("decision_id"):
            return decision
        with self.memory._event_claim_lock:
            now = number(self.clock(), "clock")
            due = max((self.state["rates"].get(key, now - period) + period
                       for key, period in decision.get("frequency_periods", {}).items()), default=now)
            if due > now:
                decision.update(status="deferred", not_before=due)
                self.state["waiting"][decision["decision_id"]] = {"not_before": due, "rule_match": deepcopy(item)}
            else:
                for key in decision.get("rate_keys", []):
                    self.state["rates"][key] = now
                decision["started_at"] = now
            self.state["decisions"][decision["decision_id"]] = deepcopy(decision)
        return decision

    def _admit_legacy(self, item: dict) -> tuple[dict, dict]:
        rule = item.get("rule", item)
        validate_rule_contract(rule)
        match = item.get("match")
        if not isinstance(match, dict) or "ok" not in match:
            match = self._fresh_match(item)
            item = {**deepcopy(item), "match": match}
        if match.get("ok") is not True:
            raise ActivationError("require_not_satisfied", str(rule.get("rule_id")))
        return item, {"status": "ready", "execution_mode": "full", "result_scope": "full",
                       "produced_indices": list(range(len(rule.get("produced", []))))}

    def complete(self, decision: dict, *, success: bool) -> None:
        cid = decision.get("decision_id")
        if not cid:
            return
        saved = self.state["decisions"].get(cid)
        if saved is None:
            raise ActivationError("unknown_execution_decision", cid)
        saved["status"] = "completed" if success else "failed"
        selection_id = saved.get("source", {}).get("selection_id")
        if selection_id:
            self.state["selections"][selection_id]["execution_outcome"] = {
                "decision_id": cid, "status": saved["status"],
                "event_id": saved.get("event_id"), "outputs": saved["mode_contract"]["outputs"]}
        self.state["waiting"].pop(cid, None)

    def register_control_action(self, rule: dict, action: dict) -> dict:
        if rule.get("rule_layer") != "control":
            raise ActivationError("invalid_activation_authority", str(rule.get("rule_id")))
        aid = digest([contract_digest(rule), action, len(self.state.setdefault("control_actions", {}))])
        self.state["control_actions"][aid] = {"control_contract_digest": contract_digest(rule),
                                             "action": deepcopy(action), "control_rule_id": rule["rule_id"]}
        return {**deepcopy(action), "control_action_id": aid}

    def ready_operations(self) -> list[dict]:
        now = number(self.clock(), "clock")
        ready = []
        for cid, pending in list(self.state["waiting"].items()):
            if pending["not_before"] <= now:
                ready.append(deepcopy(pending.get("resume_operation")) if pending.get("resume_operation") else
                             {"operation": "match", "matched_rules": [deepcopy(pending["rule_match"])]})
                del self.state["waiting"][cid]
        for cid, pending in list(self.state.get("control_waiting", {}).items()):
            if digest(self.memory.event_workspaces) != pending["state_digest"]:
                root = pending["rule_match"].get("composition_parent_frame")
                ready.append({"operation": "resume_rule_composition", "frame_id": root} if root else
                             {"operation": "match", "partial_rules": [deepcopy(pending["rule_match"])]})
                del self.state["control_waiting"][cid]
        queued_frames = {op.get("frame_id") for op in ready if op.get("operation") == "resume_rule_composition"}
        timed_frames = {p.get("resume_operation", {}).get("frame_id") for p in self.state["waiting"].values()}
        for frame in self.state.get("frames", {}).values():
            if (frame["status"] == "waiting" and frame["frame_id"] == frame["root_frame_id"]
                    and frame["frame_id"] not in queued_frames | timed_frames
                    and frame["wait_state_digest"] != digest(self.memory.event_workspaces)):
                ready.append({"operation": "resume_rule_composition", "frame_id": frame["frame_id"]})
        return ready

    def has_waiting(self) -> bool:
        return bool(self.state["waiting"] or self.state.get("control_waiting") or
                    any(f.get("status") == "waiting" for f in self.state.get("frames", {}).values()))

    def waiting_until(self) -> float | None:
        values = [p["not_before"] for p in self.state["waiting"].values()]
        return min(values) if values else None
