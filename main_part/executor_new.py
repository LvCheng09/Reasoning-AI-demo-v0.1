"""Rule-chain executor.

This module executes already-matched rules. It does not choose strategy, run
matching, call the frontend, or decide final output permission. Those decisions
belong to scheduler/controller.
"""

from __future__ import annotations
from .common.normalize import safe_float as shared_safe_float

from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any

from .data_structure import WorkingMemoryEntry, WorkingMemoryState
from .identity import stable_text_id
from .retrieval import knowledge_record_to_event_context
from .structured_arithmetic import StructuredComputationError
from .calculator import Calculator
from .rule_selector import child_rule_selectors, normalize_rule_selectors
from .rule_path_candidates import choose_candidate_rule_path
from .control_state import control_state_update, validate_task_result
from .activation_semantics import ActivationError, apply_output_semantics
from .rule_execution_contract import check_effect, check_output, execution_workspace, is_typed_rule, result_metadata, span_output_items
from .rule_execution_runtime import RuleExecutionRuntime


GRAPH_OPERATIONS = {"add", "update", "delete", "merge", "split"}
CONTROL_OPERATIONS = {"control_decide", "run_rule", "request_frontend_patch", "output", "stop_path"}
DEFAULT_RULE_CHAIN_MAX_STEPS = 64


@dataclass
class ExecutorStep:
    chain_id: str
    step_id: str
    rule_id: str
    rule_layer: str
    operation: str
    produced: dict[str, Any]
    status: str
    event_id: str = ""
    required_match: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = ""
    execution_decision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "step_id": self.step_id,
            "rule_id": self.rule_id,
            "rule_layer": self.rule_layer,
            "operation": self.operation,
            "produced": self.produced,
            "status": self.status,
            "event_id": self.event_id,
            "required_match": self.required_match,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "execution_decision": self.execution_decision,
        }


@dataclass
class ExecutorResult:
    ok: bool
    chain_id: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    produced: list[dict[str, Any]] = field(default_factory=list)
    next_rule_ids: list[str] = field(default_factory=list)
    next_rule_selectors: list[dict[str, Any]] = field(default_factory=list)
    frontend_requests: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    stopped_paths: list[dict[str, Any]] = field(default_factory=list)
    pending_operations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    claim_transaction: dict[str, Any] = field(default_factory=dict)
    control_updates: list[dict[str, Any]] = field(default_factory=list)
    activation_decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "chain_id": self.chain_id,
            "status": self.status,
            "steps": self.steps,
            "produced": self.produced,
            "next_rule_ids": self.next_rule_ids,
            "next_rule_selectors": self.next_rule_selectors,
            "frontend_requests": self.frontend_requests,
            "outputs": self.outputs,
            "stopped_paths": self.stopped_paths,
            "pending_operations": self.pending_operations,
            "errors": self.errors,
            "claim_transaction": self.claim_transaction,
            "control_updates": self.control_updates,
            "activation_decisions": self.activation_decisions,
        }


class OutputTemplateBindingError(ValueError):
    def __init__(self, missing_keys: list[str]) -> None:
        self.missing_keys = list(dict.fromkeys(str(key) for key in missing_keys if str(key)))
        super().__init__(f"output template variables are not bound: {', '.join(self.missing_keys)}")


def execute_rule_chain(
    *,
    matched_rules: list[dict[str, Any]],
    working_memory: WorkingMemoryState,
    session_id: str,
    chain_id: str | None = None,
    source_match: dict[str, Any] | None = None,
    control_context: dict[str, Any] | None = None,
    max_steps: int = DEFAULT_RULE_CHAIN_MAX_STEPS,
    resume: bool = True,
    calculator: Calculator | None = None,
    control_operation_handler=None,
    rule_runtime: RuleExecutionRuntime | None = None,
) -> dict[str, Any]:
    """Execute matched rule.produced items as a bounded logic chain.

    Input is expected to come from matcher output. Every step is recorded in
    working memory. Blocking operations return to scheduler instead of being
    handled here.
    """

    executor = RuleChainExecutor(
        working_memory=working_memory,
        session_id=session_id,
        control_context=control_context,
        calculator=calculator,
        control_operation_handler=control_operation_handler,
        rule_runtime=rule_runtime,
    )
    return executor.execute(
        matched_rules=matched_rules,
        chain_id=chain_id,
        source_match=source_match,
        max_steps=max_steps,
        resume=resume,
    ).to_dict()


class RuleChainExecutor:
    def __init__(
        self,
        *,
        working_memory: WorkingMemoryState,
        session_id: str,
        control_context: dict[str, Any] | None = None,
        calculator: Calculator | None = None,
        control_operation_handler=None,
        rule_runtime: RuleExecutionRuntime | None = None,
    ) -> None:
        self.working_memory = working_memory
        self.session_id = str(session_id or "default_session")
        self.control_context = deepcopy(control_context or {})
        self.calculator = calculator or Calculator()
        self.control_operation_handler = control_operation_handler
        self.rule_runtime = rule_runtime or RuleExecutionRuntime(working_memory, self.calculator, session_id=self.session_id)

    def execute(
        self,
        *,
        matched_rules: list[dict[str, Any]],
        chain_id: str | None = None,
        source_match: dict[str, Any] | None = None,
        max_steps: int = DEFAULT_RULE_CHAIN_MAX_STEPS,
        resume: bool = True,
    ) -> ExecutorResult:
        if any(_rule_from_match(item).get("composition_contract") for item in matched_rules):
            from .rule_composition import RuleCompositionRuntime
            return RuleCompositionRuntime(self).execute(matched_rules=matched_rules, chain_id=chain_id,
                source_match=source_match, max_steps=max_steps, resume=resume)
        normalized_chain_id = chain_id or _stable_id("chain", self.session_id, len(self.working_memory.entries), _utc_now())
        chain_state = self._load_chain_state(normalized_chain_id) if resume else {}
        start_index = int(chain_state.get("executed_step_count", 0) or 0)
        result = ExecutorResult(ok=True, chain_id=normalized_chain_id, status="completed")
        candidate_path = (source_match or {}).get("candidate_path") if isinstance((source_match or {}).get("candidate_path"), dict) else {}
        claims = _as_dict_list(candidate_path.get("claims"))
        candidate_event_ids = _string_list(candidate_path.get("event_ids"))
        reservation_id = f"{normalized_chain_id}:event-transaction" if candidate_path else ""
        if reservation_id:
            reservation = self.working_memory.begin_event_workspace_transaction(
                event_ids=candidate_event_ids,
                claims=claims,
                reservation_id=reservation_id,
            )
            result.claim_transaction = reservation
            if not reservation.get("ok"):
                result.ok = False
                result.status = str(
                    reservation.get("status") or "condition_claim_conflict"
                )
                result.errors.append(dict(reservation))
                self._finish(result)
                return result
            if reservation.get("status") == "no_claims":
                reservation_id = ""
        self._remember(
            stage="executor_chain_start",
            payload_type="executor_chain",
            payload={
                "chain_id": normalized_chain_id,
                "previous_state": chain_state,
                "source_match": dict(source_match or {}),
                "matched_rule_count": len(matched_rules or []),
                "max_steps": max_steps,
                "resume": resume,
            },
        )
        self._update_chain_state(
            normalized_chain_id,
            {
                "chain_id": normalized_chain_id,
                "status": "running",
                "session_id": self.session_id,
                "source_match": dict(source_match or chain_state.get("source_match") or {}),
                "executed_step_count": start_index,
                "started_at": chain_state.get("started_at") or _utc_now(),
                "updated_at": _utc_now(),
                "steps": list(chain_state.get("steps", []) or []),
                "continuation": dict(chain_state.get("continuation") or {}),
            },
        )

        step_count = start_index
        for rule_match in _as_dict_list(matched_rules):
            if step_count >= max_steps:
                result.status = "max_steps_reached"
                result.ok = False
                break
            try:
                rule_match, decision = self.rule_runtime.admit(rule_match, self.control_context)
            except ActivationError as exc:
                step = self._record_blocked_rule(normalized_chain_id, rule_match, exc.code, error=str(exc))
                result.steps.append(step.to_dict())
                result.errors.append({"rule_id": step.rule_id, "error": str(exc), "error_code": exc.code})
                result.ok = False
                result.status = "rule_admission_failed"
                self._finalize_claim_transaction(result, reservation_id)
                self._finish(result)
                return result
            rule = _rule_from_match(rule_match)
            if decision["status"] == "ready":
                decision = self.rule_runtime.start(decision, rule_match)
            result.activation_decisions.append(deepcopy(decision))
            if decision["status"] != "ready":
                step = self._record_blocked_rule(normalized_chain_id, rule_match, decision["status"],
                                                 status=decision["status"], decision=decision)
                result.steps.append(step.to_dict())
                result.status = decision["status"]
                continue
            rule_match["execution_decision"] = decision
            produced_items = [rule["produced"][i] for i in decision["produced_indices"]]
            if not produced_items:
                step = self._record_blocked_rule(normalized_chain_id, rule_match, "rule_has_no_produced")
                result.steps.append(step.to_dict())
                continue
            for produced in produced_items:
                if step_count >= max_steps:
                    result.status = "max_steps_reached"
                    result.ok = False
                    break
                step_count += 1
                step = self._execute_produced(
                    chain_id=normalized_chain_id,
                    step_index=step_count,
                    rule_match=rule_match,
                    produced=produced,
                )
                result.steps.append(step.to_dict())
                result.produced.append(dict(produced))
                self._accumulate_result(result, step)
                if step.status in {"needs_frontend", "output_ready", "stop_path", "error"}:
                    self.rule_runtime.complete(decision, success=step.status not in {"error", "stop_path"})
                    result.status = step.status
                    result.ok = step.status != "error"
                    self._finalize_claim_transaction(result, reservation_id)
                    self._finish(result)
                    return result
            self.rule_runtime.complete(decision, success=result.ok and result.status != "max_steps_reached")
            if result.status == "max_steps_reached":
                break

        if result.frontend_requests:
            result.status = "needs_frontend"
        elif result.outputs:
            result.status = "output_ready"
        elif result.stopped_paths:
            result.status = "stop_path"
        elif result.next_rule_ids or result.next_rule_selectors:
            result.status = "needs_rule_match"
        elif result.pending_operations:
            result.status = "operation_pending"
        elif not result.steps:
            result.status = "no_executable_rules"
            result.ok = False
        self._finalize_claim_transaction(result, reservation_id)
        self._finish(result)
        return result

    def _finalize_claim_transaction(self, result: ExecutorResult, reservation_id: str) -> None:
        if not reservation_id:
            return
        if result.ok and result.status != "max_steps_reached":
            result.claim_transaction = self.working_memory.commit_event_workspace_transaction(
                reservation_id=reservation_id,
            )
            return
        result.claim_transaction = self.working_memory.rollback_event_workspace_transaction(
            reservation_id=reservation_id,
        )

    def _execute_produced(
        self,
        *,
        chain_id: str,
        step_index: int,
        rule_match: dict[str, Any],
        produced: dict[str, Any],
    ) -> ExecutorStep:
        rule = _rule_from_match(rule_match)
        produced = dict(produced)
        produced.setdefault("rule_id", str(rule.get("rule_id") or ""))
        operation = _produced_operation(produced)
        step = ExecutorStep(
            chain_id=chain_id,
            step_id=f"{chain_id}_step_{step_index:04d}",
            rule_id=str(rule.get("rule_id") or ""),
            rule_layer=str(rule.get("rule_layer") or ""),
            operation=operation,
            produced=dict(produced),
            status="completed",
            event_id=_event_id_from_operation(produced, rule_match),
            required_match=dict(rule_match.get("match") or {}),
            created_at=_utc_now(),
            execution_decision=deepcopy(rule_match.get("execution_decision", {})),
        )
        event_before = deepcopy(self.working_memory.get_event_workspace(step.event_id)) if step.event_id else None
        try:
            if is_typed_rule(rule):
                self.rule_runtime.revalidate(rule_match, step.execution_decision)
                check_effect(rule, produced)
                step.required_match = {**deepcopy(step.required_match),
                                       "require_evaluation": deepcopy(rule_match["require_evaluation"])}
            self.control_context["binding"] = {
                "event_id": step.event_id, "slot_bindings": _slot_bindings(rule_match),
            }
            if operation in {"output", "stop_path"} and "task_result" in produced:
                if rule.get("rule_layer") != "control":
                    raise ValueError("task results may only be declared by Control rules")
                produced["task_result"] = {
                    **validate_task_result(produced["task_result"]),
                    "control_rule_id": step.rule_id,
                }
            materialized_knowledge = [] if is_typed_rule(rule) else self._materialize_bound_knowledge(step, rule_match)
            target_workspace = str(produced.get("target_workspace") or "").strip()
            if operation in GRAPH_OPERATIONS and target_workspace == "memory_workspace":
                step.result = self.working_memory.apply_memory_workspace_operation(
                    produced,
                    rule_id=step.rule_id,
                    session_id=self.session_id,
                )
            elif operation in GRAPH_OPERATIONS:
                self._execute_graph_operation(step)
            elif operation == "run_rule":
                step.status = "needs_rule_match"
                step.result = {
                    "next_rule_ids": _string_list(produced.get("rule_ids")),
                    "next_rule_selectors": _selector_list(produced, rule_match=rule_match),
                }
            elif operation == "control_decide":
                self._execute_control_decision(step, rule_match)
            elif operation == "update_control_state":
                update = control_state_update(
                    rule, _materialize_control_value(produced.get("values"), self.control_context)
                )
                self.control_context.setdefault("workflow", {}).update(deepcopy(update["values"]))
                step.status = "control_decision"
                step.result = {"control_updates": [update]}
            elif operation == "request_frontend_patch":
                step.status = "needs_frontend"
                step.result = {"frontend_request": dict(produced)}
            elif operation == "output":
                step.status = "output_ready"
                step.result = {
                    "output": _render_output_from_template_context(
                        dict(produced),
                        working_memory=self.working_memory,
                        event_id=step.event_id,
                        slot_bindings=_slot_bindings(rule_match),
                    )
                }
            elif operation == "stop_path":
                step.status = "stop_path"
                step.result = {"stopped_path": dict(produced)}
            elif operation:
                step.status = "operation_pending"
                step.result = {"pending_operation": dict(produced)}
            else:
                step.status = "error"
                step.error = "produced operation is missing"
            if materialized_knowledge:
                step.result = dict(step.result or {})
                step.result["materialized_knowledge"] = materialized_knowledge
        except OutputTemplateBindingError as exc:
            step.status = "error"
            step.error = str(exc)
            step.result = {
                "status": "output_template_binding_failed",
                "missing_template_keys": exc.missing_keys,
            }
        except StructuredComputationError as exc:
            step.status = "error"
            step.error = str(exc)
            step.result = {
                "status": "structured_computation_failed",
                "error_code": exc.code,
                "details": exc.details,
            }
        except Exception as exc:
            step.status = "error"
            step.error = repr(exc)
        if event_before is not None and operation in GRAPH_OPERATIONS and step.status in {"error", "stop_path"}:
            self.working_memory.event_workspaces[step.event_id] = event_before
        self._record_step(step)
        return step

    def _materialize_bound_knowledge(
        self,
        step: ExecutorStep,
        rule_match: dict[str, Any],
    ) -> list[dict[str, str]]:
        if not step.event_id:
            return []
        bound_span_ids = _bound_span_ids(rule_match)
        if not bound_span_ids:
            return []
        workspace = self.working_memory.get_event_workspace(step.event_id)
        context_candidates: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for context in _as_dict_list(workspace.get("context_items")):
            if str(context.get("source") or "").strip() != "knowledge_db":
                continue
            context_id = str(context.get("context_id") or "").strip()
            payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
            for position, raw_span in enumerate(_as_dict_list(payload.get("spans")), start=1):
                original_id = str(raw_span.get("span_id") or raw_span.get("id") or position)
                context_candidates[f"context_{context_id}_{original_id}"] = (context, raw_span)

        materialized: list[dict[str, str]] = []
        for bound_span_id in sorted(bound_span_ids):
            candidate = context_candidates.get(bound_span_id)
            if candidate is None:
                continue
            context, raw_span = candidate
            context_id = str(context.get("context_id") or "")
            source_record_id = str(context.get("source_record_id") or raw_span.get("knowledge_id") or "")
            existing = next(
                (
                    span
                    for span in _as_dict_list(workspace.get("spans"))
                    if str(span.get("materialized_context_id") or "") == context_id
                    and str(span.get("source_record_id") or span.get("knowledge_id") or "") == source_record_id
                ),
                None,
            )
            if existing is not None:
                materialized.append(
                    {
                        "context_id": context_id,
                        "knowledge_id": source_record_id,
                        "span_id": str(existing.get("span_id") or ""),
                    }
                )
                continue

            normalized = knowledge_record_to_event_context(
                {
                    **raw_span,
                    "knowledge_id": source_record_id,
                    "id": source_record_id,
                }
            )["spans"][0]
            event_span_id = stable_text_id(
                "knowledge_evidence",
                step.event_id,
                context_id,
                source_record_id,
            )
            evidence_span = {
                **normalized,
                "span_id": event_span_id,
                "source": "knowledge_db",
                "source_record_id": source_record_id,
                "materialized_context_id": context_id,
                "materialized_by_rule_id": step.rule_id,
                "materialized_at_step_id": step.step_id,
            }
            workspace = self.working_memory.apply_event_workspace_operation(
                step.event_id,
                {
                    "operation": "add",
                    "target": "spans",
                    "item": evidence_span,
                },
            )
            materialized.append(
                {
                    "context_id": context_id,
                    "knowledge_id": source_record_id,
                    "span_id": event_span_id,
                }
            )
        return materialized

    def _execute_control_decision(
        self,
        step: ExecutorStep,
        rule_match: dict[str, Any],
    ) -> None:
        rule = _rule_from_match(rule_match)
        if str(step.rule_layer or "").strip().lower() != "control":
            step.status = "error"
            step.error = "control_decide is only valid for control rules"
            return
        preparations = step.produced.get('before_decide', [])
        if not isinstance(preparations, list) or any(not isinstance(op, dict) for op in preparations):
            raise ValueError('before_decide must be a list of declared Control operations')
        for operation in preparations:
            if is_typed_rule(rule):
                check_effect(rule, operation)
            if self.control_operation_handler is None:
                raise ValueError('This Control decision requires an operation handler')
            self.control_context = deepcopy(self.control_operation_handler({**operation, 'rule_id': step.rule_id}))
        branches = _as_dict_list(step.produced.get("branches"))
        selected_branch: dict[str, Any] | None = None
        selected_index = -1
        evaluations: list[dict[str, Any]] = []
        for index, branch in enumerate(branches):
            predicate = branch.get("when")
            matched = _control_predicate_matches(predicate, self.control_context)
            evaluations.append({"branch_index": index, "matched": matched})
            if matched:
                selected_branch = branch
                selected_index = index
                break
        raw_actions: Any
        if selected_branch is not None:
            raw_actions = selected_branch.get("then")
        else:
            raw_actions = step.produced.get("default")
        state_update = (selected_branch or {}).get("state_update", step.produced.get("state_update"))
        if state_update is not None:
            raw_actions = [
                {"operation": "update_control_state", "values": state_update},
                *_as_dict_list(raw_actions),
            ]
        actions = _materialize_control_actions(raw_actions, self.control_context)
        if is_typed_rule(rule):
            for action in actions:
                check_effect(rule, action)
        alignment_reports = []
        expanded_actions = []
        for action in actions:
            if action.get("operation") == "select_alignment_targets":
                from .semantic_alignment import select_requirements
                selected = select_requirements(self.control_context.get("trigger", {}).get("alignment_requirements", []), action["policy"])
                values = {"alignment_requirements": selected}
                self.control_context.setdefault("workflow", {}).update(values)
                expanded_actions.append({"operation": "update_control_state", "values": values})
                continue
            if action.get("operation") != "align_event_spans":
                if (action.get("operation") == "retrieve_knowledge"
                        and action.get("schema_for_alignment_requirements")):
                    action = deepcopy(action)
                    action["alignment_requirements"] = deepcopy(
                        self.control_context.get("workflow", {}).get(
                            "alignment_requirements",
                            self.control_context.get("trigger", {}).get("alignment_requirements", []),
                        )
                    )
                expanded_actions.append(action)
                continue
            from .semantic_alignment import align_event_spans
            workflow = self.control_context.get("workflow", {})
            requirements = workflow.get("alignment_requirements", self.control_context.get("trigger", {}).get("alignment_requirements", []))
            report = align_event_spans(
                self.working_memory,
                requirements,
                action.get("knowledge_ids"), control_rule_id=step.rule_id)
            alignment_reports.append(report)
            if report["retry_rule_ids"]:
                expanded_actions.append({"operation": "run_rule", "rule_ids": report["retry_rule_ids"]})
            else:
                from .semantic_alignment import frontend_questions
                workflow = self.control_context.get("workflow", {})
                requests, attempts = frontend_questions(
                    self.working_memory, requirements,
                    workflow.get("alignment_frontend_attempts", {}),
                    max_requests=int(action.get("max_frontend_requests", 0)),
                    min_score=float(action.get("frontend_min_core_score", 1.0)),
                    max_questions=int(action.get("max_frontend_questions", 1)),
                    patch_scope=action.get("frontend_patch_scope", "attributes_only"))
                if requests:
                    expanded_actions.extend([
                        {"operation": "update_control_state", "values": {"alignment_frontend_attempts": attempts}},
                        *requests, *_as_dict_list(action.get("after_frontend"))])
                else:
                    expanded_actions.extend(_as_dict_list(action.get("on_unresolved")))
        actions = expanded_actions
        actions = [self.rule_runtime.register_control_action(rule, action)
                   if action.get("operation") == "decide_rule_activation" else action for action in actions]
        if not actions:
            step.status = "error"
            step.error = "control_decide selected no executable action"
            step.result = {
                "selected_branch_index": selected_index,
                "branch_evaluations": evaluations,
            }
            return
        step.status = "control_decision"
        step.result = {
            "selected_branch_index": selected_index,
            "branch_evaluations": evaluations,
            "selected_actions": actions,
            "semantic_alignment": alignment_reports,
            **_classify_control_actions(
                actions,
                rule_match=rule_match,
                control_context=self.control_context,
            ),
        }

    def _execute_graph_operation(self, step: ExecutorStep) -> None:
        if not step.event_id:
            step.status = "error"
            step.error = "graph operation requires event_id"
            return
        conclusion_spans = _as_dict_list(step.produced.get("conclusion"))
        explicit_item = step.produced.get("item")
        typed = bool(step.execution_decision.get("decision_id"))
        if typed:
            conclusion_spans = span_output_items(step.produced)
        if conclusion_spans and (typed or (step.operation == "add" and not isinstance(explicit_item, dict))):
            workspace: dict[str, Any] | None = None
            original = self.working_memory.get_event_workspace(step.event_id)
            expected_version = int(original.get("version", 0))
            staged_memory = WorkingMemoryState()
            staged_memory.event_workspaces[step.event_id] = deepcopy(original)
            calculations: dict[str, dict[str, Any]] = {}
            added_span_ids: list[str] = []
            computations: list[dict[str, Any]] = []
            bindings = _slot_bindings(step.required_match)
            for index, conclusion_span in enumerate(conclusion_spans):
                item = dict(conclusion_span)
                item.setdefault("span_id", f"{step.step_id}_conclusion_{index + 1}")
                current_workspace = staged_memory.get_event_workspace(step.event_id)
                if typed and step.operation == "update":
                    existing = next((s for s in current_workspace.get("spans", [])
                                     if s.get("span_id") == item["span_id"]), {})
                    # An update inherits node descriptors, not the old program
                    # or the evidence generated for its previous value.
                    descriptors = deepcopy(existing)
                    for key in ("calculation", "expression", "value_policy", "result_unit",
                                "computation", "runtime_value_resolution", "computed_by_rule_id",
                                "computed_at_step_id", "derived_by_rule_id", "derived_at_step_id",
                                "activation_fusion", "activation_effects", "confidence",
                                "source_reliability", "authority_verified", "execution_scope", "rule_execution"):
                        descriptors.pop(key, None)
                    metadata = dict(descriptors.get("metadata") or {})
                    for key in ("derivation_supports", "derived_from_span_ids", "evidence_active",
                                "confidence_semantics", "source_reliability", "authority_verified",
                                "measurement_basis", "semantic_alignment"):
                        metadata.pop(key, None)
                    descriptors["metadata"] = metadata
                    item = {**descriptors, **item}
                computation_workspace = execution_workspace(current_workspace) if step.execution_decision.get("decision_id") else current_workspace

                calculation = item.get("calculation")
                if calculation is not None:
                    if not isinstance(calculation, dict) or "expression" in item or "value_policy" in item:
                        raise StructuredComputationError("conflicting_calculation", "calculation cannot coexist with legacy fields")
                    # Multiple named outputs in one produced share one actual call.
                    import json
                    cache_key = json.dumps({k: v for k, v in calculation.items() if k != "output_name"}, sort_keys=True)
                    input_key = json.dumps([
                        s for s in computation_workspace.get("spans", [])
                        if s.get("span_id") in bindings.values()
                    ], sort_keys=True)
                    cache_key += input_key
                    if cache_key not in calculations:
                        calculations[cache_key] = self.calculator.calculate(
                            calculation, workspace=computation_workspace, slot_bindings=bindings,
                        )
                    computation = calculations[cache_key]
                    output = computation["outputs"].get(calculation.get("output_name"))
                    if output is None:
                        raise StructuredComputationError("unknown_calculation_output", str(calculation.get("output_name")))
                    item.update(value=output["value"], unit=output.get("unit", ""), computation=computation,
                                computed_by_rule_id=step.rule_id, computed_at_step_id=step.step_id)
                    _attach_derived_evidence_metadata(item, computation=computation,
                                                     spans=computation_workspace.get("spans", []), rule_id=step.rule_id)
                    computations.append(computation)

                value_policy = item.get("value_policy")
                if isinstance(value_policy, dict):
                    resolution = self.calculator.calculate(
                        {"kind": "value_policy", "value_policy": value_policy},
                        workspace=computation_workspace, slot_bindings=bindings,
                    )
                    item["runtime_value_resolution"] = {k: v for k, v in resolution.items() if k != "status"}
                    if not resolution["ok"]:
                        step.status = "stop_path"
                        step.result = {
                            "status": "repair_value_unresolved",
                            "reason": resolution["reason"],
                            "algorithm": resolution["algorithm"],
                            "feature_type": item.get("feature_type"),
                            "evidence_span_ids": list(resolution["evidence_span_ids"]),
                            "fallback": value_policy.get("fallback", "stop_path"),
                        }
                        return
                    item["value"] = resolution["value"]
                    if resolution["unit"]:
                        item["unit"] = resolution["unit"]
                    item["derived_by_rule_id"] = step.rule_id
                    item["derived_at_step_id"] = step.step_id

                expression = item.get("expression")
                if isinstance(expression, dict):
                    computation = self.calculator.calculate(
                        {"kind": "expression", "expression": expression,
                         "result_unit": str(item.get("result_unit") or "")},
                        workspace=computation_workspace, slot_bindings=bindings,
                    )
                    item["value"] = computation["value"]
                    item["unit"] = computation["unit"]
                    item["computation"] = computation
                    item["computed_by_rule_id"] = step.rule_id
                    item["computed_at_step_id"] = step.step_id
                    _attach_derived_evidence_metadata(
                        item,
                        computation=computation,
                        spans=_as_dict_list(computation_workspace.get("spans")),
                        rule_id=step.rule_id,
                    )
                    computations.append(computation)
                from .evidence_lifecycle import attach_derivation
                if step.execution_decision.get("decision_id"):
                    rule = _rule_from_match(self.rule_runtime.state["candidates"][step.execution_decision["decision_id"]])
                    check_output(rule, item, step.execution_decision["mode_contract"])
                    item = apply_output_semantics(item, q=step.execution_decision["q"],
                                                  decision=step.execution_decision, workspace=computation_workspace)
                    result_metadata(item, step.execution_decision)
                proof_bindings = dict(bindings)
                for cid, contributor in item.get("activation_fusion", {}).get("contributors", {}).items():
                    proof_bindings.update({f"fusion_{cid}_{slot}": sid for slot, sid in contributor["bindings"].items()})
                attach_derivation(item, workspace=current_workspace, bindings=proof_bindings,
                                  rule_id=step.rule_id, step_id=step.step_id)
                if typed:
                    # Typed outputs are complete newly justified results. A
                    # merging update would resurrect removed computation data.
                    current_workspace["spans"] = [s for s in current_workspace.get("spans", [])
                                                  if s.get("span_id") != item["span_id"]]
                workspace = staged_memory.apply_event_workspace_operation(
                    step.event_id,
                    {
                        # A span_id identifies one graph node. Re-executing a
                        # rule updates that node instead of creating an invalid
                        # event graph with duplicate node identifiers.
                        "operation": "update",
                        "target": "spans",
                        "id": item["span_id"],
                        "item": item,
                    },
                )
                added_span_ids.append(str(item["span_id"]))
            self.working_memory.commit_calculated_workspace(
                step.event_id, workspace, expected_version=expected_version,
            )
            step.result = {
                "event_id": step.event_id,
                "workspace_version": int((workspace or {}).get("version", 0) or 0),
                "target": "spans",
                "added_span_ids": added_span_ids,
                "computations": computations,
            }
            return
        try:
            workspace = self.working_memory.apply_event_workspace_operation(step.event_id, step.produced)
        except ValueError as exc:
            step.status = "error"
            step.error = str(exc)
            self._remember(
                stage="rejected_event_workspace_operation",
                payload_type="runtime_trace",
                payload={
                    "event_id": step.event_id,
                    "operation": dict(step.produced),
                    "reason": str(exc),
                },
                confidence=0.5,
                tags=["executor-new", "event_workspace_boundary", step.operation],
            )
            return
        step.result = {
            "event_id": step.event_id,
            "workspace_version": workspace.get("version", 0),
            "target": step.produced.get("target", "spans"),
        }

    def _record_blocked_rule(self, chain_id: str, rule_match: dict[str, Any], reason: str, *,
                             status: str = "blocked", decision: dict | None = None, error: str = "") -> ExecutorStep:
        rule = _rule_from_match(rule_match)
        step = ExecutorStep(
            chain_id=chain_id,
            step_id=stable_text_id(f"{chain_id}_blocked", rule.get("rule_id"), reason),
            rule_id=str(rule.get("rule_id") or ""),
            rule_layer=str(rule.get("rule_layer") or ""),
            operation="blocked",
            produced={},
            status=status,
            error=error,
            execution_decision=deepcopy(decision or {}),
            required_match=dict(rule_match.get("match") or {}),
            result={"reason": reason},
            created_at=_utc_now(),
        )
        self._record_step(step)
        return step

    def _record_step(self, step: ExecutorStep) -> None:
        payload = {"step": step.to_dict()}
        self._remember(
            stage="executor_step",
            payload_type="executor_step",
            payload=payload,
            tags=["executor-new", "executor_step", step.operation, step.status],
        )
        self._append_chain_step(step)

    def _accumulate_result(self, result: ExecutorResult, step: ExecutorStep) -> None:
        if step.status == "control_decision":
            result.control_updates.extend(deepcopy(step.result.get("control_updates", [])))
            result.next_rule_ids.extend(_string_list(step.result.get("next_rule_ids")))
            result.next_rule_selectors.extend(normalize_rule_selectors(step.result.get("next_rule_selectors")))
            result.frontend_requests.extend(_as_dict_list(step.result.get("frontend_requests")))
            result.outputs.extend(
                _enrich_output_with_rule_path(output, result.steps)
                for output in _as_dict_list(step.result.get("outputs"))
            )
            result.stopped_paths.extend(_as_dict_list(step.result.get("stopped_paths")))
            result.pending_operations.extend(_as_dict_list(step.result.get("pending_operations")))
        elif step.status == "needs_rule_match":
            result.next_rule_ids.extend(_string_list(step.result.get("next_rule_ids")))
            result.next_rule_selectors.extend(normalize_rule_selectors(step.result.get("next_rule_selectors")))
        elif step.status == "needs_frontend":
            request = step.result.get("frontend_request")
            if isinstance(request, dict):
                result.frontend_requests.append(request)
        elif step.status == "output_ready":
            output = step.result.get("output")
            if isinstance(output, dict):
                output = _enrich_output_with_rule_path(output, result.steps)
                result.outputs.append(output)
        elif step.status == "stop_path":
            stopped = step.result.get("stopped_path")
            if isinstance(stopped, dict):
                result.stopped_paths.append(stopped)
        elif step.status == "operation_pending":
            pending = step.result.get("pending_operation")
            if isinstance(pending, dict):
                result.pending_operations.append(pending)
        elif step.status == "error":
            result.errors.append({"step_id": step.step_id, "error": step.error})

    def _finish(self, result: ExecutorResult) -> None:
        result.next_rule_ids = _dedupe(result.next_rule_ids)
        self._update_chain_state(
            result.chain_id,
            {
                "status": result.status,
                "ok": result.ok,
                "updated_at": _utc_now(),
                "continuation": {
                    "next_rule_ids": result.next_rule_ids,
                    "next_rule_selectors": result.next_rule_selectors,
                    "frontend_requests": result.frontend_requests,
                    "outputs": result.outputs,
                    "stopped_paths": result.stopped_paths,
                    "pending_operations": result.pending_operations,
                },
                "errors": result.errors,
            },
            merge=True,
        )
        self._remember(
            stage="executor_chain_finish",
            payload_type="executor_result",
            payload={"result": result.to_dict()},
            confidence=1.0 if result.ok else 0.0,
        )

    def _remember(
        self,
        *,
        stage: str,
        payload_type: str,
        payload: dict[str, Any],
        confidence: float = 1.0,
        tags: list[str] | None = None,
    ) -> WorkingMemoryEntry:
        entry = WorkingMemoryEntry(
            entry_id=_stable_id("exec", self.session_id, stage, payload_type, len(self.working_memory.entries), _utc_now()),
            session_id=self.session_id,
            stage=stage,
            source="executor-new",
            payload_type=payload_type,
            created_at=_utc_now(),
            confidence=float(confidence),
            workspace_zone="event_workspace",
            payload=dict(payload or {}),
            tags=tags or ["executor-new", stage],
        )
        self.working_memory.add(entry, zone="event_workspace")
        return entry

    def _load_chain_state(self, chain_id: str) -> dict[str, Any]:
        for entry in reversed(self.working_memory.zone_entries.get("event_workspace", []) or []):
            if entry.payload_type != "executor_chain_state":
                continue
            state = entry.payload.get("chain_state") if isinstance(entry.payload, dict) else None
            if isinstance(state, dict) and str(state.get("chain_id") or "") == str(chain_id):
                return dict(state)
        return {}

    def _update_chain_state(self, chain_id: str, update: dict[str, Any], *, merge: bool = False) -> None:
        current = self._load_chain_state(chain_id)
        if merge:
            current.update(update)
        else:
            current = dict(update)
        current.setdefault("chain_id", chain_id)
        current["updated_at"] = update.get("updated_at") or _utc_now()
        self._remember(
            stage="executor_chain_state",
            payload_type="executor_chain_state",
            payload={"chain_state": current},
            tags=["executor-new", "executor_chain_state", str(current.get("status") or "")],
        )

    def _append_chain_step(self, step: ExecutorStep) -> None:
        state = self._load_chain_state(step.chain_id) or {"chain_id": step.chain_id}
        steps = list(state.get("steps", []) or [])
        steps.append(step.to_dict())
        state["steps"] = steps
        state["executed_step_count"] = len(steps)
        state["status"] = step.status
        state["updated_at"] = _utc_now()
        state["last_step_id"] = step.step_id
        state["continuation"] = _continuation_from_step(step)
        self._remember(
            stage="executor_chain_step",
            payload_type="executor_chain_state",
            payload={
                "chain_state": state,
                "candidate_path": {
                    "candidate_path_role": "executor_chain_step",
                    "chain_id": step.chain_id,
                    "step_id": step.step_id,
                    "operation": step.operation,
                    "status": step.status,
                    "created_at": step.created_at,
                },
            },
            tags=["executor-new", "executor_chain_step", step.operation, step.status],
        )


def _rule_from_match(rule_match: dict[str, Any]) -> dict[str, Any]:
    rule = rule_match.get("rule")
    if isinstance(rule, dict):
        return rule
    if "rule_id" in rule_match or "produced" in rule_match:
        return rule_match
    return {}


def _produced_operation(produced: dict[str, Any]) -> str:
    operation = str(produced.get("operation") or produced.get("op") or "").strip()
    if not operation and _as_dict_list(produced.get("conclusion")):
        return "add"
    return operation


def _bound_span_ids(rule_match: dict[str, Any]) -> set[str]:
    bound: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            bindings = node.get("slot_bindings")
            if isinstance(bindings, dict):
                bound.update(str(value) for value in bindings.values() if str(value))
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(rule_match.get("match"))
    return bound


def _slot_bindings(value: dict[str, Any]) -> dict[str, str]:
    """Collect matcher bindings and reject contradictory slot assignments."""

    evaluation = value.get("require_evaluation")
    if isinstance(evaluation, dict):
        return dict(evaluation["bindings"])
    bindings: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            raw_bindings = node.get("slot_bindings")
            if isinstance(raw_bindings, dict):
                for raw_slot_id, raw_span_id in raw_bindings.items():
                    slot_id = str(raw_slot_id or "").strip()
                    span_id = str(raw_span_id or "").strip()
                    if not slot_id or not span_id:
                        continue
                    existing = bindings.get(slot_id)
                    if existing and existing != span_id:
                        raise StructuredComputationError(
                            "conflicting_slot_bindings",
                            f"matcher bound slot {slot_id!r} to multiple event spans",
                            details={"slot_id": slot_id, "span_ids": [existing, span_id]},
                        )
                    bindings[slot_id] = span_id
            for nested in node.values():
                walk(nested)
        elif isinstance(node, (list, tuple)):
            for nested in node:
                walk(nested)

    walk(value)
    return bindings


def _attach_derived_evidence_metadata(
    item: dict[str, Any],
    *,
    computation: dict[str, Any],
    spans: list[dict[str, Any]],
    rule_id: str,
) -> None:
    """Preserve the provenance and weakest-link quality of computed evidence."""

    span_index = {
        str(span.get("span_id") or span.get("id") or ""): span
        for span in spans
        if str(span.get("span_id") or span.get("id") or "")
    }
    source_span_ids = [
        str(span_id)
        for span_id in computation.get("source_span_ids", [])
        if str(span_id) in span_index
    ]
    if not source_span_ids:
        return

    source_spans = [span_index[span_id] for span_id in source_span_ids]
    source_confidences = [
        max(0.0, min(_safe_float(span.get("confidence"), 0.5), 1.0))
        for span in source_spans
    ]
    source_reliabilities: list[float] = []
    authority_flags: list[bool] = []
    for span, confidence in zip(source_spans, source_confidences):
        metadata = span.get("metadata") if isinstance(span.get("metadata"), dict) else {}
        source_reliabilities.append(
            max(
                0.0,
                min(
                    _safe_float(
                        metadata.get("source_reliability", span.get("source_reliability")),
                        confidence,
                    ),
                    1.0,
                ),
            )
        )
        authority_flags.append(
            bool(metadata.get("authority_verified", span.get("authority_verified", False)))
        )

    metadata = dict(item.get("metadata") or {})
    metadata.update(
        {
            "source_reliability": min(source_reliabilities),
            "authority_verified": all(authority_flags),
            "derived_from_span_ids": source_span_ids,
            "measurement_basis": f"derived_by:{rule_id}",
            "semantic_key": str(item.get("feature_type") or item.get("feature_schema_id") or ""),
        }
    )
    item["metadata"] = metadata
    item["confidence"] = min(
        max(0.0, min(_safe_float(item.get("confidence"), 1.0), 1.0)),
        min(source_confidences),
    )


def _event_id_from_operation(produced: dict[str, Any], rule_match: dict[str, Any]) -> str:
    direct = str(produced.get("event_id") or "").strip()
    if direct:
        return direct
    match = rule_match.get("match")
    if isinstance(match, dict):
        for key in ("event_id", "matched_required"):
            value = match.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and str(item.get("event_id") or "").strip():
                        return str(item.get("event_id")).strip()
        for produced_match in _as_dict_list(match.get("produced")):
            for item in _as_dict_list(produced_match.get("matched_required")):
                event_id = str(item.get("event_id") or "").strip()
                if event_id:
                    return event_id
    return ""


def _continuation_from_step(step: ExecutorStep) -> dict[str, Any]:
    if step.status == "control_decision":
        return {
            "type": "control_decision",
            "next_rule_ids": _string_list(step.result.get("next_rule_ids")),
            "next_rule_selectors": normalize_rule_selectors(step.result.get("next_rule_selectors")),
            "frontend_requests": _as_dict_list(step.result.get("frontend_requests")),
            "outputs": _as_dict_list(step.result.get("outputs")),
            "stopped_paths": _as_dict_list(step.result.get("stopped_paths")),
            "pending_operations": _as_dict_list(step.result.get("pending_operations")),
        }
    if step.status == "needs_rule_match":
        return {
            "type": "run_rule",
            "next_rule_ids": _string_list(step.result.get("next_rule_ids")),
            "next_rule_selectors": normalize_rule_selectors(step.result.get("next_rule_selectors")),
        }
    if step.status == "needs_frontend":
        request = step.result.get("frontend_request")
        return {"type": "request_frontend_patch", "frontend_request": dict(request) if isinstance(request, dict) else {}}
    if step.status == "output_ready":
        output = step.result.get("output")
        return {"type": "output", "output": dict(output) if isinstance(output, dict) else {}}
    if step.status == "stop_path":
        stopped = step.result.get("stopped_path")
        return {"type": "stop_path", "stopped_path": dict(stopped) if isinstance(stopped, dict) else {}}
    if step.status == "operation_pending":
        pending = step.result.get("pending_operation")
        return {"type": "operation_pending", "pending_operation": dict(pending) if isinstance(pending, dict) else {}}
    if step.status == "error":
        return {"type": "error", "error": step.error}
    return {"type": "continue"}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _context_values(context: Any, path: str) -> list[Any]:
    parts = [part for part in str(path or "").split(".") if part]
    values = [context]
    for part in parts:
        next_values: list[Any] = []
        for value in values:
            if part == "*":
                if isinstance(value, dict):
                    next_values.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    next_values.extend(value)
                continue
            if isinstance(value, dict) and part in value:
                next_values.append(value[part])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict) and part in item:
                        next_values.append(item[part])
        values = next_values
        if not values:
            break
    return values


def _control_predicate_matches(predicate: Any, context: dict[str, Any]) -> bool:
    if predicate in (None, {}):
        return True
    if not isinstance(predicate, dict):
        raise ValueError("control branch predicate must be a mapping")
    if "all" in predicate:
        return all(
            _control_predicate_matches(item, context)
            for item in _as_dict_list(predicate.get("all"))
        )
    if "any" in predicate:
        return any(
            _control_predicate_matches(item, context)
            for item in _as_dict_list(predicate.get("any"))
        )
    if "not" in predicate:
        return not _control_predicate_matches(predicate.get("not"), context)

    path = str(predicate.get("path") or "").strip()
    if not path:
        raise ValueError("control branch predicate requires a context path")
    values = _context_values(context, path)
    operator = str(predicate.get("operator") or predicate.get("op") or "eq").strip().lower()
    expected = predicate.get("value")
    if operator == "exists":
        return bool(values) == bool(True if expected is None else expected)
    if operator == "truthy":
        return any(bool(value) for value in values)
    if operator == "falsy":
        return not values or all(not bool(value) for value in values)

    def compare(actual: Any) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "in":
            return actual in _list_value(expected)
        if operator == "not_in":
            return actual not in _list_value(expected)
        if operator == "contains":
            if isinstance(actual, dict):
                return expected in actual
            if isinstance(actual, (list, tuple, set, str)):
                return expected in actual
            return False
        if operator in {"gt", "gte", "lt", "lte"}:
            actual_number = float(actual)
            expected_number = float(expected)
            return {
                "gt": actual_number > expected_number,
                "gte": actual_number >= expected_number,
                "lt": actual_number < expected_number,
                "lte": actual_number <= expected_number,
            }[operator]
        raise ValueError(f"unsupported control predicate operator: {operator}")

    return any(compare(value) for value in values)


def _materialize_control_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$context."):
        resolved = _context_values(context, value[len("$context."):])
        if not resolved:
            return None
        return resolved[0] if len(resolved) == 1 else resolved
    if isinstance(value, dict):
        return {
            key: _materialize_control_value(item, context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_control_value(item, context) for item in value]
    return value


def _materialize_control_actions(value: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    actions = _as_dict_list(value)
    return [
        _materialize_control_value(action, context)
        for action in actions
    ]


def _classify_control_actions(
    actions: list[dict[str, Any]],
    *,
    rule_match: dict[str, Any],
    control_context: dict[str, Any],
) -> dict[str, Any]:
    next_rule_ids: list[str] = []
    next_rule_selectors: list[dict[str, Any]] = []
    frontend_requests: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    stopped_paths: list[dict[str, Any]] = []
    pending_operations: list[dict[str, Any]] = []
    control_updates: list[dict[str, Any]] = []
    for action in actions:
        operation = str(action.get("operation") or action.get("op") or "").strip()
        if operation == "update_control_state":
            update = control_state_update(_rule_from_match(rule_match), action.get("values"))
            control_context.setdefault("workflow", {}).update(deepcopy(update["values"]))
            control_updates.append(update)
        elif operation == "run_rule":
            next_rule_ids.extend(_string_list(action.get("rule_ids")))
            next_rule_selectors.extend(_selector_list(action, rule_match=rule_match))
        elif operation == "request_frontend_patch":
            frontend_requests.append(dict(action))
        elif operation == "output":
            output = dict(action)
            if "task_result" in output:
                output["task_result"] = {
                    **validate_task_result(output["task_result"]),
                    "control_rule_id": str(_rule_from_match(rule_match)["rule_id"]),
                }
            outputs.append(output)
        elif operation == "stop_path":
            stopped = dict(action)
            if "task_result" in stopped:
                stopped["task_result"] = {
                    **validate_task_result(stopped["task_result"]),
                    "control_rule_id": str(_rule_from_match(rule_match)["rule_id"]),
                }
            stopped_paths.append(stopped)
        elif operation == "select_candidate_path":
            trigger = control_context.get("trigger") if isinstance(control_context.get("trigger"), dict) else {}
            candidates = _as_dict_list(trigger.get("candidate_rule_paths"))
            path_state = control_context.get("path") if isinstance(control_context.get("path"), dict) else {}
            selection_policy = action.get('selection_policy') or {}
            source_selector = trigger.get('source_rule_selector') or {}
            prefer_memory = bool(selection_policy.get('inherit_memory_path_preference')) and source_selector.get('memory_path_policy') == 'prefer_then_fallback'
            selected = choose_candidate_rule_path(
                candidates,
                policy=selection_policy,
                attempted_rule_ids=_string_list(path_state.get("visited_rule_ids")),
                memory_preferred_rule_ids=_string_list(source_selector.get('memory_preferred_rule_ids')) if prefer_memory else [],
            )
            if selected is None:
                stopped_paths.append(
                    {
                        "operation": "stop_path",
                        "status": "no_compatible_candidate_path",
                        "reason": "control rule found no compatible candidate path to select",
                    }
                )
            else:
                pending_operations.append(
                    {
                        "operation": "execute_candidate_path",
                        "candidate_path": selected,
                        "selected_by_control_rule_id": str((_rule_from_match(rule_match)).get("rule_id") or ""),
                    }
                )
        elif operation == "decide_rule_activation":
            owner = _rule_from_match(rule_match)
            if owner.get("rule_layer") != "control":
                raise ActivationError("invalid_activation_authority", "activation decisions require a Control rule")
            pending_operations.append({**deepcopy(action), "control_rule_id": owner["rule_id"]})
        elif operation:
            pending = dict(action)
            # Retrieval context must be retained under the Control Rule that
            # requested it.  Rules declare the retrieval; Executor only
            # carries that declared ownership across the stage boundary.
            if operation in {
                "retrieve_memory",
                "query_memory",
                "retrieve_workspace_memory",
                "query_memory_workspace",
                "retrieve_knowledge",
                "query_knowledge",
                "enrich_knowledge",
                "inspect_goal_requirements",
                "ground_input_semantics",
                "review_event_evidence",
                "inspect_rule_frontier", "inspect_require_gaps", "bind_knowledge_context",
            }:
                pending.setdefault("rule_id", str((_rule_from_match(rule_match)).get("rule_id") or ""))
            pending_operations.append(pending)
        else:
            raise ValueError("control action is missing operation")
    return {
        "control_updates": control_updates,
        "next_rule_ids": _dedupe(next_rule_ids),
        "next_rule_selectors": next_rule_selectors,
        "frontend_requests": frontend_requests,
        "outputs": outputs,
        "stopped_paths": stopped_paths,
        "pending_operations": pending_operations,
    }


def _selector_list(produced: dict[str, Any], *, rule_match: dict[str, Any]) -> list[dict[str, Any]]:
    return child_rule_selectors(produced, rule_match=rule_match)


def _enrich_output_with_rule_path(output: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(output)
    rule_ids = [
        str(step.get("rule_id") or "")
        for step in steps
        if str(step.get("rule_id") or "")
    ]
    layers = [
        str(step.get("rule_layer") or "").strip().lower()
        for step in steps
        if str(step.get("rule_layer") or "").strip()
    ]
    if rule_ids:
        enriched.setdefault("rules_applied", list(dict.fromkeys(rule_ids)))
        enriched.setdefault("rule_path_rule_ids", list(dict.fromkeys(rule_ids)))
    if layers:
        enriched.setdefault("rule_path_layers", list(dict.fromkeys(layers)))
    if "answer_granularity" not in enriched:
        enriched["answer_granularity"] = _infer_output_granularity(enriched)
    return enriched


def _render_output_from_template_context(
    output: dict[str, Any],
    *,
    working_memory: WorkingMemoryState,
    event_id: str,
    slot_bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    rendered = dict(output)
    if str(rendered.get("answer_mode") or "").strip().lower() == "event_span":
        return _render_event_span_output(
            rendered,
            working_memory=working_memory,
            event_id=event_id,
            slot_bindings=slot_bindings or {},
        )
    explicit_answer = str(rendered.get("answer") or "").strip()
    if explicit_answer and not _template_keys(explicit_answer) and not _looks_like_render_instruction(explicit_answer):
        rendered.setdefault("conclusion", explicit_answer)
        return rendered
    template = str(
        rendered.get("output_template")
        or rendered.get("answer_template")
        or rendered.get("conclusion_template")
        or ""
    ).strip()
    selector = rendered.get("evidence_selector")
    if not template:
        return rendered
    if event_id:
        workspace = working_memory.event_workspaces.get(str(event_id or ""), {})
    elif len(working_memory.event_workspaces) == 1:
        event_id, workspace = next(iter(working_memory.event_workspaces.items()))
    else:
        workspace = {}
    evidence_span = _select_output_evidence_span(workspace, selector if isinstance(selector, dict) else {})
    if not evidence_span:
        raise OutputTemplateBindingError(_template_keys(template) or ["evidence_span"])
    context = _template_context_from_span(evidence_span)
    context.update(_template_context_from_workspace(workspace))
    answer = _render_template(template, context)
    missing_keys = _template_keys(answer)
    if missing_keys:
        raise OutputTemplateBindingError(missing_keys)
    if not answer:
        raise OutputTemplateBindingError(_template_keys(template) or ["rendered_answer"])
    rendered["answer"] = answer
    rendered["conclusion"] = answer
    rendered.setdefault("evidence_spans", [_compact_span_for_output(evidence_span)])
    rendered.setdefault("rendered_from_event_id", event_id)
    return rendered


def _render_event_span_output(
    output: dict[str, Any],
    *,
    working_memory: WorkingMemoryState,
    event_id: str,
    slot_bindings: dict[str, str],
) -> dict[str, Any]:
    rendered = dict(output)
    if event_id:
        workspace = working_memory.event_workspaces.get(str(event_id), {})
    elif len(working_memory.event_workspaces) == 1:
        event_id, workspace = next(iter(working_memory.event_workspaces.items()))
    else:
        workspace = {}
    span_index = {
        str(span.get("span_id") or span.get("id") or ""): span
        for span in _as_dict_list(workspace.get("spans"))
        if str(span.get("span_id") or span.get("id") or "")
    }
    from .evidence_lifecycle import evidence_validity
    validity = evidence_validity(span_index, workspace.get('relations', []))
    span_index = {sid: span for sid, span in span_index.items() if validity[sid]}
    requested_slot_ids = [
        str(span.get("span_id") or span.get("id") or "").strip()
        for span in _as_dict_list(output.get("conclusion"))
        if str(span.get("span_id") or span.get("id") or "").strip()
    ]
    if not requested_slot_ids:
        raise OutputTemplateBindingError(["conclusion_span"])
    result_spans: list[dict[str, Any]] = []
    missing: list[str] = []
    for slot_id in requested_slot_ids:
        # An output rule that names an existing graph node is requesting that
        # node, not merely a semantically similar binding.
        span_id = slot_id if slot_id in span_index else str(slot_bindings.get(slot_id) or slot_id)
        span = span_index.get(span_id)
        if span is None:
            missing.append(slot_id)
            continue
        result_spans.append(dict(span))
    for optional_span_id in _string_list(output.get("optional_conclusion_span_ids")):
        span = span_index.get(optional_span_id)
        if span is None:
            continue
        if not any(str(item.get("span_id") or "") == optional_span_id for item in result_spans):
            result_spans.append(dict(span))
    if missing:
        raise OutputTemplateBindingError(missing)
    rendered["result_spans"] = result_spans
    rendered["conclusion_span_ids"] = [str(span.get("span_id") or "") for span in result_spans]
    rendered["rendered_from_event_id"] = event_id
    return rendered


def _looks_like_render_instruction(text: str) -> bool:
    lower = str(text or "").strip().lower()
    if not lower:
        return False
    instruction_markers = (
        "explain the focused finance concept",
        "using the matched knowledge span",
        "render from",
        "use the matched",
        "fill the template",
        "输出模板",
        "渲染",
    )
    return any(marker in lower for marker in instruction_markers)


def _select_output_evidence_span(workspace: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    spans = _as_dict_list(workspace.get("spans"))
    scored: list[tuple[float, dict[str, Any]]] = []
    for span in spans:
        score = _output_evidence_score(span, selector)
        if score > 0:
            scored.append((score, span))
    if not scored:
        return {}
    scored.sort(key=lambda item: (item[0], float(item[1].get("confidence", 0.0) or 0.0)), reverse=True)
    return dict(scored[0][1])


def _output_evidence_score(span: dict[str, Any], selector: dict[str, Any]) -> float:
    score = 1.0
    for field in ("source", "relation", "knowledge_type", "core"):
        expected = str(selector.get(field) or "").strip().lower()
        if not expected:
            continue
        actual = str(span.get(field) or "").strip().lower()
        if actual != expected:
            return 0.0
        score += 2.0
    relation_in = [str(value).strip().lower() for value in _list_value(selector.get("relation_in")) if str(value).strip()]
    if relation_in:
        actual_relation = str(span.get("relation") or "").strip().lower()
        if actual_relation not in relation_in:
            return 0.0
        score += 1.5
    source_in = [str(value).strip().lower() for value in _list_value(selector.get("source_in")) if str(value).strip()]
    if source_in:
        actual_source = str(span.get("source") or "").strip().lower()
        if actual_source not in source_in:
            return 0.0
        score += 1.5
    labels = _parse_labeled_fields(str(span.get("nature") or span.get("text") or ""))
    for label in _list_value(selector.get("prefer_labels")):
        if str(label).strip().lower() in labels:
            score += 0.5
    return score


def _template_context_from_span(span: dict[str, Any]) -> dict[str, str]:
    nature = str(span.get("nature") or span.get("text") or "").strip()
    context = {
        "span_id": str(span.get("span_id") or span.get("id") or ""),
        "core": str(span.get("core") or ""),
        "relation": str(span.get("relation") or ""),
        "nature": nature,
        "text": str(span.get("text") or nature),
        "source": str(span.get("source") or ""),
        "knowledge_type": str(span.get("knowledge_type") or ""),
        "confidence": str(span.get("confidence") or ""),
    }
    context.update(_parse_labeled_fields(nature))
    return context


def _template_context_from_workspace(workspace: dict[str, Any]) -> dict[str, str]:
    context: dict[str, str] = {}
    for span in _as_dict_list(workspace.get("spans")):
        aliases = [
            _safe_template_key(span.get("span_id") or span.get("id")),
            _safe_template_key(span.get("relation")),
        ]
        for alias in aliases:
            if not alias:
                continue
            prefix_context = _template_context_from_span(span)
            for key, value in prefix_context.items():
                context.setdefault(f"{alias}.{key}", value)
                context.setdefault(f"{alias}_{key}", value)
    return context


def _safe_template_key(value: Any) -> str:
    key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip())
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def _parse_labeled_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for chunk in str(text or "").split("|"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def _render_template(template: str, context: dict[str, str]) -> str:
    rendered = str(template or "")
    for key, value in context.items():
        rendered = rendered.replace("{{" + key + "}}", str(value or ""))
    return " ".join(rendered.split()).strip()


def _template_keys(value: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}", str(value or ""))))


def _compact_span_for_output(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": str(span.get("span_id") or span.get("id") or ""),
        "core": str(span.get("core") or ""),
        "relation": str(span.get("relation") or ""),
        "nature": str(span.get("nature") or ""),
        "source": str(span.get("source") or ""),
        "confidence": float(span.get("confidence", 0.0) or 0.0),
    }


def _infer_output_granularity(output: dict[str, Any]) -> str:
    explicit = str(output.get("granularity") or output.get("output_granularity") or "").strip().lower()
    if explicit:
        return explicit
    if str(output.get("answer_mode") or "").strip().lower() == "event_span" and _as_dict_list(
        output.get("conclusion")
    ):
        return "structured_conclusion"
    if str(output.get("option") or output.get("predicted_option") or output.get("target_id") or "").strip():
        return "concrete_answer"
    answer = str(output.get("answer") or output.get("conclusion") or "").strip().lower()
    if any(marker in answer for marker in ("final answer", "supports target", "supported_by", "support target")):
        return "concrete_answer"
    return "high_level_strategy"


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    return shared_safe_float(value, default)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys([str(value) for value in values if str(value)]))


def _stable_id(*parts: Any) -> str:
    return stable_text_id(str(parts[0] or "id"), *parts[1:])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
