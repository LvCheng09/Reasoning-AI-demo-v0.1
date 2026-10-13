"""New controller-first scheduler runtime.

This module is intentionally independent from the old long pipeline scheduler.
Working memory is treated as a partitioned data store. SchedulerNew owns only
runtime state, controller feedback, dispatch order, interruption, and output
gating.
"""

from __future__ import annotations
from .common.normalize import safe_float as shared_safe_float

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Callable

from .data_structure import WorkingMemoryEntry, WorkingMemoryState
from .control_voc import evaluate_control_voc
from .event_graph_contract import validate_event_graph_contract
from .executor_new import DEFAULT_RULE_CHAIN_MAX_STEPS, execute_rule_chain
from .calculator import Calculator
from .identity import new_runtime_id, stable_content_id
from .matcher import match_source_to_target
from .memory_workspace import select_memory_workspace_context
from .persistence import knowledge_db as knowledge_db_persistence
from .retrieval import (
    build_missing_knowledge_queries,
    build_event_query,
    knowledge_record_to_event_context,
    retrieve_knowledge,
    retrieve_memory,
)
from .rule_layers import CONTENT_LAYERS
from .rule_selector import normalize_rule_selector, normalize_rule_selectors, selector_execution_context


ControllerFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
StageFn = Callable[[dict[str, Any], WorkingMemoryState, dict[str, Any]], dict[str, Any]]
InstructionHandlerFn = Callable[[dict[str, Any], "SchedulerNewState", WorkingMemoryState], dict[str, Any]]


@dataclass
class SchedulerInstruction:
    instruction_id: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    on_success: list[str] = field(default_factory=list)
    on_fail: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "op": self.op,
            "args": dict(self.args),
            "on_success": list(self.on_success),
            "on_fail": list(self.on_fail),
            "metadata": dict(self.metadata),
        }


@dataclass
class SchedulerProgram:
    program_id: str
    entry_instruction_id: str
    instructions: dict[str, SchedulerInstruction] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def get(self, instruction_id: str) -> SchedulerInstruction | None:
        return self.instructions.get(str(instruction_id or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "entry_instruction_id": self.entry_instruction_id,
            "instructions": {
                instruction_id: instruction.to_dict()
                for instruction_id, instruction in self.instructions.items()
            },
            "source": dict(self.source),
        }


@dataclass
class InstructionResult:
    instruction_id: str
    op: str
    ok: bool
    produced: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    next_instruction_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "op": self.op,
            "ok": self.ok,
            "produced": list(self.produced),
            "error": self.error,
            "next_instruction_ids": list(self.next_instruction_ids),
            "metadata": dict(self.metadata),
        }


class RuleStrategyCompiler:
    """Compile controller strategy/rule structures into executable instructions."""

    def compile(self, controller_feedback: dict[str, Any]) -> SchedulerProgram:
        program_id = str(controller_feedback.get("program_id") or stable_content_id("program", controller_feedback))
        instructions: dict[str, SchedulerInstruction] = {}
        raw_strategy = (
            controller_feedback.get("program")
            or controller_feedback.get("strategy")
            or controller_feedback.get("rule")
            or controller_feedback
        )
        if isinstance(controller_feedback.get("instructions"), list):
            entry_id = self._compile_instruction_list(controller_feedback, instructions)
        elif isinstance(raw_strategy, dict) and isinstance(raw_strategy.get("instructions"), list):
            entry_id = self._compile_instruction_list(raw_strategy, instructions)
        else:
            entry_id = self._compile_node(
                raw_strategy if isinstance(raw_strategy, dict) else {"operation": "run_rule", "args": {"value": raw_strategy}},
                instructions=instructions,
                fallback_fail_id="inst_frontend_patch",
            )
        self._ensure_default_fallbacks(instructions, entry_id)
        return SchedulerProgram(
            program_id=program_id,
            entry_instruction_id=entry_id,
            instructions=instructions,
            source=controller_feedback,
        )

    def _compile_instruction_list(
        self,
        raw_program: dict[str, Any],
        instructions: dict[str, SchedulerInstruction],
    ) -> str:
        previous_id = ""
        first_id = str(raw_program.get("entry_instruction_id") or "")
        for raw in raw_program.get("instructions", []) or []:
            if not isinstance(raw, dict):
                continue
            instruction_id = str(raw.get("instruction_id") or raw.get("id") or f"inst_{len(instructions) + 1}")
            if not first_id:
                first_id = instruction_id
            instructions[instruction_id] = SchedulerInstruction(
                instruction_id=instruction_id,
                op=str(raw.get("op") or raw.get("operation") or "run_rule"),
                args=dict(raw.get("args") or {}),
                on_success=[str(item) for item in raw.get("on_success", [])],
                on_fail=[str(item) for item in raw.get("on_fail", [])] or ["inst_frontend_patch"],
                metadata=dict(raw.get("metadata") or {}),
            )
            if previous_id and not instructions[previous_id].on_success:
                instructions[previous_id].on_success = [instruction_id]
            previous_id = instruction_id
        return first_id or self._compile_node({"operation": "stop_path"}, instructions=instructions, fallback_fail_id="inst_stop_path")

    def _compile_node(
        self,
        node: dict[str, Any],
        *,
        instructions: dict[str, SchedulerInstruction],
        fallback_fail_id: str,
    ) -> str:
        instruction_id = str(node.get("instruction_id") or node.get("id") or f"inst_{len(instructions) + 1}")
        if instruction_id in instructions:
            instruction_id = f"{instruction_id}_{len(instructions) + 1}"
        op = self._infer_op(node)
        instruction = SchedulerInstruction(
            instruction_id=instruction_id,
            op=op,
            args=self._build_args(node, op),
            on_success=[],
            on_fail=[] if op in {"output", "stop_path"} else [fallback_fail_id],
            metadata={
                "source_rule_id": node.get("rule_id"),
                "source_strategy": node.get("strategy"),
            },
        )
        instructions[instruction_id] = instruction
        child_ids = [
            self._compile_node(child, instructions=instructions, fallback_fail_id=fallback_fail_id)
            for child in self._iter_children(node)
        ]
        if child_ids:
            instruction.on_success = child_ids
        return instruction_id

    def _ensure_default_fallbacks(self, instructions: dict[str, SchedulerInstruction], entry_id: str) -> None:
        if "inst_frontend_patch" not in instructions:
            instructions["inst_frontend_patch"] = SchedulerInstruction(
                instruction_id="inst_frontend_patch",
                op="request_frontend_patch",
                args={"reason": "required information missing", "source": "compiler_default"},
                on_success=[entry_id],
                on_fail=["inst_stop_path"],
            )
        if "inst_stop_path" not in instructions:
            instructions["inst_stop_path"] = SchedulerInstruction(
                instruction_id="inst_stop_path",
                op="stop_path",
                args={"reason": "path blocked"},
            )

    def _infer_op(self, node: dict[str, Any]) -> str:
        raw = str(node.get("op") or node.get("operation") or "").strip()
        if raw:
            return raw
        if node.get("source") and node.get("target"):
            return "match"
        if node.get("target_rule_layer") or node.get("rule_match"):
            return "match"
        if "required" in node:
            return "match"
        if node.get("produced"):
            return "apply_produced"
        if node.get("rule_ids") or node.get("rule_id"):
            return "run_rule"
        if node.get("patch") or node.get("missing_span_query"):
            return "request_frontend_patch"
        if node.get("answer") or node.get("output"):
            return "output"
        return "run_rule"

    def _build_args(self, node: dict[str, Any], op: str) -> dict[str, Any]:
        if op == "match":
            source = node.get("source")
            target = node.get("target")
            if source is None and "required" in node:
                source = {
                    "type": "rule",
                    "rule_id": node.get("rule_id", ""),
                    "rule_layer": node.get("rule_layer", ""),
                    "required": node.get("required", []),
                }
            if target is None and "required" in node:
                target = {
                    "type": "working_memory",
                    "zone": "event_workspace",
                    "fields": [
                        "event_workspaces.*.spans",
                        "event_workspaces.*.relations",
                    ],
                }
            if source is None:
                source = {
                    "type": "working_memory",
                    "zone": "event_workspace",
                    "fields": [
                        "event_workspaces.*.spans",
                        "event_workspaces.*.relations",
                    ],
                }
            if target is None:
                target = {
                    "type": "rule_db",
                    "rule_layer": node.get("target_rule_layer") or node.get("rule_layer") or "control",
                    "rule_records": node.get("rule_records", []),
                    "rule_db_path": node.get("rule_db_path", ""),
                }
            return {
                "source": source,
                "target": target,
                "top_k": node.get("top_k", 32),
                "candidate_k": node.get("candidate_k", 256),
                "match_threshold": node.get("match_threshold", 0.55),
                "use_ann": bool(node.get("use_ann", True)),
            }
        if op == "apply_produced":
            return {"produced": node.get("produced", []), "rule_id": node.get("rule_id", "")}
        if op == "run_rule":
            rule_ids = node.get("rule_ids")
            if rule_ids is None and node.get("rule_id"):
                rule_ids = [node.get("rule_id")]
            args = {"rule_ids": rule_ids or []}
            if isinstance(node.get("rule_selector"), dict):
                args["rule_selector"] = dict(node["rule_selector"])
            return args
        if op == "request_frontend_patch":
            return {
                "event_id": node.get("event_id", ""),
                "focus_span_ids": node.get("focus_span_ids", []),
                "missing_span_query": node.get("missing_span_query", ""),
                "reason": node.get("reason", ""),
            }
        if op == "output":
            return {"output": node.get("output", {}), "answer": node.get("answer", "")}
        return dict(node.get("args") or {})

    def _iter_children(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        children: list[dict[str, Any]] = []
        for key in ("sub_rules", "children", "next", "steps"):
            raw = node.get(key)
            if isinstance(raw, list):
                children.extend(item for item in raw if isinstance(item, dict))
        produced = node.get("produced")
        if isinstance(produced, list):
            children.extend(item for item in produced if isinstance(item, dict))
        return children


class InstructionExecutor:
    """Execute compiled instructions without interpreting source strategy shape."""

    def __init__(
        self,
        *,
        source_target_matcher: InstructionHandlerFn | None = None,
        rule_runner: InstructionHandlerFn | None = None,
        produced_applier: InstructionHandlerFn | None = None,
        frontend: InstructionHandlerFn | None = None,
        output_decoder: InstructionHandlerFn | None = None,
    ) -> None:
        self.source_target_matcher = source_target_matcher
        self.rule_runner = rule_runner
        self.produced_applier = produced_applier
        self.frontend = frontend
        self.output_decoder = output_decoder

    def step(
        self,
        instruction: SchedulerInstruction,
        *,
        state: "SchedulerNewState",
        working_memory: WorkingMemoryState,
    ) -> InstructionResult:
        try:
            if instruction.op == "match":
                return self._match_source_to_target(instruction, state, working_memory)
            if instruction.op == "run_rule":
                return self._run_rule(instruction, state, working_memory)
            if instruction.op == "apply_produced":
                return self._apply_produced(instruction, state, working_memory)
            if instruction.op == "request_frontend_patch":
                return self._request_frontend_patch(instruction, state, working_memory)
            if instruction.op == "output":
                return self._output(instruction, state, working_memory)
            if instruction.op == "stop_path":
                return InstructionResult(
                    instruction_id=instruction.instruction_id,
                    op=instruction.op,
                    ok=True,
                    produced=[{"operation": "stop_path", **instruction.args}],
                )
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=False,
                error=f"unknown instruction op: {instruction.op}",
            )
        except Exception as exc:
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=False,
                error=repr(exc),
            )

    def _match_source_to_target(self, instruction: SchedulerInstruction, state: "SchedulerNewState", working_memory: WorkingMemoryState) -> InstructionResult:
        if self.source_target_matcher is None:
            return self._coerce_result(instruction, match_source_to_target(instruction.args, state, working_memory))
        return self._coerce_result(instruction, self.source_target_matcher(instruction.args, state, working_memory))

    def _run_rule(self, instruction: SchedulerInstruction, state: "SchedulerNewState", working_memory: WorkingMemoryState) -> InstructionResult:
        if self.rule_runner is None:
            produced = {"operation": "run_rule", "rule_ids": instruction.args.get("rule_ids", [])}
            if isinstance(instruction.args.get("rule_selector"), dict):
                produced["rule_selector"] = dict(instruction.args["rule_selector"])
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=True,
                produced=[produced],
            )
        return self._coerce_result(instruction, self.rule_runner(instruction.args, state, working_memory))

    def _apply_produced(self, instruction: SchedulerInstruction, state: "SchedulerNewState", working_memory: WorkingMemoryState) -> InstructionResult:
        if self.produced_applier is None:
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=True,
                produced=list(instruction.args.get("produced", []) or []),
            )
        return self._coerce_result(instruction, self.produced_applier(instruction.args, state, working_memory))

    def _request_frontend_patch(self, instruction: SchedulerInstruction, state: "SchedulerNewState", working_memory: WorkingMemoryState) -> InstructionResult:
        if self.frontend is None:
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=True,
                produced=[{"operation": "request_frontend_patch", **instruction.args}],
            )
        return self._coerce_result(instruction, self.frontend(instruction.args, state, working_memory))

    def _output(self, instruction: SchedulerInstruction, state: "SchedulerNewState", working_memory: WorkingMemoryState) -> InstructionResult:
        if self.output_decoder is None:
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=True,
                produced=[{"operation": "output", **instruction.args}],
            )
        return self._coerce_result(instruction, self.output_decoder(instruction.args, state, working_memory))

    def _coerce_result(self, instruction: SchedulerInstruction, result: Any) -> InstructionResult:
        if isinstance(result, InstructionResult):
            return result
        if not isinstance(result, dict):
            return InstructionResult(
                instruction_id=instruction.instruction_id,
                op=instruction.op,
                ok=False,
                error="instruction handler returned non-dict result",
            )
        return InstructionResult(
            instruction_id=instruction.instruction_id,
            op=instruction.op,
            ok=bool(result.get("ok", True)),
            produced=_as_dict_list(result.get("produced")),
            error=str(result.get("error", "")),
            next_instruction_ids=[str(item) for item in result.get("next_instruction_ids", []) or []],
            metadata=dict(result.get("metadata") or {}),
        )


@dataclass
class SchedulerNewState:
    session_id: str
    db_paths: dict[str, Any] = field(default_factory=dict)
    status: str = "idle"
    phase: str = "controller"
    loop_index: int = 0
    max_loops: int = 128
    max_rule_chain_steps: int = DEFAULT_RULE_CHAIN_MAX_STEPS
    active_event_ids: list[str] = field(default_factory=list)
    active_rule_ids: list[str] = field(default_factory=list)
    pending_inputs: list[dict[str, Any]] = field(default_factory=list)
    pending_operations: list[dict[str, Any]] = field(default_factory=list)
    pending_frontend_patches: list[dict[str, Any]] = field(default_factory=list)
    pending_rule_ids: list[str] = field(default_factory=list)
    pending_rule_selectors: list[dict[str, Any]] = field(default_factory=list)
    pending_output_candidates: list[dict[str, Any]] = field(default_factory=list)
    current_program: SchedulerProgram | None = None
    current_instruction_id: str = ""
    controller_history: list[dict[str, Any]] = field(default_factory=list)
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    execution_history_start: int = 0
    instruction_history: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    blocking_reason: str = ""
    last_action: str = ""
    allow_output: bool = False
    output_ready: bool = False
    interrupted: bool = False
    control_context: dict[str, Any] = field(default_factory=dict)
    control_state: dict[str, Any] = field(default_factory=dict)
    match_recovery_attempts: dict[str, int] = field(default_factory=dict)
    matcher_rule_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = field(
        default_factory=dict,
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "db_paths": dict(self.db_paths),
            "status": self.status,
            "phase": self.phase,
            "loop_index": self.loop_index,
            "max_loops": self.max_loops,
            "max_rule_chain_steps": self.max_rule_chain_steps,
            "active_event_ids": list(self.active_event_ids),
            "active_rule_ids": list(self.active_rule_ids),
            "pending_inputs": list(self.pending_inputs),
            "pending_operations": list(self.pending_operations),
            "pending_frontend_patches": list(self.pending_frontend_patches),
            "pending_rule_ids": list(self.pending_rule_ids),
            "pending_rule_selectors": list(self.pending_rule_selectors),
            "pending_output_candidates": list(self.pending_output_candidates),
            "current_program": self.current_program.to_dict() if self.current_program is not None else None,
            "current_instruction_id": self.current_instruction_id,
            "controller_history": list(self.controller_history),
            "execution_history": list(self.execution_history),
            "execution_history_start": self.execution_history_start,
            "instruction_history": list(self.instruction_history),
            "outputs": list(self.outputs),
            "errors": list(self.errors),
            "blocking_reason": self.blocking_reason,
            "last_action": self.last_action,
            "allow_output": self.allow_output,
            "output_ready": self.output_ready,
            "interrupted": self.interrupted,
            "control_context": deepcopy(self.control_context),
            "control_state": deepcopy(self.control_state),
            "match_recovery_attempts": dict(self.match_recovery_attempts),
        }


class SchedulerNew:
    """Controller-first scheduler over working memory.

    All incoming data is routed to controller before any execution. If new input
    arrives during an unfinished phase, the current state is snapshotted to
    working memory and control returns to controller. Output is emitted only
    after controller sets allow_output.
    """

    def __init__(
        self,
        working_memory: WorkingMemoryState,
        *,
        session_id: str,
        controller: ControllerFn | None = None,
        event_builder: StageFn | None = None,
        rule_matcher: StageFn | None = None,
        executor: StageFn | None = None,
        calculator: Calculator | None = None,
        frontend: StageFn | None = None,
        output_decoder: StageFn | None = None,
        compiler: RuleStrategyCompiler | None = None,
        instruction_executor: InstructionExecutor | None = None,
        instruction_rule_runner: InstructionHandlerFn | None = None,
        instruction_produced_applier: InstructionHandlerFn | None = None,
        instruction_frontend: InstructionHandlerFn | None = None,
        instruction_output_decoder: InstructionHandlerFn | None = None,
        db_paths: dict[str, Any] | None = None,
        vector_encoder: Any = None,
        memory_cache: Any = None,
        max_loops: int = 128,
        max_rule_chain_steps: int = DEFAULT_RULE_CHAIN_MAX_STEPS,
    ) -> None:
        self.working_memory = working_memory
        self.state = SchedulerNewState(
            session_id=session_id,
            db_paths=dict(db_paths or {}),
            max_loops=max(1, int(max_loops)),
            max_rule_chain_steps=max(1, int(max_rule_chain_steps)),
        )
        self.controller = controller or self._default_controller
        self.event_builder = event_builder or self._default_stage("event_build")
        self.rule_matcher = rule_matcher or self._default_rule_matcher
        self.executor = executor or self._default_executor
        self.calculator = calculator or Calculator(self._rule_db_path() or None)
        self.frontend = frontend or self._default_frontend_patcher
        self.output_decoder = output_decoder or self._default_output_decoder
        self.vector_encoder = vector_encoder
        self.memory_cache = memory_cache
        self.compiler = compiler or RuleStrategyCompiler()
        self.instruction_executor = instruction_executor or InstructionExecutor(
            rule_runner=instruction_rule_runner,
            produced_applier=instruction_produced_applier,
            frontend=instruction_frontend,
            output_decoder=instruction_output_decoder,
        )

    def receive_input(self, payload: dict[str, Any], *, source: str = "event_builder") -> None:
        if self.state.status not in {"idle", "completed", "failed"}:
            self.state.interrupted = True
            self._remember(
                zone="event_workspace",
                stage="scheduler_interrupt",
                source="scheduler-new",
                payload_type="scheduler_state_snapshot",
                payload={"state": self._runtime_state_checkpoint(), "reason": "new_input_received"},
            )
        incoming = {
            "source": str(source),
            "payload": dict(payload or {}),
            "received_at": _utc_now(),
        }
        self.state.pending_inputs.append(incoming)
        self.state.status = "waiting_controller"
        self.state.phase = "controller"
        self._remember(
            zone="input_workspace",
            stage="input",
            source=source,
            payload_type="incoming_input",
            payload=incoming,
        )

    def reset_runtime(self, *, session_id: str | None = None, clear_working_memory: bool = False) -> None:
        self.calculator.reset_versions()
        db_paths = dict(self.state.db_paths)
        max_loops = int(self.state.max_loops)
        max_rule_chain_steps = int(self.state.max_rule_chain_steps)
        next_session_id = str(session_id or self.state.session_id)
        self.state = SchedulerNewState(
            session_id=next_session_id,
            db_paths=db_paths,
            max_loops=max_loops,
            max_rule_chain_steps=max_rule_chain_steps,
        )
        if clear_working_memory:
            self.working_memory.clear()

    def clear_transient_state(
        self,
        *,
        clear_outputs: bool = True,
        clear_errors: bool = False,
        clear_working_memory: bool = False,
        clear_runtime_workspaces: bool = True,
    ) -> None:
        self.calculator.reset_versions()
        self.state.status = "idle"
        self.state.phase = "controller"
        self.state.loop_index = 0
        self.state.execution_history_start = len(self.state.execution_history)
        self.state.active_event_ids.clear()
        self.state.active_rule_ids.clear()
        self.state.pending_inputs.clear()
        self.state.pending_operations.clear()
        self.state.pending_frontend_patches.clear()
        self.state.pending_rule_ids.clear()
        self.state.pending_rule_selectors.clear()
        self.state.pending_output_candidates.clear()
        self.state.current_program = None
        self.state.current_instruction_id = ""
        self.state.blocking_reason = ""
        self.state.last_action = ""
        self.state.allow_output = False
        self.state.output_ready = False
        self.state.interrupted = False
        self.state.control_context.clear()
        self.state.control_state.clear()
        self.state.match_recovery_attempts.clear()
        if clear_outputs:
            self.state.outputs.clear()
        if clear_errors:
            self.state.errors.clear()
        if clear_working_memory:
            self.working_memory.clear()
        elif clear_runtime_workspaces and hasattr(self.working_memory, "clear_runtime_workspaces"):
            self.working_memory.clear_runtime_workspaces(
                keep_memory_workspace=True,
            )

    def submit(self, payload: dict[str, Any], *, source: str = "external") -> None:
        self.receive_input(payload, source=source)

    def process_input(self, payload: dict[str, Any], *, source: str = "external") -> dict[str, Any]:
        self.receive_input(payload, source=source)
        self.run()
        return self.read_latest_output() or {}

    def read_latest_output(self) -> dict[str, Any] | None:
        if not self.state.outputs:
            return None
        return dict(self.state.outputs[-1])

    def pop_latest_output(self) -> dict[str, Any] | None:
        if not self.state.outputs:
            return None
        return dict(self.state.outputs.pop())

    def run(self) -> SchedulerNewState:
        if self.state.status in {"idle", "waiting_controller"}:
            self.state.status = "running"
        while self.state.status == "running" and self.state.loop_index < self.state.max_loops:
            self.state.loop_index += 1
            if self.state.phase == "controller":
                self._run_controller()
                continue
            self._dispatch_current_phase()
        if self.state.status == "running":
            self.state.status = "failed"
            self.state.blocking_reason = "max_loops_exceeded"
            self._remember_scheduler_state("max_loops_exceeded")
        return self.state

    def _run_controller(self) -> None:
        self._advance_all_event_context()
        state_snapshot = self.state.to_dict()
        memory_snapshot = self.working_memory.to_dict()
        voc_state = evaluate_control_voc(state_snapshot, memory_snapshot)
        state_snapshot["control_voc"] = voc_state
        self._write_control_voc_event_graph(voc_state)
        feedback = self.controller(state_snapshot, memory_snapshot)
        if not isinstance(feedback, dict):
            feedback = {"next_phase": "failed", "reason": "controller_returned_non_dict"}
        feedback.setdefault("control_voc", voc_state)
        self._apply_controller_feedback(feedback)
        self._remember(
            zone="event_workspace",
            stage="controller",
            source="controller",
            payload_type="controller_feedback",
            payload={"feedback": feedback, "state": self._runtime_state_checkpoint(), "control_voc": voc_state},
        )

    def _advance_all_event_context(self) -> None:
        for event_id, workspace in list(self.working_memory.event_workspaces.items()):
            metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), dict) else {}
            used_context_ids = [
                str(value)
                for value in metadata.pop("used_context_ids_current_round", [])
                if str(value)
            ]
            workspace["metadata"] = metadata
            self.working_memory.advance_event_context_round(
                event_id,
                current_round=self.state.loop_index,
                used_context_ids=used_context_ids,
            )

    def _write_control_voc_event_graph(self, voc_state: dict[str, Any]) -> None:
        event_ids = [str(item) for item in self.state.active_event_ids if str(item)]
        if not event_ids:
            event_ids = [str(event_id) for event_id in self.working_memory.event_workspaces.keys() if str(event_id)]
        if not event_ids:
            event_ids = [f"control_voc_{self.state.session_id or 'session'}"]
        metric_text = (
            f"remaining_uncertainty={voc_state.get('remaining_uncertainty')} "
            f"expected_information_gain={voc_state.get('expected_information_gain')} "
            f"continuation_cost={voc_state.get('continuation_cost')} "
            f"risk_requirement={voc_state.get('risk_requirement')}"
        )
        voc_span = {
            "span_id": "control_voc_state",
            "role": "control_state",
            "core": "control value of continuation",
            "relation": str(voc_state.get("control_status") or "evaluated"),
            "nature": metric_text,
            "text": metric_text,
            "metadata": {"control_voc": dict(voc_state)},
        }
        decision_span = {
            "span_id": "control_decision",
            "role": "control_state",
            "core": "control rule decision",
            "relation": "uses",
            "nature": "VoC evaluation checkpoint",
            "text": str(voc_state.get("control_status") or ""),
            "metadata": {"evaluation_checkpoint": voc_state.get("evaluation_checkpoint", {})},
        }
        relation = {
            "relation_id": "control_voc_informs_control_decision",
            "source_span_id": "control_voc_state",
            "relation_type": "informs",
            "relation": "informs",
            "target_span_id": "control_decision",
            "metadata": {"control_voc": dict(voc_state)},
        }
        for event_id in event_ids:
            for span in (voc_span, decision_span):
                self.working_memory.apply_event_workspace_operation(
                    event_id,
                    {"operation": "update", "target": "spans", "id": span["span_id"], "item": span},
                )
            self.working_memory.apply_event_workspace_operation(
                event_id,
                {
                    "operation": "update",
                    "target": "relations",
                    "id": relation["relation_id"],
                    "item": relation,
                },
            )

    def _apply_controller_feedback(self, feedback: dict[str, Any]) -> None:
        self.state.controller_history.append({"timestamp": _utc_now(), "feedback": dict(feedback)})
        self.state.blocking_reason = str(feedback.get("reason") or "")
        self.state.allow_output = bool(feedback.get("allow_output", self.state.allow_output))
        if isinstance(feedback.get("active_event_ids"), list):
            self.state.active_event_ids = [str(item) for item in feedback["active_event_ids"]]
        if isinstance(feedback.get("active_rule_ids"), list):
            self.state.active_rule_ids = [str(item) for item in feedback["active_rule_ids"]]
        self.state.pending_operations.extend(_as_dict_list(feedback.get("pending_operations")))
        self.state.pending_frontend_patches.extend(_as_dict_list(feedback.get("pending_frontend_patches")))
        self.state.pending_rule_ids.extend([str(item) for item in feedback.get("pending_rule_ids", []) or []])
        self.state.pending_rule_selectors.extend(normalize_rule_selectors(feedback.get("pending_rule_selectors")))
        self.state.pending_output_candidates.extend(_as_dict_list(feedback.get("pending_output_candidates")))
        if isinstance(feedback.get("control_context"), dict):
            self.state.control_context = deepcopy(feedback["control_context"])
        if isinstance(feedback.get("output"), dict):
            self._append_output(dict(feedback["output"]), source="controller_feedback")
        if self._feedback_contains_program(feedback):
            program = self.compiler.compile(feedback)
            self.state.current_program = program
            self.state.current_instruction_id = program.entry_instruction_id
            self.state.phase = "execute_program"
            self._remember(
                zone="event_workspace",
                stage="compile_program",
                source="rule_strategy_compiler",
                payload_type="scheduler_program",
                payload={"program": program.to_dict()},
            )
            return
        next_phase = str(feedback.get("next_phase") or self._choose_next_phase())
        if next_phase == "completed":
            self.state.status = "completed"
            self.state.phase = "completed"
        elif next_phase == "failed":
            self.state.status = "failed"
            self.state.phase = "failed"
        else:
            self.state.phase = next_phase

    def _feedback_contains_program(self, feedback: dict[str, Any]) -> bool:
        program = feedback.get("program")
        if isinstance(program, dict) and (program.get("instructions") or program.get("strategy") or program.get("rule")):
            return True
        if isinstance(feedback.get("strategy"), dict):
            return True
        if isinstance(feedback.get("rule"), dict):
            return True
        if isinstance(feedback.get("instructions"), list):
            return True
        return False

    def _choose_next_phase(self) -> str:
        if self.state.output_ready and self.state.allow_output:
            return "output"
        if self.state.pending_inputs:
            return "event_build"
        if self.state.pending_frontend_patches:
            return "frontend_patch"
        if self.state.pending_operations:
            return "execute"
        if self.state.pending_rule_ids:
            return "rule_match"
        if self.state.pending_rule_selectors:
            return "rule_match"
        if self.state.pending_output_candidates:
            return "rule_match"
        return "rule_match"

    def _dispatch_current_phase(self) -> None:
        phase = self.state.phase
        if phase == "execute_program":
            self._execute_program_step()
            return
        stage_fn = {
            "event_build": self.event_builder,
            "rule_match": self.rule_matcher,
            "execute": self.executor,
            "frontend_patch": self.frontend,
            "output": self.output_decoder,
        }.get(phase)
        if stage_fn is None:
            self.state.status = "failed"
            self.state.errors.append({"timestamp": _utc_now(), "phase": phase, "error": "unknown_phase"})
            return
        result = stage_fn(self.state.to_dict(), self.working_memory, self._stage_payload(phase))
        self._apply_stage_result(phase, result if isinstance(result, dict) else {})

    def _execute_program_step(self) -> None:
        program = self.state.current_program
        if program is None:
            self.state.status = "failed"
            self.state.errors.append({"timestamp": _utc_now(), "phase": "execute_program", "error": "missing_program"})
            return
        instruction = program.get(self.state.current_instruction_id)
        if instruction is None:
            self.state.status = "failed"
            self.state.errors.append(
                {
                    "timestamp": _utc_now(),
                    "phase": "execute_program",
                    "error": "missing_instruction",
                    "instruction_id": self.state.current_instruction_id,
                }
            )
            return
        result = self.instruction_executor.step(
            instruction,
            state=self.state,
            working_memory=self.working_memory,
        )
        self._apply_instruction_result(instruction, result)

    def _stage_payload(self, phase: str) -> dict[str, Any]:
        if phase == "event_build":
            return {"inputs": list(self.state.pending_inputs)}
        if phase == "frontend_patch":
            return {"patches": list(self.state.pending_frontend_patches)}
        if phase == "execute":
            return {"operations": list(self.state.pending_operations)}
        if phase == "rule_match":
            return {
                "rule_ids": list(self.state.pending_rule_ids),
                "rule_selectors": list(self.state.pending_rule_selectors),
            }
        if phase == "output":
            return {"outputs": list(self.state.outputs)}
        return {}

    def _apply_stage_result(self, phase: str, result: dict[str, Any]) -> None:
        self.state.last_action = phase
        self.state.execution_history.append({"timestamp": _utc_now(), "phase": phase, "result": dict(result)})
        self._write_stage_result(phase, result)
        if phase == "event_build":
            self.state.pending_inputs.clear()
        elif phase == "frontend_patch":
            self.state.pending_frontend_patches.clear()
        elif phase == "execute":
            self.state.pending_operations.clear()
        elif phase == "rule_match":
            self.state.pending_rule_ids.clear()
            self.state.pending_rule_selectors.clear()
        self.state.active_event_ids = _merge_unique(self.state.active_event_ids, result.get("active_event_ids", []))
        self.state.active_rule_ids = _merge_unique(self.state.active_rule_ids, result.get("active_rule_ids", []))
        self.state.pending_operations.extend(_as_dict_list(result.get("pending_operations")))
        self.state.pending_frontend_patches.extend(_as_dict_list(result.get("pending_frontend_patches")))
        self.state.pending_rule_ids.extend([str(item) for item in result.get("pending_rule_ids", []) or []])
        self.state.pending_rule_selectors.extend(normalize_rule_selectors(result.get("pending_rule_selectors")))
        self.state.pending_output_candidates.extend(_as_dict_list(result.get("pending_output_candidates")))
        if phase != "output" and isinstance(result.get("output"), dict):
            self._append_output(dict(result["output"]), source=f"{phase}_stage")
        if phase == "output":
            if not self.state.allow_output:
                self.state.phase = "controller"
                self.state.blocking_reason = "controller_has_not_allowed_output"
                return
            self.state.status = "completed"
            self.state.phase = "completed"
            return
        if result.get("stop"):
            self.state.output_ready = True
        self.state.phase = "controller"

    def _apply_instruction_result(self, instruction: SchedulerInstruction, result: InstructionResult) -> None:
        self.state.last_action = instruction.op
        result_payload = result.to_dict()
        self.state.instruction_history.append({"timestamp": _utc_now(), "instruction": instruction.to_dict(), "result": result_payload})
        self._write_instruction_result(instruction, result)
        self._apply_produced_operations(result.produced)
        if not result.ok and result.error:
            self.state.errors.append(
                {
                    "timestamp": _utc_now(),
                    "phase": "execute_program",
                    "instruction_id": instruction.instruction_id,
                    "op": instruction.op,
                    "error": result.error,
                }
            )
        next_ids = result.next_instruction_ids or (instruction.on_success if result.ok else instruction.on_fail)
        if next_ids:
            self.state.current_instruction_id = str(next_ids[0])
            self.state.phase = "execute_program"
            return
        if instruction.op == "output" and self.state.allow_output:
            self.state.status = "completed"
            self.state.phase = "completed"
            return
        if instruction.op == "stop_path":
            self.state.output_ready = True
        self.state.phase = "controller"

    def _write_instruction_result(self, instruction: SchedulerInstruction, result: InstructionResult) -> None:
        self._remember(
            zone="event_workspace",
            stage="execute_instruction",
            source="instruction_executor",
            payload_type="instruction_result",
            payload={
                "instruction": instruction.to_dict(),
                "result": result.to_dict(),
                "program_id": self.state.current_program.program_id if self.state.current_program else "",
            },
        )

    def _apply_produced_operations(self, produced: list[dict[str, Any]]) -> None:
        for item in produced:
            operation = str(item.get("operation") or item.get("op") or "").strip()
            if operation == "request_frontend_patch":
                self.state.pending_frontend_patches.append(dict(item))
            elif operation == "run_rule":
                self.state.pending_rule_ids.extend([str(value) for value in item.get("rule_ids", []) or []])
                self.state.pending_rule_selectors.extend(_rule_selectors_from_run_rule_operation(item))
            elif operation == "match":
                matched_rules = [
                    value for value in item.get("matched_rules", []) or []
                    if isinstance(value, dict)
                ]
                rule_ids = [
                    str((value.get("rule") or {}).get("rule_id") or "")
                    for value in matched_rules
                    if str((value.get("rule") or {}).get("rule_id") or "")
                ]
                self.state.active_rule_ids = list(dict.fromkeys([*self.state.active_rule_ids, *rule_ids]))
                if matched_rules:
                    executor_result = execute_rule_chain(
                        calculator=self.calculator,
                        matched_rules=matched_rules,
                        working_memory=self.working_memory,
                        session_id=self.state.session_id,
                        chain_id=str(item.get("chain_id") or "") or None,
                        source_match=item,
                        control_context=self._current_control_context(),
                        max_steps=self.state.max_rule_chain_steps,
                    )
                    self._apply_executor_result(executor_result)
                else:
                    self.state.pending_rule_ids.extend(rule_ids)
            elif operation in {"add", "update", "delete", "merge", "split"}:
                if str(item.get("target_workspace") or "") == "memory_workspace":
                    try:
                        result = self.working_memory.apply_memory_workspace_operation(
                            item,
                            rule_id=str(item.get("rule_id") or ""),
                            session_id=self.state.session_id,
                        )
                        self._remember(
                            zone="event_workspace",
                            stage="memory_workspace_rule_operation",
                            source="produced_operation",
                            payload_type="memory_workspace_mutation",
                            payload={"operation": dict(item), "result": result},
                            tags=["scheduler-new", "memory_workspace_mutation"],
                        )
                    except ValueError as exc:
                        self._remember(
                            zone="event_workspace",
                            stage="rejected_memory_workspace_operation",
                            source="produced_operation",
                            payload_type="runtime_trace",
                            payload={"operation": dict(item), "reason": str(exc)},
                            confidence=0.0,
                            tags=["scheduler-new", "memory_workspace_boundary"],
                        )
                    continue
                event_id = str(item.get("event_id") or "")
                if event_id:
                    try:
                        self.working_memory.apply_event_workspace_operation(event_id, item)
                    except ValueError as exc:
                        self._remember(
                            zone="event_workspace",
                            stage="rejected_event_workspace_operation",
                            source="produced_operation",
                            payload_type="runtime_trace",
                            payload={
                                "event_id": event_id,
                                "operation": dict(item),
                                "reason": str(exc),
                            },
                            confidence=0.5,
                            tags=["scheduler-new", "event_workspace_boundary", str(operation)],
                        )
            elif operation in {"retrieve_memory", "query_memory"}:
                self._execute_memory_retrieval_operation(item)
            elif operation in {"retrieve_workspace_memory", "query_memory_workspace"}:
                self._execute_workspace_memory_retrieval_operation(item)
            elif operation in {"retrieve_knowledge", "query_knowledge", "enrich_knowledge"}:
                self._execute_knowledge_retrieval_operation(item)
            elif operation == "mark_context_used":
                context_ids = [str(value) for value in item.get("context_ids", []) if str(value)]
                for event_id in item.get("event_ids", []) or self.state.active_event_ids:
                    workspace = self.working_memory.get_event_workspace(str(event_id))
                    metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), dict) else {}
                    metadata["used_context_ids_current_round"] = list(
                        dict.fromkeys(
                            [
                                *[str(value) for value in metadata.get("used_context_ids_current_round", []) if str(value)],
                                *context_ids,
                            ]
                        )
                    )
                    workspace["metadata"] = metadata
            elif operation == "output":
                self._append_output(dict(item), source="produced_operation")
            elif operation == "stop_path":
                self._append_output(dict(item), source="produced_operation")
            elif operation:
                self.state.pending_operations.append(dict(item))

    def _apply_executor_result(self, result: dict[str, Any]) -> None:
        self._apply_control_updates(result)
        self.state.pending_rule_ids.extend([str(value) for value in result.get("next_rule_ids", []) or []])
        self.state.pending_rule_selectors.extend(normalize_rule_selectors(result.get("next_rule_selectors")))
        self.state.pending_frontend_patches.extend(_as_dict_list(result.get("frontend_requests")))
        self.state.pending_operations.extend(_as_dict_list(result.get("pending_operations")))
        for output in _as_dict_list(result.get("outputs")):
            self._append_output(output, source="executor_result")
        for stopped in _as_dict_list(result.get("stopped_paths")):
            self._append_output(stopped, source="executor_result")
        if result.get("errors"):
            self.state.errors.append(
                {
                    "timestamp": _utc_now(),
                    "phase": "executor_new",
                    "chain_id": result.get("chain_id", ""),
                    "errors": result.get("errors", []),
                }
            )
        self._remember(
            zone="event_workspace",
            stage="executor_chain_result",
            source="executor-new",
            payload_type="executor_result",
            payload={"result": result},
            confidence=1.0 if result.get("ok", False) else 0.0,
            tags=["scheduler-new", "executor-new", str(result.get("status") or "")],
        )

    def _apply_control_updates(self, result: dict[str, Any]) -> None:
        """Commit state declared by successfully executed Control actions."""
        if not result.get("ok"):
            return
        for update in result.get("control_updates", []):
            values = update["values"]
            self._synchronize_active_strategy_state(
                values,
                source_rule_id=str(update.get("rule_id") or ""),
            )
            previous_owner = self.state.control_state.get("strategy_rule_id")
            next_owner = values.get("strategy_rule_id")
            if next_owner and next_owner != previous_owner:
                # Audit Control-authored transitions; do not select a strategy.
                self.state.control_state.setdefault("original_strategy", next_owner)
                self.state.control_state.setdefault("strategy_history", []).append({
                    "from_strategy": previous_owner,
                    "to_strategy": next_owner,
                    "control_rule_id": update["rule_id"],
                    "previous_route": self.state.control_state.get("active_route"),
                    "active_route": values.get("active_route"),
                    "recovery_source": values.get("recovery_source"),
                })
            self.state.control_state.update(deepcopy(update["values"]))
            self._remember(
                zone="event_workspace", stage="control_state", source=update["rule_id"],
                payload_type="control_state_update", payload=deepcopy(update),
            )

    def _synchronize_active_strategy_state(
        self,
        values: dict[str, Any],
        *,
        source_rule_id: str,
    ) -> None:
        """Activate exactly the strategy state selected by a Control Rule.

        Strategy spans are retained as graph history.  Their lifecycle is
        separate from their semantic content: a subsequent Control decision
        marks the previously selected strategy inactive, so old route Rules
        cannot run after a recovery switch.  This method only applies the
        state declared by the Control Rule; it never selects a strategy.
        """
        active_state = str(values.get("active_strategy_state") or "").strip()
        if not active_state:
            return
        event_id = str(values.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("active_strategy_state requires event_id")
        workspace = self.working_memory.get_event_workspace(event_id)
        strategy_spans = [
            dict(span)
            for span in workspace.get("spans", [])
            if isinstance(span, dict)
            and str(span.get("span_role") or "") == "control_state"
            and str(span.get("feature_type") or "").startswith("strategy_")
        ]
        if not any(str(span.get("feature_type") or "") == active_state for span in strategy_spans):
            raise ValueError(
                f"Control Rule {source_rule_id or '<unknown>'} declared active strategy "
                f"{active_state!r}, but that strategy state is absent from event {event_id!r}"
            )
        for span in strategy_spans:
            metadata = dict(span.get("metadata") or {})
            is_active = str(span.get("feature_type") or "") == active_state
            if metadata.get("control_state_active") is is_active:
                continue
            metadata.update({
                "control_state_active": is_active,
                "control_state_updated_by_rule_id": source_rule_id,
            })
            self.working_memory.apply_event_workspace_operation(
                event_id,
                {
                    "operation": "update",
                    "target": "spans",
                    "id": str(span.get("span_id") or ""),
                    "item": {"span_id": str(span.get("span_id") or ""), "metadata": metadata},
                },
            )

    def _append_output(self, output: dict[str, Any], *, source: str) -> None:
        gated = self._apply_output_gate_decision(output)
        if gated is None:
            candidate = _prepare_output_candidate(output)
            self.state.pending_output_candidates.append(candidate)
            self.state.pending_rule_selectors.extend(_output_gate_rule_selectors(candidate))
            self._remember(
                zone="event_workspace",
                stage="output_candidate",
                source=source,
                payload_type="output_candidate",
                payload={"candidate": candidate},
                confidence=0.5,
                tags=["scheduler-new", "output_candidate", str(candidate.get("answer_granularity") or "")],
            )
            return
        if gated.get("operation") == "request_output_revision":
            candidate_output = gated.get("candidate_output")
            candidate = _prepare_output_candidate(
                dict(candidate_output) if isinstance(candidate_output, dict) else output
            )
            candidate.update(
                {
                    "status": "output_revision_required",
                    "revision_reason": str(gated.get("reason") or ""),
                    "revision_context": dict(gated.get("revision_context") or {}),
                    "revision_attempt": int(gated.get("revision_attempt") or 1),
                }
            )
            self.state.pending_output_candidates.append(candidate)
            self.state.pending_rule_selectors.extend(
                _output_revision_rule_selectors(candidate, gated)
            )
            self._remember(
                zone="event_workspace",
                stage="output_revision_required",
                source=source,
                payload_type="output_revision_request",
                payload={
                    "candidate": candidate,
                    "decision": gated,
                    "pending_selector_count": len(self.state.pending_rule_selectors),
                },
                confidence=0.0,
                tags=["scheduler-new", "output_revision_required"],
            )
            return
        if gated.get("operation") == "stop_path" and any(
            str(existing.get("operation") or "") != "stop_path"
            for existing in self.state.outputs
        ):
            self._remember(
                zone="event_workspace",
                stage="late_stop_path_after_output",
                source=source,
                payload_type="output_gate_result",
                payload={"input": output, "ignored_stop_path": gated},
                confidence=0.0,
                tags=["scheduler-new", "late_stop_path_after_output"],
            )
            return
        # Lifecycle completion is not a semantic verdict. Only copy a verdict
        # supplied by the rule; an unclassified system stop stays unknown.
        gated["task_result"] = deepcopy(gated.get("task_result") or {"status": "unknown"})
        self.state.control_state["task_result"] = deepcopy(gated["task_result"])
        gated["control_state"] = deepcopy(self.state.control_state)
        self.state.outputs.append(gated)
        self.state.output_ready = True
        if gated.get("operation") == "stop_path":
            self.state.blocking_reason = str(gated.get("reason") or gated.get("status") or "")
            # A rule-issued terminal result must be deliverable even when it
            # rejects a pending answer. Delivery is not a solved-task verdict.
            gated["discarded_output_candidate_count"] = len(self.state.pending_output_candidates)
            self.state.pending_output_candidates.clear()
            self.state.allow_output = True
        else:
            self.state.allow_output = True
        self._remember(
            zone="event_workspace",
            stage="output_gate",
            source=source,
            payload_type="output_gate_result",
            payload={"input": output, "output": gated, "pending_candidate_count": len(self.state.pending_output_candidates)},
            confidence=1.0 if gated.get("operation") != "stop_path" else 0.0,
        )

    def _apply_output_gate_decision(self, output: dict[str, Any]) -> dict[str, Any] | None:
        operation = str(output.get("operation") or "").strip()
        decision = str(
            output.get("output_gate_decision")
            or output.get("gate_decision")
            or ""
        ).strip().lower()
        if decision in {"allow", "allow_output", "allow_current_candidate", "release_current_candidate"}:
            candidate = self.state.pending_output_candidates[-1] if self.state.pending_output_candidates else {}
            released = dict(candidate.get("candidate_output") if isinstance(candidate.get("candidate_output"), dict) else candidate)
            if self.state.control_state.get("strategy_rule_id"):
                pending_count = len(self.state.pending_output_candidates)
                completion = self._active_control_completion_decision(released)
                if completion is not None:
                    if len(self.state.pending_output_candidates) == pending_count and pending_count:
                        self.state.pending_output_candidates.pop()
                    return completion
            if self.state.pending_output_candidates:
                self.state.pending_output_candidates.pop()
            if "task_result" in output:
                released["task_result"] = deepcopy(output["task_result"])
            released.update(
                {
                    "operation": "output",
                    "output_gate_checked": True,
                    "output_gate_decision": "allow_current_candidate",
                    "output_gate_rule_ids": list(dict.fromkeys(_as_text_list(output.get("rules_applied")) + _as_text_list(output.get("rule_path_rule_ids")))),
                    "output_gate_reason": str(output.get("reason") or output.get("conclusion") or ""),
                }
            )
            return released
        if decision in {"block", "block_output", "block_current_candidate"}:
            candidate = self.state.pending_output_candidates.pop() if self.state.pending_output_candidates else {}
            candidate_output = candidate.get("candidate_output") if isinstance(candidate.get("candidate_output"), dict) else candidate
            return self._output_revision_decision(
                candidate_output=dict(candidate_output),
                status=str(output.get("status") or "output_gate_blocked"),
                reason=str(output.get("reason") or "output gate rule rejected the current output candidate"),
                decision_source="fallback_output_gate",
                revision_rule_ids=_as_text_list(output.get("revision_rule_ids")),
                next_rule_selectors=normalize_rule_selectors(output.get("next_rule_selectors")),
                analysis={
                    "output_gate_rule_ids": list(
                        dict.fromkeys(
                            _as_text_list(output.get("rules_applied"))
                            + _as_text_list(output.get("rule_path_rule_ids"))
                        )
                    ),
                    "gate_result": _compact_output_for_gate(output),
                },
            )
        if operation == "stop_path":
            return output
        if bool(output.get("output_gate_checked")):
            return output
        control_completion = self._active_control_completion_decision(output)
        if control_completion is not None:
            return control_completion
        return None

    def _active_control_completion_decision(self, output: dict[str, Any]) -> dict[str, Any] | None:
        """Apply completion criteria declared by active control rules.

        A generic output-gate rule is only needed when no active control rule
        declares completion criteria. The criteria remain rule data and are
        evaluated by the normal structural matcher.
        """

        rule_db_path = self._rule_db_path()
        rule_ids = self.state.active_rule_ids
        if "strategy_rule_id" in self.state.control_state:
            owner = self.state.control_state["strategy_rule_id"]
            rule_ids = [owner] if owner else []
        if not rule_db_path or not rule_ids:
            return None
        active_rules = _load_rule_records_by_ids(rule_db_path, rule_ids)
        completion_rules: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for rule in active_rules:
            if str(rule.get("rule_layer") or "").strip().lower() != "control":
                continue
            metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
            completion_required = _as_dict_list(metadata.get("completion_required"))
            if completion_required:
                completion_rules.append((rule, completion_required))
        if not completion_rules:
            if self.state.control_state.get("strategy_rule_id"):
                return self._output_revision_decision(
                    candidate_output=output, status="control_completion_contract_missing",
                    reason="the declared strategy has no executable completion contract",
                    decision_source="active_control_completion",
                )
            return None

        evaluations: list[dict[str, Any]] = []
        for rule, completion_required in completion_rules:
            rule_id = str(rule.get("rule_id") or "")
            result = match_source_to_target(
                {
                    "source": {
                        "type": "required",
                        "rule_id": rule_id,
                        "rule_layer": "control",
                        "required": completion_required,
                    },
                    "target": {
                        "type": "working_memory",
                        "zones": ["event_workspace"],
                        "event_ids": (
                            [self.state.control_state["event_id"]]
                            if self.state.control_state.get("event_id")
                            else list(self.state.active_event_ids)
                        ),
                    },
                    "match_threshold": 0.55,
                },
                self.state,
                self.working_memory,
            )
            evaluations.append(
                {
                    "rule_id": rule_id,
                    "ok": bool(result.get("ok")),
                    "metadata": _compact_matcher_metadata(result.get("metadata", {})),
                }
            )
            if result.get("ok"):
                self._discard_latest_output_revision_candidate()
                released = dict(output)
                released.update(
                    {
                        "operation": "output",
                        "control_completion_checked": True,
                        "control_completion_decision": "allow_output",
                        "output_gate_checked": True,
                        "output_gate_decision": "allow_current_candidate",
                        "output_gate_rule_ids": [rule_id],
                        "completion_control_rule_ids": [rule_id],
                        "output_authority": "active_control_completion",
                    }
                )
                declared_result = (rule.get("metadata") or {}).get("completion_result")
                if declared_result is not None:
                    from .control_state import validate_task_result

                    released["task_result"] = {
                        **validate_task_result(declared_result), "control_rule_id": rule_id,
                    }
                return released

        return self._output_revision_decision(
            candidate_output=output,
            status="control_completion_not_satisfied",
            reason="active control rule completion criteria were not satisfied",
            decision_source="active_control_completion",
            analysis={
                "control_completion_checked": True,
                "control_completion_decision": "revise_output",
                "completion_evaluations": evaluations,
                "completion_control_rule_ids": [
                    str(rule.get("rule_id") or "") for rule, _required in completion_rules
                ],
            },
        )

    def _output_revision_decision(
        self,
        *,
        candidate_output: dict[str, Any],
        status: str,
        reason: str,
        decision_source: str,
        revision_rule_ids: list[str] | None = None,
        next_rule_selectors: list[dict[str, Any]] | None = None,
        analysis: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Create a bounded, rule-driven output revision request.

        The scheduler records the rejection and routes it back to rules. It does
        not decide how the graph should be repaired. A terminal stop is emitted
        only after repeated rule-driven revisions cannot produce a valid output.
        """

        revision_context = {
            "status": str(status),
            "decision_source": str(decision_source),
            "active_event_ids": list(self.state.active_event_ids),
            "active_rule_ids": list(self.state.active_rule_ids),
            **dict(analysis or {}),
        }
        revision_key = stable_content_id(
            "output_revision",
            {
                "status": status,
                "decision_source": decision_source,
                "active_event_ids": sorted(self.state.active_event_ids),
                "completion_control_rule_ids": sorted(
                    _as_text_list(revision_context.get("completion_control_rule_ids"))
                ),
                "answer_granularity": _output_granularity(candidate_output),
            },
        )
        attempt_count = int(self.state.match_recovery_attempts.get(revision_key, 0)) + 1
        self.state.match_recovery_attempts[revision_key] = attempt_count
        if attempt_count > max(1, int(max_attempts)):
            return {
                "operation": "stop_path",
                "status": "output_revision_exhausted",
                "reason": "rule-driven output revision did not satisfy an output completion contract",
                "blocked_output": _compact_output_for_gate(candidate_output),
                "revision_context": revision_context,
                "revision_attempt": attempt_count,
            }
        return {
            "operation": "request_output_revision",
            "status": str(status),
            "reason": str(reason),
            "candidate_output": dict(candidate_output),
            "revision_context": revision_context,
            "revision_attempt": attempt_count,
            "revision_rule_ids": list(dict.fromkeys(revision_rule_ids or [])),
            "next_rule_selectors": normalize_rule_selectors(next_rule_selectors),
        }

    def _discard_latest_output_revision_candidate(self) -> None:
        for index in range(len(self.state.pending_output_candidates) - 1, -1, -1):
            candidate = self.state.pending_output_candidates[index]
            if str(candidate.get("status") or "") == "output_revision_required":
                del self.state.pending_output_candidates[index]
                return

    def _write_stage_result(self, phase: str, result: dict[str, Any]) -> None:
        zone = {
            "event_build": "event_workspace",
            "rule_match": "event_workspace",
            "execute": "event_workspace",
            "frontend_patch": "event_workspace",
            "output": "event_workspace",
        }.get(phase, "event_workspace")
        self._remember(
            zone=zone,
            stage=phase,
            source=f"{phase}_module",
            payload_type=f"{phase}_result",
            payload={"result": result},
        )
        for item in _as_dict_list(result.get("event_workspaces")):
            event_id = str(item.get("event_id") or "")
            if event_id:
                self.working_memory.upsert_event_workspace(event_id, item)
        for item in _as_dict_list(result.get("event_operations")):
            event_id = str(item.get("event_id") or "")
            if event_id:
                try:
                    self.working_memory.apply_event_workspace_operation(event_id, item)
                except ValueError as exc:
                    self._remember(
                        zone="event_workspace",
                        stage="rejected_event_workspace_operation",
                        source=f"{phase}_module",
                        payload_type="runtime_trace",
                        payload={
                            "event_id": event_id,
                            "operation": dict(item),
                            "reason": str(exc),
                        },
                        confidence=0.5,
                        tags=["scheduler-new", "event_workspace_boundary", phase],
                    )
        self._write_stage_runtime_record(phase, result)

    def _execute_memory_retrieval_operation(self, operation: dict[str, Any]) -> None:
        memory_db_path = str((self.state.db_paths or {}).get("memory") or (self.state.db_paths or {}).get("memory_db") or "").strip()
        if not memory_db_path:
            self.state.pending_operations.append(
                {
                    "operation": "stop_path",
                    "status": "memory_db_not_configured",
                    "reason": "memory retrieval was requested, but no memory DB path is configured",
                }
            )
            return
        top_k = max(1, int(_safe_float(operation.get("top_k"), 6)))
        query = str(operation.get("query") or operation.get("debug_text") or operation.get("reason") or "").strip()
        event_ids = [str(item) for item in operation.get("event_ids", []) or self.state.active_event_ids or self.working_memory.event_workspaces.keys() if str(item)]
        retention_rule_id = str(operation.get("rule_id") or operation.get("retention_rule_id") or "").strip()
        expire_after = int(_safe_float(operation.get("expire_after_unused_rounds"), 3))
        if not retention_rule_id:
            self.state.pending_operations.append(
                {
                    "operation": "stop_path",
                    "status": "memory_retrieval_missing_rule_id",
                    "reason": "memory retrieval retention must be owned by an executed rule",
                }
            )
            return
        added = 0
        for event_id in event_ids:
            workspace = self.working_memory.event_workspaces.get(event_id)
            if not isinstance(workspace, dict):
                continue
            try:
                records = retrieve_memory(
                    self.state.db_paths,
                    event_workspace=workspace,
                    query=query,
                    top_k=top_k,
                    memory_cache=self.memory_cache,
                )
            except ValueError as exc:
                self.state.pending_operations.append(
                    {
                        "operation": "stop_path",
                        "status": "memory_query_vector_unavailable",
                        "event_id": event_id,
                        "reason": str(exc),
                    }
                )
                continue
            summary = self.working_memory.add_event_context_records(
                event_id,
                records,
                source=(
                    str(records[0].get("retrieval_source") or "memory_db")
                    if records
                    else "memory_db"
                ),
                current_round=self.state.loop_index,
                retention_rule_id=retention_rule_id,
                expire_after_unused_rounds=expire_after,
            )
            added += int(summary.get("added") or 0)
        self._remember(
            zone="event_workspace",
            stage="memory_retrieval",
            source="control_rule",
            payload_type="event_context_retrieval",
            payload={
                "operation": dict(operation),
                "query": query,
                "event_ids": event_ids,
                "record_count": added,
            },
            tags=["scheduler-new", "memory_retrieval", "event_context"],
        )

    def _execute_workspace_memory_retrieval_operation(self, operation: dict[str, Any]) -> None:
        retention_rule_id = str(operation.get("rule_id") or operation.get("retention_rule_id") or "").strip()
        expire_after = int(_safe_float(operation.get("expire_after_unused_rounds"), 3))
        if not retention_rule_id:
            raise ValueError("memory_workspace retrieval requires the executed rule_id")
        top_k = max(1, int(_safe_float(operation.get("top_k"), 6)))
        event_ids = [
            str(item)
            for item in operation.get("event_ids", [])
            or self.state.active_event_ids
            or self.working_memory.event_workspaces.keys()
            if str(item)
        ]
        added = 0
        for event_id in event_ids:
            workspace = self.working_memory.event_workspaces.get(event_id)
            if not isinstance(workspace, dict):
                continue
            query = build_event_query(workspace)
            context = select_memory_workspace_context(
                self.working_memory.query_memory_entries(),
                query_vector=query.get("query_vector"),
                query_embedding_model=str(query.get("query_embedding_model") or ""),
                query_vector_version=str(query.get("query_vector_version") or ""),
                top_k=top_k,
            )
            summary = self.working_memory.add_event_context_records(
                event_id,
                list(context.get("records") or []),
                source="memory_workspace",
                current_round=self.state.loop_index,
                retention_rule_id=retention_rule_id,
                expire_after_unused_rounds=expire_after,
            )
            added += int(summary.get("added") or 0)
        self._remember(
            zone="event_workspace",
            stage="memory_workspace_retrieval",
            source="control_rule",
            payload_type="event_context_retrieval",
            payload={
                "operation": dict(operation),
                "event_ids": event_ids,
                "record_count": added,
            },
            tags=["scheduler-new", "memory_workspace_retrieval", "event_context"],
        )

    def _execute_knowledge_retrieval_operation(self, operation: dict[str, Any]) -> None:
        knowledge_db_path = str((self.state.db_paths or {}).get("knowledge") or (self.state.db_paths or {}).get("knowledge_db") or "").strip()
        if not knowledge_db_path:
            self.state.pending_operations.append(
                {
                    "operation": "stop_path",
                    "status": "knowledge_db_not_configured",
                    "reason": "knowledge enrichment was requested, but no knowledge DB path is configured",
                }
            )
            return
        top_k = max(1, int(_safe_float(operation.get("top_k"), 3)))
        min_score = _safe_float(operation.get("min_score"), 0.28)
        event_ids = [str(item) for item in operation.get("event_ids", []) or self.state.active_event_ids or self.working_memory.event_workspaces.keys() if str(item)]
        retention_rule_id = str(operation.get("rule_id") or operation.get("retention_rule_id") or "").strip()
        expire_after = int(_safe_float(operation.get("expire_after_unused_rounds"), 3))
        if not retention_rule_id:
            self.state.pending_operations.append(
                {
                    "operation": "stop_path",
                    "status": "knowledge_retrieval_missing_rule_id",
                    "reason": "knowledge retrieval retention must be owned by an executed rule",
                }
            )
            return
        summaries: dict[str, dict[str, Any]] = {}
        for event_id in event_ids:
            workspace = self.working_memory.event_workspaces.get(event_id)
            if not isinstance(workspace, dict):
                continue
            query_specs = _knowledge_operation_query_specs(operation, workspace)
            existing_knowledge_ids = {
                str(item.get("source_record_id") or (item.get("payload") or {}).get("knowledge_id") or "")
                for item in workspace.get("context_items", []) or []
                if isinstance(item, dict)
            }
            try:
                records_by_id: dict[str, dict[str, Any]] = {}
                schema_lookup = bool(operation.get("schema_for_alignment_requirements"))
                if schema_lookup:
                    from .persistence.knowledge_db import load_knowledge_records
                    from .semantic_alignment import select_span_schema_knowledge_ids
                    all_records = load_knowledge_records(knowledge_db_path, limit=None)
                    selected_ids = select_span_schema_knowledge_ids(
                        all_records, operation.get("alignment_requirements", [])
                    )
                    records_by_id = {
                        record["id"]: {**record, "score": 1.0}
                        for record in all_records
                        if record.get("id") in set(selected_ids) and record.get("status") == "active"
                    }
                    query_specs = []
                elif operation.get("knowledge_ids"):
                    from .persistence.knowledge_db import load_knowledge_records
                    wanted = set(operation["knowledge_ids"])
                    records_by_id = {r["id"]: {**r, "score": 1.0}
                                     for r in load_knowledge_records(knowledge_db_path, limit=None, knowledge_ids=list(wanted))
                                     if r["id"] in wanted and r.get("status") == "active"}
                    query_specs = []
                for query_spec in query_specs:
                    for record in retrieve_knowledge(
                        self.state.db_paths,
                        event_workspace=workspace,
                        query=str(operation.get("query") or operation.get("debug_text") or ""),
                        query_vector=query_spec.get("query_vector"),
                        query_embedding_model=str(query_spec.get("embedding_model") or ""),
                        query_vector_version=str(query_spec.get("vector_version") or ""),
                        top_k=max(top_k * 4, 8),
                    ):
                        knowledge_id = str(record.get("knowledge_id") or record.get("id") or "")
                        if not knowledge_id or knowledge_id in existing_knowledge_ids:
                            continue
                        previous = records_by_id.get(knowledge_id)
                        if previous is None or _safe_float(record.get("score"), 0.0) > _safe_float(previous.get("score"), 0.0):
                            records_by_id[knowledge_id] = record
                records = sorted(
                    records_by_id.values(),
                    key=lambda item: (_safe_float(item.get("score"), 0.0), str(item.get("knowledge_id") or item.get("id") or "")),
                    reverse=True,
                )
                # A schema is a multi-edge definition.  Truncating it to a
                # generic retrieval top_k could turn an all_cores definition
                # into an accidental single-core definition.
                if not schema_lookup:
                    records = records[:top_k]
            except ValueError as exc:
                self.state.pending_operations.append(
                    {
                        "operation": "stop_path",
                        "status": "knowledge_query_vector_unavailable",
                        "event_id": event_id,
                        "reason": str(exc),
                    }
                )
                continue
            records = [record for record in records if _safe_float(record.get("score"), 0.0) >= min_score]
            context_records: list[dict[str, Any]] = []
            contract_errors: list[dict[str, str]] = []
            for record in records:
                try:
                    context_records.append(knowledge_record_to_event_context(record))
                except (TypeError, ValueError) as exc:
                    contract_errors.append(
                        {
                            "knowledge_id": str(record.get("knowledge_id") or record.get("id") or ""),
                            "error": str(exc),
                        }
                    )
            if contract_errors:
                self.state.pending_operations.append(
                    {
                        "operation": "stop_path",
                        "status": "knowledge_vector_contract_failed",
                        "event_id": event_id,
                        "errors": contract_errors,
                    }
                )
            summaries[event_id] = self.working_memory.add_event_context_records(
                event_id,
                context_records,
                source="knowledge_db",
                current_round=self.state.loop_index,
                retention_rule_id=retention_rule_id,
                expire_after_unused_rounds=expire_after,
            )
            summaries[event_id]["rejected_vector_contracts"] = contract_errors
            summaries[event_id]["query_count"] = len(query_specs)
        if summaries:
            self._remember(
                zone="event_workspace",
                stage="knowledge_enrichment",
                source="control_rule",
                payload_type="knowledge_enrichment_summary",
                payload={
                    "operation": dict(operation),
                    "event_summaries": summaries,
                    "event_count": len(summaries),
                },
                tags=["scheduler-new", "knowledge_enrichment", "control_requested"],
            )

    def _write_stage_runtime_record(self, phase: str, result: dict[str, Any]) -> None:
        event_ids = list(dict.fromkeys([*self.state.active_event_ids, *[str(item) for item in result.get("active_event_ids", []) or [] if str(item)]]))
        if not event_ids:
            event_ids = list(self.working_memory.event_workspaces.keys())
        feedback = {
            "id": new_runtime_id("runtime_feedback"),
            "feedback_signal": f"{phase}_result",
            "source": f"{phase}_module",
            "phase": phase,
            "status": str(result.get("status") or result.get("phase") or ""),
            "reason": str(result.get("reason") or ""),
            "active_rule_ids": [str(item) for item in result.get("active_rule_ids", []) or [] if str(item)],
            "pending_rule_ids": [str(item) for item in result.get("pending_rule_ids", []) or [] if str(item)],
            "pending_rule_selector_count": len(result.get("pending_rule_selectors", []) or []),
            "output_status": str((result.get("output") if isinstance(result.get("output"), dict) else {}).get("status") or ""),
        }
        self._remember(
            zone="event_workspace",
            stage=f"{phase}_runtime_record",
            source=f"{phase}_module",
            payload_type="runtime_feedback",
            payload={
                "event_ids": event_ids,
                "feedback": feedback,
                "phase_result": result,
            },
            confidence=1.0,
            tags=["scheduler-new", "runtime_feedback", phase],
        )

    def _remember_scheduler_state(self, reason: str) -> None:
        self._remember(
            zone="event_workspace",
            stage="scheduler_state",
            source="scheduler-new",
            payload_type="scheduler_state",
            payload={"reason": reason, "state": self._runtime_state_checkpoint()},
        )

    def _runtime_state_checkpoint(self) -> dict[str, Any]:
        """Return a bounded scheduler checkpoint for workspace audit records.

        Full execution results already have their own stage records. Repeating
        complete histories inside every controller record makes each new event
        package contain all previous records again and causes quadratic growth.
        """

        return {
            "session_id": self.state.session_id,
            "status": self.state.status,
            "phase": self.state.phase,
            "loop_index": self.state.loop_index,
            "max_loops": self.state.max_loops,
            "max_rule_chain_steps": self.state.max_rule_chain_steps,
            "active_event_ids": list(self.state.active_event_ids),
            "active_rule_ids": list(self.state.active_rule_ids),
            "pending_input_count": len(self.state.pending_inputs),
            "pending_operation_count": len(self.state.pending_operations),
            "pending_frontend_patch_count": len(self.state.pending_frontend_patches),
            "pending_rule_ids": list(self.state.pending_rule_ids),
            "pending_rule_selector_count": len(self.state.pending_rule_selectors),
            "pending_output_candidate_count": len(self.state.pending_output_candidates),
            "current_instruction_id": self.state.current_instruction_id,
            "controller_history_count": len(self.state.controller_history),
            "execution_history_count": len(self.state.execution_history),
            "instruction_history_count": len(self.state.instruction_history),
            "output_count": len(self.state.outputs),
            "error_count": len(self.state.errors),
            "blocking_reason": self.state.blocking_reason,
            "last_action": self.state.last_action,
            "allow_output": self.state.allow_output,
            "output_ready": self.state.output_ready,
            "interrupted": self.state.interrupted,
        }

    def _remember(
        self,
        *,
        zone: str,
        stage: str,
        source: str,
        payload_type: str,
        payload: dict[str, Any],
        confidence: float = 1.0,
        tags: list[str] | None = None,
    ) -> WorkingMemoryEntry:
        entry = WorkingMemoryEntry(
            entry_id=new_runtime_id("snew"),
            session_id=self.state.session_id,
            stage=stage,
            source=source,
            payload_type=payload_type,
            created_at=_utc_now(),
            confidence=float(confidence),
            workspace_zone=zone,
            payload=dict(payload or {}),
            tags=tags or ["scheduler-new", stage],
        )
        self.working_memory.add(entry, zone=zone)
        return entry

    def _request_control_recovery(
        self,
        *,
        failure_type: str,
        status: str,
        reason: str,
        target_rule_layer: str = "",
        pending_rule_ids: list[str] | None = None,
        rule_selector: dict[str, Any] | None = None,
        matcher_result: dict[str, Any] | None = None,
        executor_results: list[dict[str, Any]] | None = None,
        operations: list[dict[str, Any]] | None = None,
        max_attempts: int = 4,
    ) -> dict[str, Any] | None:
        """Record a runtime failure and ask control rules to choose the next strategy.

        Runtime state remains outside the business Event Graph. The scheduler
        records facts and invokes control rules; it does not select a recovery
        path, widen domain search, or decide whether the answer is complete.
        """

        target_rule_layer = str(target_rule_layer or "").strip().lower()
        pending_rule_ids = [str(item) for item in pending_rule_ids or [] if str(item)]
        selector_context = _compact_rule_selector(rule_selector or {})
        failure_key = "|".join(
            [
                str(failure_type),
                str(status),
                target_rule_layer,
                ",".join(pending_rule_ids),
                str(selector_context.get("rule_layer") or ""),
                ",".join(_as_text_list(selector_context.get("operator_family"))),
            ]
        )
        attempt_count = int(self.state.match_recovery_attempts.get(failure_key, 0))
        failure_payload = {
            "failure_type": failure_type,
            "outcome_status": "failed",
            "feedback_signal": "negative",
            "status": status,
            "reason": reason,
            "target_rule_layer": target_rule_layer,
            "pending_rule_ids": pending_rule_ids,
            "rule_selector": selector_context,
            "attempt_count": attempt_count,
            "failure_points": _runtime_failure_points(matcher_result, executor_results),
            "alignment_requirements": _alignment_requirements(matcher_result),
            "knowledge_expansion_requests": list(
                ((matcher_result or {}).get("metadata") or {}).get("knowledge_expansion_requests") or []
            ),
            "matcher_metadata": _compact_matcher_metadata((matcher_result or {}).get("metadata", {})),
            "executor_result_count": len(executor_results or []),
            "operation_count": len(operations or []),
            "learning_value": "runtime_recovery_and_negative_case_reference",
            "correction_status": "unverified",
        }
        self.state.control_context = self._build_control_context(
            trigger={
                "kind": "runtime_failure",
                **failure_payload,
            }
        )
        self._remember(
            zone="event_workspace",
            stage="runtime_failure",
            source="scheduler-new",
            payload_type="runtime_failure_feedback",
            payload=failure_payload,
            confidence=0.0,
            tags=[
                "scheduler-new",
                "runtime_failure",
                str(failure_type),
                str(status),
                "control_recovery",
                target_rule_layer,
            ],
        )
        if attempt_count >= max_attempts:
            return None
        self.state.match_recovery_attempts[failure_key] = attempt_count + 1
        recovery_selector = normalize_rule_selector({
            "rule_layer": "control",
            "required_rule_tags": ["runtime_control_entry"],
            "source_zones": ["event_workspace"],
            "top_k": 4,
            "execute_top_k": 1,
            "candidate_k": 256,
            "match_threshold": 0.30,
            "recovery_context": {
                "failure_type": failure_type,
                "status": status,
                "target_rule_layer": target_rule_layer,
                "attempt": attempt_count + 1,
            },
            "control_invocation": True,
        })
        return {
            "phase": "rule_match" if failure_type == "rule_match_failure" else "execute",
            "status": "control_recovery_requested",
            "reason": "runtime failure was recorded; control rules should choose recovery path",
            "runtime_failure": failure_payload,
            "pending_rule_selectors": [recovery_selector],
            "active_rule_ids": [],
        }

    def _request_control_review(
        self,
        *,
        status: str,
        reason: str,
        executor_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a completed path segment to control rules for the next decision."""

        self.state.control_context = self._build_control_context(
            trigger={
                "kind": "path_segment_completed",
                "status": str(status),
                "reason": str(reason),
                "failure_points": _runtime_failure_points(None, executor_results),
            }
        )
        return {
            "phase": "execute",
            "status": "control_review_requested",
            "reason": reason,
            "pending_rule_selectors": [
                normalize_rule_selector({
                    "rule_layer": "control",
                    "required_rule_tags": ["runtime_control_entry"],
                    "top_k": 4,
                    "execute_top_k": 1,
                    "candidate_k": 256,
                    "match_threshold": 0.30,
                    "control_invocation": True,
                })
            ],
        }

    def _request_control_candidate_path_selection(
        self,
        *,
        candidate_rule_paths: list[dict[str, Any]],
        matcher_result: dict[str, Any],
        rule_selector: dict[str, Any],
    ) -> dict[str, Any]:
        """Expose compatible alternatives to Control rules without choosing one."""

        self.state.control_context = self._build_control_context(
            trigger={
                "kind": "candidate_path_selection_required",
                "status": "compatible_alternatives_ready",
                "candidate_rule_paths": deepcopy(candidate_rule_paths),
                "source_rule_selector": _compact_rule_selector(rule_selector),
                "matcher_metadata": _compact_matcher_metadata(matcher_result.get("metadata", {})),
            }
        )
        return {
            "phase": "rule_match",
            "status": "control_candidate_path_selection_requested",
            "reason": "matcher separated conflicting matches; a control rule must select one compatible path",
            "active_rule_ids": [],
            "pending_operations": [],
            "pending_rule_selectors": [
                normalize_rule_selector(
                    {
                        "rule_layer": "control",
                        "required_rule_tags": ["candidate_path_selection"],
                        "top_k": 4,
                        "execute_top_k": 1,
                        "candidate_k": 256,
                        "match_threshold": 0.30,
                        "control_invocation": True,
                    }
                )
            ],
        }

    def _build_control_context(self, *, trigger: dict[str, Any]) -> dict[str, Any]:
        event_graphs: dict[str, Any] = {}
        for event_id in self.state.active_event_ids:
            workspace = self.working_memory.event_workspaces.get(str(event_id))
            if isinstance(workspace, dict):
                event_graphs[str(event_id)] = deepcopy(workspace)
        visited_rule_ids = _executed_rule_ids(
            self.state.execution_history[self.state.execution_history_start:]
        )
        visited_records = _load_rule_records_by_ids(self._rule_db_path(), visited_rule_ids)
        layer_by_rule_id = {
            str(record.get("rule_id") or ""): str(record.get("rule_layer") or "").strip().lower()
            for record in visited_records
        }
        visited_control_rule_ids = [
            rule_id for rule_id in visited_rule_ids
            if layer_by_rule_id.get(rule_id) == "control"
        ]
        visited_non_control_rule_ids = [
            rule_id for rule_id in visited_rule_ids
            if layer_by_rule_id.get(rule_id) != "control"
        ]
        return {
            "trigger": deepcopy(trigger),
            "workflow": deepcopy(self.state.control_state),
            "scheduler": {
                "session_id": self.state.session_id,
                "loop_index": self.state.loop_index,
                "phase": self.state.phase,
                "last_action": self.state.last_action,
            },
            "path": {
                "active_rule_ids": list(self.state.active_rule_ids),
                "visited_rule_ids": visited_rule_ids,
                "visited_control_rule_ids": visited_control_rule_ids,
                "visited_non_control_rule_ids": visited_non_control_rule_ids,
                "pending_rule_ids": list(self.state.pending_rule_ids),
                "pending_rule_selectors": deepcopy(self.state.pending_rule_selectors),
            },
            "thought_process": {
                "controller_history": deepcopy(self.state.controller_history[-12:]),
                "execution_history": deepcopy(self.state.execution_history[self.state.execution_history_start:][-12:]),
                "errors": deepcopy(self.state.errors[-12:]),
                "recovery_attempts": dict(self.state.match_recovery_attempts),
            },
            "output": {
                "output_ready": self.state.output_ready,
                "allow_output": self.state.allow_output,
                "pending_candidates": deepcopy(self.state.pending_output_candidates),
            },
            "event_graphs": event_graphs,
        }

    def _current_control_context(self) -> dict[str, Any]:
        stored = deepcopy(self.state.control_context)
        trigger = stored.get("trigger") if isinstance(stored.get("trigger"), dict) else {
            "kind": "scheduler_control_invocation",
            "status": self.state.last_action or "cold_start",
        }
        current = self._build_control_context(trigger=trigger)
        for key, value in stored.items():
            if key not in current:
                current[key] = value
        return current

    def _default_controller(self, state: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
        control_voc = state.get("control_voc") if isinstance(state.get("control_voc"), dict) else evaluate_control_voc(state, memory)
        if state.get("output_ready") and state.get("allow_output"):
            return {"next_phase": "output", "reason": "output_ready_and_allowed", "control_voc": control_voc}
        if state.get("pending_inputs"):
            return {"next_phase": "event_build", "reason": "new_input_waiting", "control_voc": control_voc}
        if state.get("last_action") == "event_build" and not state.get("active_rule_ids"):
            return {
                "next_phase": "rule_match",
                "reason": "cold_start_control_observation_rules_requested",
                "allow_output": False,
                "control_voc": control_voc,
                "control_context": {
                    "trigger": {
                        "kind": "cold_start",
                        "status": "event_graph_ready",
                    }
                },
                "pending_rule_selectors": [
                    normalize_rule_selector({
                        "rule_layer": "control",
                        "required_rule_tags": ["runtime_control_entry"],
                        "top_k": 4,
                        "execute_top_k": 1,
                        "candidate_k": 256,
                        "match_threshold": 0.30,
                        "control_invocation": True,
                    })
                ],
            }
        if state.get("pending_frontend_patches"):
            return {"next_phase": "frontend_patch", "reason": "frontend_patch_waiting", "control_voc": control_voc}
        if state.get("pending_operations"):
            return {"next_phase": "execute", "reason": "operation_waiting", "control_voc": control_voc}
        if state.get("pending_rule_ids"):
            return {"next_phase": "rule_match", "reason": "rule_waiting", "control_voc": control_voc}
        if state.get("pending_rule_selectors"):
            return {"next_phase": "rule_match", "reason": "rule_selector_waiting", "control_voc": control_voc}
        if state.get("pending_output_candidates"):
            return {"next_phase": "rule_match", "reason": "output_candidate_waiting_for_gate_rule", "control_voc": control_voc}
        meta_trigger = control_voc.get("meta_trigger") if isinstance(control_voc.get("meta_trigger"), dict) else {}
        if meta_trigger.get("triggered"):
            return {
                "next_phase": "rule_match",
                "reason": "control_voc_triggered_meta_review",
                "control_voc": control_voc,
                "pending_rule_selectors": [_control_voc_meta_rule_selector(control_voc)],
            }
        if state.get("outputs"):
            return {
                "next_phase": "output",
                "allow_output": True,
                "reason": "control_voc_allows_existing_output",
                "control_voc": control_voc,
            }
        return {"next_phase": "rule_match", "reason": "default_controller_continue_rule_match", "control_voc": control_voc}

    def _default_stage(self, phase: str) -> StageFn:
        def run(state: dict[str, Any], memory: WorkingMemoryState, payload: dict[str, Any]) -> dict[str, Any]:
            return {"phase": phase, "payload": payload, "note": "default_noop_stage"}
        return run

    def _default_rule_matcher(
        self,
        state: dict[str, Any],
        memory: WorkingMemoryState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        vectorization_status = _ensure_match_workspace_vectors(memory, self.vector_encoder)
        rule_db_path = self._rule_db_path()
        if not rule_db_path:
            return {
                "phase": "rule_match",
                "output": {
                    "operation": "stop_path",
                    "status": "rule_db_not_configured",
                    "reason": "scheduler reached rule_match, but no rule DB path is configured",
                },
                "stop": True,
            }
        rule_selectors = normalize_rule_selectors(
            payload.get("rule_selectors") or state.get("pending_rule_selectors")
        )
        rule_selector = rule_selectors[0] if rule_selectors else {}
        rule_selector, memory_path_hint_report = self._with_memory_path_hints(rule_selector)
        if rule_selector.get("prefer_goal_alignment"):
            rule_selector["goal_feature_types"] = _working_memory_goal_feature_types(memory)
        remaining_rule_selectors = rule_selectors[1:]
        pending_rule_ids = [
            str(item)
            for item in (
                payload.get("rule_ids", [])
                or state.get("pending_rule_ids", [])
                or rule_selector.get("explicit_rule_ids", [])
                or []
            )
            if str(item)
        ]
        target_rule_layer = str(
            rule_selector.get("rule_layer")
            or rule_selector.get("target_rule_layer")
            or self._next_rule_layer(state)
        ).strip().lower() or "control"
        authorization_error = _cross_path_search_authorization_error(
            rule_selector,
            rule_db_path=rule_db_path,
            target_rule_layer=target_rule_layer,
        )
        if authorization_error:
            return {
                "phase": "rule_match",
                "output": {
                    "operation": "stop_path",
                    "status": "unauthorized_cross_path_search",
                    "reason": authorization_error,
                    "rule_selector": _compact_rule_selector(rule_selector),
                },
                "stop": True,
            }
        target = {
            "type": "rule_db",
            "rule_db_path": rule_db_path,
            "rule_layer": target_rule_layer,
        }
        if pending_rule_ids:
            rule_records = _load_rule_records_by_ids(rule_db_path, pending_rule_ids)
            if rule_records:
                target = {
                    "type": "rules",
                    "rule_records": rule_records,
                    "rule_ids": pending_rule_ids,
                }
            else:
                target = {
                    "type": "rule_db",
                    "rule_db_path": rule_db_path,
                    "rule_layer": target_rule_layer,
                    "pending_rule_ids_not_found": pending_rule_ids,
                }
        scope_tags = _working_memory_routing_tags(memory)
        top_k = max(1, len(pending_rule_ids)) if pending_rule_ids else max(1, int(_safe_float(rule_selector.get("top_k"), 10)))
        source_zones = _rule_selector_source_zones(rule_selector)
        source_fields = [
            "event_workspaces.*.spans",
            "event_workspaces.*.relations",
        ]
        match_args = {
            "source": {
                "type": "working_memory",
                "zones": source_zones,
                "fields": source_fields,
            },
            "target": target,
            "rule_db_path": rule_db_path,
            "top_k": top_k,
            "candidate_k": max(1, int(_safe_float(rule_selector.get("candidate_k"), 192))),
            "match_threshold": _safe_float(rule_selector.get("match_threshold"), 0.55),
            "use_ann": True,
            "graph_vectors": _rule_selector_vectors(rule_selector),
            **_working_memory_vector_contract(memory),
            "preferred_rule_tags": _rule_selector_preferred_tags(rule_selector),
            "rule_selector": rule_selector,
            "scope_tags": scope_tags,
            "vectorization_status": vectorization_status,
        }
        coarse_knowledge = self._coarse_knowledge_before_content_match(
            target_rule_layer=target_rule_layer,
            rule_selector=rule_selector,
        )
        try:
            result = match_source_to_target(match_args, self.state, memory)
        except Exception as exc:
            return {
                "phase": "rule_match",
                "output": {
                    "operation": "stop_path",
                    "status": "rule_matcher_error",
                    "reason": repr(exc),
                    "rule_db_path": rule_db_path,
                },
                "stop": True,
            }
        result.setdefault("metadata", {})
        result["metadata"]["coarse_knowledge"] = coarse_knowledge
        result["metadata"]["memory_path_hints"] = memory_path_hint_report
        if not result.get("ok"):
            result["metadata"]["knowledge_expansion_requests"] = build_missing_knowledge_queries(result)
        produced = deepcopy(_as_dict_list(result.get("produced")))
        if rule_selector:
            for item in produced:
                if str(item.get("operation") or "") != "match":
                    continue
                item["selector_context"] = _selector_context(rule_selector)
                for match in _as_dict_list(item.get("matched_rules")):
                    match["selector_context"] = _selector_context(rule_selector)
                for candidate_path in _as_dict_list(item.get("candidate_rule_paths")):
                    for match in _as_dict_list(candidate_path.get("matched_rules")):
                        match["selector_context"] = _selector_context(rule_selector)
        if rule_selector.get("prefer_goal_reachability"):
            _annotate_goal_reachability(
                produced,
                memory=memory,
                rule_db_path=rule_db_path,
                goal_feature_types=set(rule_selector.get("goal_feature_types", [])),
                max_depth=max(1, int(_safe_float(rule_selector.get("goal_reachability_max_depth"), 8))),
            )
            _annotate_alignment_goal_reachability(
                result,
                memory=memory,
                rule_db_path=rule_db_path,
                goal_feature_types=set(rule_selector.get("goal_feature_types", [])),
                max_depth=max(1, int(_safe_float(rule_selector.get("goal_reachability_max_depth"), 8))),
            )
        if rule_selector.get("prefer_context_domain"):
            _annotate_context_domain_strategy_alignment(produced, memory=memory)
        produced, control_selection = _control_select_matched_rules(
            produced,
            rule_selector=rule_selector,
            explicit_rule_ids=pending_rule_ids,
        )
        matched_rule_ids = _matched_rule_ids(produced)
        candidate_rule_paths = [
            path
            for operation in produced
            for path in _as_dict_list(operation.get("candidate_rule_paths"))
        ]
        used_context_ids = _matched_context_ids(result, memory)
        if used_context_ids:
            for event_id in self.state.active_event_ids or memory.event_workspaces.keys():
                workspace = memory.get_event_workspace(str(event_id))
                metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), dict) else {}
                metadata["used_context_ids_current_round"] = list(
                    dict.fromkeys(
                        [
                            *[str(value) for value in metadata.get("used_context_ids_current_round", []) if str(value)],
                            *used_context_ids,
                        ]
                    )
                )
                workspace["metadata"] = metadata
        if candidate_rule_paths:
            selection = self._request_control_candidate_path_selection(
                candidate_rule_paths=candidate_rule_paths,
                matcher_result=result,
                rule_selector=rule_selector,
            )
            return {
                **selection,
                "matcher": result,
                "pending_rule_selectors": [
                    *selection.get("pending_rule_selectors", []),
                    *remaining_rule_selectors,
                ],
                "control_match_selection": {
                    "policy": "control_rule_must_select_compatible_candidate_path",
                    "candidate_path_ids": [
                        str(path.get("candidate_path_id") or "")
                        for path in candidate_rule_paths
                    ],
                },
            }
        if not result.get("ok") or not matched_rule_ids:
            if remaining_rule_selectors:
                return {
                    "phase": "rule_match",
                    "matcher": result,
                    "pending_rule_selectors": remaining_rule_selectors,
                    "active_rule_ids": [],
                    "status": "rule_selector_no_match_try_next",
                }
            recovery = self._request_control_recovery(
                failure_type="rule_match_failure",
                status="no_rules_matched",
                reason="matcher found no executable rules for requested rule layer",
                target_rule_layer=target_rule_layer,
                pending_rule_ids=pending_rule_ids,
                rule_selector=rule_selector,
                matcher_result=result,
            )
            if recovery is not None:
                return {
                    **recovery,
                    "matcher": result,
                }
            return {
                "phase": "rule_match",
                "matcher": result,
                "output": {
                    "operation": "stop_path",
                    "status": "no_rules_matched",
                    "reason": "matcher found no executable rules for requested rule layer",
                    "target_rule_layer": target_rule_layer,
                    "pending_rule_ids": pending_rule_ids,
                    "rule_selector": _compact_rule_selector(rule_selector),
                    "matcher_metadata": _compact_matcher_metadata(result.get("metadata", {})),
                },
                "stop": True,
            }
        return {
            "phase": "rule_match",
            "matcher": result,
            "pending_operations": produced,
            "pending_rule_selectors": remaining_rule_selectors,
            "active_rule_ids": matched_rule_ids,
            "control_match_selection": control_selection,
        }

    def _with_memory_path_hints(self, selector: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Attach opt-in historical path preferences without treating Memory as evidence."""
        if not isinstance(selector, dict) or not selector.get("use_memory_path_hints"):
            return selector, {"requested": False, "preferred_rule_ids": []}
        preferred: list[str] = []
        source_memory_ids: list[str] = []
        for workspace in self.working_memory.event_workspaces.values():
            if not isinstance(workspace, dict):
                continue
            for context in _as_dict_list(workspace.get("context_items")):
                if str(context.get("source") or "") not in {"memory_db", "memory_redis", "memory_workspace"}:
                    continue
                payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
                outcome = str(payload.get("outcome_status") or payload.get("outcome") or "").lower()
                completion = payload.get("completion") if isinstance(payload.get("completion"), dict) else {}
                completed = str(completion.get("status") or "").lower()
                if outcome not in {"solved", "completed", "success", "passed"} and completed not in {"completed", "solved"}:
                    continue
                trace = payload.get("execution_trace") if isinstance(payload.get("execution_trace"), dict) else {}
                execution = trace.get("rule_trace") if isinstance(trace.get("rule_trace"), dict) else {}
                graph = payload.get("event_hypergraph") if isinstance(payload.get("event_hypergraph"), dict) else {}
                graph_metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
                graph_trace = graph_metadata.get("rule_trace") if isinstance(graph_metadata.get("rule_trace"), dict) else {}
                for rule_id in (
                    execution.get("productive_rule_ids", [])
                    or execution.get("executed_rule_ids", [])
                    or graph_trace.get("productive_rule_ids", [])
                    or graph_trace.get("executed_rule_ids", [])
                ):
                    normalized = str(rule_id or "").strip()
                    if normalized and normalized not in preferred:
                        preferred.append(normalized)
                memory_id = str(payload.get("memory_id") or context.get("source_record_id") or "").strip()
                if memory_id:
                    source_memory_ids.append(memory_id)
        enriched = deepcopy(selector)
        enriched["memory_preferred_rule_ids"] = preferred
        return enriched, {
            "requested": True,
            "source_memory_ids": list(dict.fromkeys(source_memory_ids)),
            "preferred_rule_ids": preferred,
            "policy": (
                "control_may_prefer_strictly_matched_memory_path_then_fallback"
                if enriched.get("memory_path_policy") == "prefer_then_fallback"
                else "post_strict_match_tie_break_current_graph_requirements_remain_strict"
            ),
        }

    def _coarse_knowledge_before_content_match(
        self,
        *,
        target_rule_layer: str,
        rule_selector: dict[str, Any],
    ) -> dict[str, Any]:
        """Add a small general context once; deeper expansion remains control-owned."""

        if target_rule_layer not in CONTENT_LAYERS:
            return {"status": "not_content_rule_layer", "added": 0}
        knowledge_db_path = str(
            (self.state.db_paths or {}).get("knowledge")
            or (self.state.db_paths or {}).get("knowledge_db")
            or ""
        ).strip()
        if not knowledge_db_path:
            return {"status": "knowledge_db_not_configured", "added": 0}
        summary = knowledge_db_persistence.summarize_database(knowledge_db_path)
        indexed_vectors = int((summary.get("ann") or {}).get("indexed_vectors") or 0)
        if indexed_vectors <= 0:
            return {
                "status": "knowledge_index_unavailable",
                "added": 0,
                "knowledge_records": int(summary.get("knowledge_records") or 0),
                "indexed_vectors": indexed_vectors,
            }
        top_k = max(4, min(6, int(_safe_float(rule_selector.get("coarse_knowledge_top_k"), 4))))
        added = 0
        event_summaries: dict[str, dict[str, Any]] = {}
        for event_id in self.state.active_event_ids or list(self.working_memory.event_workspaces):
            workspace = self.working_memory.event_workspaces.get(str(event_id))
            if not isinstance(workspace, dict):
                continue
            metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), dict) else {}
            if metadata.get("coarse_knowledge_attempted"):
                event_summaries[str(event_id)] = {"status": "already_attempted", "added": 0}
                continue
            metadata["coarse_knowledge_attempted"] = True
            workspace["metadata"] = metadata
            try:
                query_specs = _coarse_knowledge_query_specs(workspace, max_queries=top_k)
                if not query_specs:
                    raise ValueError("coarse knowledge retrieval requires event span vectors")
                query_results = [
                    retrieve_knowledge(
                        self.state.db_paths,
                        event_workspace=workspace,
                        query_vector=query_spec["query_vector"],
                        query_embedding_model=query_spec["embedding_model"],
                        query_vector_version=query_spec["vector_version"],
                        top_k=top_k,
                    )
                    for query_spec in query_specs
                ]
                records = _round_robin_knowledge_records(query_results, top_k=top_k)
                contexts = [knowledge_record_to_event_context(record) for record in records]
            except (TypeError, ValueError) as exc:
                event_summaries[str(event_id)] = {
                    "status": "coarse_knowledge_failed",
                    "added": 0,
                    "reason": str(exc),
                }
                continue
            event_summary = self.working_memory.add_event_context_records(
                str(event_id),
                contexts,
                source="knowledge_db",
                current_round=self.state.loop_index,
                retention_rule_id="system_coarse_knowledge_context",
                expire_after_unused_rounds=2,
            )
            event_summary["query_count"] = len(query_specs)
            event_summaries[str(event_id)] = event_summary
            added += int(event_summary.get("added") or 0)
        return {
            "status": "coarse_knowledge_added" if added else "no_coarse_knowledge_added",
            "added": added,
            "top_k": top_k,
            "event_summaries": event_summaries,
        }

    def _default_frontend_patcher(
        self,
        state: dict[str, Any],
        memory: WorkingMemoryState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        del state, memory
        patches = _as_dict_list(payload.get("patches"))
        return {
            "phase": "frontend_patch",
            "status": "frontend_patch_unavailable",
            "reason": "frontend patch is required before reasoning can continue",
            "frontend_patch_unavailable": {
                "status": "frontend_patch_unavailable",
                "reason": "frontend patch is required before reasoning can continue",
                "patches": patches,
            },
        }

    def _default_executor(
        self,
        state: dict[str, Any],
        memory: WorkingMemoryState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        operations = _as_dict_list(payload.get("operations"))
        outputs: list[dict[str, Any]] = []
        pending_rule_ids: list[str] = []
        pending_rule_selectors: list[dict[str, Any]] = []
        pending_frontend_patches: list[dict[str, Any]] = []
        pending_operations: list[dict[str, Any]] = []
        consumed_operations: list[dict[str, Any]] = []
        executor_results: list[dict[str, Any]] = []

        def collect_executor_result(
            executor_result: dict[str, Any], *, suppress_stopped_paths: bool = False
        ) -> None:
            self._apply_control_updates(executor_result)
            executor_results.append(executor_result)
            pending_rule_ids.extend(
                str(value)
                for value in executor_result.get("next_rule_ids", []) or []
                if str(value)
            )
            pending_rule_selectors.extend(
                normalize_rule_selectors(executor_result.get("next_rule_selectors"))
            )
            pending_frontend_patches.extend(_as_dict_list(executor_result.get("frontend_requests")))
            pending_operations.extend(_as_dict_list(executor_result.get("pending_operations")))
            outputs.extend(_as_dict_list(executor_result.get("outputs")))
            if not suppress_stopped_paths:
                outputs.extend(_as_dict_list(executor_result.get("stopped_paths")))

        for operation in operations:
            operation_name = str(operation.get("operation") or operation.get("op") or "").strip()
            if operation_name == "execute_candidate_path":
                candidate_path = (
                    operation.get("candidate_path")
                    if isinstance(operation.get("candidate_path"), dict)
                    else {}
                )
                matched_rules = _as_dict_list(candidate_path.get("matched_rules"))
                if not matched_rules:
                    pending_operations.append(
                        {
                            **operation,
                            "status": "invalid_candidate_path",
                            "reason": "selected candidate path contains no matched rules",
                        }
                    )
                    continue
                executor_result = execute_rule_chain(
                    calculator=self.calculator,
                    matched_rules=matched_rules,
                    working_memory=memory,
                    session_id=str(state.get("session_id") or self.state.session_id),
                    source_match={
                        "operation": "match",
                        "candidate_path": candidate_path,
                        "selected_by_control_rule_id": str(
                            operation.get("selected_by_control_rule_id") or ""
                        ),
                    },
                    control_context=self._current_control_context(),
                    max_steps=self.state.max_rule_chain_steps,
                )
                collect_executor_result(executor_result)
                consumed_operations.append(
                    {
                        "operation": operation_name,
                        "status": executor_result.get("status"),
                        "candidate_path_id": str(candidate_path.get("candidate_path_id") or ""),
                    }
                )
                continue
            if operation_name != "match":
                if operation_name in {
                    "retrieve_memory",
                    "query_memory",
                    "retrieve_workspace_memory",
                    "query_memory_workspace",
                    "retrieve_knowledge",
                    "query_knowledge",
                    "enrich_knowledge",
                }:
                    before_pending_count = len(self.state.pending_operations)
                    self._apply_produced_operations([operation])
                    consumed_operations.append(
                        {
                            "operation": operation_name,
                            "status": "consumed",
                            "pending_operations_added": max(0, len(self.state.pending_operations) - before_pending_count),
                        }
                    )
                    continue
                pending_operations.append(operation)
                continue
            matched_rules = _as_dict_list(operation.get("matched_rules"))
            if not matched_rules:
                continue
            executor_result = execute_rule_chain(
                calculator=self.calculator,
                matched_rules=matched_rules,
                working_memory=memory,
                session_id=str(state.get("session_id") or self.state.session_id),
                source_match=operation,
                control_context=self._current_control_context(),
                max_steps=self.state.max_rule_chain_steps,
            )
            fallback_operation = _memory_path_fallback_operation(operation, executor_result)
            collect_executor_result(
                executor_result,
                suppress_stopped_paths=fallback_operation is not None,
            )
            if fallback_operation is not None:
                pending_operations.append(fallback_operation)
        if outputs:
            return {
                "phase": "execute",
                "executor_results": executor_results,
                "consumed_operations": consumed_operations,
                "output": outputs[-1],
                "pending_rule_ids": pending_rule_ids,
                "pending_rule_selectors": pending_rule_selectors,
                "pending_frontend_patches": pending_frontend_patches,
                "pending_operations": pending_operations,
            }
        if pending_rule_ids or pending_rule_selectors or pending_frontend_patches or pending_operations:
            return {
                "phase": "execute",
                "executor_results": executor_results,
                "consumed_operations": consumed_operations,
                "pending_rule_ids": pending_rule_ids,
                "pending_rule_selectors": pending_rule_selectors,
                "pending_frontend_patches": pending_frontend_patches,
                "pending_operations": pending_operations,
            }
        failed_executor_results = [
            result
            for result in executor_results
            if isinstance(result, dict) and not bool(result.get("ok", False))
        ]
        if failed_executor_results:
            failure_statuses = list(
                dict.fromkeys(
                    str(result.get("status") or "executor_failed")
                    for result in failed_executor_results
                )
            )
            recovery = self._request_control_recovery(
                failure_type="execution_failure",
                status="|".join(failure_statuses),
                reason="selected rule path could not acquire or execute its event-workspace transaction",
                executor_results=failed_executor_results,
                operations=operations,
            )
            if recovery is not None:
                return {
                    **recovery,
                    "executor_results": executor_results,
                    "consumed_operations": consumed_operations,
                }
            return {
                "phase": "execute",
                "executor_results": executor_results,
                "consumed_operations": consumed_operations,
                "output": {
                    "operation": "stop_path",
                    "status": "runtime_control_no_decision",
                    "reason": "executor failure remained unresolved after bounded control recovery",
                },
                "stop": True,
            }
        if _executor_results_completed_state_or_graph_step(executor_results):
            review = self._request_control_review(
                status="completed_rule_steps",
                reason="a rule path segment completed without an explicit continuation or final output",
                executor_results=executor_results,
            )
            return {
                **review,
                "executor_results": executor_results,
                "consumed_operations": consumed_operations,
            }
        if consumed_operations:
            return {
                "phase": "execute",
                "status": "consumed_operations",
                "executor_results": executor_results,
                "consumed_operations": consumed_operations,
            }
        recovery = self._request_control_recovery(
            failure_type="execution_failure",
            status="no_executable_rule_produced",
            reason="matched rules did not produce output, frontend request, graph operation, or next rule",
            executor_results=executor_results,
            operations=operations,
        )
        if recovery is not None:
            return {
                **recovery,
                "executor_results": executor_results,
                "consumed_operations": consumed_operations,
            }
        return {
            "phase": "execute",
            "executor_results": executor_results,
            "consumed_operations": consumed_operations,
            "output": {
                "operation": "stop_path",
                "status": "no_executable_rule_produced",
                "reason": "matched rules did not produce output, frontend request, graph operation, or next rule",
            },
            "stop": True,
        }

    def _rule_db_path(self) -> str:
        db_paths = self.state.db_paths or {}
        return str(db_paths.get("rule") or db_paths.get("rule_db") or "").strip()

    def _next_rule_layer(self, state: dict[str, Any]) -> str:
        del state
        return "control"

    def _default_output_decoder(
        self,
        state: dict[str, Any],
        memory: WorkingMemoryState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        outputs = payload.get("outputs") or state.get("outputs") or []
        return {
            "output": outputs[-1] if outputs else {"status": "no_output_payload"},
            "phase": "output",
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _rule_selectors_from_run_rule_operation(item: dict[str, Any]) -> list[dict[str, Any]]:
    selectors = normalize_rule_selectors(item.get("rule_selectors"))
    selector = item.get("rule_selector")
    if isinstance(selector, dict):
        selectors.append(normalize_rule_selector(selector))
    return selectors


def _prepare_output_candidate(output: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(output)
    candidate["operation"] = "output_candidate"
    candidate["candidate_output"] = dict(output)
    candidate["output_gate_checked"] = False
    candidate.setdefault("answer_granularity", _output_granularity(output))
    candidate.setdefault("status", "waiting_output_gate_rule")
    candidate.setdefault("reason", "output candidate must be reviewed by control/check_policy rule")
    return candidate


def _output_gate_rule_selectors(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    granularity = str(candidate.get("answer_granularity") or "").strip()
    base = {
        "operator_family": ["output_gate", "answer_granularity", granularity],
        "required_rule_tags": ["output_gate"],
        "required_rule_pattern": {
            "pattern_id": "output_gate_rule_selector",
            "span_slots": [
                {
                    "slot_id": "candidate",
                    "debug_text": f"output candidate waiting for gate rule with answer granularity {granularity}",
                },
                {
                    "slot_id": "path",
                    "debug_text": "rule path layers and rules applied are available for output gate review",
                },
            ],
            "relation_slots": [
                {
                    "from_slot": "candidate",
                    "to_slot": "path",
                    "debug_relation": "output gate checks answer granularity against rule path evidence level",
                    "confidence": 0.9,
                }
            ],
            "confidence": 0.9,
        },
        "top_k": 5,
        "candidate_k": 64,
        "match_threshold": 0.35,
        "search_trace": {"hop": 0, "max_hops": 3},
    }
    explicit_rule_ids = list(
        dict.fromkeys(
            str(rule_id)
            for rule_id in candidate.get("output_gate_rule_ids", []) or []
            if str(rule_id)
        )
    )
    selectors: list[dict[str, Any]] = []
    if explicit_rule_ids:
        selectors.append(
            {
                **base,
                "rule_layer": "control",
                "explicit_rule_ids": explicit_rule_ids,
                "top_k": len(explicit_rule_ids),
                "candidate_k": max(8, len(explicit_rule_ids)),
            }
        )
    selectors.extend(
        [
            {**base, "rule_layer": "check_policy"},
            {**base, "rule_layer": "control"},
        ]
    )
    return [normalize_rule_selector(selector) for selector in selectors]


def _output_revision_rule_selectors(
    candidate: dict[str, Any],
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    """Route a rejected output back through rule-based analysis and repair."""

    supplied = normalize_rule_selectors(decision.get("next_rule_selectors"))
    revision_context = dict(decision.get("revision_context") or {})
    operator_family = [
        "output_revision",
        "completion_analysis",
        "evidence_check",
        "path_repair",
        str(decision.get("status") or "output_rejected"),
    ]
    base = {
        "operator_family": operator_family,
        "source_zones": ["event_workspace"],
        "top_k": 8,
        "candidate_k": 192,
        "match_threshold": 0.30,
        "search_trace": {"hop": 0, "max_hops": 6},
        "recovery_context": {
            **revision_context,
            "revision_attempt": int(decision.get("revision_attempt") or 1),
            "candidate_status": str(candidate.get("status") or ""),
        },
    }
    explicit_rule_ids = list(
        dict.fromkeys(_as_text_list(decision.get("revision_rule_ids")))
    )
    selectors = list(supplied)
    if explicit_rule_ids:
        selectors.append(
            {
                **base,
                "rule_layer": "control",
                "explicit_rule_ids": explicit_rule_ids,
                "top_k": len(explicit_rule_ids),
            }
        )
    selectors.extend(
        [
            {
                **base,
                "rule_layer": "control",
                "preferred_rule_tags": [
                    "control_rule",
                    "output_revision",
                    "path_repair",
                    "completion_analysis",
                ],
            },
            {
                **base,
                "rule_layer": "check_policy",
                "preferred_rule_tags": [
                    "check_policy",
                    "evidence_validation",
                    "output_revision",
                ],
            },
            {
                **base,
                "rule_layer": "domain",
                "preferred_rule_tags": [
                    "domain_rule",
                    "calculation",
                    "path_repair",
                ],
            },
        ]
    )
    return [normalize_rule_selector(selector) for selector in selectors]


def _control_voc_meta_rule_selector(control_voc: dict[str, Any]) -> dict[str, Any]:
    return normalize_rule_selector({
        "rule_layer": "control",
        "operator_family": ["meta_review", "control_voc", "model_self"],
        "preferred_rule_tags": ["control_rule", "meta", "model_self", "control_voc"],
        "learning_mode": "meta",
        "target_scope": "model_self",
        "audit_level": "strict",
        "approval_required": True,
        "isolation_required": True,
        "rollback_policy": "required",
        "control_voc": dict(control_voc),
        "top_k": 8,
        "candidate_k": 128,
        "match_threshold": 0.35,
    })


def _selector_context(selector: dict[str, Any]) -> dict[str, Any]:
    return selector_execution_context(selector)


def _alignment_requirements(result):
    from .matcher.matching import alignment_requirements_from_metadata
    metadata = (result or {}).get("metadata", {})
    requirements = list(metadata.get("alignment_requirements", []))
    return requirements + alignment_requirements_from_metadata(metadata)


def _runtime_failure_points(
    matcher_result: dict[str, Any] | None,
    executor_results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    metadata = (matcher_result or {}).get("metadata") if isinstance((matcher_result or {}).get("metadata"), dict) else {}
    for required in metadata.get("per_required") or []:
        if not isinstance(required, dict):
            continue
        details = required.get("details") if isinstance(required.get("details"), dict) else {}
        required_index = required.get("required_index")
        for slot in details.get("slot_matches") or []:
            if not isinstance(slot, dict) or slot.get("matched"):
                continue
            points.append(
                {
                    "kind": "missing_required_span",
                    "required_index": required_index,
                    "slot_id": str(slot.get("slot_id") or ""),
                    "score": _safe_float(slot.get("score"), 0.0),
                    "match_threshold": _safe_float(slot.get("match_threshold"), 0.0),
                    "candidate_span_id": str(slot.get("candidate_span_id") or ""),
                    "missing_effect": str(slot.get("missing_effect") or ""),
                    "suggested_control_action": "adjust_required_boundary_or_switch_path",
                }
            )
        for relation in details.get("relation_matches") or []:
            if not isinstance(relation, dict) or relation.get("matched"):
                continue
            points.append(
                {
                    "kind": "missing_or_wrong_relation",
                    "required_index": required_index,
                    "relation_id": str(relation.get("relation_id") or relation.get("required_relation_id") or ""),
                    "required_relation": str(relation.get("required_relation") or relation.get("relation") or ""),
                    "score": _safe_float(relation.get("score"), 0.0),
                    "match_threshold": _safe_float(relation.get("match_threshold"), 0.0),
                    "candidate_source_span_id": str(relation.get("candidate_source_span_id") or ""),
                    "candidate_target_span_id": str(relation.get("candidate_target_span_id") or ""),
                    "suggested_control_action": "retrieve_knowledge_or_relation_repair_or_switch_path",
                }
            )
    for result in executor_results or []:
        if not isinstance(result, dict):
            continue
        specific_executor_failure = False
        for error in _as_dict_list(result.get("errors")):
            error_status = str(error.get("status") or "").strip()
            if error_status == "event_workspace_busy":
                points.append(
                    {
                        "kind": "event_workspace_busy",
                        "chain_id": str(result.get("chain_id") or ""),
                        "event_id": str(error.get("event_id") or ""),
                        "status": error_status,
                        "suggested_control_action": "retry_after_active_path_or_choose_disjoint_candidate_path",
                    }
                )
                specific_executor_failure = True
            elif error_status == "claim_capacity_unavailable":
                points.append(
                    {
                        "kind": "claim_capacity_unavailable",
                        "chain_id": str(result.get("chain_id") or ""),
                        "event_id": str(error.get("event_id") or ""),
                        "span_id": str(error.get("span_id") or ""),
                        "capacity": _safe_float(error.get("capacity"), 0.0),
                        "consumed": _safe_float(error.get("consumed"), 0.0),
                        "reserved": _safe_float(error.get("reserved"), 0.0),
                        "requested": _safe_float(error.get("requested"), 0.0),
                        "status": error_status,
                        "suggested_control_action": "choose_an_alternative_path_without_overcommitting_the_condition",
                    }
                )
                specific_executor_failure = True
        if specific_executor_failure:
            continue
        if result.get("outputs") or result.get("next_rule_ids") or result.get("next_rule_selectors") or result.get("pending_operations"):
            continue
        rule_ids: list[str] = []
        for step in _as_dict_list(result.get("steps")):
            rule_id = str(step.get("rule_id") or "")
            if rule_id:
                rule_ids.append(rule_id)
        points.append(
            {
                "kind": "executor_no_output",
                "chain_id": str(result.get("chain_id") or ""),
                "status": str(result.get("status") or ""),
                "rule_ids": list(dict.fromkeys(rule_ids)),
                "suggested_control_action": "switch_path_or_request_more_rules_or_stop_no_decision",
            }
        )
    return points


def _executed_rule_ids(execution_history: list[dict[str, Any]]) -> list[str]:
    rule_ids: list[str] = []
    for history_item in execution_history:
        result = history_item.get("result") if isinstance(history_item.get("result"), dict) else {}
        executor_results = _as_dict_list(result.get("executor_results"))
        if not executor_results and isinstance(result.get("result"), dict):
            executor_results = [dict(result["result"])]
        for executor_result in executor_results:
            for step in _as_dict_list(executor_result.get("steps")):
                rule_id = str(step.get("rule_id") or "").strip()
                if rule_id and rule_id not in rule_ids:
                    rule_ids.append(rule_id)
    return rule_ids


def _rule_selector_texts(selector: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            texts.append(text)

    for key in (
        "description",
        "strategy",
        "macro_strategy",
        "micro_operator",
        "reason",
        "condition",
    ):
        add(selector.get(key))
    for key in ("operator_family", "preferred_rule_tags", "scope_tags", "rule_tags"):
        value = selector.get(key)
        if isinstance(value, str):
            add(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
    for pattern_key in ("required_rule_pattern", "span_pattern"):
        pattern = selector.get(pattern_key)
        if not isinstance(pattern, dict):
            continue
        for slot in _as_dict_list(pattern.get("span_slots")):
            add(slot.get("debug_text") or slot.get("text") or slot.get("core"))
        for relation in _as_dict_list(pattern.get("relation_slots")):
            add(relation.get("debug_relation") or relation.get("relation"))
    return list(dict.fromkeys(texts))


def _ensure_match_workspace_vectors(memory: WorkingMemoryState, encoder: Any) -> dict[str, Any]:
    if encoder is None:
        return {"ok": False, "status": "encoder_unavailable", "encoded_count": 0, "requested_count": 0}
    requests: list[tuple[dict[str, Any], str, str]] = []
    for workspace in (memory.event_workspaces or {}).values():
        if not isinstance(workspace, dict):
            continue
        spans = workspace.get("spans") if isinstance(workspace.get("spans"), list) else []
        for span in spans:
            if not isinstance(span, dict):
                continue
            _append_vector_request(requests, span, "core_vector", span.get("core"))
            _append_vector_request(requests, span, "relation_vector", span.get("relation"))
            _append_vector_request(requests, span, "nature_vector", span.get("nature") or span.get("text"))
            for semantic_relation in span.get("semantic_relations", []) or []:
                if not isinstance(semantic_relation, dict):
                    continue
                debug = semantic_relation.get("debug") if isinstance(semantic_relation.get("debug"), dict) else {}
                _append_vector_request(requests, semantic_relation, "core_vector", debug.get("core"))
                _append_vector_request(requests, semantic_relation, "relation_vector", debug.get("relation"))
                _append_vector_request(requests, semantic_relation, "nature_vector", debug.get("nature"))
            _append_vector_request(
                requests,
                span,
                "vector",
                " ".join(str(span.get(key) or "").strip() for key in ("core", "relation", "nature", "text") if str(span.get(key) or "").strip()),
            )
        relations = workspace.get("relations") if isinstance(workspace.get("relations"), list) else []
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            _append_vector_request(
                requests,
                relation,
                "relation_vector",
                relation.get("relation") or relation.get("relation_type") or relation.get("debug_relation"),
            )
    if not requests:
        return {"ok": True, "status": "already_vectorized", "encoded_count": 0, "requested_count": 0}
    texts = [text for _target, _field, text in requests]
    status = _encode_match_texts(encoder, texts)
    embedding_model = str(getattr(encoder, "embedding_model", "") or getattr(encoder, "encoder_name_or_path", "") or "")
    vector_version = str(getattr(encoder, "vector_version", "") or "v1")
    for (target, field, _text), vector in zip(requests, status["vectors"], strict=False):
        if not vector:
            continue
        target[field] = vector
        target.setdefault("embedding_model", embedding_model)
        target.setdefault("vector_dimension", len(vector))
        target.setdefault("vector_version", vector_version)
    return {
        key: value
        for key, value in status.items()
        if key != "vectors"
    }


def _append_vector_request(requests: list[tuple[dict[str, Any], str, str]], target: dict[str, Any], field: str, text: Any) -> None:
    if _number_list(target.get(field)):
        return
    value = str(text or "").strip()
    if value:
        requests.append((target, field, value))


def _encode_match_texts(encoder: Any, texts: list[str]) -> dict[str, Any]:
    if hasattr(encoder, "encode_text_vectors_status"):
        result = encoder.encode_text_vectors_status(texts)
        if isinstance(result, dict) and isinstance(result.get("vectors"), list):
            return result
    if hasattr(encoder, "encode_text_vectors"):
        try:
            vectors = encoder.encode_text_vectors(texts)
            if isinstance(vectors, list) and len(vectors) == len(texts):
                normalized = [
                    [float(item) for item in vector] if isinstance(vector, list) and vector else None
                    for vector in vectors
                ]
                return _match_vector_status(normalized, [])
        except Exception as exc:
            return _match_vector_status(
                [None] * len(texts),
                [{"index": index, "status": "batch_encode_failed", "error": repr(exc)} for index, _ in enumerate(texts)],
            )
    vectors: list[list[float] | None] = []
    errors: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        try:
            vector = encoder._encode_text_vector(text)
            normalized = [float(item) for item in vector] if isinstance(vector, list) and vector else None
        except Exception as exc:
            normalized = None
            errors.append({"index": index, "status": "single_encode_failed", "error": repr(exc)})
        vectors.append(normalized)
    return _match_vector_status(vectors, errors)


def _match_vector_status(vectors: list[list[float] | None], errors: list[dict[str, Any]]) -> dict[str, Any]:
    encoded_count = sum(1 for vector in vectors if vector)
    if encoded_count == len(vectors):
        status = "ok"
    elif encoded_count > 0:
        status = "partial"
    elif vectors:
        status = "failed"
    else:
        status = "empty"
    return {
        "ok": status == "ok",
        "status": status,
        "vectors": vectors,
        "encoded_count": encoded_count,
        "requested_count": len(vectors),
        "errors": errors,
    }


def _working_memory_vector_contract(memory: WorkingMemoryState) -> dict[str, str]:
    for workspace in (memory.event_workspaces or {}).values():
        if not isinstance(workspace, dict):
            continue
        for item in [*_as_dict_list(workspace.get("spans")), *_as_dict_list(workspace.get("relations"))]:
            embedding_model = str(item.get("embedding_model") or "").strip()
            vector_version = str(item.get("vector_version") or "").strip()
            if embedding_model or vector_version:
                return {
                    "query_embedding_model": embedding_model,
                    "query_vector_version": vector_version,
                }
    return {"query_embedding_model": "", "query_vector_version": ""}


def _rule_selector_vectors(selector: dict[str, Any]) -> list[list[float]]:
    vectors: list[list[float]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"vector", "rule_vector", "condition_vector", "conclusion_vector", "span_core_vector", "relation_vector", "vector_query_vector"}:
                    vector = _number_list(item)
                    if vector:
                        vectors.append(vector)
                    continue
                walk(item)
        elif isinstance(value, list):
            vector = _number_list(value)
            if vector:
                vectors.append(vector)
                return
            for item in value:
                walk(item)

    walk(selector.get("required_rule_pattern") if isinstance(selector.get("required_rule_pattern"), dict) else selector.get("span_pattern"))
    return vectors


def _rule_selector_preferred_tags(selector: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in ("operator_family", "preferred_rule_tags", "scope_tags", "rule_tags"):
        value = selector.get(key)
        if isinstance(value, str):
            tags.append(value)
        elif isinstance(value, (list, tuple, set)):
            tags.extend(str(item) for item in value if str(item))
    for key in ("macro_strategy", "micro_operator"):
        if str(selector.get(key) or "").strip():
            tags.append(str(selector[key]))
    return list(dict.fromkeys(tag.strip().lower() for tag in tags if tag.strip()))


def _rule_selector_source_zones(selector: dict[str, Any]) -> list[str]:
    """Default matcher input is the current event graph only.

    Control rules may explicitly widen the readable zones for special matcher
    policies, but runtime traces are not included unless the rule asks for them.
    """

    raw = selector.get("source_zones") or selector.get("zones")
    if raw is None:
        return ["event_workspace"]
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw]
    else:
        return ["event_workspace"]
    zones: list[str] = []
    for value in values:
        zone = _normalize_rule_selector_source_zone(value)
        if zone and zone not in zones:
            zones.append(zone)
    return zones or ["event_workspace"]


def _normalize_rule_selector_source_zone(value: Any) -> str:
    zone = str(value or "").strip()
    return "event_workspace" if zone == "event_workspace" else ""


def _number_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            return None
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return None
    return result or None


def _knowledge_operation_query_specs(
    operation: dict[str, Any],
    workspace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve vectors explicitly selected by a control rule for knowledge expansion."""

    specs: list[dict[str, Any]] = []

    def add(vector: Any, source: dict[str, Any], reason: str) -> None:
        normalized = _number_list(vector)
        if not normalized:
            return
        specs.append(
            {
                "query_vector": normalized,
                "embedding_model": str(source.get("embedding_model") or operation.get("embedding_model") or ""),
                "vector_version": str(source.get("vector_version") or operation.get("vector_version") or ""),
                "reason": reason,
            }
        )

    add(operation.get("query_vector"), operation, "control_explicit_query")
    for vector in operation.get("query_vectors", []) or []:
        add(vector, operation, "control_explicit_query_set")

    expansion_ids = {
        str(value)
        for value in operation.get("expand_from_knowledge_ids", []) or []
        if str(value)
    }
    if expansion_ids:
        for context in workspace.get("context_items", []) or []:
            if not isinstance(context, dict):
                continue
            payload = context.get("payload") if isinstance(context.get("payload"), dict) else {}
            knowledge_id = str(payload.get("knowledge_id") or context.get("source_record_id") or "")
            if knowledge_id not in expansion_ids:
                continue
            for span in _as_dict_list(payload.get("spans")):
                if str(span.get("knowledge_node_role") or "") == "nature":
                    add(span.get("core_vector") or span.get("nature_vector"), span, f"expand_from:{knowledge_id}")

    for conclusion in _as_dict_list(operation.get("conclusion")):
        for field in ("core_vector", "relation_vector", "nature_vector"):
            add(conclusion.get(field), conclusion, f"control_conclusion:{field}")

    if not specs:
        event_query = build_event_query(workspace)
        add(event_query.get("query_vector"), {
            "embedding_model": event_query.get("query_embedding_model"),
            "vector_version": event_query.get("query_vector_version"),
        }, "event_context_fallback")

    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for spec in specs:
        key = tuple(float(value) for value in spec["query_vector"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique[:6]


def _coarse_knowledge_query_specs(
    workspace: dict[str, Any],
    *,
    max_queries: int,
) -> list[dict[str, Any]]:
    role_priority = {
        "reasoning_goal": 0,
        "guard_evidence": 1,
        "context_caveat": 1,
        "input_evidence": 2,
        "measurement_context": 3,
        "derived_result": 4,
    }
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, span in enumerate(_as_dict_list(workspace.get("spans"))):
        vector = _number_list(
            span.get("core_vector")
            or span.get("span_core_vec")
            or span.get("nature_vector")
        )
        if not vector:
            continue
        role = str(span.get("span_role") or "").strip()
        candidates.append(
            (
                role_priority.get(role, 2),
                index,
                {
                    "query_vector": vector,
                    "embedding_model": str(span.get("embedding_model") or ""),
                    "vector_version": str(span.get("vector_version") or ""),
                    "span_id": str(span.get("span_id") or ""),
                    "span_role": role,
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for _, _, spec in candidates:
        vector_key = tuple(float(value) for value in spec["query_vector"])
        if vector_key in seen:
            continue
        seen.add(vector_key)
        selected.append(spec)
        if len(selected) >= max(1, int(max_queries)):
            break
    return selected


def _round_robin_knowledge_records(
    query_results: list[list[dict[str, Any]]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    max_rank = max((len(records) for records in query_results), default=0)
    for rank in range(max_rank):
        for records in query_results:
            if rank >= len(records):
                continue
            record = records[rank]
            knowledge_id = str(record.get("knowledge_id") or record.get("id") or "")
            if not knowledge_id or knowledge_id in selected_ids:
                continue
            selected_ids.add(knowledge_id)
            selected.append(record)
            if len(selected) >= max(1, int(top_k)):
                return selected
    return selected


def _matched_context_ids(result: Any, memory: WorkingMemoryState) -> list[str]:
    known = {
        str(item.get("context_id") or "")
        for workspace in memory.event_workspaces.values()
        if isinstance(workspace, dict)
        for item in workspace.get("context_items", [])
        if isinstance(item, dict) and str(item.get("context_id") or "")
    }
    if not known:
        return []
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            context_id = str(node.get("context_id") or "")
            if context_id in known:
                found.add(context_id)
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            if node in known:
                found.add(node)
                return
            for context_id in known:
                if node.startswith(f"context_{context_id}_"):
                    found.add(context_id)

    walk(result)
    return sorted(found)


def _compact_rule_selector(selector: dict[str, Any]) -> dict[str, Any]:
    if not selector:
        return {}
    selector = normalize_rule_selector(selector)
    return {
        key: value
        for key, value in selector.items()
        if key not in {"required_rule_pattern", "span_pattern"}
    } | {
        "has_required_rule_pattern": isinstance(selector.get("required_rule_pattern") or selector.get("span_pattern"), dict),
        "selector_vector_count": len(_rule_selector_vectors(selector)),
    }


def _output_granularity(output: dict[str, Any]) -> str:
    explicit = str(output.get("answer_granularity") or output.get("granularity") or output.get("output_granularity") or "").strip().lower()
    if explicit:
        return explicit
    if str(output.get("answer_mode") or "").strip().lower() == "event_span" and _as_dict_list(output.get("conclusion")):
        return "structured_conclusion"
    if str(output.get("option") or output.get("predicted_option") or "").strip():
        return "concrete_answer"
    answer = str(output.get("answer") or output.get("conclusion") or output.get("reason") or "").strip().lower()
    if "final answer" in answer or "supports candidate" in answer:
        return "concrete_answer"
    return "high_level_strategy"


def _compact_output_for_gate(output: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in output.items()
        if key not in {"span_pattern", "vector", "condition_vector", "conclusion_vector"}
    }


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    return shared_safe_float(value, default)


def _merge_unique(existing: list[str], incoming: Any) -> list[str]:
    merged = [str(item) for item in existing if str(item)]
    for item in incoming or []:
        value = str(item)
        if value and value not in merged:
            merged.append(value)
    return merged


def _matched_rule_ids(produced: list[dict[str, Any]]) -> list[str]:
    rule_ids: list[str] = []
    for item in produced:
        for match in _as_dict_list(item.get("matched_rules")):
            rule = match.get("rule") if isinstance(match.get("rule"), dict) else {}
            rule_id = str(rule.get("rule_id") or "")
            if rule_id and rule_id not in rule_ids:
                rule_ids.append(rule_id)
    return rule_ids


def _control_select_matched_rules(
    produced: list[dict[str, Any]],
    *,
    rule_selector: dict[str, Any],
    explicit_rule_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execute_all = bool(rule_selector.get("execute_all_matches"))
    if explicit_rule_ids and not rule_selector:
        execute_all = True
    requested_top_k = max(
        1,
        int(
            _safe_float(
                rule_selector.get("execute_top_k"),
                len(explicit_rule_ids) if explicit_rule_ids else 1,
            )
        ),
    )
    memory_preferred_rule_ids = {
        str(rule_id)
        for rule_id in rule_selector.get("memory_preferred_rule_ids", [])
        if str(rule_id)
    }
    goal_feature_types = {
        str(value).strip()
        for value in rule_selector.get("goal_feature_types", [])
        if str(value).strip()
    }
    prefer_goal_reachability = bool(rule_selector.get("prefer_goal_reachability"))
    prefer_context_domain = bool(rule_selector.get("prefer_context_domain"))
    selected_ids: list[str] = []
    rejected_ids: list[str] = []
    for operation in produced:
        if str(operation.get("operation") or "") != "match":
            continue
        matched = _as_dict_list(operation.get("matched_rules"))
        ranked = sorted(
            matched,
            key=lambda item: _control_rule_execution_score(
                item,
                memory_preferred_rule_ids=memory_preferred_rule_ids,
                prefer_memory_paths=rule_selector.get("memory_path_policy") == "prefer_then_fallback",
                goal_feature_types=goal_feature_types,
                prefer_goal_reachability=prefer_goal_reachability,
                prefer_context_domain=prefer_context_domain,
            ),
            reverse=True,
        )

        selected = ranked if execute_all else ranked[:requested_top_k]
        selected_rule_ids = [
            str((item.get("rule") or {}).get("rule_id") or "")
            for item in selected
            if str((item.get("rule") or {}).get("rule_id") or "")
        ]
        selected_keys = set(selected_rule_ids)
        operation["matched_rules"] = selected
        fallback_matches = [
            deepcopy(item)
            for item in ranked
            if str((item.get("rule") or {}).get("rule_id") or "") not in selected_keys
            and str((item.get("rule") or {}).get("rule_id") or "") not in memory_preferred_rule_ids
        ]
        if memory_preferred_rule_ids and rule_selector.get("memory_path_policy") == "prefer_then_fallback":
            operation["memory_path_fallback_matched_rules"] = fallback_matches
        operation["control_selection"] = {
            "policy": (
                "selector_execute_all_matches"
                if execute_all
                else "selector_execute_top_ranked_matches"
            ),
            "execute_top_k": len(selected),
            "selected_rule_ids": selected_rule_ids,
            "rejected_matched_rule_ids": [
                str((item.get("rule") or {}).get("rule_id") or "")
                for item in ranked
                if str((item.get("rule") or {}).get("rule_id") or "") not in selected_keys
            ],
            "memory_preferred_rule_ids": sorted(memory_preferred_rule_ids),
            "memory_preference_applied_to": [
                rule_id
                for rule_id in selected_rule_ids
                if rule_id in memory_preferred_rule_ids
            ],
            "memory_preference_policy": (
                "prefer_strictly_matched_memory_path_then_fallback_to_normal_candidates"
                if memory_preferred_rule_ids and rule_selector.get("memory_path_policy") == "prefer_then_fallback"
                else "post_strict_match_tie_break_only"
                if memory_preferred_rule_ids
                else "not_requested"
            ),
            "fallback_normal_rule_ids": [
                str((item.get("rule") or {}).get("rule_id") or "")
                for item in fallback_matches
                if str((item.get("rule") or {}).get("rule_id") or "")
            ],
            "selector": _compact_rule_selector(rule_selector),
        }
        selected_ids.extend(operation["control_selection"]["selected_rule_ids"])
        rejected_ids.extend(operation["control_selection"]["rejected_matched_rule_ids"])
    return produced, {
        "selected_rule_ids": list(dict.fromkeys(selected_ids)),
        "rejected_matched_rule_ids": list(dict.fromkeys(rejected_ids)),
        "execute_all_matches": execute_all,
        "execute_top_k": requested_top_k,
    }

def _control_rule_execution_score(
    match: dict[str, Any],
    *,
    memory_preferred_rule_ids: set[str] | None = None,
    prefer_memory_paths: bool = False,
    goal_feature_types: set[str] | None = None,
    prefer_goal_reachability: bool = False,
    prefer_context_domain: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    rule = match.get("rule") if isinstance(match.get("rule"), dict) else {}
    outcome = match.get("match") if isinstance(match.get("match"), dict) else {}
    metadata = outcome.get("metadata") if isinstance(outcome.get("metadata"), dict) else {}
    rule_id = str(rule.get("rule_id") or "")
    produced_feature_types = _rule_produced_feature_types(rule)
    goal_alignment = float(bool(produced_feature_types.intersection(goal_feature_types or set())))
    reachability = match.get("control_goal_reachability") if isinstance(match.get("control_goal_reachability"), dict) else {}
    goal_reachable = float(bool(reachability.get("reachable")))
    goal_distance = _safe_float(reachability.get("distance"), 1_000_000.0)
    context_domain = match.get("control_context_domain_alignment")
    context_domain_score = _safe_float(
        context_domain.get("selection_score") if isinstance(context_domain, dict) else 0.0,
        0.0,
    )
    # Memory is a path preference, never retrieval-time evidence.  This method
    # receives only Rules that already passed strict matching.  A Control Rule
    # may explicitly prefer a successful historical path; ordinary candidates
    # remain attached for fallback if that execution fails.
    memory_preferred = float(rule_id in (memory_preferred_rule_ids or set()))
    return (
        context_domain_score if prefer_context_domain else 0.0,
        memory_preferred if prefer_memory_paths else goal_alignment,
        goal_reachable if prefer_goal_reachability else goal_alignment,
        -goal_distance if prefer_goal_reachability and goal_reachable else -1_000_000.0,
        goal_alignment,
        _safe_float(metadata.get("bundle_score"), 0.0),
        memory_preferred if not prefer_memory_paths else _safe_float(rule.get("priority"), 0.0),
        _safe_float(rule.get("priority"), 0.0),
        _safe_float(rule.get("confidence"), 0.0) + _safe_float(match.get("routing_score"), 0.0),
    )


def _annotate_context_domain_strategy_alignment(
    produced: list[dict[str, Any]],
    *,
    memory: WorkingMemoryState,
) -> None:
    """Apply a Control-authored domain routing policy to matched strategies.

    Domain evidence comes from the current frontend/event graph.  A matching
    domain strategy wins; the generic strategy is used only when no installed
    domain strategy matches.  No concrete domain name is encoded here.
    """
    signals = _working_memory_domain_signals(memory)
    matches: list[dict[str, Any]] = []
    for operation in produced:
        if str(operation.get("operation") or "") != "match":
            continue
        raw_matches = operation.get("matched_rules")
        if isinstance(raw_matches, dict):
            matches.append(raw_matches)
        elif isinstance(raw_matches, list):
            matches.extend(item for item in raw_matches if isinstance(item, dict))
    specific_scores: list[float] = []
    for match in matches:
        rule = match.get("rule") if isinstance(match.get("rule"), dict) else {}
        specific_scores.append(_domain_signal_alignment(str(rule.get("domain") or ""), signals))
    has_specific_match = any(score > 0.0 for score in specific_scores)
    for match, specific_score in zip(matches, specific_scores):
        rule = match.get("rule") if isinstance(match.get("rule"), dict) else {}
        tags = {str(value).strip().lower() for value in rule.get("scope_tags", []) if str(value).strip()}
        generic_fallback = (
            str(rule.get("domain") or "").strip().lower() in {"", "general", "generic"}
            and "generic_domain_fallback" in tags
        )
        selection_score = specific_score
        if not has_specific_match and generic_fallback:
            selection_score = 1.0
        match["control_context_domain_alignment"] = {
            "signals": signals,
            "rule_domain": str(rule.get("domain") or ""),
            "specific_alignment": specific_score,
            "generic_fallback": generic_fallback,
            "specific_strategy_available": has_specific_match,
            "selection_score": selection_score,
        }


def _working_memory_domain_signals(memory: WorkingMemoryState) -> list[str]:
    signals: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip().lower()
        if text and text not in signals:
            signals.append(text)

    def add_frontend_document(value: Any) -> None:
        if not isinstance(value, dict):
            return
        add(value.get("domain"))
        for mainline in _as_dict_list(value.get("mainlines")):
            add(mainline.get("title"))
        for event in _as_dict_list(value.get("events")):
            add(event.get("event_type"))

    for entry in memory.zone_entries.get("input_workspace", []):
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        add(payload.get("domain"))
        game_input = payload.get("game_input") if isinstance(payload.get("game_input"), dict) else {}
        add(game_input.get("domain"))
        frontend_state = payload.get("frontend_state") if isinstance(payload.get("frontend_state"), dict) else {}
        add((frontend_state.get("metadata") or {}).get("domain") if isinstance(frontend_state.get("metadata"), dict) else "")
        for event in _as_dict_list(frontend_state.get("events")):
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            add(metadata.get("domain"))
            add(event.get("event_type"))
            add_frontend_document(metadata.get("frontend_document"))
    for workspace in (memory.event_workspaces or {}).values():
        if not isinstance(workspace, dict):
            continue
        metadata = workspace.get("metadata") if isinstance(workspace.get("metadata"), dict) else {}
        add(metadata.get("domain"))
        add_frontend_document(metadata.get("frontend_document"))
    return signals


def _domain_signal_alignment(rule_domain: str, signals: list[str]) -> float:
    domain = str(rule_domain or "").strip().lower()
    if domain in {"", "general", "generic"}:
        return 0.0
    domain_tokens = set(re.findall(r"[a-z0-9]+", domain))
    if not domain_tokens:
        return 0.0
    best = 0.0
    for signal in signals:
        normalized = str(signal or "").strip().lower()
        signal_tokens = set(re.findall(r"[a-z0-9]+", normalized))
        if not signal_tokens:
            continue
        if domain == normalized:
            best = max(best, 3.0)
            continue
        overlap = len(domain_tokens.intersection(signal_tokens))
        if overlap:
            best = max(best, 1.0 + overlap / len(domain_tokens))
    return best


def _annotate_goal_reachability(
    produced: list[dict[str, Any]],
    *,
    memory: WorkingMemoryState,
    rule_db_path: str,
    goal_feature_types: set[str],
    max_depth: int,
) -> None:
    """Score strictly matched candidates by bounded structural reachability.

    This is a generic execution tool for a Control-authored policy.  It reads
    only Rule require/produced feature types and the current Event Graph; it
    does not read descriptions, expected paths, domains, or evaluator data.
    """

    if not goal_feature_types or not rule_db_path:
        return
    from .persistence.rule_db import load_rule_records

    all_rules = [
        rule for rule in load_rule_records(rule_db_path, limit=None)
        if str(rule.get("rule_layer") or "").strip().lower() != "control"
        and str(rule.get("status") or "active").strip().lower() not in {
            "inactive", "deprecated", "superseded", "archived", "deleted", "rejected", "blocked",
        }
    ]
    available = set(_working_memory_feature_types(memory))
    rule_specs = [
        (
            str(rule.get("rule_id") or ""),
            _rule_required_feature_sets(rule),
            _rule_produced_feature_types(rule),
        )
        for rule in all_rules
    ]
    for operation in produced:
        if str(operation.get("operation") or "") != "match":
            continue
        raw_matches = operation.get("matched_rules")
        matched_items = (
            [raw_matches]
            if isinstance(raw_matches, dict)
            else [item for item in raw_matches if isinstance(item, dict)]
            if isinstance(raw_matches, list)
            else []
        )
        for item in matched_items:
            rule = item.get("rule") if isinstance(item.get("rule"), dict) else {}
            rule_id = str(rule.get("rule_id") or "")
            start = available | _rule_produced_feature_types(rule)
            distance = _shortest_goal_distance(
                start,
                goal_feature_types=goal_feature_types,
                rule_specs=rule_specs,
                excluded_rule_ids={rule_id},
                max_depth=max_depth,
            )
            item["control_goal_reachability"] = {
                "reachable": distance is not None,
                "distance": distance,
                "max_depth": max_depth,
                "policy": "bounded_structural_rule_graph_search",
            }


def _annotate_alignment_goal_reachability(
    matcher_result: dict[str, Any],
    *,
    memory: WorkingMemoryState,
    rule_db_path: str,
    goal_feature_types: set[str],
    max_depth: int,
) -> None:
    """Annotate rejected semantic gaps using only the current Rule graph.

    This is the failure-path counterpart of ``_annotate_goal_reachability``.
    It does not make a failed Rule executable and does not inspect evaluator
    answers.  It only tells Control which strictly unproven Rule, if repaired
    through Knowledge, can structurally lead to the goal already present in
    the Event Graph.
    """

    metadata = matcher_result.get("metadata") if isinstance(matcher_result.get("metadata"), dict) else {}
    requirements = metadata.get("alignment_requirements")
    if not isinstance(requirements, list) or not requirements or not goal_feature_types or not rule_db_path:
        return
    from .persistence.rule_db import load_rule_records

    all_rules = [
        rule for rule in load_rule_records(rule_db_path, limit=None)
        if str(rule.get("rule_layer") or "").strip().lower() != "control"
        and str(rule.get("status") or "active").strip().lower() not in {
            "inactive", "deprecated", "superseded", "archived", "deleted", "rejected", "blocked",
        }
    ]
    rules_by_id = {str(rule.get("rule_id") or ""): rule for rule in all_rules}
    rule_specs = [
        (
            str(rule.get("rule_id") or ""),
            _rule_required_feature_sets(rule),
            _rule_produced_feature_types(rule),
        )
        for rule in all_rules
    ]
    available = set(_working_memory_feature_types(memory))
    annotations: dict[str, dict[str, Any]] = {}
    for rule_id in {str(item.get("rule_id") or "") for item in requirements if isinstance(item, dict)}:
        rule = rules_by_id.get(rule_id)
        if not rule:
            continue
        produced = _rule_produced_feature_types(rule)
        goal_alignment = bool(produced.intersection(goal_feature_types))
        distance = 0 if goal_alignment else _shortest_goal_distance(
            available | produced,
            goal_feature_types=goal_feature_types,
            rule_specs=rule_specs,
            excluded_rule_ids={rule_id},
            max_depth=max_depth,
        )
        annotations[rule_id] = {
            "goal_alignment": goal_alignment,
            "goal_reachable": distance is not None,
            "goal_distance": distance,
            "goal_reachability_policy": "bounded_structural_rule_graph_search",
        }
    for requirement in requirements:
        if isinstance(requirement, dict):
            requirement.update(annotations.get(str(requirement.get("rule_id") or ""), {}))


def _shortest_goal_distance(
    start_features: set[str],
    *,
    goal_feature_types: set[str],
    rule_specs: list[tuple[str, list[set[str]], set[str]]],
    excluded_rule_ids: set[str],
    max_depth: int,
    max_states: int = 4096,
) -> int | None:
    if start_features.intersection(goal_feature_types):
        return 0
    queue: list[tuple[frozenset[str], frozenset[str], int]] = [
        (frozenset(start_features), frozenset(excluded_rule_ids), 0)
    ]
    seen: set[tuple[frozenset[str], frozenset[str]]] = {(queue[0][0], queue[0][1])}
    cursor = 0
    while cursor < len(queue) and len(seen) <= max_states:
        features, used_rule_ids, depth = queue[cursor]
        cursor += 1
        if depth >= max_depth:
            continue
        feature_set = set(features)
        for rule_id, required_options, outputs in rule_specs:
            if not rule_id or rule_id in used_rule_ids or not outputs.difference(feature_set):
                continue
            if not any(required.issubset(feature_set) for required in required_options):
                continue
            next_features = frozenset(feature_set | outputs)
            next_used = frozenset((*used_rule_ids, rule_id))
            next_depth = depth + 1
            if set(next_features).intersection(goal_feature_types):
                return next_depth
            key = (next_features, next_used)
            if key not in seen:
                seen.add(key)
                queue.append((next_features, next_used, next_depth))
    return None


def _rule_required_feature_sets(rule: dict[str, Any]) -> list[set[str]]:
    options: list[set[str]] = []
    for required in _as_dict_list(rule.get("required")):
        features = {
            str(slot.get("feature_type") or "").strip()
            for slot in _as_dict_list(required.get("condition"))
            if str(slot.get("feature_type") or "").strip()
        }
        if features:
            options.append(features)
    return options or [set()]


def _rule_produced_feature_types(rule: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for produced in _as_dict_list(rule.get("produced")):
        for span in _as_dict_list(produced.get("conclusion")):
            feature_type = str(span.get("feature_type") or "").strip()
            if feature_type:
                values.add(feature_type)
        for field in ("result_feature_type", "goal_result_feature_type"):
            feature_type = str(produced.get(field) or "").strip()
            if feature_type:
                values.add(feature_type)
    return values


def _working_memory_goal_feature_types(memory: WorkingMemoryState) -> list[str]:
    goals: list[str] = []
    for workspace in (memory.event_workspaces or {}).values():
        for span in _as_dict_list(workspace.get("spans")):
            feature_type = str(span.get("feature_type") or "").strip()
            if feature_type.startswith("goal_") and len(feature_type) > len("goal_"):
                goals.append(feature_type[len("goal_"):])
                continue
            metadata = span.get("metadata") if isinstance(span.get("metadata"), dict) else {}
            declared_target = str(
                metadata.get("target_feature_type")
                or metadata.get("requested_feature_type")
                or span.get("target_feature_type")
                or ""
            ).strip()
            role = str(span.get("span_role") or span.get("role") or "").strip().lower()
            relation = str(span.get("relation") or "").strip().lower()
            if declared_target and (role in {"goal", "objective"} or relation in {"requests", "targets", "objective"}):
                goals.append(declared_target)
    return list(dict.fromkeys(goals))


def _working_memory_feature_types(memory: WorkingMemoryState) -> list[str]:
    return list(dict.fromkeys(
        str(span.get("feature_type") or "").strip()
        for workspace in (memory.event_workspaces or {}).values()
        for span in _as_dict_list(workspace.get("spans"))
        if str(span.get("feature_type") or "").strip()
    ))


def _memory_path_fallback_operation(
    operation: dict[str, Any],
    executor_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Return retained ordinary candidates if a preferred historical path fails.

    The fallback set is built only from Rules that already passed strict
    matching against the current Event Graph.  It is not reconstructed from
    Memory, so a failed historical path cannot be retried through this branch.
    """
    selection = operation.get("control_selection") if isinstance(operation.get("control_selection"), dict) else {}
    if selection.get("memory_preference_policy") != "prefer_strictly_matched_memory_path_then_fallback_to_normal_candidates":
        return None
    if not selection.get("memory_preference_applied_to"):
        return None
    status = str(executor_result.get("status") or "").strip()
    if bool(executor_result.get("ok", False)) and status not in {"stop_path", "no_executable_rules"}:
        return None
    fallback_matches = _as_dict_list(operation.get("memory_path_fallback_matched_rules"))
    if not fallback_matches:
        return None
    failed_rule_ids = [str(rule_id) for rule_id in selection.get("selected_rule_ids", []) if str(rule_id)]
    return {
        "operation": "match",
        "match_type": str(operation.get("match_type") or "rule_to_working_memory"),
        "source": deepcopy(operation.get("source") or {}),
        "target": deepcopy(operation.get("target") or {}),
        "matched_rules": fallback_matches,
        "rejected_rules": [],
        "control_selection": {
            "policy": "memory_preferred_path_failed_execute_retained_normal_fallback",
            "failed_memory_rule_ids": failed_rule_ids,
            "selected_rule_ids": [
                str((item.get("rule") or {}).get("rule_id") or "")
                for item in fallback_matches
                if str((item.get("rule") or {}).get("rule_id") or "")
            ],
        },
    }


def _compact_matcher_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    compact = dict(metadata)
    target = compact.get("target")
    if isinstance(target, dict) and target.get("rule_records"):
        target = dict(target)
        target["rule_records"] = [
            str((rule or {}).get("rule_id") or "")
            for rule in target.get("rule_records", [])[:24]
            if isinstance(rule, dict)
        ]
        compact["target"] = target
    return compact


def _load_rule_records_by_ids(rule_db_path: str, rule_ids: list[str]) -> list[dict[str, Any]]:
    wanted = {str(item) for item in rule_ids if str(item)}
    if not wanted:
        return []
    from .persistence.rule_db import load_rule_records
    records = load_rule_records(rule_db_path, rule_ids=rule_ids, limit=len(wanted))
    by_id = {str(rule.get("rule_id") or ""): rule for rule in records}
    return [by_id[rule_id] for rule_id in rule_ids if rule_id in by_id]


def _cross_path_search_authorization_error(
    selector: dict[str, Any],
    *,
    rule_db_path: str,
    target_rule_layer: str,
) -> str:
    selector = normalize_rule_selector(selector)
    search_mode = str(selector.get("search_mode") or "").strip().lower()
    if search_mode not in {"control_governed_cross_path", "cross_path_composition"}:
        return ""
    if target_rule_layer == "control":
        return "cross-path composition must target executable non-control rules"
    authorization = selector.get("authorization") if isinstance(selector.get("authorization"), dict) else {}
    if str(authorization.get("kind") or "").strip() != "control_rule":
        return "cross-path composition requires control-rule authority"
    control_rule_id = str(authorization.get("rule_id") or "").strip()
    if not control_rule_id:
        return "cross-path composition is missing the authorizing control rule id"
    records = _load_rule_records_by_ids(rule_db_path, [control_rule_id])
    if not records:
        return f"authorizing control rule does not exist: {control_rule_id}"
    if str(records[0].get("rule_layer") or "").strip().lower() != "control":
        return f"authorizing rule is not a control rule: {control_rule_id}"
    return ""


def _state_payload_text(value: Any) -> str:
    parts: list[str] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {"vector", "rule_vector", "condition_vector", "conclusion_vector", "span_core_vector", "relation_vector", "nature_vector"}:
                    continue
                if key in {"status", "reason", "phase", "stage", "payload_type", "source", "missing_span_query", "operation"}:
                    if isinstance(item, (str, int, float, bool)):
                        parts.append(str(item))
                walk(item, depth + 1)
        elif isinstance(node, list):
            for item in node[:24]:
                walk(item, depth + 1)
        elif isinstance(node, (str, int, float, bool)):
            text = str(node).strip()
            if text and len(text) <= 240:
                parts.append(text)

    walk(value)
    return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))[:4000]


def _working_memory_routing_tags(memory: WorkingMemoryState) -> list[str]:
    tags: list[str] = []

    def add_span_tags(span: dict[str, Any]) -> None:
        core = str(span.get("core") or "").strip().lower()
        relation = str(span.get("relation") or "").strip().lower()
        nature = str(span.get("nature") or "").strip().lower()
        text = str(span.get("text") or "").strip().lower()
        if "question_type" in core or relation == "classify":
            tags.append("question_type")
            for tag in ("multiple_choice", "open_question", "yes_no", "explanation"):
                if tag in nature or tag in text:
                    tags.append(tag)
        if "question_goal" in core or relation == "question_goal":
            tags.append("question_goal")

    for entry in memory.zone_entries.get("input_workspace", []):
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        domain = str(payload.get("domain") or "").strip()
        if domain:
            tags.append(domain)
        frontend_state = payload.get("frontend_state") if isinstance(payload.get("frontend_state"), dict) else {}
        for span in _as_dict_list(frontend_state.get("spans")):
            add_span_tags(span)
    for workspace in (memory.event_workspaces or {}).values():
        for span in _as_dict_list(workspace.get("spans")):
            add_span_tags(span)
    for zone in ("event_workspace",):
        for entry in memory.zone_entries.get(zone, [])[-12:]:
            payload = entry.payload if isinstance(entry.payload, dict) else {}
            text = _state_payload_text(payload).lower()
            if "frontend_patch_failed" in text or "frontend_patch_unavailable" in text:
                tags.extend(["frontend_patch_required", "missing_information", "no_decision"])
            if "missing" in text and "information" in text:
                tags.append("missing_information")
            if "no completed evidence chain" in text:
                tags.extend(["domain_chain_missing", "no_decision"])
    return list(dict.fromkeys(tags))


def _executor_results_completed_state_or_graph_step(executor_results: list[dict[str, Any]]) -> bool:
    for result in executor_results:
        for step in _as_dict_list(result.get("steps")):
            if str(step.get("status") or "") != "completed":
                continue
            operation = str(step.get("operation") or "").strip()
            if operation in {"add", "update", "delete", "merge", "split"}:
                return True
    return False
