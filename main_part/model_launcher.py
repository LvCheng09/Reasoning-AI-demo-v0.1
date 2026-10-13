"""Top-level launcher for the main reasoning model.

This is the normal path outside demos:
raw input -> frontend LLM -> working memory -> runtime launcher -> scheduler.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from .data_structure import WorkingMemoryState
from .event_graph_contract import validate_event_graph_contract
from .frontend_schema import normalize_spans, normalize_events, confidence as normalize_frontend_confidence
from .frontend_event_builder import FrontendEventBuilder
from .frontend_semantic import frontend_from_env, validate_frontend_selection, REQUIRED_FRONTEND_MODEL
from .frontend_working_memory_bridge import submit_frontend_state_to_working_memory
from .memory_redis import build_memory_redis_cache_from_env
from .memory_workspace import select_memory_workspace_context
from .rule_vector_encoder import RuleVectorEncoder
from .runtime_launcher import RuntimeLauncher
from .runtime_databases import default_db_paths, validate_runtime_db_paths
from .scheduler_new import SchedulerNew


class MainPartModelLauncher:
    """One long-lived model launcher.

    It owns one working memory, one scheduler, and one runtime launcher. Each
    new raw input is sent to the frontend LLM first, then stored through the
    frontend-working-memory bridge. Scheduler execution is driven by
    RuntimeLauncher.
    """

    def __init__(
        self,
        *,
        working_memory: WorkingMemoryState | None = None,
        scheduler: SchedulerNew | None = None,
        frontend_adapter: FrontendEventBuilder | None = None,
        vector_encoder: Any | None = None,
        memory_cache: Any | None = None,
        session_id: str = "main_runtime",
        auto_run: bool = True,
        runtime_max_attempts: int = 3,
        clear_previous_transient_state: bool = True,
        db_paths: Mapping[str, Any] | None = None,
    ) -> None:
        validate_frontend_selection()
        if frontend_adapter is not None:
            llm = getattr(getattr(frontend_adapter, "frontend", None), "llm", None)
            if getattr(llm, "model", REQUIRED_FRONTEND_MODEL) != REQUIRED_FRONTEND_MODEL:
                raise ValueError(f"frontend adapter must use {REQUIRED_FRONTEND_MODEL}")
        if scheduler is not None:
            scheduler_paths = validate_runtime_db_paths(scheduler.state.db_paths)
            resolved_db_paths = validate_runtime_db_paths(scheduler_paths if db_paths is None else db_paths)
            if resolved_db_paths != scheduler_paths:
                raise ValueError("launcher and provided scheduler must use the same runtime databases")
        else:
            resolved_db_paths = validate_runtime_db_paths(_default_db_paths() if db_paths is None else db_paths)
        self.working_memory = working_memory or WorkingMemoryState()
        self.session_id = str(session_id)
        self.auto_run = bool(auto_run)
        self.clear_previous_transient_state = bool(clear_previous_transient_state)
        self.frontend_adapter = frontend_adapter
        if vector_encoder is not None:
            self.vector_encoder = vector_encoder
            self.vector_encoder_status = {"ok": True, "status": "provided"}
        else:
            encoder_result = _maybe_build_vector_encoder()
            self.vector_encoder = encoder_result.get("encoder")
            self.vector_encoder_status = {key: value for key, value in encoder_result.items() if key != "encoder"}
        self.db_paths = resolved_db_paths
        if memory_cache is not None:
            self.memory_cache = memory_cache
            self.memory_cache_status = {"ok": True, "status": "provided"}
        else:
            cache_result = build_memory_redis_cache_from_env()
            self.memory_cache = cache_result.get("cache")
            self.memory_cache_status = {
                key: value
                for key, value in cache_result.items()
                if key != "cache"
            }
        self.scheduler = scheduler or SchedulerNew(
            self.working_memory,
            session_id=session_id,
            db_paths=resolved_db_paths,
            frontend=self._frontend_patch_stage,
            vector_encoder=self.vector_encoder,
            memory_cache=self.memory_cache,
        )
        self.scheduler.working_memory = self.working_memory
        self.scheduler.vector_encoder = self.vector_encoder
        self.scheduler.memory_cache = self.memory_cache
        self.runtime = RuntimeLauncher(self.scheduler, working_memory=self.working_memory, max_attempts=runtime_max_attempts)

    def submit_input(self, input_payload: Mapping[str, Any] | str) -> dict[str, Any]:
        frontend_started = time.perf_counter()
        game_input = self._normalize_input(input_payload)
        _reject_supervision_at_frontend_boundary(game_input)
        self._prepare_submission()
        original_game_input = deepcopy(game_input)
        input_context = self._build_input_context(original_game_input)
        frontend_input = deepcopy(original_game_input)
        frontend_input["input_context"] = input_context
        frontend_state = self.build_frontend_state(frontend_input)
        frontend_source = "frontend_adapter"
        frontend_seconds = time.perf_counter() - frontend_started
        result = self._submit_event_graph(
            original_game_input,
            frontend_state,
            input_context=input_context,
            source=frontend_source,
        )
        result["pipeline_timing"]["input_context_and_frontend_seconds"] = frontend_seconds
        return result

    def submit_event_graph(
        self,
        game_input: Mapping[str, Any] | str,
        event_graph: Mapping[str, Any],
        *,
        input_context: Mapping[str, Any] | None = None,
        source: str = "frontend_event_graph",
    ) -> dict[str, Any]:
        """Submit a frontend-produced event graph through the stable backend boundary."""

        normalized_input = self._normalize_input(game_input)
        _reject_supervision_at_frontend_boundary(normalized_input)
        self._prepare_submission()
        context = dict(input_context) if input_context is not None else self._build_input_context(normalized_input)
        return self._submit_event_graph(
            normalized_input,
            event_graph,
            input_context=context,
            source=source,
        )

    def build_input_context(self, input_payload: Mapping[str, Any] | str) -> dict[str, Any]:
        """Build the same ANN-backed context used before frontend event construction."""

        normalized_input = self._normalize_input(input_payload)
        _reject_supervision_at_frontend_boundary(normalized_input)
        return self._build_input_context(normalized_input)

    def _prepare_submission(self) -> None:
        if self.clear_previous_transient_state:
            if self.working_memory.event_workspaces:
                self.working_memory.transfer_event_workspace_to_memory(
                    session_id=self.session_id,
                    completion={
                        "status": "superseded",
                        "reason": "new_input_received_before_previous_event_was_closed",
                    },
                )
            self.scheduler.clear_transient_state(
                clear_outputs=True,
                clear_errors=True,
                clear_working_memory=False,
                clear_runtime_workspaces=True,
            )

    def _submit_event_graph(
        self,
        game_input: Mapping[str, Any],
        event_graph: Mapping[str, Any],
        *,
        input_context: Mapping[str, Any],
        source: str,
    ) -> dict[str, Any]:
        boundary_started = time.perf_counter()
        frontend_state = self._normalize_frontend_state(dict(event_graph), game_input)
        contract = validate_event_graph_contract(frontend_state)
        if not contract["ok"]:
            raise ValueError(f"frontend event graph contract rejected: {contract['errors']}")
        bridge_result = submit_frontend_state_to_working_memory(
            self.working_memory,
            game_input=game_input,
            frontend_state=frontend_state,
            input_context=input_context,
            session_id=self.session_id,
            source=source,
        )
        boundary_seconds = time.perf_counter() - boundary_started
        backend_started = time.perf_counter()
        backend_cpu_started = time.process_time()
        outputs = self.runtime.run_once() if self.auto_run else []
        backend_cpu_seconds = time.process_time() - backend_cpu_started
        backend_seconds = time.perf_counter() - backend_started
        return {
            "frontend_state": frontend_state,
            "event_graph_contract": contract,
            "working_memory_submit": bridge_result,
            "input_context": input_context,
            "outputs": outputs,
            "runtime_summary": self.runtime_summary(),
            "pipeline_timing": {
                "graph_boundary_seconds": boundary_seconds,
                "backend_wall_seconds": backend_seconds,
                "backend_process_cpu_seconds": backend_cpu_seconds,
                "backend_executed": self.auto_run,
            },
        }

    def run_once(self) -> list[dict[str, Any]]:
        return self.runtime.run_once()

    def run_forever(self) -> None:
        self.runtime.run_forever()

    def read_output(self) -> dict[str, Any] | None:
        return self.scheduler.read_latest_output()

    def build_output_runtime_state(self) -> tuple[dict[str, Any], list[str]]:
        """Expose the final output-gated graph for a presentation layer.

        This is a read-only projection.  It deliberately does not belong to
        the Scheduler, Matcher, or Executor, because rendering text must not
        affect a reasoning decision.
        """
        from .output_service import build_runtime_output_state

        return build_runtime_output_state(
            self.working_memory,
            outputs=self.scheduler.state.outputs,
            task_result=self.scheduler.state.control_state.get("task_result"),
        )

    def render_final_answer(
        self,
        answer_service: Any,
        *,
        user_query: str = "",
        language: str = "zh-CN",
        style_instruction: str = "",
    ) -> Any:
        """Render a user answer from the already completed runtime state.

        ``answer_service`` must expose ``answer_runtime``.  It is injected so
        a deployment can choose deterministic-only presentation or an
        OpenAI-compatible verbalizer without coupling the reasoning core to an
        HTTP client.
        """
        render = getattr(answer_service, "answer_runtime", None)
        if not callable(render):
            raise TypeError("answer_service must expose answer_runtime(...)")
        return render(
            self.working_memory,
            outputs=self.scheduler.state.outputs,
            task_result=self.scheduler.state.control_state.get("task_result"),
            user_query=user_query,
            language=language,
            style_instruction=style_instruction,
        )

    def runtime_summary(self) -> dict[str, Any]:
        state = self.scheduler.state
        return {
            "scheduler_status": state.status,
            "scheduler_phase": state.phase,
            "task_result": deepcopy(state.control_state.get("task_result") or {"status": "unknown"}),
            "control_state": deepcopy(state.control_state),
            "loop_index": state.loop_index,
            "active_rule_ids": list(state.active_rule_ids),
            "pending_rule_ids": list(state.pending_rule_ids),
            "pending_rule_selectors": list(state.pending_rule_selectors),
            "vector_encoder_status": dict(getattr(self, "vector_encoder_status", {}) or {}),
            "memory_cache_status": dict(getattr(self, "memory_cache_status", {}) or {}),
            "output_count": len(state.outputs),
            "working_memory_zone_counts": self.working_memory.zone_counts(session_id=self.session_id),
            "execution_phases": [str(item.get("phase") or "") for item in state.execution_history],
            "last_execution_results": [
                _compact_execution_result(item)
                for item in state.execution_history[-6:]
            ],
        }

    def _build_input_context(
        self,
        game_input: Mapping[str, Any],
        *,
        top_k: int = 6,
    ) -> dict[str, Any]:
        query = self._input_text(game_input)
        vector_result = _safe_encode_vectors_status(self.vector_encoder, [query])
        query_vector = (vector_result.get("vectors") or [None])[0]
        if not query_vector:
            return {
                "status": "input_context_vector_failed",
                "query": query,
                "memory_workspace": [],
                "vectorization": vector_result,
            }
        embedding_model = str(
            getattr(self.vector_encoder, "embedding_model", "")
            or getattr(self.vector_encoder, "encoder_name_or_path", "")
            or ""
        )
        vector_version = str(getattr(self.vector_encoder, "vector_version", "") or "v1")
        memory_context = select_memory_workspace_context(
            self.working_memory.query_memory_entries(),
            query_vector=query_vector,
            query_embedding_model=embedding_model,
            query_vector_version=vector_version,
            top_k=top_k,
        )
        return {
            "status": "input_context_ready",
            "query": query,
            "embedding_model": embedding_model,
            "vector_version": vector_version,
            "memory_workspace": list(memory_context.get("records") or []),
            "memory_workspace_status": memory_context.get("status"),
            "vectorization": {
                "ok": bool(vector_result.get("ok")),
                "status": vector_result.get("status"),
                "errors": list(vector_result.get("errors") or []),
            },
        }

    def _get_frontend_adapter(self) -> FrontendEventBuilder:
        # Graph-only callers do not need an API client or credentials.
        if self.frontend_adapter is None:
            self.frontend_adapter = frontend_from_env()
        return self.frontend_adapter

    def build_frontend_state(self, game_input: Mapping[str, Any]) -> dict[str, Any]:
        adapter = self._get_frontend_adapter()
        return dict(adapter.build_event_graph(game_input))

    def build_frontend_patch_state(
        self, *, game_input: Mapping[str, Any],
        current_frontend_state: Mapping[str, Any], patches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        adapter = self._get_frontend_adapter()
        return dict(adapter.patch_event_graph(
            game_input=game_input, current_event_graph=current_frontend_state,
            patches=patches,
        ))

    def _frontend_patch_stage(
        self,
        state: dict[str, Any],
        memory: WorkingMemoryState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        patches = _as_dict_list(payload.get("patches"))
        try:
            game_input = _latest_game_input(memory)
            current_frontend_state = _latest_frontend_state(memory)
            patched_state = self._normalize_frontend_state(
                self.build_frontend_patch_state(
                    game_input=game_input,
                    current_frontend_state=current_frontend_state,
                    patches=patches,
                ), game_input,
            )
            patch_source = "frontend_adapter_patch"
            contract = validate_event_graph_contract(patched_state)
            if not contract["ok"]:
                raise ValueError(f"frontend event graph patch rejected: {contract['errors']}")
            bridge_result = submit_frontend_state_to_working_memory(
                memory,
                game_input=game_input,
                frontend_state=patched_state,
                session_id=str(state.get("session_id") or self.session_id),
                source=patch_source,
                record_input_entry=False,
            )
            return {
                "phase": "frontend_patch",
                "frontend_patch_result": {
                    "status": "patched",
                    "patches": patches,
                    "bridge_result": bridge_result,
                    "frontend_state": patched_state,
                    "event_graph_contract": contract,
                },
                "active_event_ids": bridge_result.get("event_ids", []),
            }
        except Exception as exc:
            return {
                "phase": "frontend_patch",
                "status": "frontend_patch_failed",
                "reason": repr(exc),
                "frontend_patch_error": {
                    "status": "frontend_patch_failed",
                    "reason": repr(exc),
                    "patches": patches,
                },
            }

    def _normalize_frontend_state(self, state: dict[str, Any], game_input: Mapping[str, Any]) -> dict[str, Any]:
        del game_input
        spans = _normalize_spans(state.get("spans", []))
        events = _normalize_events(state.get("events", []), spans=spans)
        if not events and spans:
            events = [_single_event_from_spans(spans)]
        state["vectorization"] = self._attach_frontend_vectors(spans, events)
        state["spans"] = spans
        state["events"] = events
        state["confidence"] = normalize_frontend_confidence(state.get("confidence", 1.0), field="frontend", default=1.0)
        state["event_workspaces"] = _event_workspaces_from_events(events, spans)
        state.setdefault("frontend_event_source", "provided")
        state.setdefault("frontend_mode", "provided")
        return state

    def _normalize_input(self, input_payload: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(input_payload, str):
            return {"input": input_payload}
        payload = dict(input_payload or {})
        if "input" not in payload and "frontend_state" not in payload and "frontend_output" not in payload:
            raise ValueError("MainPartModelLauncher input requires a single 'input' field.")
        return payload

    def _attach_frontend_vectors(self, spans: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
        encoder = self.vector_encoder
        if encoder is None or not hasattr(encoder, "_encode_text_vector"):
            return {
                "ok": False,
                "status": "encoder_unavailable",
                "requested": 0,
                "encoded": 0,
                "errors": [{"status": "encoder_unavailable"}],
            }
        requests: list[tuple[dict[str, Any], str, str]] = []
        for span in spans:
            core_text = str(span.get("core") or "").strip()
            nature_text = str(span.get("nature") or "").strip()
            relation_text = str(span.get("relation") or "").strip()
            combined_text = " ".join(
                str(span.get(key) or "").strip()
                for key in ("core", "relation", "nature", "text")
                if str(span.get(key) or "").strip()
            )
            requests.extend(
                [
                    (span, "vector", combined_text),
                    (span, "core_vector", core_text),
                    (span, "nature_vector", nature_text),
                    (span, "relation_vector", relation_text),
                ]
            )
            for semantic_relation in span.get("semantic_relations", []) or []:
                if not isinstance(semantic_relation, dict):
                    continue
                debug = semantic_relation.get("debug") if isinstance(semantic_relation.get("debug"), Mapping) else {}
                requests.extend(
                    [
                        (
                            semantic_relation,
                            "core_vector",
                            str(semantic_relation.get("core") or debug.get("core") or "").strip(),
                        ),
                        (
                            semantic_relation,
                            "relation_vector",
                            str(semantic_relation.get("relation") or debug.get("relation") or "").strip(),
                        ),
                        (
                            semantic_relation,
                            "nature_vector",
                            str(semantic_relation.get("nature") or debug.get("nature") or "").strip(),
                        ),
                    ]
                )
        for event in events:
            for relation in event.get("relations", []) or []:
                if not isinstance(relation, Mapping):
                    continue
                relation_text = str(relation.get("relation_type") or relation.get("relation") or "").strip()
                if isinstance(relation, dict):
                    requests.append((relation, "relation_vector", relation_text))
        model_identity = str(getattr(encoder, "embedding_model", "") or getattr(encoder, "encoder_name_or_path", "") or "")
        encoder_version = str(getattr(encoder, "vector_version", "") or "v1")
        reusable = [(target, field, text) for target, field, text in requests
                    if target.get(field) and target.get("embedding_model") == model_identity
                    and target.get("vector_version") == encoder_version
                    and target.get("_frontend_vector_texts", {}).get(field) == text]
        cached_keys = {(id(target), field) for target, field, _ in reusable}
        requests = [(target, field, text) for target, field, text in requests
                    if (id(target), field) not in cached_keys]
        vectorization = _safe_encode_vectors_status(encoder, [text for _, _, text in requests])
        vectors = list(vectorization.get("vectors") or [])
        embedding_model = str(getattr(encoder, "embedding_model", "") or getattr(encoder, "encoder_name_or_path", "") or "")
        vector_version = str(getattr(encoder, "vector_version", "") or "v1")
        errors_by_index = {
            int(error.get("index")): dict(error)
            for error in vectorization.get("errors", [])
            if isinstance(error, dict) and str(error.get("index", "")).isdigit()
        }
        for index, ((target, field, _text), vector) in enumerate(zip(requests, vectors, strict=False)):
            if vector:
                target[field] = vector
                target["embedding_model"] = embedding_model
                target["vector_dimension"] = len(vector)
                target["vector_version"] = vector_version
                target.setdefault("_frontend_vector_texts", {})[field] = _text
                continue
            error = {"field": field, **errors_by_index.get(index, {"status": "missing_vector"})}
            target.setdefault("vectorization_errors", []).append(error)
        return {
            "ok": bool(vectorization.get("ok")),
            "status": str(vectorization.get("status") or "failed"),
            "requested": len(requests),
            "reused": len(reusable),
            "encoded": sum(1 for vector in vectors if vector),
            "embedding_model": embedding_model,
            "vector_version": vector_version,
            "errors": list(vectorization.get("errors") or []),
        }

    def _input_text(self, game_input: Mapping[str, Any]) -> str:
        parts = [
            str(game_input.get("title") or ""),
            str(game_input.get("input") or ""),
            json.dumps(game_input.get("options") or [], ensure_ascii=False),
        ]
        return "\n".join(part for part in parts if part.strip())




def _latest_game_input(memory: WorkingMemoryState) -> dict[str, Any]:
    entries = memory.query_entries(
        zone="input_workspace",
        payload_type="frontend_ready_input",
        reverse=True,
    )
    for entry in entries:
        payload = entry.payload if isinstance(entry.payload, Mapping) else {}
        game_input = payload.get("game_input")
        if isinstance(game_input, Mapping):
            return dict(game_input)
    return {}


def _latest_frontend_state(memory: WorkingMemoryState) -> dict[str, Any]:
    if memory.event_workspaces:
        spans_by_id: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        for event_id, workspace in memory.event_workspaces.items():
            if not isinstance(workspace, Mapping):
                continue
            event_spans = [dict(span) for span in workspace.get("spans", []) or [] if isinstance(span, Mapping)]
            for span in event_spans:
                span_id = str(span.get("span_id") or span.get("id") or "").strip()
                if span_id:
                    spans_by_id[span_id] = span
            events.append(
                {
                    "event_id": event_id,
                    "event_type": workspace.get("event_type", ""),
                    "evidence_span_ids": list(dict.fromkeys(
                        list(workspace.get("evidence_span_ids", [])) +
                        [str(s.get("span_id") or s.get("id")) for s in event_spans
                         if s.get("span_id") or s.get("id")]
                    )),
                    "relations": workspace.get("relations", []),
                    "candidate_paths": workspace.get("candidate_paths", []),
                    "feedback": workspace.get("feedback", []),
                    "metadata": workspace.get("metadata", {}),
                    "confidence": workspace.get("confidence", 0.0),
                }
            )
        if spans_by_id and events:
            return {"spans": list(spans_by_id.values()), "events": events}
    entries = memory.query_entries(
        zone="input_workspace",
        payload_type="frontend_ready_input",
        reverse=True,
    )
    for entry in entries:
        payload = entry.payload if isinstance(entry.payload, Mapping) else {}
        frontend_state = payload.get("frontend_state")
        if isinstance(frontend_state, Mapping):
            return dict(frontend_state)
    return {"spans": [], "events": []}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _drop_legacy_event_fields(value: dict[str, Any]) -> None:
    for key in (
        "span_patterns",
        "relation_patterns",
        "known_conditions",
        "needed_conditions",
        "intermediate_states",
        "final_goals",
        "blocked_paths",
        "operations",
        "rules",
        "produced",
        "history",
        "workspace",
    ):
        value.pop(key, None)


def _normalize_spans(value: Any) -> list[dict[str, Any]]:
    return normalize_spans(value)


def _normalize_events(value: Any, *, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del spans  # References must survive normalization and be checked by the contract.
    events = normalize_events(value)
    for event in events:
        _drop_legacy_event_fields(event)
    return events


def _event_workspaces_from_events(events: list[dict[str, Any]], spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    span_by_id = {
        str(span.get("span_id") or span.get("id") or ""): dict(span)
        for span in spans
        if str(span.get("span_id") or span.get("id") or "")
    }
    workspaces: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event.get("event_id") or event.get("id") or "").strip()
        if not event_id:
            continue
        evidence_ids = [
            str(span_id)
            for span_id in event.get("evidence_span_ids", []) or []
            if str(span_id) in span_by_id
        ]
        if not evidence_ids and event.get("use_all_spans") is True:
            evidence_ids = list(span_by_id)
        workspace = dict(event.get("workspace") or {})
        workspace["event_id"] = event_id
        workspace["event_type"] = event.get("event_type", workspace.get("event_type", ""))
        workspace["actor_display"] = event.get("actor_display", workspace.get("actor_display", ""))
        workspace["evidence_span_ids"] = evidence_ids
        workspace["spans"] = _dedupe_spans(
            [
                *[dict(span) for span in workspace.get("spans", []) or [] if isinstance(span, Mapping)],
                *[dict(span_by_id[span_id]) for span_id in evidence_ids if span_id in span_by_id],
            ]
        )
        workspace["relations"] = _dedupe_relations(
            [
                *[dict(rel) for rel in workspace.get("relations", []) or [] if isinstance(rel, Mapping)],
                *[dict(rel) for rel in event.get("relations", []) or [] if isinstance(rel, Mapping)],
            ],
            evidence_ids=evidence_ids,
        )
        workspace.setdefault("focus", {})
        workspace["candidate_paths"] = [
            *_as_dict_list(workspace.get("candidate_paths")),
            *_as_dict_list(event.get("candidate_paths")),
        ]
        workspace["feedback"] = [
            *_as_dict_list(workspace.get("feedback")),
            *_as_dict_list(event.get("feedback")),
        ]
        metadata = dict(workspace.get("metadata") or {}) if isinstance(workspace.get("metadata"), Mapping) else {}
        event_metadata = dict(event.get("metadata") or {}) if isinstance(event.get("metadata"), Mapping) else {}
        metadata.update(event_metadata)
        workspace["metadata"] = metadata
        workspace.setdefault("confidence", event.get("confidence", 0.0))
        _drop_legacy_event_fields(workspace)
        workspaces[event_id] = workspace
    return workspaces


def _dedupe_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for span in spans:
        span_id = str(span.get("span_id") or span.get("id") or "").strip()
        key = span_id or json.dumps(span, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def _dedupe_relations(relations: list[dict[str, Any]], *, evidence_ids: list[str]) -> list[dict[str, Any]]:
    evidence = set(evidence_ids)
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for relation in relations:
        source = str(relation.get("source_span_id") or relation.get("source") or "").strip()
        target = str(relation.get("target_span_id") or relation.get("target") or "").strip()
        if evidence and source and target and (source not in evidence or target not in evidence):
            continue
        key = str(relation.get("relation_id") or "") or json.dumps(relation, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result




def _single_event_from_spans(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_id": "e1",
        "event_type": "frontend_cluster",
        "actor_display": "",
        "evidence_span_ids": [str(span.get("span_id") or span.get("id") or "") for span in spans],
        "relations": [],
        "candidate_paths": [],
        "feedback": [],
        "metadata": {},
        "confidence": 0.6,
    }




def _default_db_paths() -> dict[str, str]:
    return default_db_paths()


def _maybe_build_vector_encoder() -> dict[str, Any]:
    flag = os.getenv("MAIN_PART_FRONTEND_VECTORIZE", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return {"ok": False, "status": "disabled", "encoder": None}
    model = os.getenv("MAIN_PART_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    local_files_only = os.getenv("MAIN_PART_EMBEDDING_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
    try:
        encoder = RuleVectorEncoder(
            encoder_name_or_path=model,
            local_files_only=local_files_only,
            load_encoder=True,
        )
        return {"ok": True, "status": "ready", "encoder": encoder, "embedding_model": encoder.embedding_model}
    except Exception as exc:
        return {"ok": False, "status": "encoder_build_failed", "encoder": None, "error": repr(exc), "embedding_model": model}


def _safe_encode_vectors_status(encoder: Any, texts: list[str]) -> dict[str, Any]:
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
                return {
                    "ok": all(bool(vector) for vector in normalized),
                    "status": "ok" if all(bool(vector) for vector in normalized) else "partial",
                    "vectors": normalized,
                    "errors": [
                        {"index": index, "status": "missing_vector"}
                        for index, vector in enumerate(normalized)
                        if not vector
                    ],
                }
        except Exception as exc:
            return {
                "ok": False,
                "status": "batch_encode_failed",
                "vectors": [None] * len(texts),
                "errors": [{"index": index, "status": "batch_encode_failed", "error": repr(exc)} for index, _ in enumerate(texts)],
            }
    vectors: list[list[float] | None] = []
    errors: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        try:
            raw_vector = encoder._encode_text_vector(text)
            vector = [float(item) for item in raw_vector] if isinstance(raw_vector, list) and raw_vector else None
        except Exception as exc:
            vector = None
            errors.append({"index": index, "status": "single_encode_failed", "error": repr(exc)})
        if vector is None and not any(error.get("index") == index for error in errors):
            errors.append({"index": index, "status": "missing_vector"})
        vectors.append(vector)
    return {
        "ok": all(bool(vector) for vector in vectors),
        "status": "ok" if all(bool(vector) for vector in vectors) else "partial" if any(vectors) else "failed",
        "vectors": vectors,
        "errors": errors,
    }


_FRONTEND_FORBIDDEN_SUPERVISION_KEYS = frozenset(
    {
        "answer_key",
        "correct_answer",
        "correct_option",
        "diagnostic_feedback",
        "expected_answer",
        "expected_conclusion",
        "expected_path_supervision",
        "expected_produced",
        "expected_result",
        "expected_rule_path",
        "feedback_signal",
        "gold",
        "gold_answer",
        "gold_option",
        "agent_review",
        "required_knowledge_ids",
        "reward",
    }
)
_FRONTEND_FORBIDDEN_PACKET_MARKERS = frozenset(
    {"expert_feedback", "diagnostic_feedback"}
)


def _reject_supervision_at_frontend_boundary(game_input: Mapping[str, Any]) -> None:
    violations = _supervision_field_paths(game_input)
    markers = {
        str(game_input.get(key) or "").strip().lower()
        for key in ("input_type", "packet_type", "learning_route")
    }
    packet_markers = sorted(markers & _FRONTEND_FORBIDDEN_PACKET_MARKERS)
    if not violations and not packet_markers:
        return
    details = [*violations, *(f"packet:{marker}" for marker in packet_markers)]
    raise ValueError(
        "frontend inference input cannot contain supervision: "
        + ", ".join(details[:12])
    )


def _supervision_field_paths(value: Any) -> list[str]:
    paths: list[str] = []
    visited: set[int] = set()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for key, item in node.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text.strip().lower() in _FRONTEND_FORBIDDEN_SUPERVISION_KEYS:
                    paths.append(child_path)
                    continue
                walk(item, child_path)
        elif isinstance(node, (list, tuple)):
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(value, "")
    return paths




def _compact_execution_result(item: Mapping[str, Any]) -> dict[str, Any]:
    result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
    output = result.get("output") if isinstance(result.get("output"), Mapping) else {}
    executor_results = result.get("executor_results") if isinstance(result.get("executor_results"), list) else []
    matched_rule_ids: list[str] = []
    for executor_result in executor_results:
        if not isinstance(executor_result, Mapping):
            continue
        for step in executor_result.get("steps", []) or []:
            if isinstance(step, Mapping) and str(step.get("rule_id") or ""):
                matched_rule_ids.append(str(step.get("rule_id")))
    return {
        "phase": str(item.get("phase") or ""),
        "output_status": str(output.get("status") or ""),
        "output_option": str(output.get("option") or output.get("predicted_option") or ""),
        "output_reason": str(output.get("reason") or "")[:240],
        "active_rule_ids": [str(rule_id) for rule_id in result.get("active_rule_ids", []) or []][:24],
        "pending_rule_ids": [str(rule_id) for rule_id in result.get("pending_rule_ids", []) or []][:24],
        "pending_rule_selectors": len(result.get("pending_rule_selectors", []) or []),
        "executor_rule_ids": list(dict.fromkeys(matched_rule_ids))[:24],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["MainPartModelLauncher"]
