"""Top-level launcher for the main reasoning model.

This is the normal path outside demos:
raw input -> frontend LLM -> working memory -> runtime launcher -> scheduler.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .data_structure import WorkingMemoryState
from .event_graph_contract import validate_event_graph_contract
from .frontend_schema import normalize_spans, normalize_events, confidence as normalize_frontend_confidence
from .common.normalize import finite_confidence
from .frontend_event_builder import FrontendEventBuilder
from .frontend_semantic import frontend_from_env
from .frontend_working_memory_bridge import submit_frontend_state_to_working_memory
from .llm_clients import maybe_build_instruction_client_from_env, maybe_build_remote_instruction_clients_from_env
from .llm_routing import DEFAULT_CHINESE_INSTRUCT_MODEL, DEFAULT_ENGLISH_INSTRUCT_MODEL, select_instruction_model
from .memory_redis import build_memory_redis_cache_from_env
from .memory_workspace import select_memory_workspace_context
from .rule_vector_encoder import RuleVectorEncoder
from .runtime_launcher import RuntimeLauncher
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
        frontend_client: Any | None = None,
        frontend_span_client: Any | None = None,
        frontend_event_client: Any | None = None,
        frontend_adapter: FrontendEventBuilder | None = None,
        vector_encoder: Any | None = None,
        memory_cache: Any | None = None,
        session_id: str = "main_runtime",
        frontend_model: str = "",
        auto_run: bool = True,
        runtime_max_attempts: int = 3,
        clear_previous_transient_state: bool = True,
        allow_heuristic_frontend_fallback: bool = False,
        db_paths: Mapping[str, Any] | None = None,
    ) -> None:
        self.working_memory = working_memory or WorkingMemoryState()
        self.session_id = str(session_id)
        self.frontend_model = str(frontend_model or "")
        self.auto_run = bool(auto_run)
        self.clear_previous_transient_state = bool(clear_previous_transient_state)
        self.allow_heuristic_frontend_fallback = bool(allow_heuristic_frontend_fallback)
        self.llm_event_chinese_model = os.getenv("MAIN_PART_LLM_CHINESE_MODEL", DEFAULT_CHINESE_INSTRUCT_MODEL)
        self.llm_event_english_model = os.getenv("MAIN_PART_LLM_ENGLISH_MODEL", DEFAULT_ENGLISH_INSTRUCT_MODEL)
        remote_clients = maybe_build_remote_instruction_clients_from_env()
        fallback_client = maybe_build_instruction_client_from_env()
        self.frontend_span_client = frontend_span_client or frontend_client or (
            remote_clients.get("span") if isinstance(remote_clients, dict) else fallback_client
        )
        self.frontend_event_client = frontend_event_client or frontend_client or (
            remote_clients.get("event") if isinstance(remote_clients, dict) else fallback_client
        )
        self.frontend_client = self.frontend_event_client
        self.frontend_adapter = frontend_adapter if frontend_adapter is not None else frontend_from_env()
        if vector_encoder is not None:
            self.vector_encoder = vector_encoder
            self.vector_encoder_status = {"ok": True, "status": "provided"}
        else:
            encoder_result = _maybe_build_vector_encoder()
            self.vector_encoder = encoder_result.get("encoder")
            self.vector_encoder_status = {key: value for key, value in encoder_result.items() if key != "encoder"}
        resolved_db_paths = dict(_default_db_paths() if db_paths is None else db_paths)
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
        game_input = self._normalize_input(input_payload)
        _reject_supervision_at_frontend_boundary(game_input)
        self._prepare_submission()
        original_game_input = deepcopy(game_input)
        input_context = self._build_input_context(original_game_input)
        frontend_input = deepcopy(original_game_input)
        frontend_input["input_context"] = input_context
        if self.frontend_adapter is not None:
            build = getattr(self.frontend_adapter, "build_event_graph", None)
            if not callable(build):
                raise TypeError("frontend_adapter must expose build_event_graph(game_input)")
            frontend_state = dict(build(frontend_input))
            frontend_source = "frontend_adapter"
        else:
            frontend_state = self.build_frontend_state(frontend_input)
            frontend_source = "frontend_llm"
        return self._submit_event_graph(
            original_game_input,
            frontend_state,
            input_context=input_context,
            source=frontend_source,
        )

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
        outputs = self.runtime.run_once() if self.auto_run else []
        return {
            "frontend_state": frontend_state,
            "event_graph_contract": contract,
            "working_memory_submit": bridge_result,
            "input_context": input_context,
            "outputs": outputs,
            "runtime_summary": self.runtime_summary(),
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

    def build_frontend_state(self, game_input: Mapping[str, Any]) -> dict[str, Any]:
        provided = game_input.get("frontend_state") or game_input.get("frontend_output")
        if isinstance(provided, Mapping):
            return self._normalize_frontend_state(dict(provided), game_input)
        if self.frontend_span_client is None or self.frontend_event_client is None:
            raise RuntimeError("Frontend LLM is not configured. Set MAIN_PART_LLM_BACKEND or pass frontend_client.")
        text = self._input_text(game_input)
        model = self.frontend_model or select_instruction_model(self, text, task_prefix="llm_event")
        span_raw = self._complete_frontend(self._span_prompt(game_input), model=model, role="span")
        span_payload = _parse_json_object(span_raw)
        spans = _normalize_spans(span_payload.get("spans", []))
        if not spans:
            if not self.allow_heuristic_frontend_fallback:
                raise RuntimeError("frontend_event_graph_generation_failed:no_valid_spans")
            fallback_state = self._heuristic_frontend_state(
                game_input,
                model=model,
                raw_span_response=str(span_raw),
                fallback_reason="no_valid_spans",
            )
            return fallback_state
        spans = _augment_spans_with_story_sentences(spans, game_input)
        if _frontend_spans_need_text_fallback(spans):
            if not self.allow_heuristic_frontend_fallback:
                raise RuntimeError("frontend_event_graph_generation_failed:no_valid_evidence_spans")
            fallback_state = self._heuristic_frontend_state(
                game_input,
                model=model,
                raw_span_response=str(span_raw),
                fallback_reason="no_valid_story_spans",
            )
            return fallback_state
        event_raw = self._complete_frontend(self._event_prompt(game_input, spans), model=model, role="event")
        event_payload = _parse_json_object(event_raw)
        events = _normalize_events(event_payload.get("events", []), spans=spans)
        if not events:
            if not self.allow_heuristic_frontend_fallback:
                raise RuntimeError("frontend_event_graph_generation_failed:no_valid_events")
            events = [_single_event_from_spans(spans)]
        vectorization = self._attach_frontend_vectors(spans, events)
        return {
            "spans": spans,
            "events": events,
            "event_workspaces": _event_workspaces_from_events(events, spans),
            "frontend_event_source": "llm",
            "frontend_mode": "model_launcher_two_pass",
            "model": model,
            "vectorization": vectorization,
            "raw_span_response_preview": str(span_raw)[:1200],
            "raw_event_response_preview": str(event_raw)[:1200],
        }

    def _heuristic_frontend_state(
        self,
        game_input: Mapping[str, Any],
        *,
        model: str,
        raw_span_response: str = "",
        fallback_reason: str = "heuristic_fallback",
    ) -> dict[str, Any]:
        spans = _heuristic_spans_from_input(game_input)
        if not spans:
            raise RuntimeError("Frontend LLM returned no valid spans.")
        events = [_single_event_from_spans(spans)]
        events[0]["event_type"] = "heuristic_text_cluster"
        events[0]["metadata"] = {
            "frontend_note": "heuristic sentence spans from raw input",
            "final_goal": "choose final answer from options",
        }
        vectorization = self._attach_frontend_vectors(spans, events)
        return {
            "spans": spans,
            "events": events,
            "event_workspaces": _event_workspaces_from_events(events, spans),
            "frontend_event_source": "heuristic_fallback",
            "frontend_mode": "model_launcher_heuristic_fallback",
            "frontend_fallback_reason": fallback_reason,
            "vectorization": vectorization,
            "model": model,
            "raw_span_response_preview": str(raw_span_response)[:1200],
            "raw_event_response_preview": "",
        }

    def _complete_frontend(self, prompt: str, *, model: str, role: str = "event") -> str:
        client = self.frontend_span_client if role == "span" else self.frontend_event_client
        if hasattr(client, "complete"):
            return str(client.complete(prompt=prompt, text=prompt, model=model))
        if callable(client):
            return str(client(prompt))
        raise RuntimeError("Frontend client must expose complete(...) or be callable.")

    def build_frontend_patch_state(
        self,
        *,
        game_input: Mapping[str, Any],
        current_frontend_state: Mapping[str, Any],
        patches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.frontend_span_client is None or self.frontend_event_client is None:
            raise RuntimeError("Frontend LLM is not configured for frontend patch.")
        text = self._input_text(game_input)
        model = self.frontend_model or select_instruction_model(self, text, task_prefix="llm_event")
        base_spans = _normalize_spans(current_frontend_state.get("spans", []))
        span_raw = self._complete_frontend(
            self._span_patch_prompt(game_input, current_frontend_state, patches),
            model=model,
            role="span",
        )
        span_payload = _parse_json_object(span_raw)
        spans = _merge_patch_spans(base_spans, span_payload)
        if not spans:
            raise RuntimeError("Frontend patch LLM returned no valid spans.")
        base_events = _normalize_events(current_frontend_state.get("events", []), spans=spans)
        event_raw = self._complete_frontend(
            self._event_patch_prompt(game_input, spans, current_frontend_state, patches),
            model=model,
            role="event",
        )
        event_payload = _parse_json_object(event_raw)
        events = _merge_patch_events(base_events, event_payload, spans=spans)
        if not events:
            events = [_single_event_from_spans(spans)]
        vectorization = self._attach_frontend_vectors(spans, events)
        return {
            "spans": spans,
            "events": events,
            "event_workspaces": _event_workspaces_from_events(events, spans),
            "frontend_event_source": "llm_patch",
            "frontend_mode": "model_launcher_patch_two_pass",
            "model": model,
            "patches": patches,
            "vectorization": vectorization,
            "raw_span_response_preview": str(span_raw)[:1200],
            "raw_event_response_preview": str(event_raw)[:1200],
        }

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
            if self.frontend_adapter is not None:
                patch = getattr(self.frontend_adapter, "patch_event_graph", None)
                if not callable(patch):
                    raise TypeError("frontend_adapter must expose patch_event_graph(...) for runtime graph revision")
                patched_state = self._normalize_frontend_state(
                    dict(
                        patch(
                            game_input=game_input,
                            current_event_graph=current_frontend_state,
                            patches=patches,
                        )
                    ),
                    game_input,
                )
                patch_source = "frontend_adapter_patch"
            else:
                patched_state = self.build_frontend_patch_state(
                    game_input=game_input,
                    current_frontend_state=current_frontend_state,
                    patches=patches,
                )
                patch_source = "frontend_patch_llm"
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

    def _span_prompt(self, game_input: Mapping[str, Any]) -> str:
        domain = _frontend_prompt_domain(game_input)
        domain_addendum = _span_domain_addendum(domain)
        return f"""你是前端 span 抽取器。只输出 JSON，不要回答问题。

任务：先自己区分正文、背景信息、问题目标、候选项、约束和已知条件，再把输入拆成可推理 span。
span 是三元组，文本字段只用于 debug，后续匹配依赖 core/relation/nature 的语义和向量。

核心要求：
- 题目类型 span 必须有，但不能只抽题目类型、问题目标、候选项。
- 必须抽取正文故事里的具体事实；对非故事文本，必须抽取正文里的具体事实、数据、定义、条件、变化、比较、异常、约束和问题目标。
- 每个关键推理线索至少一个 span；如果一句话里有多个事实或多个数值关系，拆成多个 span。
- core 必须尽量写具体主体、指标、对象、概念或事件；不要用泛化的 person/object/event/metric 代替具体内容。
- relation 用清晰动词或关系，例如 reports/increases/decreases/equals/causes/implies/compares_with/depends_on/constrains/asks_for/contradicts/supports/excludes，不要都写 is。
- text 必须是原文短证据，不要写推理结论；core/relation/nature 要能单独表达事实。
- 如果正文很长，优先保留与解题相关的 20-80 个高价值事实 span，不要只保留开头或结尾。
- 不确定是否关键的带时间、主体、指标、数值、单位、定义、假设、比较、异常状态的句子，也要抽成 span；宁可多抽正文事实，不要漏掉线索。

必须抽取题目类型 span：
- core 写 question_type
- relation 写 classify
- nature 写题型，例如 calculation / explanation / comparison / diagnosis / forecast / multiple_choice / open_question / yes_no
- text 写判断题型的原文短证据
候选项规则：
- 不要把候选项列表本身抽成正文事实 span；候选项是最终答案约束，不是正文事实。
- 如果候选项里的实体、指标或概念也出现在正文里，必须抽正文中的事实，但不要只抽“(A) xxx”这种选项行。
如果输入里有“选择正确答案/从选项中选择”等目标，写成一个 question_goal span。

当前领域补充：
{domain_addendum}

质量检查：
- 输出前检查 spans 中是否包含正文关键事实；如果只有 question_type/question_goal/option，必须重做。
- 输出前检查是否覆盖主要主体、关键指标、约束条件、变化关系和问题目标。

输出格式：
{{
  "spans": [
    {{"span_id":"s1","core":"实体/对象","relation":"关系/动作/状态","nature":"具体内容","text":"原文短证据","confidence":0.0}}
  ]
}}

完整输入：
{json.dumps(_frontend_prompt_game_input(game_input), ensure_ascii=False)}
"""

    def _event_prompt(self, game_input: Mapping[str, Any], spans: list[dict[str, Any]]) -> str:
        domain = _frontend_prompt_domain(game_input)
        domain_addendum = _event_domain_addendum(domain)
        return f"""你是前端 event 构建器。只输出 JSON，不要回答问题。

event 是相关 span 的聚类。你要先自己区分正文、背景信息、问题目标、候选项、约束和已知条件。
每个 event 要把属于同一条推理链、计算链、比较链、风险链、定义链或动态变化链的 span 放一起，并保留 span 之间的 relation。
不要为选项列表建立候选项 event；event 只承载正文事实、问题目标、约束和这些事实之间的关系。
后续推理层会在证据链完成后，再把结果与选项做对齐。

核心要求：
- 至少建立覆盖正文证据的 event，不能只建立 question event。
- 按链条组织 event，例如 calculation_chain/comparison_chain/risk_diagnosis/forecast_update/definition/constraint/change_update。
- 对同一主体、指标、时期、假设、变化、风险信号要尽量放入可连接的 event，必要时用 relations 标出 depends_on/equals/increases/decreases/supports/contradicts/constrains/temporal_before/temporal_after。
- event 只能包含 spans、relations、candidate_paths、feedback 和 metadata；不要输出 span_patterns、relation_patterns、produced、history 或 workspace 的旧条件字段。
- 已知事实、缺失环节、中间推导、最终目标如果需要记录，写入 metadata，不要建旧字段。
- 如果 spans 已经包含正文事实但 event 没引用它们，必须重做；高价值正文 spans 不应孤立。

当前领域补充：
{domain_addendum}

质量检查：
- events 的 evidence_span_ids 必须覆盖大部分关键正文 spans。
- 如果只有题型/目标 event，没有证据 event，则输出不合格，必须重做。

输出格式：
{{
  "events": [
    {{
      "event_id":"e1",
      "event_type":"calculation_chain/comparison_chain/risk_diagnosis/forecast_update/definition/constraint/change_update",
      "actor_display":"",
      "evidence_span_ids":["s1"],
      "relations":[{{"relation_id":"r1","source_span_id":"s1","target_span_id":"s2","relation_type":"depends_on/supports/contradicts/constrains","relation":"depends_on/supports/contradicts/constrains","confidence":0.0}}],
      "candidate_paths":[],
      "feedback":[],
      "metadata":{{"known_facts":[],"missing_links":[],"intermediate_states":[],"final_goals":[]}},
      "confidence":0.0
    }}
  ]
}}

完整输入：
{json.dumps(_frontend_prompt_game_input(game_input), ensure_ascii=False)}

spans：
{json.dumps(_prompt_safe_spans(spans), ensure_ascii=False)}
"""

    def _span_patch_prompt(
        self,
        game_input: Mapping[str, Any],
        current_frontend_state: Mapping[str, Any],
        patches: list[dict[str, Any]],
    ) -> str:
        prompt_state = _prompt_safe_frontend_state(current_frontend_state)
        prompt_patches = _prompt_safe_patch_payload(patches)
        return f"""你是前端 span 修复器。只输出 JSON，不要回答问题。

你收到的是推理层 control rule 的结构化 patch 请求。你的任务是根据 patch 重点重新补全/修正 spans。
优先输出 span_updates / span_additions，不要重写完整 frontend_state。只在必须全量重建时输出 spans。
span 仍然是三元组：core / relation / nature。文本字段只用于 debug，后续匹配依赖这三个字段的向量。

关键规则：
- 如果 patch 要求识别题目类型，必须加入 span：
  core="question_type"
  relation="classify"
  nature 为 multiple_choice / open_question / yes_no / explanation 之一
  text 写判断题型的原文短证据
- 如果输入有问题目标，加入 question_goal span：
  core="question_goal"
  relation="question_goal"
  nature 写要解决的目标
- 不要把候选项列表本身抽成正文事实 span；候选项只作为最终对齐约束。
- 如果候选人/候选物在正文事实中出现，只抽正文事实，不抽选项列表。
- patch 只围绕 focus_span_ids 和 missing_span_query 相关信息补充。
- 如果修正旧 span，必须保留原 span_id。
- 如果新增 span，使用 s_patch_1 / s_patch_2 这种新 span_id。
- 禁止输出 vector / *_vector / *_vec。

输出格式：
{{
  "span_updates": [
    {{"operation":"update","span_id":"s1","core":"实体/对象","relation":"关系/动作/状态","nature":"具体内容","text":"原文短证据","confidence":0.0}}
  ],
  "span_additions": [
    {{"span_id":"s_patch_1","core":"实体/对象","relation":"关系/动作/状态","nature":"具体内容","text":"原文短证据","confidence":0.0}}
  ]
}}

patch 请求：
{json.dumps(prompt_patches, ensure_ascii=False)}

当前 span 索引（只含 span_id 和短文本，不含向量）：
{json.dumps(prompt_state["span_index"], ensure_ascii=False)}

当前 event 索引（只含 event_id 与 span_id 关系，不含向量）：
{json.dumps(prompt_state["event_index"], ensure_ascii=False)}

	完整输入：
	{json.dumps(_frontend_prompt_game_input(game_input), ensure_ascii=False)}
"""

    def _event_patch_prompt(
        self,
        game_input: Mapping[str, Any],
        spans: list[dict[str, Any]],
        current_frontend_state: Mapping[str, Any],
        patches: list[dict[str, Any]],
    ) -> str:
        prompt_state = _prompt_safe_frontend_state(current_frontend_state)
        prompt_patches = _prompt_safe_patch_payload(patches)
        return f"""你是前端 event 修复器。只输出 JSON，不要回答问题。

你收到的是修复后的完整 spans。你的任务是基于 patch 请求重建 event hypergraph。
优先输出 event_updates / event_additions，不要重写完整 frontend_state。只在必须全量重建时输出 events。

event 要求：
- event 是相关 span 的聚类，把属于同一件事的 span 放到同一个 event。
- event 内必须保留 span 之间的 relations。
- 如果存在 question_type/question_goal，要放入能支持题目调度的 event 中。
- 不要为候选项列表建立候选项 event。
	- event 只输出 spans 引用、relations、candidate_paths、feedback、metadata；不要输出 workspace、span_patterns、relation_patterns、produced、history。
	- event 只能通过 span_id 引用 span。
- 禁止输出 vector / *_vector / *_vec。

输出格式：
{{
  "event_updates": [
    {{
      "event_id":"e1",
      "event_type":"state/change/action/evidence/conflict/question",
      "actor_display":"",
      "evidence_span_ids":["s1"],
	      "relations":[{{"relation_id":"r1","source_span_id":"s1","target_span_id":"s2","relation_type":"supports","relation":"supports","confidence":0.0}}],
	      "candidate_paths":[],
	      "feedback":[],
	      "metadata":{{"known_facts":[],"missing_links":[],"final_goals":[]}},
	      "confidence":0.0
    }}
  ],
  "event_additions": [
    {{
      "event_id":"e_patch_1",
      "event_type":"state/change/action/evidence/conflict/question",
	      "actor_display":"",
	      "evidence_span_ids":["s1"],
	      "relations":[],
	      "candidate_paths":[],
	      "feedback":[],
	      "metadata":{{"known_facts":[],"missing_links":[],"final_goals":[]}},
	      "confidence":0.0
    }}
  ]
}}

patch 请求：
{json.dumps(prompt_patches, ensure_ascii=False)}

当前 event 索引（只含 event_id 与 span_id 关系，不含向量）：
{json.dumps(prompt_state["event_index"], ensure_ascii=False)}

修复后 spans：
{json.dumps(_prompt_safe_spans(spans), ensure_ascii=False)}

	完整输入：
	{json.dumps(_frontend_prompt_game_input(game_input), ensure_ascii=False)}
"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_thinking(str(raw or "")).strip()
    for candidate in reversed(_balanced_json_candidates(text)):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


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


def _frontend_spans_need_text_fallback(spans: list[dict[str, Any]]) -> bool:
    if not spans:
        return True
    story_like = 0
    option_like = 0
    question_like = 0
    for span in spans:
        text = " ".join(
            str(span.get(key) or "")
            for key in ("core", "relation", "nature", "text")
        ).strip().lower()
        if not text:
            continue
        if "question_type" in text or "question_goal" in text or "multiple_choice" in text:
            question_like += 1
            continue
        if re.search(r"\([a-d]\)", text) or " option" in f" {text}":
            option_like += 1
            continue
        story_like += 1
    return story_like == 0 and (question_like > 0 or option_like > 0)


def _augment_spans_with_story_sentences(spans: list[dict[str, Any]], game_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    story = _story_text_from_input(_input_text_from_payload(game_input))
    if not story:
        return spans
    existing_text = "\n".join(
        str(span.get("text") or span.get("nature") or span.get("core") or "")
        for span in spans
    ).lower()
    augmented = [dict(span) for span in spans]
    for sentence in _split_story_sentences(story):
        if not _is_salient_story_sentence(sentence):
            continue
        for unit in _atomic_story_units(sentence):
            if not _is_salient_story_sentence(unit):
                continue
            normalized = unit.lower()
            if _sentence_already_covered(normalized, existing_text):
                continue
            span_id = f"s_aug_{len(augmented) + 1}"
            augmented.append(
                {
                    "span_id": span_id,
                    "id": span_id,
                    "core": _sentence_core_hint(unit),
                    "relation": _relation_hint(unit),
                    "nature": unit,
                    "text": unit,
                    "confidence": 0.58,
                    "source": "heuristic_story_augmentation",
                }
            )
            existing_text += "\n" + normalized
    return _normalize_spans(augmented)


def _atomic_story_units(sentence: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(sentence or "")).strip()
    if not text:
        return []
    units: list[str] = [text]
    quote_parts = [
        part.strip(" ,;")
        for part in re.split(r"(?<=[.!?])\s+(?=[\"“])|(?<=[\"”])\s+(?=[A-Z\"“])", text)
        if len(part.strip(" ,;")) >= 12
    ]
    if len(quote_parts) > 1:
        units.extend(quote_parts)
    clause_parts = [
        part.strip(" ,;")
        for part in re.split(r"\s*(?:;|--|—|\.\.\.|,\s+(?=(?:and|but|because|when|after|before|then|so|while|if)\b))\s*", text, flags=re.I)
        if len(part.strip(" ,;")) >= 12
    ]
    if 1 < len(clause_parts) <= 12:
        units.extend(clause_parts)
    direct_speech = [
        match.group(1).strip()
        for match in re.finditer(r"[“\"]([^”\"]{8,240})[”\"]", text)
        if match.group(1).strip()
    ]
    units.extend(direct_speech)
    return list(dict.fromkeys(units))


def _relation_hint(text: str) -> str:
    lower = str(text or "").lower()
    if any(word in lower for word in ("said", "asked", "replied", "claimed", "told")):
        return "says"
    if any(word in lower for word in ("saw", "noticed", "observed", "heard")):
        return "observes"
    if any(word in lower for word in ("arrived", "entered", "left", "went", "came back")):
        return "moves"
    if any(word in lower for word in ("has", "had", "ticket", "coffee", "shirt", "heart")):
        return "has"
    if any(word in lower for word in ("happened", "at 9", "a.m.", "p.m.")):
        return "happens_at"
    return "states"


def _sentence_already_covered(sentence: str, existing_text: str) -> bool:
    compact = re.sub(r"\s+", " ", sentence).strip().lower()
    existing = re.sub(r"\s+", " ", existing_text).strip().lower()
    if compact and compact in existing:
        return True
    anchor = compact[:120].strip()
    return bool(anchor and len(anchor) >= 40 and anchor in existing)


def _is_salient_story_sentence(sentence: str) -> bool:
    text = str(sentence or "")
    if len(text) < 12:
        return False
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b|\b\d+\s*(?:minutes?|hours?|a\.m\.|p\.m\.)\b", text, flags=re.I):
        return True
    if '"' in text or "“" in text or "”" in text:
        return True
    return bool(re.search(r"[.!?。！？;；:]", text))


def _sentence_core_hint(sentence: str) -> str:
    text = str(sentence or "").strip()
    names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text)
    useful_names = [name for name in names if name.lower() not in {"Case", "Story", "Question", "Options"}]
    if useful_names:
        return " / ".join(dict.fromkeys(useful_names[:3]))
    first_clause = re.split(r"[,，;；:.。!?！？]", text, maxsplit=1)[0].strip()
    return (first_clause or text)[:120]


def _heuristic_spans_from_input(game_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = _input_text_from_payload(game_input)
    story = _story_text_from_input(text)
    spans: list[dict[str, Any]] = []
    if re.search(r"\b(choose|option|answer|final answer)\b|选择|选项|答案", text, flags=re.I):
        spans.append(
            {
                "span_id": "s1",
                "id": "s1",
                "core": "question_type",
                "relation": "classify",
                "nature": "multiple_choice",
                "text": "Choose the correct answer from the options.",
                "confidence": 0.85,
            }
        )
        spans.append(
            {
                "span_id": "s2",
                "id": "s2",
                "core": "question_goal",
                "relation": "choose_answer",
                "nature": "select the correct option from evidence in the story",
                "text": "Return the final option as Final Answer: <A/B/C/D>.",
                "confidence": 0.85,
            }
        )
    for sentence in _split_story_sentences(story):
        span_id = f"s{len(spans) + 1}"
        spans.append(
            {
                "span_id": span_id,
                "id": span_id,
                "core": sentence[:120],
                "relation": "states",
                "nature": sentence,
                "text": sentence,
                "confidence": 0.62,
            }
        )
    return _normalize_spans(spans)


def _input_text_from_payload(game_input: Mapping[str, Any]) -> str:
    parts = [
        str(game_input.get("title") or ""),
        str(game_input.get("input") or ""),
        str(game_input.get("story") or ""),
        json.dumps(game_input.get("options") or [], ensure_ascii=False),
    ]
    return "\n".join(part for part in parts if part.strip())


def _story_text_from_input(text: str) -> str:
    value = str(text or "")
    story_match = re.search(r"(?:^|\n)Story:\s*(.*?)(?:\n\s*Question:|\n\s*Options:|\Z)", value, flags=re.S | re.I)
    if story_match:
        return story_match.group(1).strip()
    before_options = re.split(r"\n\s*(?:Question|Options):", value, maxsplit=1, flags=re.I)[0]
    return re.sub(r"^Case:\s*.*?(?:\n\s*)", "", before_options, count=1, flags=re.I | re.S).strip()


def _split_story_sentences(story: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(story or "")).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s+(?=[\"“A-Z0-9])", text)
    sentences: list[str] = []
    for piece in pieces:
        sentence = piece.strip()
        if len(sentence) < 8:
            continue
        if len(sentence) > 420:
            chunks = re.split(r"(?<=[,;，；])\s+", sentence)
            buffer = ""
            for chunk in chunks:
                candidate = f"{buffer} {chunk}".strip()
                if len(candidate) <= 360:
                    buffer = candidate
                    continue
                if buffer:
                    sentences.append(buffer)
                buffer = chunk.strip()
            if buffer:
                sentences.append(buffer)
            continue
        sentences.append(sentence)
    return sentences[:80]


def _merge_patch_spans(base_spans: list[dict[str, Any]], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    full = _normalize_spans(payload.get("spans", []))
    if full:
        return full
    merged_by_id = {str(span.get("span_id") or span.get("id") or ""): dict(span) for span in base_spans}
    for raw in _as_dict_list(payload.get("span_updates")):
        span_id = str(raw.get("span_id") or raw.get("id") or "").strip()
        if not span_id:
            continue
        existing = dict(merged_by_id.get(span_id) or {"span_id": span_id, "id": span_id})
        semantic_changed = False
        for key in ("core", "relation", "nature", "text", "confidence"):
            if key in raw and raw.get(key) not in (None, ""):
                existing[key] = raw[key]
                semantic_changed = semantic_changed or key in {"core", "relation", "nature", "text"}
        if semantic_changed:
            _drop_prompt_vector_fields(existing)
        existing["span_id"] = span_id
        existing.setdefault("id", span_id)
        merged_by_id[span_id] = existing
    for raw in _as_dict_list(payload.get("span_additions")):
        span = dict(raw)
        span_id = str(span.get("span_id") or span.get("id") or f"s_patch_{len(merged_by_id) + 1}").strip()
        if not span_id:
            continue
        span["span_id"] = span_id
        span.setdefault("id", span_id)
        merged_by_id[span_id] = span
    return _normalize_spans(list(merged_by_id.values()))


def _merge_patch_events(base_events: list[dict[str, Any]], payload: Mapping[str, Any], *, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = _normalize_events(payload.get("events", []), spans=spans)
    if full:
        return full
    merged_by_id = {str(event.get("event_id") or event.get("id") or ""): dict(event) for event in base_events}
    for raw in _as_dict_list(payload.get("event_updates")):
        event_id = str(raw.get("event_id") or raw.get("id") or "").strip()
        if not event_id:
            continue
        existing = dict(merged_by_id.get(event_id) or {"event_id": event_id})
        for key in ("event_type", "actor_display", "evidence_span_ids", "relations", "candidate_paths", "feedback", "metadata", "confidence"):
            if key in raw and raw.get(key) not in (None, ""):
                existing[key] = raw[key]
        existing["event_id"] = event_id
        merged_by_id[event_id] = existing
    for raw in _as_dict_list(payload.get("event_additions")):
        event = dict(raw)
        event_id = str(event.get("event_id") or event.get("id") or f"e_patch_{len(merged_by_id) + 1}").strip()
        if not event_id:
            continue
        event["event_id"] = event_id
        merged_by_id[event_id] = event
    return _normalize_events(list(merged_by_id.values()), spans=spans)


def _drop_prompt_vector_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value.keys()):
            if _is_prompt_vector_key(str(key)):
                value.pop(key, None)
                continue
            _drop_prompt_vector_fields(value[key])
    elif isinstance(value, list):
        for item in value:
            _drop_prompt_vector_fields(item)


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


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S | re.I)
    return re.sub(r"</?think>", "", text, flags=re.I)


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escape = False
    for idx, ch in enumerate(str(text or "")):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if not stack:
                start = idx
            stack.append("}")
            continue
        if ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : idx + 1].strip())
                start = None
    return candidates


def _default_db_paths() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    # The current product entry is the prepared DSO demo. Keep the normal CLI
    # and the validated finance runtime on the same databases so a successful
    # integration run cannot differ from real CLI behavior.
    configured_dir = os.getenv("MAIN_PART_DB_DIR", "").strip()
    data_dir = (
        Path(configured_dir).expanduser()
        if configured_dir
        else project_root / "data" / "finance_dso_medium_v1" / "db"
    )
    return {
        "knowledge": str(data_dir / "knowledge.sqlite"),
        "rule": str(data_dir / "rule.sqlite"),
        "memory": str(data_dir / "memory.sqlite"),
    }


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


def _prompt_safe_frontend_value(value: Any, *, max_depth: int, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return ""
    if isinstance(value, Mapping):
        return {
            str(key): _prompt_safe_frontend_value(item, max_depth=max_depth, _depth=_depth + 1)
            for key, item in value.items()
            if str(key).strip().lower() not in _FRONTEND_FORBIDDEN_SUPERVISION_KEYS
            and not _is_prompt_vector_key(str(key))
            and not str(key).startswith("raw_")
        }
    if isinstance(value, list):
        if _looks_like_vector(value):
            return []
        return [
            _prompt_safe_frontend_value(item, max_depth=max_depth, _depth=_depth + 1)
            for item in value[:80]
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _clip_text(value, 800) if isinstance(value, str) else value
    return _clip_text(value, 160)


def _frontend_prompt_game_input(game_input: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist only user-visible task text for the remote frontend prompt."""
    allowed_keys = [
        "input",
        "title",
        "story",
        "text",
        "narrative",
        "question",
        "options",
        "choices",
        "answer_options",
        "domain",
        "input_context",
    ]
    safe: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in game_input:
            continue
        value = game_input.get(key)
        if key == "input_context" and isinstance(value, Mapping):
            safe[key] = _frontend_memory_context(value)
        elif isinstance(value, str):
            safe[key] = _clip_text(value, 32000)
        else:
            safe[key] = _prompt_safe_frontend_value(value, max_depth=4)
    return safe


def _frontend_memory_context(input_context: Mapping[str, Any]) -> dict[str, Any]:
    """Expose recalled memory to extraction while keeping knowledge backend-only."""

    allowed_keys = {
        "status",
        "query",
        "embedding_model",
        "vector_version",
        "memory_workspace",
        "memory_workspace_status",
        "vectorization",
    }
    return {
        key: _prompt_safe_frontend_value(value, max_depth=4)
        for key, value in input_context.items()
        if key in allowed_keys
    }


def _frontend_prompt_domain(game_input: Mapping[str, Any]) -> str:
    explicit = str(game_input.get("domain") or game_input.get("task_domain") or "").strip().lower()
    return explicit or "general"


def _span_domain_addendum(domain: str) -> str:
    normalized = str(domain or "general").strip().lower()
    if normalized in {"finance", "financial", "company_analysis", "company-analysis", "capital_markets", "financial_analysis"}:
        return """当前领域是 finance / company analysis / financial reasoning。
- 只抽取用户输入或 input_context.memory_workspace 明确提供的公司、期间、指标、数值、单位、条件、变化、比较、公式和约束。
- 用户输入与召回记忆是不同来源；不得把召回记忆伪装成用户事实。
- 不要根据模型自身金融知识补充输入中不存在的公式、定义、风险信号或结论。
- 保留指标的主体、报告期、币种、单位和会计口径，避免把不同口径的数据合并。
- 如果输入先给条件再给变化，分别抽取条件、变化、受影响对象和问题目标，并用来源字段标明证据来自哪里。"""
    return """当前领域是 general。
- 保持领域中立，重点抽取主体、指标、对象、时间、条件、变化、比较、定义、约束、证据、问题目标。
- 不要默认使用侦探题、金融题或其它专门领域术语；只有输入文本明确出现时才抽取。"""


def _event_domain_addendum(domain: str) -> str:
    normalized = str(domain or "general").strip().lower()
    if normalized in {"finance", "financial", "company_analysis", "company-analysis", "capital_markets", "financial_analysis"}:
        return """当前领域是 finance / company analysis / financial reasoning。
- 只根据当前 spans 组织事件，不得补充 spans 中没有的金融知识或答案。
- 计算、比较、风险、定义和变化可以拆成不同 event，但每个 event 必须引用明确证据 span。
- 用户事实与 knowledge context 必须保留来源差异；推导结论不能伪装成原始证据。
- 使用能准确表达当前 spans 关系的 relation，不要为了套用固定金融模板而制造关系。"""
    return """当前领域是 general。
- event_type 根据输入内容选择 calculation_chain/comparison_chain/risk_diagnosis/forecast_update/definition/constraint/change_update。
- 不要默认建立某个专门领域的 event；只根据当前 spans 的事实关系组织。"""


def _prompt_safe_patch_payload(patches: list[dict[str, Any]]) -> dict[str, Any]:
    safe_patches = _prompt_safe_frontend_value(patches, max_depth=5)
    focus_span_ids: list[str] = []
    missing_queries: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                normalized_key = str(key or "")
                if normalized_key in {"span_id", "source_span_id", "target_span_id"}:
                    text = str(value or "").strip()
                    if text and text not in focus_span_ids:
                        focus_span_ids.append(text)
                elif normalized_key in {"span_ids", "focus_span_ids", "evidence_span_ids"} and isinstance(value, list):
                    for item in value:
                        text = str(item or "").strip()
                        if text and text not in focus_span_ids:
                            focus_span_ids.append(text)
                elif normalized_key in {"missing_span_query", "reason", "patch_query"}:
                    text = _clip_text(value, 360)
                    if text:
                        missing_queries.append(text)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(safe_patches)
    return {
        "focus_span_ids": focus_span_ids[:24],
        "missing_span_queries": list(dict.fromkeys(missing_queries))[:12],
        "patches": safe_patches,
    }


def _prompt_safe_frontend_state(state: Mapping[str, Any]) -> dict[str, Any]:
    spans = _prompt_safe_spans(_as_dict_list(state.get("spans")))
    span_ids = {str(span.get("span_id") or "") for span in spans}
    event_index: list[dict[str, Any]] = []
    for event in _as_dict_list(state.get("events"))[:48]:
        relations: list[dict[str, Any]] = []
        for relation in _as_dict_list(event.get("relations"))[:32]:
            relations.append(
                {
                    "relation_id": str(relation.get("relation_id") or relation.get("id") or "")[:80],
                    "source_span_id": str(relation.get("source_span_id") or "")[:80],
                    "target_span_id": str(relation.get("target_span_id") or "")[:80],
                    "relation_type": _clip_text(relation.get("relation_type") or relation.get("relation"), 80),
                    "confidence": _safe_confidence(relation.get("confidence")),
                }
            )
        event_index.append(
            {
                "event_id": str(event.get("event_id") or event.get("id") or "")[:80],
                "event_type": _clip_text(event.get("event_type"), 80),
                "evidence_span_ids": [
                    str(span_id)
                    for span_id in event.get("evidence_span_ids", []) or []
                    if str(span_id) in span_ids
                ][:64],
                "relations": relations,
                "workspace": _prompt_safe_workspace(event.get("workspace")),
                "confidence": _safe_confidence(event.get("confidence")),
            }
        )
    return {"span_index": spans, "event_index": event_index}


def _prompt_safe_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_spans: list[dict[str, Any]] = []
    for span in spans[:128]:
        safe_spans.append(
            {
                "span_id": str(span.get("span_id") or span.get("id") or "")[:80],
                "core": _clip_text(span.get("core"), 120),
                "relation": _clip_text(span.get("relation"), 120),
                "nature": _clip_text(span.get("nature"), 220),
                "text": _clip_text(span.get("text"), 260),
                "confidence": _safe_confidence(span.get("confidence")),
            }
        )
    return safe_spans


def _prompt_safe_workspace(value: Any) -> dict[str, Any]:
    safe = _prompt_safe_value(value, max_depth=3)
    return safe if isinstance(safe, dict) else {}


def _prompt_safe_value(value: Any, *, max_depth: int, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return ""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_prompt_vector_key(key_text) or key_text.startswith("raw_"):
                continue
            result[key_text] = _prompt_safe_value(item, max_depth=max_depth, _depth=_depth + 1)
        return result
    if isinstance(value, list):
        if _looks_like_vector(value):
            return []
        return [_prompt_safe_value(item, max_depth=max_depth, _depth=_depth + 1) for item in value[:80]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _clip_text(value, 800) if isinstance(value, str) else value
    return _clip_text(value, 160)


def _is_prompt_vector_key(key: str) -> bool:
    normalized = str(key or "").lower()
    return normalized == "vector" or normalized.endswith("_vector") or normalized.endswith("_vectors") or normalized.endswith("_vec")


def _looks_like_vector(value: list[Any]) -> bool:
    if len(value) < 16:
        return False
    sample = value[:16]
    return all(isinstance(item, int | float) for item in sample)


def _clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _safe_confidence(value: Any) -> float:
    return round(finite_confidence(value, 0.0), 4)


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
