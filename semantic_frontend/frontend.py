from __future__ import annotations

"""Frontend workflow: translate, validate and apply local refinements."""
import json
import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Optional

# Re-export existing imports for callers of the original single-file module.
from .graph import Event, Edge, Mainline, EventGraph, event_from_dict, edge_from_dict, mainline_from_dict
from .validation import GraphContractValidator
from .prompts import build_initial_prompt, build_refinement_prompt
from .llm_client import SingleSmallLLM
from .embedding import SentenceTransformerEmbedder

# ============================================================
# 7. Simple frontend
# ============================================================

class SimpleSemanticFrontend:
    def __init__(
        self,
        llm: SingleSmallLLM,
        embedder: SentenceTransformerEmbedder,
    ):
        self.llm = llm
        self.embedder = embedder
        self.validator = GraphContractValidator()

    def _build_graph(
        self,
        doc_id: str,
        raw_text: str,
        data: dict[str, Any],
    ) -> EventGraph:
        if not isinstance(data, dict) or set(data) - {"mainlines", "events", "edges"}:
            raise ValueError("invalid graph object")
        for name in ("mainlines", "events", "edges"):
            if not isinstance(data.get(name, []), list):
                raise ValueError(f"invalid graph collection: {name}")
        graph = EventGraph(
            doc_id=doc_id,
            raw_text=raw_text,
            mainlines=[
                mainline_from_dict(x)
                for x in data.get("mainlines", [])
            ],
            events=[
                event_from_dict(x)
                for x in data.get("events", [])
            ],
            edges=[
                edge_from_dict(x)
                for x in data.get("edges", [])
            ],
        )

        self.validator.validate(graph)
        if self.embedder is not None:
            self.embed_events(graph)
        return graph

    def embed_events(
        self,
        graph: EventGraph,
        only_event_ids: Optional[set[str]] = None,
    ):
        targets = [
            event
            for event in graph.events
            if only_event_ids is None or event.id in only_event_ids
        ]

        vectors = self.embedder.encode(
            [self.embedding_text(event) for event in targets]
        )
        if len(vectors) != len(targets):
            raise ValueError("embedding count mismatch")
        dimensions = {len(vector) for vector in vectors}
        if vectors and (len(dimensions) != 1 or 0 in dimensions or
                        any(not all(math.isfinite(v) for v in vector) or not any(vector) for vector in vectors)):
            raise ValueError("invalid embedding vectors")
        for event, vector in zip(targets, vectors):
            event.feature = vector
            event.embedding_model = self.embedder.model_name
            event.vector_version = "event_fields_v1"

    @staticmethod
    def embedding_text(event: Event) -> str:
        return json.dumps({name: getattr(event, name) for name in (
            "canonical_text", "event_type", "actor", "object", "state", "direction",
            "magnitude", "time", "modality", "polarity", "attribution", "condition", "span")},
            ensure_ascii=False, sort_keys=True)

    def initial_parse(
        self,
        doc_id: str,
        text: str,
    ) -> EventGraph:
        data = self.llm.chat_json(build_initial_prompt(text))
        return self._build_graph(doc_id, text, data)

    @staticmethod
    def _local_context(
        raw_text: str,
        source_span: str,
        radius: int = 600,
    ) -> str:
        if not source_span:
            return raw_text[:1500]

        pos = raw_text.find(source_span)
        if pos < 0:
            raise ValueError("source span not found")
        if raw_text.find(source_span, pos + 1) >= 0:
            # Do not silently choose the wrong occurrence. Full context keeps both visible.
            return raw_text

        start = max(0, pos - radius)
        end = min(
            len(raw_text),
            pos + len(source_span) + radius,
        )
        return raw_text[start:end]

    def refine(
        self,
        graph: EventGraph,
        target_event_id: str,
        need: dict[str, Any],
    ) -> EventGraph:
        target = graph.event_by_id(target_event_id)
        if target is None:
            raise ValueError(
                f"target event not found: {target_event_id}"
            )

        context = self._local_context(
            graph.raw_text,
            target.source_span,
        )

        patch = self.llm.chat_json(
            build_refinement_prompt(
                target_event=target,
                need=need,
                context=context,
            )
        )

        scope = need.get("patch_scope", "subtree")
        if scope not in {"subtree", "attributes_only"}:
            raise ValueError("unknown patch_scope")
        if scope == "attributes_only" and (
            patch.get("events_to_add") or patch.get("edges_to_add") or
            any(item.get("id") != target_event_id for item in patch.get("events_to_update", []))
        ):
            raise ValueError("attribute-only patch cannot change graph structure or other events")
        candidate, changed_ids = self.prepare_patch(graph, target_event_id, patch)
        if self.embedder is not None:
            self.embed_events(candidate, only_event_ids=changed_ids)
        return candidate

    def prepare_patch(self, graph: EventGraph, target_event_id: str, patch: dict[str, Any]):
        """Stage and validate local edits without mutating the original graph."""
        if not isinstance(patch, dict) or set(patch) - {"events_to_update", "events_to_add", "edges_to_add", "unresolved"}:
            raise ValueError("invalid patch object")
        for name in ("events_to_update", "events_to_add", "edges_to_add", "unresolved"):
            if not isinstance(patch.get(name, []), list) or any(not isinstance(v, dict) for v in patch.get(name, [])):
                raise ValueError(f"invalid patch collection: {name}")
        original_version = graph.version
        graph = deepcopy(graph)
        target = graph.event_by_id(target_event_id)
        if target is None:
            raise ValueError("unknown patch target")
        scope = {target_event_id}
        while True:
            expanded = scope | {e.id for e in graph.events if e.parent_id in scope}
            if expanded == scope:
                break
            scope = expanded
        changed_ids: set[str] = set()

        for update in patch.get("events_to_update", []):
            event_id = update.get("id")
            event = graph.event_by_id(event_id)
            if event is None or event_id not in scope:
                raise ValueError("patch update outside target subtree")

            for key, value in update.items():
                if key == "id":
                    continue
                if key not in Event.__dataclass_fields__ or key in {"feature", "embedding_model", "vector_version", "mainline_id", "parent_id", "level"}:
                    raise ValueError(f"patch cannot update {key}")
                setattr(event, key, value)

            changed_ids.add(event.id)

        existing_ids = {e.id for e in graph.events}

        for item in patch.get("events_to_add", []):
            event = event_from_dict(item)
            if event.id in existing_ids:
                raise ValueError("duplicate new event ID")
            if event.mainline_id != target.mainline_id or not event.parent_id:
                raise ValueError("new event must refine the target mainline")
            existing_ids.add(event.id)
            graph.events.append(event)
            changed_ids.add(event.id)

        existing_edge_keys = {
            (e.source, e.target, e.relation)
            for e in graph.edges
        }
        allowed = scope | changed_ids
        for event in graph.events:
            if event.id in changed_ids and event.id not in scope and event.parent_id not in allowed:
                raise ValueError("new event parent outside patch scope")

        for item in patch.get("edges_to_add", []):
            edge_data = dict(item)
            edge = edge_from_dict(edge_data)
            if edge.source not in allowed or edge.target not in allowed:
                raise ValueError("edge outside patch scope")

            key = (edge.source, edge.target, edge.relation)
            if key not in existing_edge_keys:
                graph.edges.append(edge)
                existing_edge_keys.add(key)

        graph.version = original_version + int(bool(changed_ids or patch.get("edges_to_add")))
        # Patches are transactional: do not let initial-parse cleanup delete an
        # existing event when an invalid replacement is returned by the LLM.
        self.validator._check_structure(graph)
        self.validator.validate(graph)

        surviving_changed = {
            eid
            for eid in changed_ids
            if graph.event_by_id(eid) is not None
        }

        graph.validation["unresolved"] = deepcopy(patch.get("unresolved", []))
        return graph, surviving_changed
