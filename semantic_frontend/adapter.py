"""Pure Python data boundary; no backend imports and no HTTP service."""
from copy import deepcopy
import hashlib

from .frontend import SimpleSemanticFrontend
from .graph import EventGraph, event_from_dict, edge_from_dict, mainline_from_dict
from .validation import GraphContractValidator

CONTRACT_VERSION = "semantic_frontend.v1"


def document_from_dict(data, text):
    if not isinstance(data, dict) or data.get("raw_text") != text:
        raise ValueError("document/source mismatch")
    graph = EventGraph(doc_id=data["doc_id"], raw_text=text,
        events=[event_from_dict(e) for e in data.get("events", [])],
        edges=[edge_from_dict(e) for e in data.get("edges", [])],
        mainlines=[mainline_from_dict(e) for e in data.get("mainlines", [])],
        version=data.get("version", 0))
    if type(graph.version) is not int or graph.version < 0:
        raise ValueError("invalid document version")
    GraphContractValidator().validate(graph)
    return graph


def document_dict(graph):
    data = graph.to_dict()
    for e in data["events"]:
        for key in ("feature", "embedding_model", "vector_version"):
            e.pop(key, None)
    return data


def runtime_graph(graph):
    """Lossless text fields; grouping is structural, not business inference."""
    document = document_dict(graph)
    spans = []
    for e in graph.events:
        spans.append({"span_id": e.id, "core": e.canonical_text,
            "nature": SimpleSemanticFrontend.embedding_text(e), "text": e.source_span,
            "feature_type": e.event_type or "event", "confidence": e.confidence,
            **deepcopy(e.span or {}),
            "metadata": {"semantic_event": next(x for x in document["events"] if x["id"] == e.id),
                         "source_span": e.source_span, "origin": CONTRACT_VERSION}})
    events = []
    for mainline in graph.mainlines:
        own = {e.id for e in graph.events if e.mainline_id == mainline.id}
        if not own:
            continue
        edges = [e for e in graph.edges if e.source in own or e.target in own]
        evidence = own | {v for e in edges for v in (e.source, e.target)}
        events.append({"event_id": mainline.id, "event_type": mainline.title,
            "evidence_span_ids": sorted(evidence),
            "relations": [{"source_span_id": e.source, "target_span_id": e.target,
                           "relation_type": e.relation, "confidence": e.confidence,
                           "metadata": {"source_span": e.source_span, "explicit": e.explicit}}
                          for e in edges],
            "metadata": {"mainline": mainline.id, "summary": mainline.summary}})
    if events:
        # Retained by the backend workspace, including after frontend patches.
        events[0]["metadata"]["frontend_document"] = document
        events[0]["metadata"]["source_text"] = graph.raw_text
    return {"spans": spans, "events": events}


class SemanticFrontendAdapter:
    def __init__(self, frontend):
        self.frontend = frontend

    def build_event_graph(self, game_input):
        # Only original text crosses the semantic boundary, not backend rules/results.
        return self._translate("parse", {"contract_version": CONTRACT_VERSION,
                           "text": game_input["input"]})["event_graph"]

    def patch_event_graph(self, *, game_input, current_event_graph, patches):
        document = next((e.get("metadata", {}).get("frontend_document")
                         for e in current_event_graph.get("events", [])
                         if e.get("metadata", {}).get("frontend_document")), None)
        if document is None:
            raise ValueError("frontend patch requires its original versioned document")
        needs = []
        for patch in patches:
            need = {k: deepcopy(patch[k]) for k in ("target_span_id", "target_event_id", "event_id",
                "missing_span_query", "required_fields", "missing", "target_semantics", "patch_scope") if k in patch}
            # A question is a search request, never evidence or an expected answer.
            if patch.get("question"):
                need.setdefault("missing_span_query", patch["question"])
            if not any(need.get(k) for k in ("target_span_id", "target_event_id", "event_id")):
                if not need.get("missing_span_query"):
                    raise ValueError("Source recovery requires a target or a concrete question")
                need["patch_scope"] = "document"
            needs.append(need)
        result = self._translate("refine", {"contract_version": CONTRACT_VERSION,
            "text": game_input["input"], "document": document, "patches": needs})["event_graph"]
        old_spans = {s["span_id"]: s for s in current_event_graph.get("spans", [])}
        for index, span in enumerate(result["spans"]):
            old = old_spans.get(span["span_id"], {})
            conflicts = [field for field in ("value", "unit", "polarity")
                if old.get(field) not in (None, "") and span.get(field) not in (None, "")
                and old[field] != span[field]]
            if conflicts:
                preserved = deepcopy(old)
                evidence = {"source": "frontend_source_refinement", "target_span_id": span["span_id"],
                    "fields": conflicts, "existing": {k: old[k] for k in conflicts},
                    "candidate": {k: span[k] for k in conflicts},
                    "source_span": span.get("metadata", {}).get("source_span")}
                retained = preserved.setdefault("metadata", {}).setdefault("require_conflicts", [])
                if evidence not in retained:
                    retained.append(evidence)
                result["spans"][index] = preserved
                continue
            if old.get("metadata", {}).get("semantic_event") == span["metadata"]["semantic_event"]:
                result["spans"][index] = deepcopy(old)
        # The LLM only sees source data. Preserve backend additions in the returned
        # workspace because the existing bridge replaces its span collection.
        new_ids = {s["span_id"] for s in result["spans"]}
        for span in current_event_graph.get("spans", []):
            if span["span_id"] not in new_ids:
                result["spans"].append(deepcopy(span))
                new_ids.add(span["span_id"])
        existing = {e["event_id"]: e for e in current_event_graph.get("events", [])}
        for event in result["events"]:
            old = existing.pop(event["event_id"], {})
            event["metadata"] = {**deepcopy(old.get("metadata", {})), **event["metadata"]}
            event["evidence_span_ids"] = sorted(set(event["evidence_span_ids"]) |
                set(old.get("evidence_span_ids", [])))
            for key in ("candidate_paths", "feedback"):
                if key in old:
                    event[key] = deepcopy(old[key])
            def edge_key(edge):
                return (edge.get("source_span_id") or edge.get("source"),
                        edge.get("target_span_id") or edge.get("target"),
                        edge.get("relation_type") or edge.get("relation"))
            keys = {edge_key(r) for r in event["relations"]}
            event["relations"].extend(deepcopy(r) for r in old.get("relations", []) if edge_key(r) not in keys)
        result["events"].extend(deepcopy(list(existing.values())))
        return result

    def _translate(self, mode, payload):
        if not isinstance(payload, dict) or payload.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("unsupported frontend data contract version")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        unresolved = []
        if mode == "parse":
            graph = self.frontend.initial_parse(hashlib.sha256(text.encode()).hexdigest()[:24], text)
        elif mode == "refine":
            graph = document_from_dict(payload.get("document"), text)
            patches = payload.get("patches")
            if not isinstance(patches, list) or not patches or len(patches) > 16:
                raise ValueError("patches must contain 1..16 requests")
            for patch in patches:
                if not isinstance(patch, dict):
                    raise ValueError("patch must be an object")
                if patch.get("patch_scope") == "document":
                    if not patch.get("missing_span_query"):
                        raise ValueError("Document recovery requires a concrete question")
                    # Snapshot roots before refinement adds children. All original
                    # source roots are eligible; no arbitrary first-root fallback.
                    roots = [e.id for e in graph.events if not e.parent_id]
                    if not roots:
                        unresolved.append({"reason": "target_not_identified", "request": patch})
                    for root in roots:
                        graph = self.frontend.refine(graph, root, {**patch, "target_span_id": root})
                        unresolved.extend(graph.validation.get("unresolved", []))
                    continue
                target = patch.get("target_span_id") or patch.get("target_event_id") or patch.get("event_id")
                if not graph.event_by_id(target):
                    roots = [e for e in graph.events if e.mainline_id == target and not e.parent_id]
                    target = roots[0].id if len(roots) == 1 else None
                if target is None or not graph.event_by_id(target):
                    unresolved.append({"reason": "target_not_identified", "request": patch})
                    continue
                graph = self.frontend.refine(graph, target, patch)
                unresolved.extend(graph.validation.get("unresolved", []))
        else:
            raise ValueError("unknown translation operation")
        result = runtime_graph(graph)
        result["frontend_translation"] = {"version": CONTRACT_VERSION,
            "status": "unresolved" if unresolved else "ok", "unresolved": unresolved}
        return {"event_graph": result}
