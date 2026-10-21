"""Two source-only translation stages; backend owns Knowledge interpretation."""
from copy import deepcopy
import math
from jsonschema import Draft202012Validator
from .source_contract import BASIC_SCHEMA, VERSION, render_basic_request, source_units
from .property_translation import render_property_request, merge_property_reply


def project_basic_reply(data, text):
    errors = list(Draft202012Validator(BASIC_SCHEMA).iter_errors(data))
    if errors:
        raise ValueError("Basic schema: " + "; ".join(
            "/".join(map(str, e.absolute_path)) + ": " +
            ("Duplicate facts" if e.validator == "uniqueItems" else e.message[:200]) for e in errors[:8]))
    units = {u["id"]: u for u in source_units(text)}
    covered = set()
    def cite(record):
        endpoints = record["source_ref"].split(":")
        if any(k not in units for k in endpoints):
            raise ValueError("Unknown source_ref: " + record["source_ref"])
        first, last = units[endpoints[0]], units[endpoints[-1]]
        if first["start"] > last["start"]:
            raise ValueError("Reversed source range")
        source = text[first["start"]:last["end"]]
        for k, v in record.get("scope", {}).items():
            if v not in source:
                raise ValueError("scope." + k + " must be a literal identifier in its own source")
        covered.update(k for k, u in units.items() if first["start"] <= u["start"] <= last["start"])
        return source, {"source_ref": record["source_ref"], "start": first["start"], "end": last["end"]}
    spans = []
    source, reference = cite(data["decision"])
    spans.append({"span_id": "decision_context", "feature_type": "unresolved_context",
        "core": "决策上下文", "relation": "描述", "nature": "当前决策范围", "value": 1, "unit": "",
        **data["decision"]["scope"], "text": source, "modality": "fact", "polarity": "positive",
        "semantic_relations": [], "metadata": {"origin": VERSION, "translation_role": "context", "source_reference": reference}, "confidence": 1.})
    for i, fact in enumerate(data["facts"]):
        source, reference = cite(fact)
        value = fact["value"]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite fact value")
        if fact["value_type"] == "number" and type(value) not in (int, float):
            raise ValueError("number requires a numeric scalar")
        if fact["value_type"] == "boolean" and (type(value) not in (bool, int) or value not in (0, 1) or fact["polarity"] != "positive"):
            raise ValueError("boolean requires true/false/0/1 with positive polarity, no duplicate negation")
        triple = {k: fact[k] for k in ("core", "relation", "nature")}
        spans.append({"span_id": f"observation_{i}", "feature_type": "unresolved_observation", **triple,
            "semantic_concept": fact["concept"], "value": int(value) if fact["value_type"] == "boolean" else value,
            "value_type": fact["value_type"], "unit": fact["unit"], **fact["scope"], "text": source,
            "modality": fact["modality"], "polarity": fact["polarity"], "semantic_relations": [triple],
            "metadata": {"origin": VERSION, "translation_role": "fact", "source_reference": reference, "qualifier_evidence": []}, "confidence": 1.})
    goal = data["goal"]
    source, reference = cite(goal)
    spans.append({"span_id": "requested_goal", "feature_type": "unresolved_goal",
        **{k: goal[k] for k in ("core", "relation", "nature")}, "semantic_concept": goal["concept"],
        "span_role": "objective", "unit": "", "text": source, "modality": "request", "polarity": "positive",
        "semantic_relations": [], "metadata": {"origin": VERSION, "translation_role": "goal", "source_reference": reference}, "confidence": 1.})
    if set(units) != covered:
        raise ValueError("Unrepresented source units: " + ",".join(sorted(set(units) - covered)))
    return {"spans": spans, "events": [{"event_id": "case", "event_type": "reasoning_request",
        "evidence_span_ids": [s["span_id"] for s in spans], "relations": [],
        "metadata": {"source_text": text, "frontend_contract": VERSION}}]}


class SourceTranslationFrontend:
    def __init__(self, llm, *, max_format_corrections=2):
        if type(max_format_corrections) is not int or not 0 <= max_format_corrections <= 2:
            raise ValueError("max_format_corrections must be an integer from 0 to 2")
        self.llm = llm
        self.max_format_corrections = max_format_corrections
        self.translations = []
        self.validation_attempts = []

    def build_event_graph(self, game_input):
        text = game_input["input"]
        corrections = 0
        attempts = 0
        def translate(stage, render, project):
            nonlocal corrections, attempts
            correction = None
            while True:
                attempts += 1
                try:
                    data = self.llm.chat_json(render(correction))
                    self.translations.append({"stage": stage, "reply": deepcopy(data)})
                    result = project(data)
                except ValueError as exc:
                    self.validation_attempts.append({"stage": stage, "attempt": attempts, "status": "rejected", "error": str(exc)})
                    if corrections >= self.max_format_corrections:
                        raise ValueError(f"Translation rejected after {attempts} requests; shared correction budget exhausted: {exc}") from exc
                    corrections += 1
                    correction = {"validation_error": str(exc)}
                else:
                    self.validation_attempts.append({"stage": stage, "attempt": attempts, "status": "accepted"})
                    return result
        basic = translate("basic_facts", lambda c: render_basic_request(text, c), lambda d: project_basic_reply(d, text))
        return translate("fact_properties", lambda c: render_property_request(basic, c), lambda d: merge_property_reply(basic, d))

    def patch_event_graph(self, *, game_input, current_event_graph, patches):
        ids = {p.get("target_span_id") for p in patches if isinstance(p, dict)}
        facts = [s for s in current_event_graph.get("spans", []) if s.get("metadata", {}).get("translation_role") == "fact" and s["span_id"] in ids]
        if not ids or len(facts) != len(ids):
            raise ValueError("Targeted source refinement requires existing source fact ids")
        # Do not pass the requested Rule requirement, expected properties or
        # Knowledge candidates to the model. It only re-reads these sources.
        selected = deepcopy(facts)
        for s in selected:
            s["semantic_relations"] = [{k: s[k] for k in ("core", "relation", "nature")}]
            s["metadata"]["qualifier_evidence"] = []
        subset = {"spans": [deepcopy(current_event_graph["spans"][0]), *selected, deepcopy(current_event_graph["spans"][-1])]}
        correction = None
        for attempt in range(self.max_format_corrections + 1):
            try:
                data = self.llm.chat_json(render_property_request(subset, correction))
                self.translations.append({"stage": "targeted_properties", "reply": deepcopy(data)})
                updated = merge_property_reply(subset, data)
            except ValueError as exc:
                self.validation_attempts.append({"stage": "targeted_properties", "attempt": attempt + 1, "status": "rejected", "error": str(exc)})
                if attempt == self.max_format_corrections:
                    raise ValueError("Targeted properties correction budget exhausted: " + str(exc)) from exc
                correction = {"validation_error": str(exc)}
            else:
                self.validation_attempts.append({"stage": "targeted_properties", "attempt": attempt + 1, "status": "accepted"})
                result = deepcopy(current_event_graph)
                by_id = {s["span_id"]: s for s in updated["spans"][1:-1]}
                result["spans"] = [by_id.get(s["span_id"], s) for s in result["spans"]]
                return result
