"""Control-requested vector interpretation of a source frontend graph.

No domain aliases or label comparisons. Knowledge stores recognition vectors,
roles and vector triples. Ambiguous/low-score observations remain unresolved.
Identity recognition does not prove a Rule require or invent missing facts.
"""
from copy import deepcopy
import hashlib
import json
from .vector_contracts import canonical_embedding_model, rescale_cosine_strict, coerce_vector
from .persistence import knowledge_db


def _choose(vector, records, kind, policy, model, version, components=None):
    ranked = []
    for r in records:
        d = r.get("metadata", {}).get("semantic_definition", {})
        if d.get("kind") != kind or r.get("status", "active") != "active":
            continue
        if canonical_embedding_model(r.get("embedding_model")) != canonical_embedding_model(model) or str(r.get("vector_version")) != str(version):
            continue
        score = rescale_cosine_strict(vector, d.get("recognition_vector"))
        if score["ok"]:
            component_scores = {}
            if kind in {"observation", "qualifier"}:
                for field in ("core_vector", "relation_vector", "nature_vector"):
                    matched = rescale_cosine_strict((components or {}).get(field), r.get(field))
                    if not matched["ok"]:
                        break
                    component_scores[field] = float(matched["score"])
                if len(component_scores) != 3 or min(component_scores.values()) < policy["component_minimum_score"]:
                    continue
            ranked.append((float(score["score"]), str(r.get("id") or r.get("knowledge_id")), r, component_scores))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if not ranked:
        return None, {"reason": "no_vector_candidates"}
    margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else 1.
    audit = {"score": ranked[0][0], "margin": margin, "candidate_knowledge_id": ranked[0][1], "component_scores": ranked[0][3]}
    if ranked[0][0] < policy["minimum_score"] or margin < policy["minimum_margin"]:
        return None, {**audit, "reason": "uncertain_vector_identity"}
    return ranked[0][2], audit


def _edge(r, debug):
    result = {field: deepcopy(r[field]) for field in ("core_vector", "relation_vector", "nature_vector", "embedding_model", "vector_version")}
    result["debug"] = debug
    return result


def ground_input_semantics(memory, db_path, encoder, *, control_rule_id, policy):
    if not control_rule_id or set(policy) != {"minimum_score", "minimum_margin", "component_minimum_score"}:
        raise ValueError("Input semantics requires an owning Control rule and explicit vector policy")
    if not 0 <= policy["minimum_score"] <= 1 or not 0 < policy["minimum_margin"] <= 1 or not 0 <= policy["component_minimum_score"] <= 1:
        raise ValueError("Invalid input semantic recognition policy")
    records = knowledge_db.load_knowledge_records(db_path, limit=None)
    definitions = [r for r in records if r.get("metadata", {}).get("semantic_definition")]
    frames = [r for r in definitions if r["metadata"]["semantic_definition"]["kind"] == "frame"]
    if len(frames) != 1:
        raise ValueError("A vector decision frame definition is required")
    properties = {r["metadata"]["semantic_definition"]["property"]: r for r in definitions if r["metadata"]["semantic_definition"]["kind"] == "property"}
    accepted, rejected = [], []
    for event_id, original_workspace in list(memory.event_workspaces.items()):
        workspace = deepcopy(original_workspace)
        originals = list(workspace.get("spans", []))
        existing = {s["span_id"] for s in originals}
        mappings = {}
        for source in originals:
            metadata = source.get("metadata", {})
            role = metadata.get("translation_role")
            if role not in {"fact", "context", "goal"} or metadata.get("control_semantic_binding"):
                continue
            stamp = hashlib.sha256(json.dumps([event_id, source["span_id"], source.get("concept_vector"), source.get("semantic_relations")], sort_keys=True).encode()).hexdigest()[:20]
            sid = "semantic_input_" + stamp
            if sid in existing:
                continue
            if role == "context":
                record, audit = frames[0], {"reason": "explicit_context_role"}
            else:
                record, audit = _choose(source.get("concept_vector"), definitions, "observation" if role == "fact" else "goal", policy, source.get("embedding_model"), source.get("vector_version"), source)
            if record is None:
                rejected.append({"source_span_id": source["span_id"], "event_id": event_id, **audit})
                continue
            definition = record["metadata"]["semantic_definition"]
            if role == "fact":
                expected_type = definition["value_type"]
                value = source.get("value")
                compatible = (type(value) in (int, float) and source.get("value_type") == "number") if expected_type == "number" else (source.get("value_type") == "boolean" and type(value) in (bool, int) and value in (0, 1))
                if not compatible or source.get("unit", "") != definition.get("unit", ""):
                    rejected.append({"source_span_id": source["span_id"], "reason": "identity_value_or_unit_conflict", **audit})
                    continue
            derived = deepcopy(source)
            derived.update(span_id=sid, feature_type=definition["runtime_feature_type"], **{k: deepcopy(record[k]) for k in ("core_vector", "relation_vector", "nature_vector")})
            # Labels are output/debug rendering, never used by _choose.
            derived.update(record["debug"])
            relations = [_edge(record, record["debug"])] if role == "fact" else []
            for key, prop in properties.items():
                if key not in source:
                    continue
                nature = encoder._encode_text_vector(str(source[key]))
                if not coerce_vector(nature):
                    raise ValueError("Cannot encode current literal scope")
                edge = _edge(prop, {**prop["debug"], "nature": str(source[key])})
                edge["nature_vector"] = nature
                relations.append(edge)
            property_audits = []
            for raw in source.get("semantic_relations", [])[1:]:
                # Negated/uncertain properties never establish positive evidence.
                if raw.get("polarity", "positive") != "positive":
                    property_audits.append({"reason": "non_positive_property", "polarity": raw.get("polarity")})
                    relations.append(deepcopy(raw))
                    continue
                candidate, pa = _choose(raw.get("concept_vector"), definitions, "qualifier", policy, source.get("embedding_model"), source.get("vector_version"), raw)
                property_audits.append(pa)
                if candidate is not None:
                    edge = _edge(candidate, candidate["debug"])
                    edge["source_span"] = raw["source_span"]
                    relations.append(edge)
                else:
                    relations.append(deepcopy(raw))
            derived["semantic_relations"] = relations
            if role == "goal":
                derived["metadata"]["target_feature_type"] = definition["target_feature_type"]
            provenance = {"control_rule_id": control_rule_id, "source_span_id": source["span_id"], "knowledge_id": record["id"],
                          "recognition": audit, "property_recognition": property_audits,
                          "preserved_fields": ["value", "unit", "entity_id", "period", "decision_id", "currency", "policy_version", "modality", "polarity"]}
            derived["metadata"]["control_semantic_binding"] = provenance
            workspace["spans"].append(derived)
            mappings[source["span_id"]] = sid
            accepted.append({"event_id": event_id, "span_id": sid, "feature_type": derived["feature_type"], **provenance})
            existing.add(sid)
        if mappings:
            workspace["evidence_span_ids"] = list(dict.fromkeys([*workspace.get("evidence_span_ids", []), *mappings.values()]))
            for raw in original_workspace.get("relations", []):
                if any(raw.get(k) in mappings for k in ("source_span_id", "target_span_id")):
                    mirrored = deepcopy(raw)
                    for key in ("source_span_id", "target_span_id"):
                        mirrored[key] = mappings.get(mirrored.get(key), mirrored.get(key))
                    mirrored["relation_id"] = str(raw.get("relation_id", "edge")) + "_interpreted"
                    workspace["relations"].append(mirrored)
            old_version = int(original_workspace.get("version", 0))
            workspace["version"] = old_version + 1
            memory.commit_calculated_workspace(event_id, workspace, expected_version=old_version)
    return {"accepted": accepted, "rejected": rejected, "control_rule_id": control_rule_id,
            "scope": "vector_identity_recognition_not_business_rule_proof"}
