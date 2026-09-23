"""Execute rule-declared, evidence-preserving type alignment; no domain aliases here."""
from copy import deepcopy
import hashlib
import json

from .matcher.matching import structured_slot_match
from .alignment_contract import fingerprint
from .semantic_subgraph import has_required_subgraph, normalize_semantic_relations
from .vector_contracts import canonical_embedding_model, coerce_vector, rescale_cosine_strict


def is_fact(span):
    semantics = span.get("metadata", {}).get("semantic_event", {})
    return (semantics.get("modality", span.get("modality", "fact")) == "fact"
            and semantics.get("polarity", span.get("polarity", "positive")) == "positive")


def select_requirements(requirements, policy):
    """Rank goal-relevant structural gaps as requested by Control.

    The matcher can report gaps for many unrelated Rules.  Coverage alone is
    not enough once several domains share one Rule DB: a nearly matching Rule
    that cannot reach the requested goal must not crowd out a less complete
    Rule that can.  Goal annotations are computed by the Scheduler from the
    current goal and the generic Rule graph; this function only applies the
    Control policy to those annotations.
    """
    ranks = {}
    for request in requirements:
        matched, total = request.get("matched_slot_count", 0), request.get("total_slot_count", 0)
        coverage = matched / total if total else 0
        if coverage >= policy["minimum_coverage"]:
            goal_reachable = bool(request.get("goal_reachable"))
            goal_distance = request.get("goal_distance")
            try:
                distance_rank = -float(goal_distance) if goal_reachable else float("-inf")
            except (TypeError, ValueError):
                distance_rank = float("-inf")
            ranks[request["rule_id"]] = (
                float(goal_reachable),
                float(bool(request.get("goal_alignment"))),
                distance_rank,
                coverage,
                matched,
                -total,
            )
    selected = sorted(ranks, key=lambda key: (ranks[key], key), reverse=True)[:policy["max_rules"]]
    return [r for r in requirements if r["rule_id"] in selected]


def select_span_schema_knowledge_ids(records, requirements):
    """Select Knowledge schema edges from the Rule requirements they can prove.

    This is deliberately a structural lookup, rather than an ANN similarity
    lookup or a Control-rule-owned list of database IDs.  A schema is eligible
    only when its declared completion has exactly the same vector contract as
    the currently missing Rule slot.
    """
    targets = []
    for request in requirements or []:
        slot = (request or {}).get("slot") or {}
        required = slot.get("alignment_required") or {}
        if not isinstance(required, dict):
            continue
        if str(required.get("feature_type") or "").strip():
            targets.append(required)

    selected = set()
    for record in records or []:
        schema = dict((record.get("metadata") or {}).get("span_schema") or {})
        completion = dict(schema.get("completion") or {})
        target_type = str(schema.get("target_feature_type") or "").strip()
        if not target_type or set(completion) != {"core_vector", "relation_vector", "nature_vector"}:
            continue
        for required in targets:
            if target_type != str(required.get("feature_type") or "").strip():
                continue
            # Rule slots persist vectors but, unlike Event spans, do not carry
            # an embedding contract.  The only safe lookup is therefore an
            # exact, dimension-preserving vector equality check; text is never
            # used as a fallback identity.
            if all(
                bool(coerce_vector(required.get(field)))
                and bool(coerce_vector(completion.get(field)))
                and (match := rescale_cosine_strict(
                    coerce_vector(required.get(field)), coerce_vector(completion.get(field))
                ))["ok"]
                and float(match["score"]) >= .995
                for field in ("core_vector", "relation_vector", "nature_vector")
            ):
                selected.add(str(record.get("knowledge_id") or record.get("id") or ""))
                break
    return sorted(item for item in selected if item)


def _knowledge_span_schemas(workspace, knowledge_ids=None):
    """Load declarative Knowledge schemas selected by Control.

    Knowledge owns the within-span graph: a core, its relation(s), and its
    nature(s).  A schema defines the structure required to recognize a target
    span: either explicit triples, one core, or a set of cores.  It never
    contains a business Rule or chooses an output; it only proves that the
    current event already contains a better-described copy of the same fact.
    """
    schemas = {}
    allowed_ids = set(knowledge_ids or [])
    for context in workspace.get("context_items", []):
        payload = context.get("payload", {})
        if (context.get("status", "active") != "active"
                or (allowed_ids and payload.get("knowledge_id") not in allowed_ids)
                or payload.get("retrieval_source") != "knowledge_db"):
            continue
        schema = payload.get("span_schema", {})
        edge = payload.get("knowledge_edge", {})
        if not schema or not edge:
            continue
        schema_id = str(schema.get("schema_id") or "").strip()
        target_type = str(schema.get("target_feature_type") or "").strip()
        if not schema_id or not target_type:
            continue
        mode = str(schema.get("activation_mode") or "explicit_components").strip()
        if mode not in {"explicit_components", "single_core", "all_cores", "relation_context"}:
            raise ValueError(f"unsupported Knowledge activation mode: {mode}")
        completion = dict(schema.get("completion") or {})
        if set(completion) - {"core_vector", "relation_vector", "nature_vector"}:
            raise ValueError("Knowledge completion may only set semantic vectors")
        if set(completion) != {"core_vector", "relation_vector", "nature_vector"}:
            raise ValueError("Knowledge schema completion requires three semantic vectors")
        group = schemas.setdefault(schema_id, {
            "policy_id": schema_id,
            "policy_kind": "knowledge_span_schema",
            "target_type": target_type,
            "min_score": float(schema.get("min_score", 0.85)),
            "activation_mode": mode,
            "completion": completion,
            "evidence_components": [],
            "activation_components": [],
            "activation_relations": [],
            "context_components": [],
            "knowledge_ids": [],
            "knowledge_edge_ids": [],
        })
        if (group["target_type"] != target_type
                or group["completion"] != completion
                or group["activation_mode"] != mode):
            raise ValueError(f"inconsistent Knowledge span schema: {schema_id}")
        edge_role = str(schema.get("edge_role") or "evidence").strip()
        if edge_role == "evidence" and mode == "explicit_components":
            group["evidence_components"].append(dict(edge))
        elif edge_role == "activation_core" and mode in {"single_core", "all_cores"}:
            group["activation_components"].append(dict(edge))
        elif edge_role == "activation_relation" and mode == "relation_context":
            group["activation_relations"].append(dict(edge))
        elif edge_role == "context_core" and mode == "relation_context":
            group["context_components"].append(dict(edge))
        else:
            raise ValueError(f"unsupported Knowledge schema edge role: {edge_role}")
        group["knowledge_ids"].append(payload["knowledge_id"])
        group["knowledge_edge_ids"].append(str(edge.get("edge_id") or payload["knowledge_id"]))
    return [
        value
        for value in schemas.values()
        if (
            value["evidence_components"]
            or value["activation_components"]
            or (value["activation_relations"] and value["context_components"])
        )
    ]


def _core_match(vector, model, version, component):
    candidate = coerce_vector(vector)
    expected = coerce_vector(component.get("core_vector"))
    if not candidate or not expected:
        return None
    if (canonical_embedding_model(model) != canonical_embedding_model(component.get("embedding_model"))
            or str(version or "") != str(component.get("vector_version") or "")):
        return None
    result = rescale_cosine_strict(candidate, expected)
    return float(result["score"]) if result["ok"] and float(result["score"]) >= .995 else None


def _core_locations(workspace, component):
    """Find actual core nodes in the event, using vectors only."""
    locations = []
    for span in workspace.get("spans", []):
        span_id = str(span.get("span_id") or "")
        score = _core_match(span.get("core_vector"), span.get("embedding_model"), span.get("vector_version"), component)
        if score is not None:
            locations.append({"span_id": span_id, "relation_index": None, "score": score})
        for index, relation in enumerate(normalize_semantic_relations(span)):
            score = _core_match(relation.get("core_vector"), relation.get("embedding_model"), relation.get("vector_version"), component)
            if score is not None:
                locations.append({"span_id": span_id, "relation_index": index, "score": score})
    return locations


def _component_match(actual, component, field):
    candidate = coerce_vector(actual.get(field))
    expected = coerce_vector(component.get(field))
    if not candidate or not expected:
        return None
    if (
        canonical_embedding_model(actual.get("embedding_model"))
        != canonical_embedding_model(component.get("embedding_model"))
        or str(actual.get("vector_version") or "")
        != str(component.get("vector_version") or "")
    ):
        return None
    result = rescale_cosine_strict(candidate, expected)
    return float(result["score"]) if result["ok"] and float(result["score"]) >= .995 else None


def _adjacent_span_ids(workspace, source_span_id):
    """Return the source and spans connected to it by an explicit Event edge."""
    adjacent = {str(source_span_id or "")}
    for relation in workspace.get("relations", []):
        source = str(relation.get("source_span_id") or "")
        target = str(relation.get("target_span_id") or "")
        if source == source_span_id and target:
            adjacent.add(target)
        elif target == source_span_id and source:
            adjacent.add(source)
    return adjacent


def _activate_by_relation_context(workspace, source, relation_components, context_components):
    """Prove a sense from the source relation plus adjacent core evidence.

    The nature on the Event relation is deliberately ignored: it is the value
    being resolved.  Knowledge may complete it only when the same source core
    and relation are present and every declared context core occurs in the
    source span or a directly connected span.
    """
    if not relation_components or not context_components:
        return False, []
    try:
        nested_relations = normalize_semantic_relations(source)
    except ValueError:
        return False, []
    # The span's primary core-relation-nature triple is part of the same local
    # semantic graph as semantic_relations.  Frontend documents keep that
    # primary triple at the span top level and only place additional context
    # triples in semantic_relations; requiring a duplicated copy would make a
    # valid public frontend graph impossible to align.
    source_relations = [(None, source)] + [
        (index, relation) for index, relation in enumerate(nested_relations)
    ]
    evidence = []
    used_relations = set()
    for component in relation_components:
        selected = None
        for index, relation in source_relations:
            if index in used_relations:
                continue
            core_score = _component_match(relation, component, "core_vector")
            relation_score = _component_match(relation, component, "relation_vector")
            if core_score is not None and relation_score is not None:
                selected = (index, core_score, relation_score)
                break
        if selected is None:
            return False, []
        index, core_score, relation_score = selected
        used_relations.add(index)
        evidence.append({
            "knowledge_edge_id": str(component.get("edge_id") or component.get("knowledge_id") or ""),
            "span_id": str(source.get("span_id") or ""),
            "relation_index": index,
            "core_score": round(core_score, 6),
            "relation_score": round(relation_score, 6),
            "evidence_role": "activation_relation",
        })

    adjacent_ids = _adjacent_span_ids(workspace, str(source.get("span_id") or ""))
    used_locations = {(item["span_id"], item["relation_index"]) for item in evidence}
    for component in context_components:
        locations = [
            item
            for item in _core_locations(workspace, component)
            if item["span_id"] in adjacent_ids
            and (item["span_id"], item["relation_index"]) not in used_locations
        ]
        if not locations:
            return False, []
        location = locations[0]
        used_locations.add((location["span_id"], location["relation_index"]))
        evidence.append({
            "knowledge_edge_id": str(component.get("edge_id") or component.get("knowledge_id") or ""),
            "span_id": location["span_id"],
            "relation_index": location["relation_index"],
            "core_score": round(location["score"], 6),
            "evidence_role": "context_core",
        })
    return True, evidence


def _activate_by_cores(workspace, source, components, mode):
    """A single core activates alone; all_cores requires distinct event evidence."""
    if not components:
        return False, []
    evidence, used = [], set()
    for index, component in enumerate(components):
        locations = _core_locations(workspace, component)
        if index == 0:
            locations = [item for item in locations if item["span_id"] == source.get("span_id")]
        location = next((item for item in locations if (item["span_id"], item["relation_index"]) not in used), None)
        if location is None:
            return False, []
        used.add((location["span_id"], location["relation_index"]))
        evidence.append({"knowledge_edge_id": str(component.get("edge_id") or component.get("knowledge_id") or ""),
                         "span_id": location["span_id"], "relation_index": location["relation_index"],
                         "core_score": round(location["score"], 6)})
        if mode == "single_core":
            return True, evidence
    return True, evidence


def align_event_spans(memory, requirements, knowledge_ids, *, control_rule_id):
    """Only unique, policy-authorized, vector-compatible evidence can be added.

    Originals are never edited. A derived span retains its original value, unit,
    entity, period and vectors. Relations are mirrored within the same workspace.
    """
    accepted, rejected, retries = [], [], []
    for request in requirements:
        event_id, rule_id = request.get("event_id"), request.get("rule_id")
        workspace = memory.event_workspaces.get(event_id)
        if not workspace or not rule_id:
            rejected.append({"reason": "missing_target", "request": request})
            continue
        originals = {s["span_id"]: s for s in workspace.get("spans", [])}
        policies = _knowledge_span_schemas(workspace, knowledge_ids)
        eligible = {}
        for candidate in request["slot"].get("alignment_candidates", []):
            source = originals.get(candidate["span_id"])
            required = request["slot"]["alignment_required"]
            if not source or source.get("metadata", {}).get("semantic_alignment"):
                continue
            if not is_fact(source):
                continue
            for policy in policies:
                linked = []
                component_evidence = []
                if required.get("feature_type") != policy["target_type"]:
                    continue
                if policy["activation_mode"] == "explicit_components":
                    proven, component_evidence = has_required_subgraph(source, policy["evidence_components"])
                elif policy["activation_mode"] == "relation_context":
                    proven, component_evidence = _activate_by_relation_context(
                        workspace,
                        source,
                        policy["activation_relations"],
                        policy["context_components"],
                    )
                else:
                    proven, component_evidence = _activate_by_cores(
                        workspace, source, policy["activation_components"], policy["activation_mode"]
                    )
                if not proven:
                    continue
                completion = policy.get("completion", {})
                if set(completion) - {"core_vector", "relation_vector", "nature_vector"}:
                    raise ValueError("alignment knowledge cannot change values, units or identities")
                probe = {**source, **completion, "feature_type": policy["target_type"]}
                score = structured_slot_match(required, probe)
                threshold = max(float(policy["min_score"]), float(request["slot"]["match_threshold"]))
                if score["vector_status"] == "ok" and score["score"] >= threshold:
                    eligible[(source["span_id"], fingerprint(probe))] = (source, required, policy, score["score"], linked, component_evidence)
        if len(eligible) != 1:
            rejected.append({"rule_id": rule_id, "event_id": event_id,
                             "reason": "ambiguous" if eligible else "no_supported_alignment"})
            continue
        source, required, policy, score, linked, component_evidence = next(iter(eligible.values()))
        key = json.dumps([event_id, source["span_id"], required["feature_type"], rule_id, request["slot"]["slot_id"]], ensure_ascii=False)
        span_id = "alignment_" + hashlib.sha256(key.encode()).hexdigest()[:20]
        if span_id in originals:
            rejected.append({"rule_id": rule_id, "reason": "already_attempted"})
            continue
        derived = deepcopy(source)
        derived.update(deepcopy(policy.get("completion", {})))
        derived.update(span_id=span_id, feature_type=required["feature_type"])
        derived.setdefault("metadata", {})["semantic_alignment"] = {
            "control_rule_id": control_rule_id, "policy_id": policy["policy_id"],
            "source_span_id": source["span_id"], "target_rule_id": rule_id,
            "target_slot_id": request["slot"]["slot_id"], "score": score,
            "original_feature_type": source.get("feature_type"),
            "knowledge_id": policy.get("knowledge_id") or policy.get("knowledge_ids", [""])[0], "supporting_span_ids": linked,
            "knowledge_ids": list(dict.fromkeys(policy.get("knowledge_ids", [policy.get("knowledge_id", "")]))) ,
            "knowledge_edge_ids": list(dict.fromkeys(policy.get("knowledge_edge_ids", []))),
            "semantic_component_evidence": component_evidence,
            "activation_mode": policy["activation_mode"],
            "source_fingerprints": {sid: fingerprint(originals[sid]) for sid in [source["span_id"], *linked]},
            "source_relations": [deepcopy(r) for r in workspace.get("relations", [])
                                 if r.get("source_span_id") == source["span_id"] and r.get("target_span_id") in linked],
        }
        # One atomic workspace replacement; preserved unrelated facts and relations.
        updated = deepcopy(workspace)
        updated.setdefault("spans", []).append(derived)
        if "evidence_span_ids" in updated:
            updated["evidence_span_ids"] = list(dict.fromkeys([*updated["evidence_span_ids"], span_id]))
        for relation in workspace.get("relations", []):
            if source["span_id"] not in (relation.get("source_span_id"), relation.get("target_span_id")):
                continue
            edge = deepcopy(relation)
            for endpoint in ("source_span_id", "target_span_id"):
                if edge.get(endpoint) == source["span_id"]:
                    edge[endpoint] = span_id
            edge["relation_id"] = str(edge.get("relation_id", "relation")) + "_" + span_id
            updated.setdefault("relations", []).append(edge)
        updated["version"] = int(workspace.get("version", 0)) + 1
        memory.commit_calculated_workspace(event_id, updated, expected_version=int(workspace.get("version", 0)))
        accepted.append(derived["metadata"]["semantic_alignment"])
        retries.append(rule_id)
    return {"accepted": accepted, "rejected": rejected, "retry_rule_ids": list(dict.fromkeys(retries))}


def frontend_questions(memory, requirements, attempts, *, max_requests, min_score=0.8, max_questions=3, patch_scope="attributes_only"):
    """Build bounded source-only questions, never present a desired answer as fact."""
    questions, updated = [], dict(attempts)
    for request in requirements:
        workspace = memory.event_workspaces.get(request.get("event_id"), {})
        for candidate in sorted(request["slot"].get("alignment_candidates", []), key=lambda c: c.get("core_score", 0), reverse=True)[:1]:
            if candidate.get("core_score", 0) < min_score:
                continue
            source = next((s for s in workspace.get("spans", []) if s.get("span_id") == candidate["span_id"]), {})
            if not source.get("metadata", {}).get("semantic_event"):
                continue
            key = json.dumps([request.get("event_id"), candidate["span_id"]])
            if updated.get(key, 0) >= max_requests:
                continue
            updated[key] = updated.get(key, 0) + 1
            questions.append({"operation": "request_frontend_patch", "target_span_id": candidate["span_id"],
                "patch_scope": patch_scope,
                "missing_span_query": "Re-read only the original source. Clarify this event's concept, scope, entity, period and stated conditions. Where relevant, distinguish an average from an ending balance and a gross/total amount from a net or credit-basis amount, but only when the source explicitly says so. Correct its structured span if warranted. Otherwise return unresolved. Do not infer a business result or use a downstream rule as evidence.",
                "required_fields": ["core", "nature", "feature_type"],
                "reason": "control_semantic_alignment_unclear"})
            if len(questions) >= max_questions:
                return questions, updated
    return questions, updated
