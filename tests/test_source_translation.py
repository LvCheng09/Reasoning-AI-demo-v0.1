"""Offline boundary tests, not evidence of real model translation accuracy."""
from copy import deepcopy
import pytest
from semantic_frontend.source_translation import SourceTranslationFrontend, project_basic_reply
from semantic_frontend.source_contract import render_basic_request
from semantic_frontend.property_translation import merge_property_reply, render_property_request
from main_part.input_semantics import _choose
from main_part.semantic_subgraph import normalize_semantic_relations, has_required_subgraph


def basic():
    triple = {"concept": "温度", "core": "设备", "relation": "温度", "nature": "观测值"}
    return {"decision": {"scope": {"entity_id": "A"}, "source_ref": "s0"},
        "facts": [{**triple, "value": 3, "value_type": "number", "unit": "", "scope": {"entity_id": "A"},
                   "source_ref": "s0", "modality": "fact", "polarity": "positive"}],
        "goal": {**triple, "source_ref": "s0"}}


def properties(status="not_stated", *, extra=None):
    return {"properties": [{"fact_id": "observation_0", "status": status,
        "semantic_relations": extra or [], "note": "难以确定" if status == "uncertain" else ""}]}


def test_two_stages_share_correction_budget_and_never_send_knowledge():
    prompts = []
    bad = basic();bad["facts"][0]["source_ref"] = "s99"
    replies = [bad, basic(), properties()]
    class OfflineTransport:
        def chat_json(self, prompt):
            prompts.append(prompt)
            return deepcopy(replies.pop(0))
    frontend = SourceTranslationFrontend(OfflineTransport())
    graph = frontend.build_event_graph({"input": "A observed question"})
    assert graph["spans"][1]["value"] == 3
    assert len(prompts) == 3
    assert "basic_facts" in prompts[0] and "fact_properties" in prompts[-1]
    assert all("<concept_definitions>" not in p and "<property_definitions>" not in p and '"q0"' not in p for p in prompts)
    assert "question" in prompts[0]
    assert '"goal"' not in prompts[-1]
    assert [r["status"] for r in frontend.validation_attempts] == ["rejected", "accepted", "accepted"]
    assert graph["spans"][1]["metadata"]["property_translation"]["status"] == "not_stated"


def test_property_errors_do_not_get_an_independent_extra_retry_budget():
    prompts = []
    bad = basic();bad["facts"][0]["source_ref"] = "s99"
    replies = [bad, basic(), {}, {}]
    class OfflineTransport:
        def chat_json(self, prompt):
            prompts.append(prompt);return deepcopy(replies.pop(0))
    frontend = SourceTranslationFrontend(OfflineTransport())
    with pytest.raises(ValueError, match="shared correction budget exhausted"):
        frontend.build_event_graph({"input": "A observed question"})
    assert len(prompts) == 4
    assert all(r["stage"] == "fact_properties" for r in frontend.validation_attempts[-2:])


@pytest.mark.parametrize("status", ["not_stated", "uncertain"])
def test_property_states_preserved_without_inventing_a_triple(status):
    graph = project_basic_reply(basic(), "A observed question")
    result = merge_property_reply(graph, properties(status))
    assert result["spans"][1]["metadata"]["property_translation"]["status"] == status
    assert result["spans"][1]["semantic_relations"] == graph["spans"][1]["semantic_relations"]
    assert "property_translation" not in graph["spans"][1]["metadata"]


def test_properties_must_cover_all_fact_ids_and_cannot_change_values_or_scopes():
    graph = project_basic_reply(basic(), "A observed question")
    for reply in [{"properties": []}, {"properties": properties()["properties"] * 2}]:
        with pytest.raises(ValueError, match="exactly once"):
            merge_property_reply(graph, reply)
    for field in ["value", "scope", "unit", "qualifier_refs"]:
        reply = properties();reply["properties"][0][field] = 99
        with pytest.raises(ValueError, match="Property schema"):
            merge_property_reply(graph, reply)


def test_missing_property_processing_is_not_an_empty_property_set():
    graph = project_basic_reply(basic(), "A observed question")
    reply = properties();reply["properties"][0].pop("status")
    with pytest.raises(ValueError, match="Property schema"):
        merge_property_reply(graph, reply)
    with pytest.raises(ValueError, match="requires a property"):
        merge_property_reply(graph, properties("extracted"))


def test_property_evidence_is_local_and_negative_is_preserved():
    graph = project_basic_reply(basic(), "A observed question")
    extra = {"concept": "木制", "core": "设备", "relation": "材料", "nature": "木制", "polarity": "negative", "source_span": "observed"}
    result = merge_property_reply(graph, properties("extracted", extra=[extra]))
    assert result["spans"][1]["semantic_relations"][-1]["polarity"] == "negative"
    extra["source_span"] = "unrelated"
    with pytest.raises(ValueError, match="own fact source"):
        merge_property_reply(graph, properties("extracted", extra=[extra]))


def test_basic_stage_does_not_accept_domain_feature_keys_or_qualifier_ids():
    d = basic();d["facts"][0]["feature"] = "quote_amount"
    with pytest.raises(ValueError, match="Basic schema"):
        project_basic_reply(d, "A observed question")
    assert "vocabulary" not in render_basic_request("A observed question")


def test_boolean_old_policy_scope_and_source_coverage():
    d = basic();d["facts"][0].update(value=False, value_type="boolean", polarity="negative")
    with pytest.raises(ValueError, match="duplicate negation"):
        project_basic_reply(d, "A observed question")
    d["facts"][0]["polarity"] = "positive"
    assert project_basic_reply(d, "A observed question")["spans"][1]["value"] == 0
    d["facts"][0]["scope"]["policy_version"] = "current"
    with pytest.raises(ValueError, match="scope.policy_version"):
        project_basic_reply(d, "A old observed question")
    with pytest.raises(ValueError, match="Unrepresented"):
        project_basic_reply(basic(), "A observed question\nold record")


def test_targeted_property_refinement_uses_selected_source_not_requested_answer():
    graph = project_basic_reply(basic(), "A observed question")
    prompts = []
    class OfflineTransport:
        def chat_json(self, prompt):
            prompts.append(prompt);return properties()
    frontend = SourceTranslationFrontend(OfflineTransport())
    result = frontend.patch_event_graph(game_input={"input": "Do not replay full text"}, current_event_graph=graph,
        patches=[{"target_span_id": "observation_0", "expected_answer": "not_for_llm"}])
    assert len(prompts) == 1 and "not_for_llm" not in prompts[0] and "Do not replay full text" not in prompts[0]
    assert result["spans"][1]["value"] == 3 and result["spans"][-1] == graph["spans"][-1]


def test_vector_identity_is_unchanged_by_debug_text_and_rejects_ambiguity():
    # Tiny vectors test comparison logic only; real encoding is tested in the gate.
    record = {"id": "k1", "status": "active", "embedding_model": "test", "vector_version": "v1",
        "debug": {"core": "misleading text"}, "metadata": {"semantic_definition": {"kind": "observation", "recognition_vector": [1., 0.]}}}
    record.update({field: [1.,0.] for field in ("core_vector", "relation_vector", "nature_vector")})
    policy = {"minimum_score": .9, "minimum_margin": .025, "component_minimum_score": .8}
    chosen, audit = _choose([1., 0.], [record], "observation", policy, "test", "v1", record)
    assert chosen["id"] == "k1"
    record["debug"]["core"] = "a different domain"
    assert _choose([1., 0.], [record], "observation", policy, "test", "v1", record)[0]["id"] == "k1"
    other = deepcopy(record);other["id"] = "k2"
    assert _choose([1., 0.], [record, other], "observation", policy, "test", "v1", record)[0] is None
    assert _choose([1., 0.], [record], "observation", policy, "different", "v1", record)[0] is None
    incompatible = {**record, "nature_vector": [0.,1.]}
    assert _choose([1., 0.], [record], "observation", policy, "test", "v1", incompatible)[0] is None


def test_negative_semantic_relation_does_not_prove_positive_require():
    edge = {"core_vector": [1.,0.], "relation_vector": [1.,0.], "nature_vector": [1.,0.],
            "embedding_model": "test", "vector_version": "v1", "debug": {"core": "material", "relation": "type", "nature": "wooden"}}
    span = {"semantic_relations": [{**edge, "polarity": "negative"}]}
    assert has_required_subgraph(span, [edge])[0] is False
    span["semantic_relations"].append({**edge, "polarity": "positive"})
    assert len(normalize_semantic_relations(span)) == 2
    assert has_required_subgraph(span, [edge])[0] is True


def test_frontend_bridge_preserves_source_semantics_and_identity_vector():
    from main_part.data_structure import WorkingMemoryState
    from main_part.frontend_working_memory_bridge import submit_frontend_state_to_working_memory
    graph = project_basic_reply(basic(), "A observed question")
    graph["spans"][1]["concept_vector"] = [1.,0.]
    memory = WorkingMemoryState()
    submit_frontend_state_to_working_memory(memory, game_input={"input":"A observed question"}, frontend_state=graph)
    delivered = memory.event_workspaces["case"]["spans"][1]
    for key in ("entity_id", "semantic_concept", "value_type", "polarity", "modality", "concept_vector"):
        assert delivered[key] == graph["spans"][1][key]


def test_generic_format_example_is_valid_and_quoted_numbers_rejected():
    from jsonschema import Draft202012Validator
    from semantic_frontend.source_contract import BASIC_SCHEMA, FORMAT_EXAMPLE
    assert not list(Draft202012Validator(BASIC_SCHEMA).iter_errors(FORMAT_EXAMPLE))
    d = basic();d["facts"][0]["value"] = "3"
    with pytest.raises(ValueError, match="not of type 'number'"):
        project_basic_reply(d, "A observed question")
