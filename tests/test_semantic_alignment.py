"""Vector-only semantic alignment checks."""

from main_part.data_structure import WorkingMemoryState
from main_part.matcher.matching import match_required_bundle_against_graph
from main_part.matcher.routing import (
    graph_bundle_from_spec,
    normalize_required_pattern,
    required_bundle_from_rule,
)
from main_part.scheduler_new import _alignment_requirements
from main_part.semantic_alignment import (
    align_event_spans,
    select_requirements,
    select_span_schema_knowledge_ids,
)


def test_alignment_target_selection_prefers_rules_that_reach_current_goal():
    requirements = [
        {
            "rule_id": "unrelated_nearly_complete",
            "matched_slot_count": 2,
            "total_slot_count": 3,
            "goal_reachable": False,
            "goal_alignment": False,
        },
        {
            "rule_id": "goal_rule_with_more_missing_inputs",
            "matched_slot_count": 0,
            "total_slot_count": 2,
            "goal_reachable": True,
            "goal_alignment": True,
            "goal_distance": 0,
        },
    ]
    selected = select_requirements(
        requirements,
        {"minimum_coverage": 0.0, "max_rules": 1},
    )
    assert [item["rule_id"] for item in selected] == [
        "goal_rule_with_more_missing_inputs"
    ]
from main_part.semantic_subgraph import normalize_semantic_relations
from main_part.executor_new import execute_rule_chain


MODEL, VERSION = "numerical-test", "v1"
CORE, RELATION, NATURE = [1.0, 0.0], [0.0, 1.0], [0.7, 0.7]
PROJECTED = [0.6, 0.8]
OTHER = [-1.0, 0.0]


def _rule():
    return {"rule_id": "average_rule", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "needed", "feature_type": "avg_ar", "core_vector": PROJECTED,
        "relation_vector": PROJECTED, "nature_vector": PROJECTED, "match_threshold": .99,
    }]}]}


def _match(memory, rule):
    return match_required_bundle_against_graph(
        required_bundle_from_rule(rule, {}), graph_bundle_from_spec({"type": "event_workspace"}, {}, memory), {}
    )


def _schema_context(*, activation_mode="explicit_components", edge_role="evidence", core=CORE, relation=RELATION, nature=NATURE):
    return {"status": "active", "payload": {
        "knowledge_id": f"schema_{edge_role}", "retrieval_source": "knowledge_db",
        "knowledge_edge": {"edge_id": f"schema_{edge_role}", "core_vector": core, "relation_vector": relation,
                           "nature_vector": nature, "embedding_model": MODEL, "vector_version": VERSION,
                           "vector_dimension": 2},
        "span_schema": {"schema_id": "knowledge_span_avg_ar_v1", "target_feature_type": "avg_ar",
                        "activation_mode": activation_mode, "edge_role": edge_role, "min_score": .99,
                        "completion": {"core_vector": PROJECTED, "relation_vector": PROJECTED,
                                       "nature_vector": PROJECTED}},
    }}


def _memory(nature=NATURE):
    memory = WorkingMemoryState()
    source = {"span_id": "s", "feature_type": "unresolved_receivable", "value": 120, "unit": "万元",
              "core_vector": CORE, "relation_vector": RELATION, "nature_vector": NATURE,
              "embedding_model": MODEL, "vector_version": VERSION,
              "semantic_relations": [{"debug": {"core": "accounts_receivable", "relation": "aggregation", "nature": "average"},
                                      "core_vector": CORE, "relation_vector": RELATION, "nature_vector": nature,
                                      "embedding_model": MODEL, "vector_version": VERSION}]}
    memory.event_workspaces["e"] = {"event_id": "e", "version": 0, "spans": [source], "relations": [], "context_items": [_schema_context()]}
    return memory


def test_alignment_requires_vector_subgraph_and_strict_rematch():
    memory, rule = _memory(), _rule()
    requirements = _alignment_requirements(_match(memory, rule))
    result = align_event_spans(memory, requirements, ["schema_evidence"], control_rule_id="control")
    assert result["accepted"]
    derived = memory.event_workspaces["e"]["spans"][-1]
    assert derived["value"] == 120 and derived["unit"] == "万元"
    assert derived["metadata"]["semantic_alignment"]["semantic_component_evidence"]
    assert _match(memory, rule)["ok"]


def test_alignment_rejects_different_vector_component_even_with_same_debug_label():
    memory, rule = _memory(nature=[-1.0, 0.0]), _rule()
    result = align_event_spans(memory, _alignment_requirements(_match(memory, rule)), ["prototype_avg"], control_rule_id="control")
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "no_supported_alignment"


def test_frontend_relation_has_no_text_derived_identity_before_vectorization():
    raw = {"semantic_relations": [{"core": "same label", "relation": "means", "nature": "same value"}]}
    relation = normalize_semantic_relations(raw)[0]
    assert relation["debug"]["core"] == "same label"
    assert "core_node_id" not in relation


def test_persistent_node_identity_is_vector_contract_not_debug_text(tmp_path):
    from main_part.persistence.knowledge_db import load_knowledge_network, upsert_knowledge_records
    path = tmp_path / "knowledge.sqlite"
    rows = [
        {"id": "one", "debug": {"core": "first wording", "relation": "r", "nature": "n"}, "core_vector": CORE, "relation_vector": RELATION, "nature_vector": NATURE, "embedding_model": MODEL, "vector_version": VERSION},
        {"id": "two", "debug": {"core": "different wording", "relation": "r2", "nature": "n2"}, "core_vector": CORE, "relation_vector": RELATION, "nature_vector": PROJECTED, "embedding_model": MODEL, "vector_version": VERSION},
    ]
    result = upsert_knowledge_records(path, rows)
    graph = load_knowledge_network(path)
    assert result["knowledge_graph"]["edges"] == 2
    assert len({edge["core_node_id"] for edge in graph["edges"]}) == 1
    assert "knowledge_spans" not in {row[0] for row in __import__("sqlite3").connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_knowledge_all_cores_requires_every_declared_core():
    memory, rule = _memory(), _rule()
    memory.event_workspaces["e"]["spans"].append({
        "span_id": "qualifier", "core_vector": PROJECTED,
        "embedding_model": MODEL, "vector_version": VERSION, "semantic_relations": [],
    })
    memory.event_workspaces["e"]["context_items"] = [
        _schema_context(activation_mode="all_cores", edge_role="activation_core", core=CORE),
        _schema_context(activation_mode="all_cores", edge_role="activation_core", core=PROJECTED),
    ]
    result = align_event_spans(memory, _alignment_requirements(_match(memory, rule)), ["schema_activation_core"], control_rule_id="control")
    assert result["accepted"]
    derived = memory.event_workspaces["e"]["spans"][-1]
    assert derived["metadata"]["semantic_alignment"]["activation_mode"] == "all_cores"
    assert len(derived["metadata"]["semantic_alignment"]["semantic_component_evidence"]) == 2

    incomplete, rule = _memory(), _rule()
    incomplete.event_workspaces["e"]["context_items"] = [
        _schema_context(activation_mode="all_cores", edge_role="activation_core", core=CORE),
        _schema_context(activation_mode="all_cores", edge_role="activation_core", core=PROJECTED),
    ]
    rejected = align_event_spans(incomplete, _alignment_requirements(_match(incomplete, rule)), ["schema_activation_core"], control_rule_id="control")
    assert not rejected["accepted"]
    assert rejected["rejected"][0]["reason"] == "no_supported_alignment"


def test_knowledge_single_core_can_complete_a_declared_single_core_definition():
    memory, rule = _memory(), _rule()
    memory.event_workspaces["e"]["context_items"] = [
        _schema_context(activation_mode="single_core", edge_role="activation_core", core=CORE),
    ]
    result = align_event_spans(memory, _alignment_requirements(_match(memory, rule)), ["schema_activation_core"], control_rule_id="control")
    assert result["accepted"]
    assert memory.event_workspaces["e"]["spans"][-1]["metadata"]["semantic_alignment"]["activation_mode"] == "single_core"


def _relation_context_schema(schema_id, target_type, completion, edge_role, core, relation=RELATION):
    return {"status": "active", "payload": {
        "knowledge_id": f"{schema_id}_{edge_role}", "retrieval_source": "knowledge_db",
        "knowledge_edge": {
            "edge_id": f"{schema_id}_{edge_role}", "core_vector": core,
            "relation_vector": relation, "nature_vector": NATURE,
            "embedding_model": MODEL, "vector_version": VERSION, "vector_dimension": 2,
        },
        "span_schema": {
            "schema_id": schema_id, "target_feature_type": target_type,
            "activation_mode": "relation_context", "edge_role": edge_role,
            "min_score": .99, "completion": {
                "core_vector": completion, "relation_vector": completion, "nature_vector": completion,
            },
        },
    }}


def _ambiguous_turnover_memory(context_core):
    memory = WorkingMemoryState()
    memory.event_workspaces["e"] = {
        "event_id": "e", "version": 0, "relations": [],
        "spans": [{
            "span_id": "turnover", "feature_type": "unresolved_turnover",
            "core_vector": CORE, "relation_vector": RELATION, "nature_vector": OTHER,
            "embedding_model": MODEL, "vector_version": VERSION,
            "semantic_relations": [
                {"debug": {"core": "turnover", "relation": "measured_for", "nature": "unresolved"},
                 "core_vector": CORE, "relation_vector": RELATION, "nature_vector": OTHER,
                 "embedding_model": MODEL, "vector_version": VERSION},
                {"debug": {"core": "context", "relation": "describes", "nature": "entity"},
                 "core_vector": context_core, "relation_vector": OTHER, "nature_vector": NATURE,
                 "embedding_model": MODEL, "vector_version": VERSION},
            ],
        }],
        "context_items": [
            _relation_context_schema("inventory_sense", "inventory_turnover", PROJECTED, "activation_relation", CORE),
            _relation_context_schema("inventory_sense", "inventory_turnover", PROJECTED, "context_core", [0.2, 0.98]),
            _relation_context_schema("employee_sense", "employee_turnover", NATURE, "activation_relation", CORE),
            _relation_context_schema("employee_sense", "employee_turnover", NATURE, "context_core", [-0.2, 0.98]),
        ],
    }
    return memory


def test_relation_and_adjacent_core_select_nature_for_same_core():
    memory = _ambiguous_turnover_memory([0.2, 0.98])
    rule = {"rule_id": "inventory_rule", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "inventory_turnover", "feature_type": "inventory_turnover",
        "core_vector": PROJECTED, "relation_vector": PROJECTED,
        "nature_vector": PROJECTED, "match_threshold": .99,
    }]}]}
    before = _match(memory, rule)
    result = align_event_spans(memory, _alignment_requirements(before), None, control_rule_id="control")
    assert result["accepted"]
    assert memory.event_workspaces["e"]["spans"][-1]["feature_type"] == "inventory_turnover"
    roles = {
        item["evidence_role"]
        for item in result["accepted"][0]["semantic_component_evidence"]
    }
    assert roles == {"activation_relation", "context_core"}
    assert _match(memory, rule)["ok"]


def test_relation_context_accepts_primary_span_triple_without_duplicate_relation():
    memory = _ambiguous_turnover_memory([0.2, 0.98])
    # Public frontend graphs keep the primary triple on the span and only put
    # the adjacent context triple in semantic_relations.
    memory.event_workspaces["e"]["spans"][0]["semantic_relations"] = [
        memory.event_workspaces["e"]["spans"][0]["semantic_relations"][1]
    ]
    rule = {"rule_id": "inventory_rule", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "inventory_turnover", "feature_type": "inventory_turnover",
        "core_vector": PROJECTED, "relation_vector": PROJECTED,
        "nature_vector": PROJECTED, "match_threshold": .99,
    }]}]}
    result = align_event_spans(
        memory, _alignment_requirements(_match(memory, rule)), None,
        control_rule_id="control",
    )
    assert result["accepted"]
    evidence = result["accepted"][0]["semantic_component_evidence"]
    primary = next(item for item in evidence if item["evidence_role"] == "activation_relation")
    assert primary["relation_index"] is None
    assert _match(memory, rule)["ok"]


def test_same_core_does_not_cross_to_other_nature_without_its_adjacent_core():
    memory = _ambiguous_turnover_memory([0.2, 0.98])
    rule = {"rule_id": "employee_rule", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "employee_turnover", "feature_type": "employee_turnover",
        "core_vector": NATURE, "relation_vector": NATURE,
        "nature_vector": NATURE, "match_threshold": .99,
    }]}]}
    result = align_event_spans(
        memory, _alignment_requirements(_match(memory, rule)), None, control_rule_id="control"
    )
    assert not result["accepted"]
    assert result["rejected"][0]["reason"] == "no_supported_alignment"


def test_control_knowledge_recovery_then_strict_two_rule_chain():
    """Control proves a missing require, then Rules execute one at a time."""
    memory, first_rule = _memory(), _rule()
    first_rule["produced"] = [{"operation": "add", "event_id": "e", "conclusion": [{
        "span_id": "intermediate", "feature_type": "intermediate",
        "core_vector": [0.2, 0.98], "relation_vector": [0.2, 0.98], "nature_vector": [0.2, 0.98],
        "embedding_model": MODEL, "vector_version": VERSION, "value": 42, "unit": "u",
    }]}]
    memory.event_workspaces["e"]["context_items"] = [_schema_context()]

    initial_match = _match(memory, first_rule)
    assert not initial_match["ok"]
    requirements = _alignment_requirements(initial_match)
    control_rule = {"rule_id": "control_alignment", "rule_layer": "control", "produced": [{
        "operation": "control_decide", "default": [{
            "operation": "align_event_spans", "knowledge_ids": ["schema_evidence"],
        }],
    }]}
    control = execute_rule_chain(
        matched_rules=[{"rule": control_rule, "match": {"ok": True}}],
        working_memory=memory,
        session_id="knowledge-progressive-test",
        control_context={"trigger": {"alignment_requirements": requirements}},
    )
    assert control["next_rule_ids"] == ["average_rule"]
    strict_first = _match(memory, first_rule)
    assert strict_first["ok"]
    first_execution = execute_rule_chain(
        matched_rules=[{"rule": first_rule, "match": strict_first}],
        working_memory=memory, session_id="knowledge-progressive-test",
    )
    assert first_execution["ok"]

    second_rule = {"rule_id": "final_rule", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "intermediate", "feature_type": "intermediate",
        "core_vector": [0.2, 0.98], "relation_vector": [0.2, 0.98], "nature_vector": [0.2, 0.98],
        "match_threshold": .99,
    }]}], "produced": [{"operation": "add", "event_id": "e", "conclusion": [{
        "span_id": "final", "feature_type": "final", "value": 42,
    }]}]}
    strict_second = match_required_bundle_against_graph(
        required_bundle_from_rule(second_rule, {}),
        graph_bundle_from_spec({"type": "event_workspace"}, {}, memory), {},
    )
    assert strict_second["ok"]
    second_execution = execute_rule_chain(
        matched_rules=[{"rule": second_rule, "match": strict_second}],
        working_memory=memory, session_id="knowledge-progressive-test",
    )
    assert second_execution["ok"]
    assert memory.event_workspaces["e"]["spans"][-1]["span_id"] == "final"


def test_control_switches_to_strict_backup_when_knowledge_cannot_prove_require():
    memory, blocked_rule = _memory(), _rule()
    memory.event_workspaces["e"]["context_items"] = [_schema_context(nature=OTHER)]
    requirements = _alignment_requirements(_match(memory, blocked_rule))
    control_rule = {"rule_id": "control_alignment", "rule_layer": "control", "produced": [{
        "operation": "control_decide", "default": [{
            "operation": "align_event_spans", "knowledge_ids": ["schema_evidence"],
            "on_unresolved": [{"operation": "run_rule", "rule_ids": ["backup_raw"]}],
        }],
    }]}
    recovery = execute_rule_chain(
        matched_rules=[{"rule": control_rule, "match": {"ok": True}}],
        working_memory=memory, session_id="knowledge-fallback-test",
        control_context={"trigger": {"alignment_requirements": requirements}},
    )
    assert recovery["next_rule_ids"] == ["backup_raw"]
    assert [span["span_id"] for span in memory.event_workspaces["e"]["spans"]] == ["s"]
    backup = {"rule_id": "backup_raw", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "raw", "feature_type": "unresolved_receivable",
        "core_vector": CORE, "relation_vector": RELATION, "nature_vector": NATURE, "match_threshold": .99,
    }]}]}
    assert _match(memory, backup)["ok"]


def test_schema_retrieval_is_selected_from_required_vectors_not_database_ids():
    required = {
        "feature_type": "avg_ar", "core_vector": PROJECTED,
        "relation_vector": PROJECTED, "nature_vector": PROJECTED,
    }
    matching = {
        "id": "schema_matching", "embedding_model": MODEL, "vector_version": VERSION,
        "metadata": {"span_schema": {"target_feature_type": "avg_ar", "completion": {
            "core_vector": PROJECTED, "relation_vector": PROJECTED, "nature_vector": PROJECTED,
        }}},
    }
    wrong_target = {**matching, "id": "schema_other", "metadata": {"span_schema": {
        "target_feature_type": "sales", "completion": matching["metadata"]["span_schema"]["completion"],
    }}}
    wrong_vectors = {**matching, "id": "schema_wrong_vector", "metadata": {"span_schema": {
        "target_feature_type": "avg_ar", "completion": {
            "core_vector": CORE, "relation_vector": RELATION, "nature_vector": NATURE,
        },
    }}}
    assert select_span_schema_knowledge_ids(
        [matching, wrong_target, wrong_vectors], [{"slot": {"alignment_required": required}}]
    ) == ["schema_matching"]


def test_schema_lookup_keeps_every_edge_of_a_matching_composite_definition():
    required = {"feature_type": "avg_ar", "core_vector": PROJECTED,
                "relation_vector": PROJECTED, "nature_vector": PROJECTED}
    completion = {"core_vector": PROJECTED, "relation_vector": PROJECTED, "nature_vector": PROJECTED}
    records = [
        {"id": identifier, "embedding_model": MODEL, "vector_version": VERSION,
         "metadata": {"span_schema": {"schema_id": "composite", "target_feature_type": "avg_ar", "completion": completion}}}
        for identifier in ("composite_edge_one", "composite_edge_two")
    ]
    assert select_span_schema_knowledge_ids(records, [{"slot": {"alignment_required": required}}]) == [
        "composite_edge_one", "composite_edge_two"
    ]


def test_rule_evidence_constraints_survive_routing_and_block_weak_span():
    required = {
        "condition": [{
            "span_id": "caveat",
            "feature_type": "no_trend_caveat",
            "core_vector": CORE,
            "min_source_reliability": 0.90,
            "require_authority_verified": True,
        }]
    }
    pattern = normalize_required_pattern(required)
    slot = pattern["span_slots"][0]
    assert slot["min_source_reliability"] == 0.90
    assert slot["require_authority_verified"] is True

    candidate = {
        "span_id": "caveat",
        "feature_type": "no_trend_caveat",
        "core_vector": CORE,
        "metadata": {"source_reliability": 0.45, "authority_verified": False},
    }
    memory = WorkingMemoryState()
    memory.event_workspaces["e"] = {"event_id": "e", "spans": [candidate], "relations": []}
    rule = {"rule_id": "requires_verified_caveat", "rule_layer": "domain", "required": [required]}
    result = match_required_bundle_against_graph(
        required_bundle_from_rule(rule, {}),
        graph_bundle_from_spec({"type": "event_workspace"}, {}, memory),
        {},
    )
    assert not result["ok"]
    slot_match = result["metadata"]["per_required"][0]["details"]["slot_matches"][0]
    assert slot_match["vector_status"] == "source_reliability_below_required_minimum"


def test_rule_requires_shared_vectorized_measurement_context():
    def amount(span_id, nature):
        return {
            "span_id": span_id,
            "feature_type": span_id,
            "core_vector": CORE,
            "relation_vector": RELATION,
            "nature_vector": NATURE,
            "embedding_model": MODEL,
            "vector_version": VERSION,
            "semantic_relations": [{
                "core": "measurement context", "relation": "denominated in", "nature": "debug only",
                "core_vector": CORE, "relation_vector": RELATION, "nature_vector": nature,
                "embedding_model": MODEL, "vector_version": VERSION,
            }],
        }

    required = {
        "condition": [
            {"span_id": "begin", "feature_type": "begin", "core_vector": CORE},
            {"span_id": "end", "feature_type": "end", "core_vector": CORE},
        ],
        "shared_semantic_relation_constraints": [{
            "slot_ids": ["begin", "end"],
            "core_vector": CORE,
            "relation_vector": RELATION,
            "match_threshold": .99,
        }],
    }
    rule = {"rule_id": "same_context", "rule_layer": "domain", "required": [required]}

    matching = WorkingMemoryState()
    matching.event_workspaces["e"] = {
        "event_id": "e", "spans": [amount("begin", NATURE), amount("end", NATURE)], "relations": [],
    }
    assert match_required_bundle_against_graph(
        required_bundle_from_rule(rule, {}), graph_bundle_from_spec({"type": "event_workspace"}, {}, matching), {}
    )["ok"]

    mismatched = WorkingMemoryState()
    mismatched.event_workspaces["e"] = {
        "event_id": "e", "spans": [amount("begin", NATURE), amount("end", OTHER)], "relations": [],
    }
    result = match_required_bundle_against_graph(
        required_bundle_from_rule(rule, {}), graph_bundle_from_spec({"type": "event_workspace"}, {}, mismatched), {}
    )
    assert not result["ok"]
    assert result["metadata"]["per_required"][0]["details"]["shared_semantic_relation_constraint_matches"][0]["matched"] is False


def test_inactive_control_state_is_not_matchable_but_remains_in_event_history():
    state = {
        "span_id": "old_strategy",
        "span_role": "control_state",
        "feature_type": "strategy_direct_endpoints",
        "core_vector": CORE,
        "metadata": {"control_state_active": False},
    }
    memory = WorkingMemoryState()
    memory.event_workspaces["e"] = {"event_id": "e", "spans": [state], "relations": []}
    rule = {
        "rule_id": "old_route",
        "rule_layer": "control",
        "required": [{"condition": [{
            "span_id": "strategy_direct_endpoints",
            "feature_type": "strategy_direct_endpoints",
            "core_vector": CORE,
        }]}],
    }
    result = match_required_bundle_against_graph(
        required_bundle_from_rule(rule, {}),
        graph_bundle_from_spec({"type": "event_workspace"}, {}, memory),
        {},
    )
    assert not result["ok"]
    assert memory.event_workspaces["e"]["spans"][0]["metadata"]["control_state_active"] is False


def test_retrieved_memory_cannot_supply_current_rule_evidence():
    memory = WorkingMemoryState()
    memory.event_workspaces["e"] = {
        "event_id": "e",
        "spans": [],
        "relations": [],
        "context_items": [{
            "context_id": "past_case",
            "source": "memory_db",
            "payload": {
                "retrieval_source": "memory_db",
                "event_hypergraph": {"spans": [{
                    "span_id": "historical_sales",
                    "feature_type": "sales",
                    "core_vector": CORE,
                }]},
            },
        }],
    }
    rule = {"rule_id": "requires_current_sales", "rule_layer": "domain", "required": [{"condition": [{
        "span_id": "sales", "feature_type": "sales", "core_vector": CORE,
    }]}]}
    result = match_required_bundle_against_graph(
        required_bundle_from_rule(rule, {}),
        graph_bundle_from_spec({"type": "event_workspace"}, {}, memory),
        {},
    )
    assert not result["ok"]
