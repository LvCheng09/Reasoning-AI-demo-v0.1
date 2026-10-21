"""Executable contract tests. Basis vectors test matcher geometry, not an encoder."""
from copy import deepcopy
import json
import math

import pytest

from main_part.activation_semantics import ActivationError, evaluate_activation, validate_activation
from main_part.calculator import Calculator
from main_part.data_structure import ReasoningRule, WorkingMemoryState
from main_part.executor_new import execute_rule_chain
from main_part.matcher.matching import match_rules_against_graph
from main_part.matcher.routing import graph_bundle_from_spec
from main_part.persistence import rule_db
from main_part.rule_execution_contract import validate_rule_contract
from main_part.rule_execution_runtime import RuleExecutionRuntime
from main_part.scheduler_new import SchedulerNew


def signal(value, domain="real"):
    return {"domain": domain, "source": {"kind": "constant", "value": value}}


def graph_memory(value=3):
    memory = WorkingMemoryState()
    memory.event_workspaces["e"] = {"event_id": "e", "version": 0, "relations": [], "spans": [{
        "span_id": "input", "feature_type": "x", "value": value, "unit": "",
        "core_vector": [1., 0.], "relation_vector": [1., 0.], "nature_vector": [1., 0.],
        "embedding_model": "numerical-test", "vector_version": "v1"}]}
    return memory


def typed_rule(*, partial=False, control=False):
    rule = {"rule_id": "compute", "rule_layer": "domain", "execution_contract_version": 2,
        "required": [{"required_id": "input", "condition": [{"span_id": "x", "feature_type": "x",
            "core_vector": [1., 0.], "relation_vector": [1., 0.], "nature_vector": [1., 0.],
            "embedding_model": "numerical-test", "vector_version": "v1", "match_threshold": .99}]}],
        "produced": [{"operation": "add", "conclusion": [{"span_id": "answer", "feature_type": "answer",
            "expression": {"operator": "multiply", "operands": ["x", 2]}}]}],
        "require_policy": {"partial_mode": "disabled"},
        "activation_spec": {"primary": "response", "signals": {"response": {
            "domain": "real", "source": {"kind": "input", "slot": "x"}}}},
        "q_semantics": {}, "io_contract": {"required_slots": ["x"], "outputs": ["answer"]},
        "effect_contract": {"kind": "graph", "write_targets": ["spans"]}}
    if partial:
        rule["required"].append({"required_id": "optional", "condition": [{"span_id": "optional",
            "feature_type": "optional",
            "core_vector": [-1., 0.], "relation_vector": [-1., 0.], "nature_vector": [-1., 0.],
            "embedding_model": "numerical-test", "vector_version": "v1", "match_threshold": .99}]})
        rule["require_policy"] = ({"partial_mode": "control"} if control else
                                  {"partial_mode": "declared", "execution_mode": "local"})
        rule["io_contract"]["partial_modes"] = {"local": {"required_ids": ["input"], "required_slots": ["x"],
            "produced_indices": [0], "outputs": ["answer"], "result_scope": "partial"}}
    return rule


def match_operation(memory, rule):
    graph = graph_bundle_from_spec({"type": "event_workspace", "event_ids": ["e"]}, None, memory)
    result = match_rules_against_graph([rule], graph, {}, target_spec={"type": "rules"}, routing_metadata={})
    return result["produced"][0]


def actual_value(memory):
    return next(s for s in memory.event_workspaces["e"]["spans"] if s["span_id"] == "answer")


def test_activation_is_a_signed_vector_computation_graph():
    spec = {"primary": "signed", "signals": {
        "input": signal([2., -3.], "vector"),
        "weights": signal([[1., 2.], [-2., 1.]], "matrix"),
        "projection": {"domain": "vector", "size": 2, "source": {"kind": "operation", "op": "matmul",
            "inputs": [{"signal": "weights"}, {"signal": "input"}]}},
        "signed": {"domain": "real", "source": {"kind": "operation", "op": "dot",
            "inputs": [{"signal": "projection"}, {"value": [1., -1.]}]}},
        "probability": {"domain": "unit_interval", "source": {"kind": "operation", "op": "sigmoid",
            "inputs": [{"signal": "signed"}]}}}}
    validate_activation(spec, {})
    result = evaluate_activation(spec, workspace={}, bindings={}, context={}, require_degree=.5, calculator=Calculator())
    assert result["signals"]["projection"]["value"] == [-4., -7.]
    assert result["value"] == 3.
    assert result["signals"]["probability"]["value"] == pytest.approx(1 / (1 + math.exp(-3)))


def test_activation_calls_an_actual_registered_function_and_preserves_provenance():
    calculator = Calculator()
    calculator.register("response", "v1", lambda inputs, request: {
        "outputs": {"signal": {"value": inputs["x"]["value"] ** 2 - 10, "unit": ""}}})
    spec = {"primary": "a", "signals": {"a": {"domain": "real", "source": {
        "kind": "function", "request": {"schema_version": 1, "function_id": "response", "function_version": "v1",
            "inputs": {"x": {"slot": "x"}}}, "output_name": "signal"}}}}
    result = evaluate_activation(spec, workspace=graph_memory().event_workspaces["e"], bindings={"x": "input"},
                                 context={}, require_degree=1, calculator=calculator)
    assert result["value"] == -1
    assert result["signals"]["a"]["computation"]["function_version"] == "v1"


@pytest.mark.parametrize("problem", ["cycle", "nan", "unknown_operation", "shape"])
def test_invalid_activation_is_visible(problem):
    spec = {"primary": "a", "signals": {"a": signal(1)}}
    if problem == "cycle":
        spec["signals"]["a"]["source"] = {"kind": "operation", "op": "add", "inputs": [{"signal": "a"}]}
    elif problem == "nan":
        spec["signals"]["a"] = signal(float("nan"))
    elif problem == "shape":
        spec["signals"]["a"] = {"domain": "vector", "size": 3, "source": {"kind": "constant", "value": [1, 2]}}
    else:
        spec["signals"]["a"]["source"] = {"kind": "operation", "op": "ignored", "inputs": []}
    with pytest.raises(ActivationError):
        validate_activation(spec, {})
        evaluate_activation(spec, workspace={}, bindings={}, context={}, require_degree=1, calculator=Calculator())


@pytest.mark.parametrize("caller", ["direct", "apply", "default", "path"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("effect,factor,expected", [("signed_effect", -.5, -3), ("intensity", .5, 3)])
def test_actual_match_activation_and_output_in_every_execution_entry(caller, partial, effect, factor, expected):
    memory, rule = graph_memory(), typed_rule(partial=partial)
    rule["activation_spec"]["signals"]["factor"] = signal(factor)
    rule["q_semantics"] = {"transform": [{"id": "effect", "type": effect, "source": {"signal": "factor"},
        "target": {"span_id": "answer", "field": "value"}}]}
    operation = match_operation(memory, rule)
    candidates = operation["partial_rules"] if partial else operation["matched_rules"]
    assert len(candidates) == 1
    assert candidates[0]["match"]["ok"] is not partial
    if caller == "direct":
        result = execute_rule_chain(matched_rules=candidates, working_memory=memory, session_id="test")
        assert result["ok"], result
    else:
        scheduler = SchedulerNew(memory, session_id="test")
        if caller == "apply":
            scheduler._apply_produced_operations([operation])
        else:
            if caller == "path":
                operation = {"operation": "execute_candidate_path", "candidate_path": {
                    "matched_rules": candidates, "event_ids": ["e"]}}
            scheduler._default_executor({"session_id": "test"}, memory, {"operations": [operation]})
    output = actual_value(memory)
    assert output["value"] == expected
    assert output["execution_scope"] == ("partial" if partial else "full")
    decision = memory.rule_runtime["test"]["decisions"][output["rule_execution"]["decision_id"]]
    assert decision["activation"]["value"] == 3
    assert decision["require_degree"] < 1 if partial else decision["require_degree"] == 1
    assert decision["status"] == "completed"


def test_partial_cannot_execute_without_its_mandatory_input():
    memory, rule = graph_memory(), typed_rule(partial=True)
    rule["io_contract"]["partial_modes"]["local"]["required_ids"] = ["optional"]
    operation = match_operation(memory, rule)
    assert not operation["matched_rules"] and not operation["partial_rules"]
    assert len(operation["rejected_rules"]) == 1
    forged = {"rule": rule, "match": {"ok": True}, "execution_authorization": {"approved": True}}
    result = execute_rule_chain(matched_rules=[forged], working_memory=memory, session_id="test")
    assert not result["ok"]
    assert len(memory.event_workspaces["e"]["spans"]) == 1


def test_state_changes_are_rechecked_instead_of_trusting_match_ok():
    memory, rule = graph_memory(), typed_rule()
    candidates = match_operation(memory, rule)["matched_rules"]
    memory.event_workspaces["e"]["spans"][0]["status"] = "retracted"
    result = execute_rule_chain(matched_rules=candidates, working_memory=memory, session_id="test")
    assert not result["ok"]
    assert len(memory.event_workspaces["e"]["spans"]) == 1


@pytest.mark.parametrize("probability,expected", [(0., False), (1., True)])
def test_probability_gates_actual_produced(probability, expected):
    memory, rule = graph_memory(), typed_rule()
    rule["activation_spec"]["signals"]["p"] = signal(probability, "unit_interval")
    rule["q_semantics"] = {"dispatch": [{"id": "sample", "type": "probability", "source": {"signal": "p"}}]}
    result = execute_rule_chain(matched_rules=match_operation(memory, rule)["matched_rules"], working_memory=memory, session_id="test")
    assert result["ok"]
    assert any(s["span_id"] == "answer" for s in memory.event_workspaces["e"]["spans"]) == expected


def test_probability_and_completed_decision_survive_json_restore():
    memory, rule = graph_memory(), typed_rule()
    rule["activation_spec"]["signals"]["p"] = signal(.5, "unit_interval")
    rule["q_semantics"] = {"dispatch": [{"id": "sample", "type": "probability", "source": {"signal": "p"}}]}
    item = match_operation(memory, rule)["matched_rules"][0]
    runtime = RuleExecutionRuntime(memory, Calculator(), session_id="test")
    _, first = runtime.admit(item)
    snapshot = json.loads(json.dumps(memory.to_dict()))
    restored = WorkingMemoryState(event_workspaces=snapshot["event_workspaces"], rule_runtime=snapshot["rule_runtime"])
    _, second = RuleExecutionRuntime(restored, Calculator(), session_id="test").admit(item)
    assert first["samples"] == second["samples"]
    assert first["status"] == second["status"]


def test_frequency_defers_and_resumes_real_produced():
    memory, rule = graph_memory(), typed_rule()
    rule["activation_spec"]["signals"]["rate"] = signal(2)
    rule["q_semantics"] = {"dispatch": [{"id": "rate", "type": "frequency", "unit": "calls/s", "source": {"signal": "rate"}}]}
    now = [100.]
    scheduler = SchedulerNew(memory, session_id="test", activation_clock=lambda: now[0])
    item = match_operation(memory, rule)["matched_rules"][0]
    first = execute_rule_chain(matched_rules=[item], working_memory=memory, session_id="test", rule_runtime=scheduler.rule_runtime)
    assert first["ok"] and actual_value(memory)["value"] == 6
    memory.event_workspaces["e"]["spans"][0]["value"] = 4
    second = execute_rule_chain(matched_rules=[item], working_memory=memory, session_id="test", rule_runtime=scheduler.rule_runtime)
    assert second["status"] == "deferred"
    assert actual_value(memory)["value"] == 6
    assert scheduler.rule_runtime.ready_operations() == []
    now[0] += .5
    due = scheduler.rule_runtime.ready_operations()
    assert len(due) == 1
    scheduler._apply_produced_operations(due)
    assert actual_value(memory)["value"] == 8


def test_confidence_is_independent_of_match_and_activation():
    memory, rule = graph_memory(), typed_rule()
    rule["activation_spec"]["signals"]["score"] = signal(.2, "unit_interval")
    rule["q_semantics"] = {"assessment": [{"id": "score", "type": "confidence", "source": {"signal": "score"},
        "target": {"span_id": "answer", "field": "confidence"}, "score_definition": "declared model score", "calibrated": False}]}
    result = execute_rule_chain(matched_rules=match_operation(memory, rule)["matched_rules"], working_memory=memory, session_id="test")
    assert result["ok"], result
    output = actual_value(memory)
    assert output["confidence"] == .2
    assert output["rule_execution"]["require_degree"] == 1
    assert output["metadata"]["confidence_semantics"]["calibrated"] is False


def test_contract_roundtrip_and_no_silent_unknown_q(tmp_path):
    rule = typed_rule(partial=True)
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [rule], rule_kind="content")
    loaded = rule_db.load_rule_records(path, rule_ids=[rule["rule_id"]])[0]
    assert loaded["activation_spec"] == rule["activation_spec"]
    assert loaded["io_contract"] == rule["io_contract"]
    invalid = deepcopy(rule)
    invalid["q_semantics"] = {"dispatch": [{"id": "bad", "type": "unknown", "source": {"signal": "response"}}]}
    with pytest.raises(ActivationError):
        rule_db.upsert_rule_records(path, [invalid], rule_kind="content")
    dataclass_rule = ReasoningRule(rule_id="r", name="r", description="r")
    assert dataclass_rule.to_dict()["execution_contract_version"] == 1


def test_missing_match_does_not_default_to_pass():
    memory = graph_memory()
    rule = {"rule_id": "invalid", "rule_layer": "domain", "required": [], "produced": [{"operation": "stop_path"}]}
    result = execute_rule_chain(matched_rules=[{"rule": rule}], working_memory=memory, session_id="test")
    assert not result["ok"]


def test_control_rule_really_authorizes_partial_in_scheduler(tmp_path):
    memory, rule = graph_memory(), typed_rule(partial=True, control=True)
    owner = {"rule_id": "partial_policy", "rule_layer": "control", "scope_tags": ["rule_activation_decision"],
        "required": [], "produced": [{"operation": "control_decide", "default": [
            {"operation": "decide_rule_activation", "action": "execute_partial", "execution_mode": "local"}]}]}
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    owner = rule_db.load_rule_records(path, rule_ids=[owner["rule_id"]])[0]
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)})
    operation = match_operation(memory, rule)
    scheduler._apply_produced_operations([operation])
    assert len(memory.event_workspaces["e"]["spans"]) == 1
    # Execute the actual stored Control Rule through Matcher and Executor.
    control_operation = match_operation(memory, owner)
    result = scheduler._default_executor({"session_id": "test"}, memory, {"operations": [control_operation]})
    assert result["pending_operations"][0]["operation"] == "decide_rule_activation"
    result = scheduler._default_executor({"session_id": "test"}, memory, {"operations": result["pending_operations"]})
    scheduler._apply_produced_operations(result["pending_operations"])
    assert actual_value(memory)["value"] == 6
    assert actual_value(memory)["execution_scope"] == "partial"


def composite(child_id, *, rule_id="parent", output="returned", source="answer", max_steps=16):
    parent = typed_rule()
    parent["rule_id"] = rule_id
    parent["produced"] = [{"operation": "run_rule", "rule_ids": [child_id]}]
    parent["io_contract"]["outputs"] = [output]
    parent["composition_contract"] = {"kind": "sequential", "max_depth": 4, "max_steps": max_steps,
                                      "output_bindings": {output: source}}
    return parent


def test_nested_composition_uses_real_children_and_checks_returns(tmp_path):
    memory, child = graph_memory(), typed_rule()
    middle = composite(child["rule_id"], rule_id="middle")
    parent = composite("middle", output="final", source="returned")
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [child, middle, parent], rule_kind="content")
    calculator = Calculator(str(path))
    result = execute_rule_chain(matched_rules=match_operation(memory, parent)["matched_rules"],
                               working_memory=memory, calculator=calculator, session_id="test")
    assert result["ok"], result
    assert next(s for s in memory.event_workspaces["e"]["spans"] if s["span_id"] == "final")["value"] == 6
    frames = memory.rule_runtime["test"]["frames"]
    assert len(frames) == 2 and all(f["status"] == "completed" for f in frames.values())
    assert all(f["budget"]["used"] == 3 for f in frames.values())


def test_composition_shared_budget_blocks_child_before_effect(tmp_path):
    memory, child = graph_memory(), typed_rule()
    parent = composite(child["rule_id"], max_steps=1)
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [child, parent], rule_kind="content")
    result = execute_rule_chain(matched_rules=match_operation(memory, parent)["matched_rules"],
                               working_memory=memory, calculator=Calculator(str(path)), session_id="test")
    assert not result["ok"]
    assert result["errors"][0]["error_code"] == "composition_budget_exhausted"
    assert len(memory.event_workspaces["e"]["spans"]) == 1


def test_composition_waits_for_frequency_and_resumes_without_repeating_child(tmp_path):
    memory, first = graph_memory(), typed_rule()
    rate_child = typed_rule()
    rate_child["rule_id"] = "rate_child"
    rate_child["produced"][0]["conclusion"][0]["span_id"] = "timed"
    rate_child["io_contract"]["outputs"] = ["timed"]
    rate_child["activation_spec"]["signals"]["rate"] = signal(1)
    rate_child["q_semantics"] = {"dispatch": [{"id": "rate", "type": "frequency", "unit": "calls/s", "source": {"signal": "rate"}}]}
    parent = composite(first["rule_id"], source="timed")
    parent["produced"].append({"operation": "run_rule", "rule_ids": ["rate_child"]})
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [first, rate_child, parent], rule_kind="content")
    now = [100.]
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)}, activation_clock=lambda: now[0])
    scheduler._apply_produced_operations([match_operation(memory, rate_child)])
    memory.event_workspaces["e"]["spans"][0]["value"] = 4
    result = scheduler._execute_rule_chain(matched_rules=match_operation(memory, parent)['matched_rules'])
    assert result["ok"] and result["status"] == "deferred", result
    frame = next(iter(memory.rule_runtime["test"]["frames"].values()))
    assert frame["position"] == 1 and frame["status"] == "waiting"
    # Restore through JSON, including rate deadline, parent frame and budget.
    snapshot = json.loads(json.dumps(memory.to_dict()))
    restored = WorkingMemoryState(event_workspaces=snapshot["event_workspaces"], rule_runtime=snapshot["rule_runtime"])
    resumed = SchedulerNew(restored, session_id="test", db_paths={"rule": str(path)}, activation_clock=lambda: now[0])
    now[0] += 1
    operations = resumed.rule_runtime.ready_operations()
    assert operations == [{"operation": "resume_rule_composition", "frame_id": frame["frame_id"]}]
    before = len([e for e in restored.entries if e.stage == "executor_step"])
    result = resumed._default_executor({"session_id": "test"}, restored, {"operations": operations})
    root = restored.rule_runtime["test"]["frames"][frame["frame_id"]]
    assert root["status"] == "completed", result
    assert root["position"] == 2
    assert next(s for s in restored.event_workspaces["e"]["spans"] if s["span_id"] == "returned")["value"] == 8
    assert root["budget"]["used"] == 4


def test_cycle_and_parallel_composition_are_rejected(tmp_path):
    memory = graph_memory()
    parent = composite("parent")
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [parent], rule_kind="content")
    result = execute_rule_chain(matched_rules=match_operation(memory, parent)["matched_rules"],
                               working_memory=memory, calculator=Calculator(str(path)), session_id="test")
    assert not result["ok"]
    assert any(e["error_code"] == "recursive_composition_cycle" for e in result["errors"])
    parent["composition_contract"]["kind"] = "parallel"
    with pytest.raises(ActivationError, match="sequential"):
        validate_rule_contract(parent)


def test_weighted_fusion_uses_actual_results_and_deduplicates_replay():
    memory, first = graph_memory(), typed_rule()
    second = deepcopy(first)
    second["rule_id"] = "second"
    second["produced"][0]["conclusion"][0]["expression"]["operands"][1] = 4
    for rule, weight in ((first, 1.), (second, 3.)):
        rule["activation_spec"]["signals"]["weight"] = signal(weight)
        rule["q_semantics"] = {"transform": [{"id": "fusion", "type": "weight", "group": "ensemble",
            "normalization": "sum", "source": {"signal": "weight"}, "target": {"span_id": "answer", "field": "value"}}]}
    for rule in (first, second, first):
        result = execute_rule_chain(matched_rules=match_operation(memory, rule)["matched_rules"],
                                    working_memory=memory, session_id="fusion")
        assert result["ok"], result
    output = actual_value(memory)
    assert output["value"] == 10.5
    assert len(output["activation_fusion"]["contributors"]) == 2
    memory.event_workspaces["e"]["spans"][0]["value"] = 4
    result = execute_rule_chain(matched_rules=match_operation(memory, first)["matched_rules"],
                                working_memory=memory, session_id="fusion")
    assert result["ok"], result
    assert actual_value(memory)["value"] == 8
    assert len(actual_value(memory)["activation_fusion"]["invalidated_contributors"]) == 2


def test_partial_results_require_explicit_downstream_compatibility():
    memory, first = graph_memory(), typed_rule(partial=True)
    first["produced"][0]["conclusion"][0].update(core_vector=[0., 1.], relation_vector=[0., 1.],
        nature_vector=[0., 1.], embedding_model="numerical-test", vector_version="v1")
    result = execute_rule_chain(matched_rules=match_operation(memory, first)["partial_rules"],
                                working_memory=memory, session_id="test")
    assert result["ok"]
    downstream = typed_rule()
    downstream["rule_id"] = "downstream"
    slot = downstream["required"][0]["condition"][0]
    slot.update(feature_type="answer", core_vector=[0., 1.], relation_vector=[0., 1.], nature_vector=[0., 1.])
    downstream["produced"][0]["conclusion"][0]["span_id"] = "second_answer"
    downstream["io_contract"]["outputs"] = ["second_answer"]
    candidates = match_operation(memory, downstream)["matched_rules"]
    assert len(candidates) == 1
    result = execute_rule_chain(matched_rules=candidates, working_memory=memory, session_id="test")
    assert not result["ok"] and result["errors"][0]["error_code"] == "partial_input_not_accepted"
    downstream["io_contract"]["accepts_partial_inputs"] = True
    result = execute_rule_chain(matched_rules=match_operation(memory, downstream)["matched_rules"],
                                working_memory=memory, session_id="test")
    assert result["ok"], result
    assert next(s for s in memory.event_workspaces["e"]["spans"] if s["span_id"] == "second_answer")["value"] == 12


def test_result_score_can_come_from_the_actual_function_output():
    memory, rule = graph_memory(), typed_rule()
    calculator = Calculator()
    calculator.register("prediction", "v1", lambda values, request: {"outputs": {
        "answer": {"value": values["x"]["value"] * 2}, "score": {"value": .37}}})
    out = rule["produced"][0]["conclusion"][0]
    out.pop("expression")
    out["calculation"] = {"schema_version": 1, "function_id": "prediction", "function_version": "v1",
                          "inputs": {"x": {"slot": "x"}}, "output_name": "answer"}
    rule["q_semantics"] = {"assessment": [{"id": "model_score", "type": "confidence", "source": {
        "result_field": "outputs.score.value"}, "target": {"span_id": "answer", "field": "confidence"},
        "score_definition": "actual independent model output", "calibrated": False}]}
    result = execute_rule_chain(matched_rules=match_operation(memory, rule)["matched_rules"], working_memory=memory,
                                session_id="test", calculator=calculator)
    assert result["ok"], result
    assert actual_value(memory)["confidence"] == .37


def test_control_actions_cannot_be_forged_using_an_existing_rule_id(tmp_path):
    from main_part.activation_policies import activation_control_rule
    path = tmp_path / "rules.sqlite"
    owner = activation_control_rule("policy", action="execute_partial", execution_mode="local")
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    scheduler = SchedulerNew(graph_memory(), session_id="test", db_paths={"rule": str(path)})
    with pytest.raises(ActivationError, match="actual Control execution"):
        scheduler._resolve_rule_activation({"operation": "decide_rule_activation", "control_rule_id": "policy",
                                            "action": "execute_partial", "execution_mode": "local"})


@pytest.mark.parametrize("change", ["inactive", "deprecated", "archived", "changed"])
def test_retained_authorization_is_rechecked_after_control_contract_changes(tmp_path, change):
    from main_part.activation_policies import activation_control_rule
    memory, rule = graph_memory(), typed_rule(partial=True, control=True)
    owner = activation_control_rule("policy", action="execute_partial", execution_mode="local")
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    runtime = RuleExecutionRuntime(memory, Calculator(str(path)), session_id="test")
    candidate = runtime.inspect(match_operation(memory, rule)["partial_rules"][0])
    admitted = runtime.authorize(candidate["candidate_id"], mode="local", control_rule=owner)
    if change != "changed":
        owner["status"] = change
    else:
        owner["produced"][0]["branches"][0]["then"][0]["action"] = "reject"
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    before = deepcopy(memory.event_workspaces)
    result = execute_rule_chain(matched_rules=[admitted], working_memory=memory, session_id="test",
                                calculator=runtime.calculator, rule_runtime=runtime)
    assert not result["ok"]
    assert result["errors"][0]["error_code"] == "stale_activation_authority"
    assert memory.event_workspaces == before


def test_control_can_defer_without_busy_retry_and_revisit_after_input_changes(tmp_path):
    from main_part.activation_policies import activation_control_rule
    path = tmp_path / "rules.sqlite"
    owner = activation_control_rule("policy", action="defer")
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    memory, rule = graph_memory(), typed_rule(partial=True, control=True)
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)})
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    action = scheduler._default_executor({"session_id": "test"}, memory,
        {"operations": [match_operation(memory, rule_db.load_rule_records(path, rule_ids=["policy"])[0])]})
    result = scheduler._default_executor({"session_id": "test"}, memory, {"operations": action["pending_operations"]})
    assert scheduler.rule_runtime.has_waiting()
    assert scheduler.rule_runtime.ready_operations() == []
    assert len(memory.event_workspaces["e"]["spans"]) == 1
    memory.event_workspaces["e"]["spans"][0]["value"] = 4
    assert len(scheduler.rule_runtime.ready_operations()) == 1




def test_new_task_reset_reexecutes():
    memory, rule = graph_memory(), typed_rule()
    scheduler = SchedulerNew(memory, session_id="test")
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    first_generation = scheduler.rule_runtime.state["generation"]
    scheduler.clear_transient_state(clear_runtime_workspaces=False)
    assert scheduler.rule_runtime.state["generation"] == first_generation + 1
    assert scheduler.rule_runtime.state["decisions"] == {}
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    assert actual_value(memory)["value"] == 6


def test_partial_output_cannot_be_declared_solved():
    memory = graph_memory()
    scheduler = SchedulerNew(memory, session_id="test")
    scheduler._append_output({"operation": "output", "output_gate_checked": True,
        "result_spans": [{"span_id": "answer", "execution_scope": "partial"}],
        "task_result": {"status": "solved"}}, source="contract_test")
    assert scheduler.state.outputs[-1]["status"] == "partial_result_cannot_complete"
    assert scheduler.state.errors[-1]["error_code"] == "partial_result_cannot_complete"


def test_control_selects_a_declared_q_profile_and_changes_actual_execution(tmp_path):
    from main_part.activation_policies import activation_control_rule
    memory, rule = graph_memory(), typed_rule()
    rule["require_policy"]["q_selection"] = "control"
    rule["activation_spec"]["signals"]["gain"] = signal(-2)
    rule["q_semantics"] = {"default_profile": "plain", "profiles": {"plain": {}, "scaled": {
        "transform": [{"id": "gain", "type": "signed_effect", "source": {"signal": "gain"},
                       "target": {"span_id": "answer", "field": "value"}}]}}}
    owner = activation_control_rule("policy", action="execute_full", q_profile="scaled")
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)})
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    assert len(memory.event_workspaces["e"]["spans"]) == 1
    action = scheduler._default_executor({"session_id": "test"}, memory, {"operations": [match_operation(memory, owner)]})
    decision = scheduler._default_executor({"session_id": "test"}, memory, {"operations": action["pending_operations"]})
    scheduler._apply_produced_operations(decision["pending_operations"])
    assert actual_value(memory)["value"] == -12
    saved = memory.rule_runtime["test"]["decisions"][actual_value(memory)["rule_execution"]["decision_id"]]
    assert saved["q_profile"] == "scaled"


def test_scheduler_combines_compatible_weighted_rules_in_one_actual_path():
    memory, first = graph_memory(), typed_rule()
    second = deepcopy(first)
    second["rule_id"] = "second"
    second["produced"][0]["conclusion"][0]["expression"]["operands"][1] = 4
    for rule, weight in ((first, 1), (second, 3)):
        rule["activation_spec"]["signals"]["weight"] = signal(weight)
        rule["q_semantics"] = {"transform": [{"id": "fusion", "type": "weight", "group": "ensemble",
            "normalization": "sum", "source": {"signal": "weight"}, "target": {"span_id": "answer", "field": "value"}}]}
    graph = graph_bundle_from_spec({"type": "event_workspace", "event_ids": ["e"]}, None, memory)
    result = match_rules_against_graph([first, second], graph, {}, target_spec={}, routing_metadata={})
    scheduler = SchedulerNew(memory, session_id="test")
    scheduler._apply_produced_operations(result["produced"])
    assert not scheduler.state.pending_rule_selectors
    assert actual_value(memory)["value"] == 10.5


def test_activation_cache_tracks_actual_inputs_and_not_unrelated_state():
    memory, rule = graph_memory(), typed_rule()
    calls = []
    calculator = Calculator()
    def response(inputs, request):
        x = inputs["x"]["value"]
        calls.append(x)
        return {"outputs": {"activation": {"value": x * x}}}
    calculator.register("activation", "v1", response)
    rule["activation_spec"] = {"primary": "network", "signals": {"network": {"domain": "real", "source": {
        "kind": "function", "request": {"schema_version": 1, "function_id": "activation", "function_version": "v1",
            "inputs": {"x": {"slot": "x"}}}, "output_name": "activation"}}}}
    runtime = RuleExecutionRuntime(memory, calculator, session_id="test")
    candidate = match_operation(memory, rule)["matched_rules"][0]
    assert runtime.inspect(candidate)["activation"]["value"] == 9
    memory.event_workspaces["e"]["spans"].append({"span_id": "irrelevant", "value": 100})
    assert runtime.inspect(candidate)["activation"]["value"] == 9
    assert calls == [3]
    memory.event_workspaces["e"]["spans"][0]["value"] = 4
    assert runtime.inspect(candidate)["activation"]["value"] == 16
    assert calls == [3, 4]


def test_frequency_wait_uses_the_real_scheduler_clock():
    import time
    memory, rule = graph_memory(), typed_rule()
    rule["activation_spec"]["signals"]["rate"] = signal(2.)
    rule["q_semantics"] = {"dispatch": [{"id": "clock", "type": "frequency", "unit": "calls/s", "source": {"signal": "rate"}}]}
    scheduler = SchedulerNew(memory, session_id="clock")
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    memory.event_workspaces["e"]["spans"][0]["value"] = 4
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    deadline = scheduler.rule_runtime.waiting_until()
    assert deadline is not None
    assert scheduler.run().status == "waiting_activation"
    assert actual_value(memory)["value"] == 6
    time.sleep(max(0, deadline - time.time()) + .01)
    scheduler.run()
    assert actual_value(memory)["value"] == 8


def test_registered_contract_changes_invalidate_old_candidates(tmp_path):
    memory, rule = graph_memory(), typed_rule()
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [rule], rule_kind="content")
    candidate = match_operation(memory, rule)["matched_rules"][0]
    changed = deepcopy(rule)
    changed["activation_spec"]["signals"]["response"] = signal(20)
    rule_db.upsert_rule_records(path, [changed], rule_kind="content")
    result = execute_rule_chain(matched_rules=[candidate], working_memory=memory, calculator=Calculator(str(path)), session_id="test")
    assert not result["ok"] and result["errors"][0]["error_code"] == "stale_rule_contract"


def test_nested_local_budget_does_not_replace_the_global_ledger(tmp_path):
    memory, child = graph_memory(), typed_rule()
    middle = composite(child["rule_id"], rule_id="middle", max_steps=2)
    parent = composite("middle", output="final", source="returned", max_steps=3)
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [child, middle, parent], rule_kind="content")
    result = execute_rule_chain(matched_rules=match_operation(memory, parent)["matched_rules"],
                               working_memory=memory, calculator=Calculator(str(path)), session_id="test")
    assert result["ok"], result
    assert all(f["budget"] == {"limit": 3, "used": 3} for f in memory.rule_runtime["test"]["frames"].values())


def test_composed_partial_child_goes_through_control_and_shared_budget(tmp_path):
    from main_part.activation_policies import activation_control_rule
    memory, child = graph_memory(), typed_rule(partial=True, control=True)
    parent = composite(child["rule_id"])
    parent["io_contract"]["outputs"] = []
    parent["composition_contract"]["output_bindings"] = {}
    owner = activation_control_rule("policy", action="execute_partial", execution_mode="local")
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [child, parent], rule_kind="content")
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)})
    first = scheduler._execute_rule_chain(matched_rules=match_operation(memory, parent)["matched_rules"])
    assert first["ok"] and first["status"] == "waiting_composition", first
    frame = next(iter(scheduler.rule_runtime.state["frames"].values()))
    assert frame["budget"]["used"] == 1
    request = scheduler._default_executor({"session_id": "test"}, memory, {"operations": first["pending_operations"]})
    assert request["status"] == "activation_control_requested"
    control = scheduler._default_executor({"session_id": "test"}, memory, {"operations": [match_operation(memory, owner)]})
    assert frame["budget"]["used"] == 2
    decision = scheduler._default_executor({"session_id": "test"}, memory, {"operations": control["pending_operations"]})
    assert decision["pending_operations"][0]["operation"] == "resume_rule_composition"
    returned = scheduler._default_executor({"session_id": "test"}, memory, {"operations": decision["pending_operations"]})
    assert frame["status"] == "completed", returned
    assert frame["budget"]["used"] == 3
    assert actual_value(memory)["execution_scope"] == "partial"


def test_public_rule_to_memory_match_retains_the_full_contract():
    from main_part.matcher import match_source_to_target
    memory, rule = graph_memory(), typed_rule(partial=True)
    result = match_source_to_target({"source": {"type": "rule", **rule},
                                     "target": {"type": "event_workspace", "event_ids": ["e"]}}, None, memory)
    assert result["ok"] is False
    assert 0 < result["require_evaluation"]["require_degree"] < 1
    assert result["require_evaluation"]["missing_ids"] == ["optional"]


def test_activation_and_produced_read_the_same_knowledge_context():
    memory, rule = graph_memory(), typed_rule()
    source = memory.event_workspaces["e"]["spans"].pop()
    memory.event_workspaces["e"]["context_items"] = [{"context_id": "knowledge", "source": "knowledge_db",
        "status": "active", "payload": {"retrieval_source": "knowledge_db", "spans": [source]}}]
    operation = match_operation(memory, rule)
    assert len(operation["matched_rules"]) == 1
    result = execute_rule_chain(matched_rules=operation["matched_rules"], working_memory=memory, session_id="test")
    assert result["ok"], result
    assert actual_value(memory)["value"] == 6
    assert result["activation_decisions"][0]["activation"]["value"] == 3
    assert [s["span_id"] for s in memory.event_workspaces["e"]["spans"]] == ["answer"]
    assert actual_value(memory)["metadata"]["derived_from_span_ids"] == ["context_knowledge_input"]


def test_unsupported_computed_update_is_rejected_before_any_write():
    rule = typed_rule()
    rule["produced"][0]["operation"] = "update"
    with pytest.raises(ActivationError, match="computed conclusions use add"):
        validate_rule_contract(rule)


@pytest.mark.parametrize("operation", ["add", "update"])
@pytest.mark.parametrize("partial", [False, True])
def test_explicit_span_items_use_the_same_output_contract_q_and_scope(operation, partial):
    memory, rule = graph_memory(), typed_rule(partial=partial)
    item = deepcopy(rule["produced"][0]["conclusion"][0])
    rule["produced"] = [{"operation": operation, "item": item}]
    if operation == "update":
        rule["produced"][0]["id"] = "answer"
        memory.event_workspaces["e"]["spans"].append({"span_id": "answer", "value": 100, "feature_type": "answer"})
    rule["activation_spec"]["signals"]["gain"] = signal(-.5)
    rule["q_semantics"] = {"transform": [{"id": "gain", "type": "signed_effect", "source": {"signal": "gain"},
        "target": {"span_id": "answer", "field": "value"}}]}
    if partial:
        # Partial modes identify the same actual write, regardless of payload form.
        rule["io_contract"]["partial_modes"]["local"]["produced_indices"] = [0]
    result = execute_rule_chain(matched_rules=[{"rule": rule}], working_memory=memory, session_id="test")
    assert result["ok"], result
    output = actual_value(memory)
    assert output["value"] == -3
    assert output["execution_scope"] == ("partial" if partial else "full")
    assert output["rule_execution"]["rule_id"] == rule["rule_id"]
    assert output["metadata"]["derived_from_span_ids"] == ["input"]
    assert len([s for s in memory.event_workspaces["e"]["spans"] if s.get("span_id") == "answer"]) == 1


@pytest.mark.parametrize("payload", ["undeclared", "wrong_target", "both_forms", "update_id", "merge", "split"])
def test_typed_graph_payload_cannot_bypass_output_or_effect_contract(payload):
    memory, rule = graph_memory(), typed_rule()
    item = deepcopy(rule["produced"][0]["conclusion"][0])
    rule["produced"] = [{"operation": "add", "item": item}]
    if payload == "undeclared":
        rule["io_contract"]["outputs"] = []
    elif payload == "wrong_target":
        rule["produced"] = [{"operation": "add", "target": "relations", "conclusion": [item]}]
        rule["effect_contract"]["write_targets"] = ["relations"]
    elif payload == "both_forms":
        rule["produced"][0]["conclusion"] = [item]
    elif payload == "update_id":
        rule["produced"][0].update(operation="update", id="different")
    else:
        rule["produced"][0]["operation"] = payload
    before = deepcopy(memory.event_workspaces)
    result = execute_rule_chain(matched_rules=[{"rule": rule, "match": {"ok": True}}],
                                working_memory=memory, session_id="test")
    assert not result["ok"], result
    assert result["status"] == "rule_admission_failed"
    assert memory.event_workspaces == before


@pytest.mark.parametrize("container", ["default", "branch", "before_decide", "after_frontend"])
def test_control_decision_containers_cannot_hide_undeclared_graph_writes(container):
    memory, rule = graph_memory(), typed_rule()
    rule.update(rule_id="control", rule_layer="control", effect_contract={"kind": "control"})
    rule["io_contract"]["outputs"] = []
    hidden = {"operation": "add", "event_id": "e", "item": {"span_id": "unauthorized", "value": 99}}
    decision = {"operation": "control_decide", "default": [{"operation": "stop_path"}]}
    if container == "branch":
        decision["branches"] = [{"when": {"path": "trigger.kind", "op": "eq", "value": "test"}, "then": [hidden]}]
    elif container == "after_frontend":
        decision["default"] = [{"operation": "align_event_spans", "after_frontend": [hidden]}]
    else:
        decision[container] = [hidden]
    rule["produced"] = [decision]
    result = execute_rule_chain(matched_rules=[{"rule": rule}], working_memory=memory, session_id="test")
    assert not result["ok"] and result["errors"][0]["error_code"] == "undeclared_rule_effect"
    assert len(memory.event_workspaces["e"]["spans"]) == 1


def test_add_and_update_item_writes_to_the_same_node_require_control_selection():
    from main_part.rule_path_candidates import attach_candidate_rule_paths
    memory, add = graph_memory(), typed_rule()
    update = deepcopy(add)
    update["rule_id"] = "update"
    update["produced"] = [{"operation": "update", "item": update["produced"][0]["conclusion"][0]}]
    candidates = [match_operation(memory, r)["matched_rules"][0] for r in (add, update)]
    operation = attach_candidate_rule_paths({"operation": "match", "matched_rules": candidates})
    assert len(operation["candidate_rule_paths"]) == 2
    assert operation["matched_rules"] == []
    assert operation["path_conflicts"][0]["reasons"] == ["competing_graph_mutation"]


def test_composition_rejects_child_effects_outside_the_parent_contract(tmp_path):
    memory, child = graph_memory(), typed_rule()
    parent = composite(child["rule_id"])
    parent["io_contract"]["outputs"] = []
    parent["composition_contract"]["output_bindings"] = {}
    parent["effect_contract"]["write_targets"] = ["relations"]
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [child, parent], rule_kind="content")
    result = execute_rule_chain(matched_rules=match_operation(memory, parent)["matched_rules"],
                                working_memory=memory, calculator=Calculator(str(path)), session_id="test")
    assert not result["ok"] and result["errors"][0]["error_code"] == "undeclared_child_effect"
    assert len(memory.event_workspaces["e"]["spans"]) == 1


@pytest.mark.parametrize("action", ["retry", "reject"])
def test_control_retry_and_reject_are_actual_actions(tmp_path, action):
    from main_part.activation_policies import activation_control_rule
    memory, rule = graph_memory(), typed_rule(partial=True, control=True)
    owner = activation_control_rule("policy", action=action)
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [rule], rule_kind="content")
    rule_db.upsert_rule_records(path, [owner], rule_kind="control")
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)})
    scheduler.state.active_event_ids = ["e"]
    scheduler._apply_produced_operations([match_operation(memory, rule)])
    control_match = scheduler._default_rule_matcher(scheduler.state.to_dict(), memory,
        {"rule_selectors": list(scheduler.state.pending_rule_selectors)})
    scheduler._apply_stage_result("rule_match", control_match)
    control = scheduler._default_executor(scheduler.state.to_dict(), memory,
        {"operations": list(scheduler.state.pending_operations)})
    scheduler._apply_stage_result("execute", control)
    result = scheduler._default_executor({"session_id": "test"}, memory, {"operations": control["pending_operations"]})
    scheduler._apply_stage_result("execute", result)
    if action == "reject":
        assert result["output"]["status"] == "rule_activation_rejected"
        assert len(memory.event_workspaces["e"]["spans"]) == 1
    else:
        assert result["pending_rule_ids"] == [rule["rule_id"]]
        slot = deepcopy(rule["required"][1]["condition"][0])
        slot["span_id"] = "optional_fact"
        memory.event_workspaces["e"]["spans"].append(slot)
        retried = scheduler._default_rule_matcher(scheduler.state.to_dict(), memory, {"rule_ids": result["pending_rule_ids"]})
        assert retried.get("pending_operations"), retried
        scheduler._apply_produced_operations(retried["pending_operations"])
        assert actual_value(memory)["value"] == 6
        assert actual_value(memory)["execution_scope"] == "full"


def test_composition_propagates_declared_control_effects_and_context(tmp_path):
    memory, child = graph_memory(), typed_rule()
    child["activation_spec"] = {"primary": "flag", "signals": {"flag": {
        "domain": "boolean", "source": {"kind": "context", "path": "workflow.flag"}}}}
    controller = typed_rule()
    controller.update(rule_id="child_control", rule_layer="control", effect_contract={"kind": "control"},
        io_contract={"required_slots": ["x"], "outputs": []},
        produced=[{"operation": "control_decide", "default": [{"operation": "update_control_state", "values": {"flag": True}}]}])
    parent = composite("child_control")
    parent["produced"].append({"operation": "run_rule", "rule_ids": [child["rule_id"]]})
    parent["effect_contract"] = {"kind": "composite", "child_effects": ["graph", "control"], "write_targets": ["spans"]}
    path = tmp_path / "rules.sqlite"
    rule_db.upsert_rule_records(path, [child, parent], rule_kind="content")
    rule_db.upsert_rule_records(path, [controller], rule_kind="control")
    scheduler = SchedulerNew(memory, session_id="test", db_paths={"rule": str(path)})
    scheduler._apply_produced_operations([match_operation(memory, parent)])
    assert scheduler.state.control_state["flag"] is True
    assert next(s for s in memory.event_workspaces["e"]["spans"] if s["span_id"] == "returned")["value"] == 6


@pytest.mark.parametrize("operator,operands,expected", [
    ("or", [("gt", 100), ("eq", 3)], True),
    ("and", [("gt", 0), ("eq", 3)], True),
    ("and", [("gt", 100), ("eq", 3)], False),
])
def test_outer_coverage_preserves_original_boolean_predicates(operator, operands, expected):
    from main_part.matcher import match_source_to_target
    memory, rule = graph_memory(), typed_rule()
    rule["required"][0]["predicates"] = [{"operator": operator, "operands": [
        {"operator": op, "left": "x", "right": value} for op, value in operands]}]
    result = match_source_to_target({"source": {"type": "rule", **rule},
                                     "target": {"type": "event_workspace", "event_ids": ["e"]}}, None, memory)
    assert result["ok"] is expected
    assert (result["require_evaluation"]["require_degree"] == 1) is expected


@pytest.mark.parametrize("version", [None, 3])
def test_public_match_does_not_ignore_invalid_contract_versions(version):
    from main_part.matcher import match_source_to_target
    memory, rule = graph_memory(), typed_rule()
    if version is None:
        rule.pop("execution_contract_version")
    else:
        rule["execution_contract_version"] = version
    with pytest.raises(ActivationError):
        match_source_to_target({"source": {"type": "rule", **rule},
                               "target": {"type": "event_workspace", "event_ids": ["e"]}}, None, memory)


def test_active_function_version_is_pinned_across_json_restore(tmp_path):
    import sqlite3
    from main_part.persistence import function_db
    memory, rule = graph_memory(), typed_rule()
    path = tmp_path / "rules.sqlite"
    for version, coefficient in (("p1", 2), ("p2", 3)):
        function_db.store(path, "local_linear", version, json.dumps({"coefficient": coefficient}).encode(), {"function_version": "1"})
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO function_active(function_id,version) VALUES(?,?)", ("local_linear", "p1"))
    rule["activation_spec"] = {"primary": "response", "signals": {"response": {"domain": "real", "source": {
        "kind": "function", "output_name": "response", "request": {"schema_version": 1,
            "function_id": "local_linear", "function_version": "1", "parameter_version": "active", "inputs": {"x": {"slot": "x"}}}}}}}
    rule_db.upsert_rule_records(path, [rule], rule_kind="content")
    def calculator():
        actual = Calculator(str(path))
        def linear(inputs, request):
            parameters, _, _ = function_db.load(path, "local_linear", request["parameter_version"])
            coefficient = json.loads(parameters)["coefficient"]
            return {"outputs": {"response": {"value": coefficient * inputs["x"]["value"]}}}
        actual.register("local_linear", "1", linear)
        return actual
    item = match_operation(memory, rule)["matched_rules"][0]
    runtime = RuleExecutionRuntime(memory, calculator(), session_id="task")
    assert runtime.inspect(item)["activation"]["value"] == 6
    snapshot = json.loads(json.dumps(memory.to_dict()))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE function_active SET version='p2' WHERE function_id='local_linear'")
    restored = WorkingMemoryState(event_workspaces=snapshot["event_workspaces"], rule_runtime=snapshot["rule_runtime"])
    resumed = RuleExecutionRuntime(restored, calculator(), session_id="task")
    assert resumed.inspect(item)["activation"]["value"] == 6
    independent = RuleExecutionRuntime(restored, calculator(), session_id="new_task")
    assert independent.inspect(item)["activation"]["value"] == 9


@pytest.mark.parametrize('field', ['expression', 'calculation', 'value_policy'])
@pytest.mark.parametrize('payload', ['x * 2', None, []])
def test_invalid_computation_cannot_fall_back_to_literal(field, payload):
    memory, rule = graph_memory(), typed_rule()
    output = rule['produced'][0]['conclusion'][0]
    output.pop('expression')
    output.update({field: payload, 'value': 999})
    before = deepcopy(memory.event_workspaces)
    result = execute_rule_chain(matched_rules=[{'rule': rule}], working_memory=memory, session_id="audit")
    assert not result["ok"]
    assert result["errors"][0]['error_code'] == 'invalid_calculation'
    assert memory.event_workspaces == before


@pytest.mark.parametrize('other', ['calculation', 'value_policy'])
def test_multiple_output_computation_mechanisms_are_rejected(other):
    memory, rule = graph_memory(), typed_rule()
    rule['produced'][0]['conclusion'][0][other] = {}
    before = deepcopy(memory.event_workspaces)
    result = execute_rule_chain(matched_rules=[{'rule': rule}], working_memory=memory, session_id="audit")
    assert not result["ok"]
    assert result["errors"][0]['error_code'] == 'conflicting_calculation'
    assert memory.event_workspaces == before


@pytest.mark.parametrize('operation', ['add', 'update'])
def test_literal_overwrite_replaces_old_computation_and_provenance(operation):
    memory, rule = graph_memory(), typed_rule()
    assert execute_rule_chain(matched_rules=[{'rule': rule}], working_memory=memory, session_id="audit")["ok"]
    old = next(s for s in memory.event_workspaces['e']['spans'] if s['span_id'] == 'answer')
    assert old['computation']['value'] == 6
    old['metadata']['confidence_semantics'] = {'definition': 'previous value only'}
    old['activation_effects'] = {'q_ids': ['old']}
    replacement = deepcopy(rule)
    replacement['rule_id'] = 'overwrite'
    replacement['produced'] = [{'operation': operation, 'item': {
        'span_id': 'answer', 'feature_type': 'answer', 'value': 100, 'unit': ''}}]
    result = execute_rule_chain(matched_rules=[{'rule': replacement}], working_memory=memory, session_id="audit")
    assert result["ok"], result["errors"]
    spans = [s for s in memory.event_workspaces['e']['spans'] if s['span_id'] == 'answer']
    assert len(spans) == 1
    output = spans[0]
    assert output['value'] == 100
    for key in ('expression', 'computation', 'computed_by_rule_id', 'activation_effects'):
        assert key not in output
    assert 'confidence_semantics' not in output['metadata']
    assert {s['rule_id'] for s in output['metadata']['derivation_supports']} == {'overwrite'}


def test_admission_error_stops_chain_before_later_output_or_write():
    memory, bad = graph_memory(), typed_rule()
    bad['required'][0]['condition'][0]['core_vector'] = [-1., 0.]
    write = typed_rule()
    write['rule_id'] = 'later-write'
    finish = typed_rule()
    finish.update(rule_id='finish', rule_layer='control', effect_contract={'kind': 'control'})
    finish['io_contract']['outputs'] = []
    finish['produced'] = [{'operation': 'output', 'answer': 'done'}]
    memory.get_event_workspace("e")
    before = deepcopy(memory.event_workspaces)
    result = execute_rule_chain(matched_rules=[{'rule': r} for r in (bad, write, finish)], working_memory=memory, session_id="audit")
    assert not result["ok"]
    assert result["status"] == 'rule_admission_failed'
    assert len(result["errors"]) == len(result["steps"]) == 1
    assert not result["outputs"]
    assert memory.event_workspaces == before


@pytest.mark.parametrize('op,args', [('sigmoid', []), ('dot', [{'value': [1]}]), ('concat', []), ('clip', [{'value': 1}])])
def test_invalid_activation_arity_rejected_at_contract_validation(op, args):
    spec = {'primary': 'x', 'signals': {'x': {'domain': 'real', 'source': {'kind': 'operation', 'op': op, 'inputs': args}}}}
    with pytest.raises(ActivationError, match='invalid operands'):
        validate_activation(spec, {})


@pytest.mark.parametrize('normalization', [{'op': 'clip'}, {'op': 'clip', 'lower': 2, 'upper': 1}, {'op': 'sigmoid', 'lower': 0}])
def test_invalid_normalization_rejected_before_execution(normalization):
    spec = {'primary': 'x', 'signals': {'x': {**signal(1), 'normalization': normalization}}}
    with pytest.raises(ActivationError):
        validate_activation(spec, {})
