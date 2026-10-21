"""Small deterministic experiments through the real Scheduler and SQLite stores.

Basis vectors isolate control/matching behavior. These experiments do not measure
language understanding or embedding quality. No replacement model is used.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from main_part.data_structure import WorkingMemoryState
from main_part.persistence import rule_db, knowledge_db
from main_part.require_gaps import recovery_control_network
from main_part.scheduler_new import SchedulerNew


def span(sid, feature, value=3):
    return {"span_id": sid, "feature_type": feature, "value": value, "unit": "",
        "core_vector": [1., 0.], "relation_vector": [1., 0.], "nature_vector": [1., 0.],
        "embedding_model": "numerical-test", "vector_version": "v1"}


def calculation(rid="compute", feature="input", output="answer"):
    return {"rule_id": rid, "rule_layer": "domain", "required": [{
        "required_id": "inputs", "condition": [span("x", feature)]}],
        "produced": [{"operation": "add", "conclusion": [{
            **span(output, output), "expression": {"operator": "multiply", "operands": ["x", 2]}}]}]}


def relation(nature="USD"):
    return {"core": "measurement context", "relation": "denominated in", "nature": nature,
        "core_vector": [1., 0.], "relation_vector": [1., 0.],
        "nature_vector": [1., 0.] if nature == "USD" else [-1., 0.],
        "embedding_model": "numerical-test", "vector_version": "v1"}


def scenario(kind):
    target = calculation()
    memory = WorkingMemoryState(event_workspaces={"e": {
        "event_id": "e", "spans": [span("observed", "input")], "relations": []}})
    extra, knowledge = [], []
    if kind in {"unknown", "reextract", "derivation"}:
        target["required"][0]["condition"][0]["feature_type"] = "missing"
        if kind == "reextract":
            memory.event_workspaces["e"]["metadata"] = {"source_text": "Observed input is 3."}
        if kind == "derivation":
            extra = [calculation("derive", "input", "missing")]
            target["metadata"] = {"require_recovery": {"inputs": {
                "kind": "derivation", "repair_rule_ids": ["derive"]}}}
    if kind == "condition_false":
        target["required"][0]["predicates"] = [{"operator": "gt", "left": "x", "right": 100}]
    if kind in {"missing_relation", "conflict"}:
        target["required"][0]["condition"].append(span("y", "second"))
        target["required"][0]["shared_semantic_relation_constraints"] = [{
            "slot_ids": ["x", "y"], "core_vector": [1., 0.],
            "relation_vector": [1., 0.], "match_threshold": .99}]
        spans = memory.event_workspaces["e"]["spans"]
        spans.append({**span("second", "second"), "semantic_relations": [relation()]})
        if kind == "conflict":
            spans[0]["semantic_relations"] = [relation("EUR")]
        else:
            spans.append({**span("scope", "scope"), "metadata": {"applies_to": ["observed"]},
                "semantic_relations": [relation()]})
            target["metadata"] = {"require_recovery": {"inputs": {
                "kind": "missing_relation", "repair_rule_ids": ["bind_context"]}}}
            extra = [{"rule_id": "bind_context", "rule_layer": "control", "required": [],
                "produced": [{"operation": "bind_knowledge_context", "policy": {"knowledge_id": "context"}}]}]
            knowledge = [{"knowledge_id": "context", **relation(), "metadata": {"context_binding": {
                "scope": "explicit_applies_to", "relation_types": [relation()]}}}]
    return memory, target, extra, knowledge


def scheduler_for(root, kind):
    memory, target, extra, knowledge = scenario(kind)
    entry, network = recovery_control_network({
        "rule_id": "entry", "rule_layer": "control", "scope_tags": ["runtime_control_entry"],
        "required": [], "produced": [{"operation": "control_decide", "branches": [{
            "when": {"all": [
                {"path": "trigger.kind", "operator": "eq", "value": "path_segment_completed"},
                {"path": "path.visited_rule_ids.*", "operator": "eq", "value": "compute"}]},
            "then": {"operation": "stop_path", "reason": "calculation_finished",
                "task_result": {"status": "solved", "reason": "calculation_finished"}}}],
            "default": {"operation": "run_rule", "rule_ids": ["compute"]}}]})
    paths = {"rule": str(root / "rules.sqlite"), "knowledge": str(root / "knowledge.sqlite")}
    rules = [target, *extra, entry, *network]
    for layer in ("domain", "control"):
        rule_db.upsert_rule_records(paths["rule"], [r for r in rules if r["rule_layer"] == layer],
            rule_kind="control" if layer == "control" else "content")
    knowledge_db.upsert_knowledge_records(paths["knowledge"], knowledge)
    scheduler = SchedulerNew(memory, session_id=kind, db_paths=paths, max_loops=80)
    scheduler.state.active_event_ids = ["e"]
    scheduler.state.pending_rule_ids = ["entry"]
    scheduler.state.control_context = scheduler._build_control_context(trigger={
        "kind": "runtime_failure", "pending_rule_ids": ["compute"]})
    return scheduler


def run_case(root, kind):
    scheduler = scheduler_for(root, kind)
    original = deepcopy(scheduler.working_memory.event_workspaces)
    state = scheduler.run()
    result = state.control_state.get("task_result", {})
    spans = scheduler.working_memory.event_workspaces["e"]["spans"]
    answers = [s["value"] for s in spans if s["span_id"] == "answer"]
    assert state.status == "completed", (kind, state.status, state.blocking_reason)
    if kind in {"missing_relation", "derivation"}:
        assert answers == ([6] if kind == "missing_relation" else [12]), (kind, answers)
        assert result["status"] == "solved", result
        if kind == "missing_relation":
            source = next(s for s in spans if s["span_id"] == "observed")
            assert source["semantic_relations"][0]["provenance"]["knowledge_id"] == "context"
    else:
        assert not answers
        assert result["status"] == ("blocked" if kind == "condition_false" else "pending"), result
        if kind == "conflict":
            assert spans[0]["semantic_relations"] == original["e"]["spans"][0]["semantic_relations"]
        if kind == "reextract":
            # No frontend configured: the genuine failed request must terminate,
            # never be replaced by a fabricated model response.
            assert sum(h["phase"] == "frontend_patch" for h in state.execution_history) == 1
    return {"kind": kind, "status": "PARTIAL" if kind == "reextract" else "PASS",
        "scheduler_status": state.status, "loops": state.loop_index, "task_status": result["status"],
        "answers": answers, "reason": result.get("reason"),
        "limitation": "Real LLM extraction not run: no configured frontend" if kind == "reextract" else None}


def main():
    results = []
    for kind in ("reextract", "missing_relation", "derivation", "unknown", "condition_false", "conflict"):
        with tempfile.TemporaryDirectory(prefix="require-network-") as directory:
            results.append(run_case(Path(directory), kind))
    print(json.dumps({"scope": "real scheduler; authored numeric events; no LLM accuracy claim",
        "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
