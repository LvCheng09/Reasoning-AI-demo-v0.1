"""Real matcher/executor acceptance before a trained version can be published."""
import math

from .service import Calculator
from .model_runtime import FEATURES, CLASSIFIER


def validate_pipeline(db_path, function_id, version, rows, vectors, *, embedding_model, vector_version):
    from ..data_structure import WorkingMemoryState
    from ..matcher.routing import graph_bundle_from_spec, required_bundle_from_rule
    from ..matcher.matching import match_required_bundle_against_graph
    from ..scheduler_new import SchedulerNew
    from ..persistence import function_db

    if not embedding_model or not vector_version:
        raise ValueError("pipeline acceptance requires the real encoder identity and vector version")
    if not rows or len(vectors) != len(FEATURES) + 1 or any(not v for v in vectors):
        raise ValueError("pipeline acceptance needs cases and actual feature embeddings")
    calculator = Calculator(db_path)
    outputs = (["class_id", "severe_overdue_probability"] if function_id == CLASSIFIER
               else ["predicted_delay_days"])
    checked = []
    for index, row in enumerate(rows):
        memory = WorkingMemoryState()
        event_id = row["sample_id"]
        spans = [{"span_id": name, "feature_type": name, "core": name, "core_vector": vector,
                  **row["inputs"][name]} for name, vector in zip(FEATURES, vectors)]
        spans.append({"span_id": "prediction_goal", "feature_type": f"goal_{function_id}",
                      "core": function_id, "core_vector": vectors[-1]})
        memory.event_workspaces[event_id] = {"event_id": event_id, "spans": spans, "relations": [], "version": 0}
        request = {"schema_version": 1, "function_id": function_id, "function_version": "1",
                   "parameter_version": version, "inputs": {n: {"slot": n} for n in FEATURES}}
        rule = {"rule_id": f"predict_{function_id}", "rule_layer": "domain",
                "required": [{"condition": [{k: v for k, v in s.items() if k not in ("value", "unit")} for s in spans]}],
                "produced": [{"operation": "add", "conclusion": [
                    {"span_id": f"prediction_{name}", "feature_type": name,
                     "calculation": {**request, "output_name": name}} for name in outputs]}]}
        graph = graph_bundle_from_spec({"type": "event_workspace", "event_ids": [event_id]}, None, memory)
        match = match_required_bundle_against_graph(required_bundle_from_rule(rule, {}), graph, {})
        if not match["ok"]:
            raise ValueError(f"real feature matching failed: {match}")
        operation = {"operation": "match", "matched_rules": [{"rule": rule, "match": match}]}
        scheduler = SchedulerNew(memory, session_id=event_id, db_paths={"rule": db_path})
        # Cover every production execute_rule_chain caller, not a second executor.
        if index % 3 == 0:
            scheduler._apply_produced_operations([operation])
        else:
            if index % 3 == 2:
                operation = {"operation": "execute_candidate_path", "candidate_path": {
                    "matched_rules": operation["matched_rules"], "event_ids": [event_id]}}
            scheduler._default_executor({"session_id": event_id}, memory, {"operations": [operation]})
        direct = calculator.calculate(request, workspace={"spans": spans}, slot_bindings={n: n for n in FEATURES})
        actual = {s["span_id"]: s for s in memory.get_event_workspace(event_id)["spans"]}
        for name in outputs:
            result = actual.get(f"prediction_{name}")
            if result is None or not math.isclose(result["value"], direct["outputs"][name]["value"], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("pipeline did not write actual model output")
            if result["computation"]["parameter_version"] != version:
                raise ValueError("pipeline parameter version drift")
        checked.append(event_id)
    # Reusable rule has a goal condition and no case-specific event or input values.
    rule.update(rule_vector=vectors[-1], embedding_model=embedding_model, vector_version=vector_version,
                status="active", scope_tags=["calculator_micro_models"])
    report = {"status": "PASS", "case_count": len(checked), "case_ids": checked, "rule_template": rule,
              "checks": ["real_matcher", "three_scheduler_callers", "model_values", "version", "graph_write"]}
    function_db.record_acceptance(db_path, function_id, version, report)
    return report
