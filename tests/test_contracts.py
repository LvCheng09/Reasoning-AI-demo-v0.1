import math
import pytest
from pathlib import Path
from main_part.model_launcher import MainPartModelLauncher, _default_db_paths
from main_part.frontend_working_memory_bridge import submit_frontend_state_to_working_memory
from main_part.event_graph_contract import validate_event_graph_contract
from main_part.data_structure import WorkingMemoryState


def test_default_db_paths_use_dso_product_database(monkeypatch):
    monkeypatch.delenv("MAIN_PART_DB_DIR", raising=False)
    paths = _default_db_paths()
    assert all("data/finance_dso_medium_v1/db" in value for value in paths.values())


def test_default_db_paths_allow_isolated_runtime_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MAIN_PART_DB_DIR", str(tmp_path))
    paths = _default_db_paths()
    assert paths == {
        "knowledge": str(Path(tmp_path) / "knowledge.sqlite"),
        "rule": str(Path(tmp_path) / "rule.sqlite"),
        "memory": str(Path(tmp_path) / "memory.sqlite"),
    }


def graph():
    return {"spans": [{"span_id": "a", "core": "first"}, {"span_id": "b", "core": "second"}],
            "events": [{"event_id": "e1", "evidence_span_ids": ["a"]}, {"event_id": "e2", "evidence_span_ids": ["b"]}]}


def test_invalid_evidence_rejected_before_workspace_mutation():
    for evidence in [["missing_span"], ["a", "missing_span"], [], "a", None]:
        value = graph()
        value["events"][0]["evidence_span_ids"] = evidence
        assert not validate_event_graph_contract(value)["ok"]
        memory = WorkingMemoryState()
        with pytest.raises(ValueError):
            submit_frontend_state_to_working_memory(memory, game_input={}, frontend_state=value)
        assert not memory.entries and not memory.event_workspaces
        launcher = MainPartModelLauncher(auto_run=False)
        with pytest.raises(ValueError):
            launcher.submit_event_graph("test", value)
        assert not launcher.working_memory.entries


def test_explicit_all_spans_and_event_isolation():
    value = graph()
    memory = WorkingMemoryState()
    submit_frontend_state_to_working_memory(memory, game_input={}, frontend_state=value)
    assert [s["span_id"] for s in memory.event_workspaces["e1"]["spans"]] == ["a"]
    assert [s["span_id"] for s in memory.event_workspaces["e2"]["spans"]] == ["b"]
    value["events"][0].update(evidence_span_ids=[], use_all_spans=True)
    fresh = WorkingMemoryState()
    submit_frontend_state_to_working_memory(fresh, game_input={}, frontend_state=value)
    assert {s["span_id"] for s in fresh.event_workspaces["e1"]["spans"]} == {"a", "b"}
    result = MainPartModelLauncher(auto_run=False).submit_event_graph("test", value)
    assert result["frontend_state"]["event_workspaces"]["e1"]["evidence_span_ids"] == ["a", "b"]
    value["events"][0]["evidence_span_ids"] = ["missing_span"]
    assert not validate_event_graph_contract(value)["ok"]


def test_all_frontend_confidences_share_a_finite_boundary():
    for confidence, expected in [("high", .7), (None, .7), (float("nan"), .7), (float("inf"), .7), (-2, 0.), (2, 1.), (0, 0.)]:
        value = graph()
        value["spans"][0]["confidence"] = confidence
        event = value["events"][0]
        event["confidence"] = confidence
        event["relations"] = [{"source": "a", "target": "a", "confidence": confidence}]
        result = MainPartModelLauncher(auto_run=False).submit_event_graph("test", value)
        normalized = result["frontend_state"]
        assert normalized["spans"][0]["confidence"] == expected
        assert normalized["events"][0]["confidence"] == expected
        assert normalized["events"][0]["relations"][0]["confidence"] == expected
        assert math.isfinite(expected)
