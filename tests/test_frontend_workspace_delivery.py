"""Real bridge + receive_input + event_build stage, no replacement stage."""
import pytest
from main_part.data_structure import WorkingMemoryState
from main_part.frontend_working_memory_bridge import submit_frontend_state_to_working_memory
from main_part.scheduler_new import SchedulerNew


def test_real_delivered_graph_becomes_active_not_default_noop():
    graph = {"spans": [{"span_id": "observation", "core": "observed fact"}],
             "events": [{"event_id": "decision_5", "evidence_span_ids": ["observation"]}]}
    memory = WorkingMemoryState()
    submit_frontend_state_to_working_memory(memory, game_input={}, frontend_state=graph)
    scheduler = SchedulerNew(memory, session_id="delivery-test", db_paths={})
    scheduler.receive_input({"frontend_state": graph}, source="runtime_launcher")
    result = scheduler.event_builder(scheduler.state.to_dict(), memory, scheduler._stage_payload("event_build"))
    scheduler._apply_stage_result("event_build", result)
    assert scheduler.state.active_event_ids == ["decision_5"]
    assert not scheduler.state.pending_inputs
    assert scheduler.state.last_action == "event_build"
    assert memory.event_workspaces["decision_5"]["spans"][0]["span_id"] == "observation"


def test_missing_delivered_workspace_is_not_silently_accepted():
    graph = {"spans": [{"span_id": "s", "core": "fact"}],
             "events": [{"event_id": "missing", "evidence_span_ids": ["s"]}]}
    scheduler = SchedulerNew(WorkingMemoryState(), session_id="missing-test", db_paths={})
    scheduler.receive_input({"frontend_state": graph})
    with pytest.raises(ValueError, match="not submitted"):
        scheduler.event_builder(scheduler.state.to_dict(), scheduler.working_memory, scheduler._stage_payload("event_build"))
