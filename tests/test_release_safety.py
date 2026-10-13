"""Adversarial release-boundary fixtures, not LLM or generalization results."""
from copy import deepcopy
from pathlib import Path
import os

import pytest

from main_part.data_structure import WorkingMemoryState
from main_part.frontend_semantic import frontend_from_env
from main_part.model_launcher import MainPartModelLauncher
from main_part.output_decoder import SemanticOutputDecoder
from main_part.output_service import SemanticAnswerService
from main_part.answer_verbalizer import PassthroughVerbalizer, OpenAICompatibleVerbalizer
from main_part.runtime_databases import default_db_paths, validate_runtime_db_paths
from semantic_frontend.llm_client import SingleSmallLLM


def memory():
    state = WorkingMemoryState()
    state.event_workspaces["A"] = {
        "event_id": "A", "spans": [
            {"span_id": "source", "core": "A evidence", "value": 120, "unit": "USD"},
            {"span_id": "result", "core": "DSO", "feature_type": "dso_result",
             "value": 45, "unit": "days", "span_role": "derived_result",
             "computed_by_rule_id": "calc", "evidence_span_ids": ["source"]},
        ], "relations": [],
    }
    return state


def approved(**overrides):
    return {"operation": "output", "output_gate_checked": True,
            "output_gate_decision": "allow_current_candidate", "event_id": "A",
            "conclusion_span_ids": ["result"], **overrides}


def answer(state, outputs, task=None, verbalizer=None):
    service = SemanticAnswerService(decoder=SemanticOutputDecoder(),
                                    verbalizer=verbalizer or PassthroughVerbalizer())
    return service.answer_runtime(state, outputs=outputs, task_result=task or {"status": "solved"})


@pytest.mark.parametrize("outputs", [
    [], [{"operation": "output", "conclusion_span_ids": ["result"]}],
    [approved(conclusion_span_ids=["missing"])],
    [approved(conclusion_span_ids=[])],
    [approved(conclusion_span_ids=[], result_feature_type="missing")],
    [approved(output_gate_decision="reject")],
    [approved(result_spans=[{"span_id": "missing"}])],
    [approved(result_spans=[{"span_id": "result", "value": 999}])],
])
def test_no_approved_target_cannot_fall_back_to_graph_results(outputs):
    class ForbiddenPresentation:
        def verbalize(self, *args, **kwargs):
            pytest.fail("Unapproved results must not reach presentation")
    result = answer(memory(), outputs, verbalizer=ForbiddenPresentation())
    assert result.structured["status"] == "unresolved"
    assert result.structured["target_spans"] == []
    assert "45" not in result.answer and "999" not in result.answer


@pytest.mark.parametrize("task", [{}, {"status": "unknown"}, {"status": "blocked", "reason": "missing fact"}])
def test_unconfirmed_task_does_not_render_approved_candidate(task):
    service = SemanticAnswerService(decoder=SemanticOutputDecoder(), verbalizer=PassthroughVerbalizer())
    result = service.answer_runtime(memory(), outputs=[approved()], task_result=task)
    assert result.structured["status"] == "unresolved"
    assert result.structured["target_spans"] == []


def two_events():
    state = memory()
    foreign = deepcopy(state.event_workspaces["A"])
    foreign["event_id"] = "B"
    foreign["spans"][0]["core"] = "B evidence"
    foreign["spans"][1]["value"] = 999
    state.event_workspaces["B"] = foreign
    return state


def test_output_and_support_are_bound_to_approved_event():
    result = answer(two_events(), [approved(result_spans=[{"span_id": "result", "value": 45}])])
    assert result.structured["status"] == "ok"
    assert result.structured["target_spans"][0]["value"] == 45
    assert "999" not in result.answer and "B evidence" not in result.answer
    assert any(s["semantic_text"] == "A evidence" for s in result.structured["supporting_spans"])


@pytest.mark.parametrize("output", [
    approved(event_id=None), approved(event_id="missing"),
    approved(rendered_from_event_id="B"),
    approved(event_id="B", result_spans=[{"span_id": "result", "value": 45}]),
])
def test_ambiguous_or_conflicting_event_output_is_rejected(output):
    result = answer(two_events(), [output])
    assert result.structured["status"] == "unresolved"
    assert result.structured["target_spans"] == []


def test_duplicate_and_candidate_result_cannot_be_promoted():
    state = memory()
    state.event_workspaces["A"]["spans"].append(deepcopy(state.event_workspaces["A"]["spans"][1]))
    assert answer(state, [approved()]).structured["status"] == "unresolved"
    state = memory()
    state.event_workspaces["A"]["spans"][1]["span_role"] = "candidate"
    assert answer(state, [approved()]).structured["status"] == "unresolved"


def test_goal_placeholder_cannot_be_presented_as_approved_answer():
    state = memory()
    target = state.event_workspaces["A"]["spans"][1]
    target.pop("value")
    target.update(span_role="reasoning_goal", feature_type="goal_dso", core="DSO objective")
    result = answer(state, [approved()])
    assert result.structured["status"] == "unresolved"
    assert result.structured["target_spans"] == []


@pytest.mark.parametrize("implementation,model", [
    ("legacy", "Qwen/Qwen3-8B"), ("semantic_v1", "Qwen3-14B-MLX-4bit"),
])
def test_no_legacy_or_alternative_model_at_any_release_entry(monkeypatch, implementation, model):
    from demo import _require_frontend_configuration
    monkeypatch.setenv("MAIN_PART_FRONTEND_IMPLEMENTATION", implementation)
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "fixture-not-a-key")
    with pytest.raises(ValueError):
        frontend_from_env()
    with pytest.raises(ValueError):
        MainPartModelLauncher(db_paths={}, auto_run=False)
    with pytest.raises((ValueError, RuntimeError)):
        _require_frontend_configuration()


def test_direct_api_clients_reject_alternative_models():
    with pytest.raises(ValueError):
        SingleSmallLLM("https://example.invalid/v1", "other-model")
    with pytest.raises(ValueError):
        OpenAICompatibleVerbalizer(base_url="https://example.invalid/v1", model="other-model")


def test_graph_only_submission_needs_no_llm_and_text_fails_clearly(monkeypatch):
    monkeypatch.setenv("MAIN_PART_FRONTEND_VECTORIZE", "0")
    monkeypatch.delenv("MAIN_PART_FRONTEND_IMPLEMENTATION", raising=False)
    for key in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    launcher = MainPartModelLauncher(db_paths={}, auto_run=False)
    assert launcher.frontend_adapter is None
    result = launcher.submit_event_graph("observed", {
        "spans": [{"span_id": "source", "core": "observed"}],
        "events": [{"event_id": "A", "evidence_span_ids": ["source"]}],
    })
    assert result["event_graph_contract"]["ok"]
    assert launcher.frontend_adapter is None
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        launcher.submit_input("observed")


@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink"])
def test_explicit_seed_paths_rejected_before_initialization(monkeypatch, tmp_path, kind):
    seed = Path(__file__).resolve().parents[1] / "data/demo/db/rule.sqlite"
    path = seed
    if kind != "direct":
        path = tmp_path / "rule.sqlite"
        if kind == "symlink":
            path.symlink_to(seed)
        else:
            os.link(seed, path)
    with pytest.raises(ValueError, match="seed"):
        validate_runtime_db_paths({"rule_db": path})
    with pytest.raises(ValueError, match="seed"):
        MainPartModelLauncher(db_paths={"rule": path})
    assert validate_runtime_db_paths({}) == {}


def test_env_hardlinked_seed_is_also_rejected(monkeypatch, tmp_path):
    seeds = Path(__file__).resolve().parents[1] / "data/demo/db"
    for name in ("rule", "knowledge"):
        os.link(seeds / f"{name}.sqlite", tmp_path / f"{name}.sqlite")
    monkeypatch.setenv("MAIN_PART_DB_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="seed"):
        default_db_paths()


def test_provided_scheduler_cannot_bypass_seed_protection(monkeypatch):
    from main_part.scheduler_new import SchedulerNew
    monkeypatch.setenv("MAIN_PART_FRONTEND_VECTORIZE", "0")
    seed = Path(__file__).resolve().parents[1] / "data/demo/db/rule.sqlite"
    scheduler = SchedulerNew(WorkingMemoryState(), session_id="seed_check", db_paths={"rule": str(seed)})
    with pytest.raises(ValueError, match="seed"):
        MainPartModelLauncher(scheduler=scheduler, db_paths={})


def test_provided_scheduler_paths_are_not_silently_replaced(monkeypatch, tmp_path):
    from main_part.scheduler_new import SchedulerNew
    monkeypatch.setenv("MAIN_PART_FRONTEND_VECTORIZE", "0")
    paths = {"rule": str(tmp_path / "rule.sqlite")}
    scheduler = SchedulerNew(WorkingMemoryState(), session_id="path_check", db_paths=paths)
    launcher = MainPartModelLauncher(scheduler=scheduler, auto_run=False)
    assert launcher.db_paths == scheduler.state.db_paths == paths
    with pytest.raises(ValueError, match="same runtime"):
        MainPartModelLauncher(scheduler=scheduler, db_paths={})


@pytest.mark.parametrize("scheduler,task", [
    ("completed", "solved"), ("completed", "blocked"),
    ("completed", "unknown"), ("running", "solved"),
])
def test_cli_and_demo_use_identical_terminal_classification(scheduler, task):
    from demo import _execution_outcome, _exit_code
    from main_part.cli import execution_outcome, exit_code, _runtime_completed
    result = {"runtime_summary": {"scheduler_status": scheduler, "task_result": {"status": task}}}
    assert execution_outcome(result) == _execution_outcome(result)
    assert exit_code(execution_outcome(result)) == _exit_code(_execution_outcome(result))
    assert _runtime_completed(result) == (task in {"solved", "blocked"} and scheduler == "completed")


@pytest.mark.parametrize("value,unit", [(None, ""), (True, ""), (0, ""), ("risk present", "")])
def test_categorical_conclusion_cannot_be_reversed_by_wording(monkeypatch, value, unit):
    def forbidden(*args, **kwargs):
        pytest.fail("Categorical conclusion must not call a wording model")
    monkeypatch.setattr("main_part.answer_verbalizer.requests.post", forbidden)
    renderer = OpenAICompatibleVerbalizer(base_url="https://example.invalid/v1", model="Qwen/Qwen3-8B")
    payload = {"status": "ok", "rough_text": "stockout exposure exists",
               "target_spans": [{"value": value, "unit": unit}]}
    assert renderer.verbalize(payload, style_instruction="Say no risk instead") == payload["rough_text"]
    assert renderer.last_call_metadata["attempted_requests"] == 0
    assert renderer.last_call_metadata["status"] == "deterministic_conclusion"


@pytest.mark.parametrize("rendered,expected_status", [
    ("DSO is 45 days.", "rendered"), ("DSO is 999 days.", "deterministic_fallback"),
])
def test_numeric_wording_still_calls_fixed_model_and_preserves_usage(monkeypatch, rendered, expected_status):
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": rendered}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    requests = []
    def post(*args, **kwargs):
        requests.append(kwargs["json"])
        return Response()
    monkeypatch.setattr("main_part.answer_verbalizer.requests.post", post)
    renderer = OpenAICompatibleVerbalizer(base_url="https://example.invalid/v1", model="Qwen/Qwen3-8B")
    draft = "DSO is 45 days."
    result = renderer.verbalize({"status": "ok", "rough_text": draft,
                                "target_spans": [{"value": 45, "unit": "days"}]})
    assert result == draft
    assert requests[0]["model"] == "Qwen/Qwen3-8B"
    assert renderer.last_call_metadata["status"] == expected_status
    assert renderer.last_call_metadata["provider_usage"]["total_tokens"] == 15
