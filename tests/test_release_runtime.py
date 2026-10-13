"""Release plumbing checks; not evidence of real LLM semantic quality."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import sqlite3

import pytest

from demo import _execution_outcome, _exit_code
from main_part.runtime_databases import default_db_paths, prepare_runtime_databases
from main_part.persistence import knowledge_db
from main_part.retrieval import retrieve_knowledge
from semantic_frontend.llm_client import SingleSmallLLM


def test_real_knowledge_retrieval_only_changes_runtime_copy(tmp_path):
    seeds = Path(__file__).resolve().parents[1] / "data/demo/db"
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in seeds.glob("*.sqlite")}
    paths = prepare_runtime_databases(runtime_root=tmp_path)
    second = prepare_runtime_databases(runtime_root=tmp_path)
    assert paths != second
    record = knowledge_db.load_knowledge_records(paths["knowledge"], limit=1)[0]
    def counters(path):
        with sqlite3.connect(path) as conn:
            return conn.execute("SELECT SUM(usage_count),SUM(hit_count) FROM knowledge_edges").fetchone()
    before = counters(paths["knowledge"])
    hits = retrieve_knowledge(paths, query_vector=record["core_vector"],
                              query_embedding_model=record["embedding_model"],
                              query_vector_version=record["vector_version"], top_k=1)
    assert hits
    assert counters(paths["knowledge"]) == (before[0] + 1, before[1] + 1)
    assert counters(second["knowledge"]) == before
    assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in seeds.glob("*.sqlite")}


def test_missing_configured_database_is_not_silently_created(monkeypatch, tmp_path):
    monkeypatch.setenv("MAIN_PART_DB_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        default_db_paths()
    assert not list(tmp_path.iterdir())


def test_seed_directory_cannot_be_configured_as_writable_runtime(monkeypatch):
    monkeypatch.setenv("MAIN_PART_DB_DIR", str(Path(__file__).resolve().parents[1] / "data/demo/db"))
    with pytest.raises(ValueError, match="seed"):
        default_db_paths()


@pytest.mark.parametrize("scheduler,task,outcome,code", [
    ("completed", "solved", "solved", 0),
    ("completed", "blocked", "blocked", 0),
    ("completed", "unknown", "unresolved", 2),
    ("running", "solved", "incomplete", 2),
])
def test_exit_status_does_not_promote_partial_or_unknown(scheduler, task, outcome, code):
    result = {"runtime_summary": {"scheduler_status": scheduler, "task_result": {"status": task}}, "outputs": []}
    assert _execution_outcome(result) == outcome
    assert _exit_code(outcome) == code


def test_delivery_failure_takes_priority_over_rolled_back_scheduler_state():
    result = {"runtime_summary": {"scheduler_status": "completed", "task_result": {"status": "solved"}},
              "outputs": [{"operation": "runtime_delivery_failure", "error_type": "RuntimeError"}]}
    assert _execution_outcome(result) == "runtime_failed"
    assert _exit_code(_execution_outcome(result)) == 1


def test_actual_runtime_failure_is_reported_after_state_rollback(monkeypatch):
    # Deliberate fault injection tests the real delivery host, not inference ability.
    from main_part.data_structure import WorkingMemoryState
    from main_part.model_launcher import MainPartModelLauncher
    from main_part.scheduler_new import SchedulerNew
    monkeypatch.setenv("MAIN_PART_FRONTEND_VECTORIZE", "0")
    monkeypatch.setenv("MAIN_PART_FRONTEND_IMPLEMENTATION", "semantic_v1")
    def fault(*args, **kwargs):
        raise RuntimeError("deliberately injected delivery fault")
    memory = WorkingMemoryState()
    scheduler = SchedulerNew(memory, session_id="failure_check", controller=fault, db_paths={})
    launcher = MainPartModelLauncher(working_memory=memory, scheduler=scheduler, db_paths={})
    graph = {"spans": [{"span_id": "source", "core": "observed source"}],
             "events": [{"event_id": "event", "evidence_span_ids": ["source"]}]}
    result = launcher.submit_event_graph("observed source", graph)
    failures = [o for o in result["outputs"] if o.get("operation") == "runtime_delivery_failure"]
    assert len(failures) == 1 and failures[0]["status"] == "retry_pending"
    assert failures[0]["error_type"] == "RuntimeError"
    assert _execution_outcome(result) == "runtime_failed"
    assert _exit_code(_execution_outcome(result)) == 1


def test_api_attempts_and_provider_tokens_are_recorded_without_network(monkeypatch):
    # Authored HTTP fixture tests transport accounting only, not model extraction.
    class Response:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": '{"events": []}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}}
    requests = []
    def post(*args, **kwargs):
        requests.append(kwargs)
        return Response()
    monkeypatch.setattr("semantic_frontend.llm_client.requests.post", post)
    llm = SingleSmallLLM("https://example.invalid/v1", "Qwen/Qwen3-8B", "fixture-only", timeout=900, max_retries=1)
    assert llm.chat_json("source") == {"events": []}
    assert requests[0]["timeout"] == (20, 900)
    usage = asdict(llm.usage)
    assert usage["attempted_requests"] == usage["calls"] == usage["token_usage_calls"] == 1
    assert usage["total_tokens"] == 25 and usage["api_wall_seconds"] >= 0
    assert "fixture-only" not in str(usage)


def test_truncated_complete_json_is_rejected_and_unknown_usage_is_not_zero(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": '{"events": []}'}, "finish_reason": "length"}]}
    llm = SingleSmallLLM("https://example.invalid/v1", "Qwen/Qwen3-8B")
    monkeypatch.setattr(llm, "_post", lambda payload: Response())
    with pytest.raises(ValueError, match="truncated"):
        llm.chat_json("source")
    assert llm.usage.calls == 1
    assert llm.usage.total_tokens is None


def test_json_fragments_are_not_accepted_as_whole_responses():
    with pytest.raises(ValueError, match="fragments"):
        SingleSmallLLM._extract_json('Here is a fragment: {"events": []} and other text')
    assert SingleSmallLLM._extract_json('```json\n{"events": []}\n```') == {"events": []}


def test_frontend_environment_limits_reach_transport(monkeypatch):
    from main_part.frontend_semantic import frontend_from_env
    monkeypatch.setenv("MAIN_PART_FRONTEND_IMPLEMENTATION", "semantic_v1")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("LLM_READ_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "8192")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    llm = frontend_from_env().frontend.llm
    assert (llm.timeout, llm.max_tokens, llm.max_retries, llm.model) == (900, 8192, 1, "Qwen/Qwen3-8B")


def test_output_api_failure_exposes_fallback_and_unknown_cost(monkeypatch):
    import requests
    from main_part.answer_verbalizer import OpenAICompatibleVerbalizer
    def fail(*args, **kwargs):
        raise requests.Timeout("authored transport timeout")
    monkeypatch.setattr("main_part.answer_verbalizer.requests.post", fail)
    renderer = OpenAICompatibleVerbalizer(base_url="https://example.invalid/v1", model="Qwen/Qwen3-8B")
    assert renderer.verbalize({"status": "ok", "rough_text": "verified draft", "target_spans": [{"value": 45, "unit": "days"}]}) == "verified draft"
    assert renderer.last_call_metadata["status"] == "deterministic_fallback"
    assert renderer.last_call_metadata["error_type"] == "Timeout"
    assert renderer.last_call_metadata["provider_usage"] is None
    assert renderer.last_call_metadata["api_wall_seconds"] >= 0
