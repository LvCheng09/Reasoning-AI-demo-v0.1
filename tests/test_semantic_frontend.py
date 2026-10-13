"""Authored JSON fixtures test plumbing only; never evidence of LLM semantic quality."""
from copy import deepcopy
import pytest

from semantic_frontend import SemanticFrontendAdapter, SimpleSemanticFrontend
from semantic_frontend.frontend import GraphContractValidator
from main_part.model_launcher import MainPartModelLauncher, _latest_frontend_state

TEXT = "甲厂十月减产。乙厂没有减产。"
GRAPH = {"mainlines": [{"id": "M1", "title": "甲厂", "summary": "", "event_ids": ["E1"]},
                       {"id": "M2", "title": "乙厂", "summary": "", "event_ids": ["E2"]}],
         "events": [{"id": "E1", "canonical_text": "甲厂减产", "mainline_id": "M1", "source_span": "甲厂十月减产。"},
                    {"id": "E2", "canonical_text": "乙厂没有减产", "mainline_id": "M2", "polarity": "negative", "source_span": "乙厂没有减产。"}],
         "edges": []}


class AuthoredResponses:
    """Explicit offline test fixture, not a model implementation."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def chat_json(self, prompt):
        self.prompts.append(prompt)
        return deepcopy(next(self.responses))


def test_structure_and_patch_transaction():
    frontend = SimpleSemanticFrontend(None, None)
    graph = frontend._build_graph("doc", TEXT, deepcopy(GRAPH))
    original = graph.to_dict()
    candidate, changed = frontend.prepare_patch(graph, "E1", {"events_to_update": [{"id": "E1", "time": "十月"}]})
    assert graph.to_dict() == original
    assert changed == {"E1"} and candidate.version == 1
    assert candidate.event_by_id("E1").time == "十月"
    for patch in ({"events_to_update": [{"id": "E2", "time": "十月"}]},
                  {"events_to_update": [{"id": "E1", "source_span": "不存在"}]},
                  {"events_to_add": [deepcopy(GRAPH["events"][0])]}):
        with pytest.raises(ValueError):
            frontend.prepare_patch(graph, "E1", patch)
        assert graph.to_dict() == original
    graph.events[0].parent_id = "E1"
    with pytest.raises(ValueError, match="cyclic"):
        GraphContractValidator().validate(graph)


def test_launcher_build_and_patch_data_boundary(monkeypatch):
    monkeypatch.setenv("MAIN_PART_FRONTEND_VECTORIZE", "0")
    responses = AuthoredResponses([GRAPH, {"events_to_update": [{"id": "E1", "time": "十月"}]}])
    adapter = SemanticFrontendAdapter(SimpleSemanticFrontend(responses, None))
    launcher = MainPartModelLauncher(frontend_adapter=adapter, auto_run=False, db_paths={})
    result = launcher.submit_input(TEXT)
    assert result["event_graph_contract"]["ok"]
    assert set(launcher.working_memory.event_workspaces) == {"M1", "M2"}
    launcher.working_memory.event_workspaces["M1"]["spans"].append(
        {"span_id": "derived", "core": "BACKEND_ONLY_CONCLUSION", "computed_by_rule_id": "rule"})
    patch_result = launcher._frontend_patch_stage({"session_id": launcher.session_id}, launcher.working_memory,
        {"patches": [{"target_span_id": "E1", "missing_span_query": "原文中的时间"}]})
    assert patch_result["frontend_patch_result"]["status"] == "patched"
    workspace = launcher.working_memory.event_workspaces["M1"]
    e = next(s for s in workspace["spans"] if s["span_id"] == "E1")
    assert e["metadata"]["semantic_event"]["time"] == "十月"
    assert workspace["metadata"]["frontend_document"]["version"] == 1
    assert any(s["span_id"] == "derived" for s in workspace["spans"])
    assert "BACKEND_ONLY_CONCLUSION" not in responses.prompts[-1]
    current = _latest_frontend_state(launcher.working_memory)
    unresolved = adapter.patch_event_graph(game_input={"input": TEXT}, current_event_graph=current,
                                          patches=[{"target_span_id": "missing"}])
    assert unresolved["frontend_translation"]["status"] == "unresolved"
    assert len(responses.prompts) == 2


def test_explicit_configuration(monkeypatch):
    from main_part.frontend_semantic import frontend_from_env
    monkeypatch.setenv("MAIN_PART_FRONTEND_IMPLEMENTATION", "semantic_v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        frontend_from_env()
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-8B")
    adapter = frontend_from_env()
    assert isinstance(adapter, SemanticFrontendAdapter)
    assert adapter.frontend.llm.model == "Qwen/Qwen3-8B"


def test_no_semantic_judge_added():
    frontend = SimpleSemanticFrontend(None, None)
    data = deepcopy(GRAPH)
    data["events"][0]["canonical_text"] = "This authored fixture is semantically wrong"
    # Structural validity does not imply semantic accuracy; intentionally no semantic checker.
    assert frontend._build_graph("doc", TEXT, data).events


def test_vector_refresh_only_for_changed_text():
    class CountingVectorFixture:
        embedding_model = "offline-vector-shape-fixture"
        vector_version = "v1"
        def __init__(self):
            self.calls = []
        def _encode_text_vector(self, text):
            self.calls.append(text)
            return [float(len(text)), 1.0]
    encoder = CountingVectorFixture()
    launcher = MainPartModelLauncher(vector_encoder=encoder, auto_run=False, db_paths={})
    spans = [{"span_id": "x", "core": "original", "nature": "time unknown", "text": "source"}]
    launcher._attach_frontend_vectors(spans, [])
    count = len(encoder.calls)
    launcher._attach_frontend_vectors(spans, [])
    assert len(encoder.calls) == count
    spans[0]["nature"] = "time October"
    launcher._attach_frontend_vectors(spans, [])
    assert len(encoder.calls) == count + 2  # nature and combined vector only
    assert spans[0]["_frontend_vector_texts"]["nature_vector"] == "time October"


def test_frontend_semantic_relations_vectorize_public_fields():
    class CountingVectorFixture:
        embedding_model = "offline-vector-shape-fixture"
        vector_version = "v1"

        def _encode_text_vector(self, text):
            return [float(len(text)), 1.0]

    launcher = MainPartModelLauncher(
        vector_encoder=CountingVectorFixture(), auto_run=False, db_paths={}
    )
    relation = {
        "core": "customer demand",
        "relation": "is adjacent context for",
        "nature": "volatility",
        "source_span": "客户需求波动较大",
    }
    spans = [{
        "span_id": "demand_volatility_high",
        "core": "volatility",
        "relation": "observed for",
        "nature": "demand changes sharply",
        "semantic_relations": [relation],
    }]
    result = launcher._attach_frontend_vectors(spans, [])
    assert result["ok"]
    assert relation["_frontend_vector_texts"] == {
        "core_vector": "customer demand",
        "relation_vector": "is adjacent context for",
        "nature_vector": "volatility",
    }
    assert all(relation[field] for field in (
        "core_vector", "relation_vector", "nature_vector"
    ))


def test_dso_prompt_example_reaches_backend_without_field_loss():
    from semantic_frontend.examples import DSO_SOURCE, dso_example
    from semantic_frontend.frontend import build_initial_prompt, build_refinement_prompt
    from main_part.event_graph_contract import validate_event_graph_contract
    example = dso_example()
    responses = AuthoredResponses([example])
    adapter = SemanticFrontendAdapter(SimpleSemanticFrontend(responses, None))
    result = adapter.build_event_graph({"input": DSO_SOURCE})
    assert validate_event_graph_contract(result)["ok"]
    by_id = {s["span_id"]: s for s in result["spans"]}
    for event in example["events"]:
        assert all(by_id[event["id"]][k] == v for k, v in event["span"].items())
    assert by_id["avg_ar"]["value"] == 100
    assert by_id["sales"]["unit"] == "百万美元"
    assert "value" not in by_id["goal_dso"]
    graph = adapter.frontend._build_graph("doc", DSO_SOURCE, example)
    for prompt in (build_initial_prompt("新原文"),
                   build_refinement_prompt(graph.events[0], {"missing": ["unit"]}, "新原文")):
        assert "average accounts receivable" in prompt
        assert "net credit sales" in prompt
        assert "Never copy example values" in prompt


def test_semantic_contract_examples_and_schemas():
    import json
    from semantic_frontend.examples import PLAIN_SALES_SOURCE, plain_sales_example
    from semantic_frontend.prompts import GRAPH_SCHEMA, PATCH_SCHEMA
    example = plain_sales_example()
    adapter = SemanticFrontendAdapter(SimpleSemanticFrontend(AuthoredResponses([example]), None))
    result = adapter.build_event_graph({"input": PLAIN_SALES_SOURCE})
    by_id = {s["span_id"]: s for s in result["spans"]}
    assert by_id["days"]["value"] == 365
    assert by_id["avg_ar"]["value"] == 100
    assert by_id["avg_ar"]["unit"] == "万元"
    assert by_id["total_sales"]["value"] == 1200
    assert by_id["total_sales"]["unit"] == "万元"
    assert "sales" not in by_id and "currency" not in by_id
    assert by_id["goal_dso"]["metadata"]["semantic_event"]["modality"] == "request"
    assert "value" not in by_id["goal_dso"]
    assert example["edges"] == []
    for event in example["events"]:
        assert event["source_span"] in PLAIN_SALES_SOURCE
        assert len(event["source_span"]) < len(PLAIN_SALES_SOURCE)
    for raw in (GRAPH_SCHEMA, PATCH_SCHEMA):
        schema = json.loads(raw)
        props = schema["$defs"]["event"]["properties"]
        assert "request" in props["modality"]["enum"]
        assert props["confidence"] == {"type": "number", "minimum": 0, "maximum": 1}
        assert schema["$defs"]["edge"]["properties"]["explicit"] == {"const": True}


def test_supply_chain_prompt_contract_and_example():
    from semantic_frontend.examples import (
        SUPPLY_SINGLE_SOURCE,
        SUPPLY_SOURCE,
        supply_chain_example,
        supply_chain_single_example,
    )
    from semantic_frontend.prompts import SYSTEM_PROMPT, build_initial_prompt
    from main_part.event_graph_contract import validate_event_graph_contract

    required_inputs = {
        "inventory_turnover_slow", "receivables_turnover_slow",
        "employee_turnover_high", "supplier_lead_time_long",
        "customer_lead_time_long", "stockout_rate_high", "return_rate_high",
        "on_time_delivery_low", "safety_stock_low", "safety_stock_high",
        "procurement_cost_up", "freight_cost_up", "inventory_days_high",
        "payment_terms_extended", "production_yield_low", "order_backlog_high",
        "forecast_error_high", "demand_volatility_high",
        "capacity_utilization_high", "supplier_concentration_high",
    }
    for feature_type in required_inputs:
        assert feature_type in SYSTEM_PROMPT

    example = supply_chain_example()
    adapter = SemanticFrontendAdapter(
        SimpleSemanticFrontend(AuthoredResponses([example]), None)
    )
    result = adapter.build_event_graph({"input": SUPPLY_SOURCE})
    assert validate_event_graph_contract(result)["ok"]
    by_id = {span["span_id"]: span for span in result["spans"]}
    assert set(by_id) == {
        "demand_volatility_high", "safety_stock_low",
        "goal_stockout_exposure",
    }
    assert "value" not in by_id["goal_stockout_exposure"]
    assert by_id["goal_stockout_exposure"]["metadata"]["semantic_event"]["modality"] == "request"
    prompt = build_initial_prompt("只有预测误差较高，请判断是否存在过量库存风险。")
    full_prompt = SYSTEM_PROMPT + prompt
    assert "forecast_error_high" in full_prompt
    assert "safety_stock_high" in full_prompt
    assert "does NOT" in full_prompt
    single = supply_chain_single_example()
    single_adapter = SemanticFrontendAdapter(
        SimpleSemanticFrontend(AuthoredResponses([single]), None)
    )
    single_result = single_adapter.build_event_graph({"input": SUPPLY_SINGLE_SOURCE})
    single_by_id = {span["span_id"]: span for span in single_result["spans"]}
    turnover = single_by_id["inventory_turnover_slow"]
    assert turnover["core"] == "turnover"
    assert turnover["nature"] == "inventory moves more slowly"
    assert turnover["semantic_relations"][0]["core"] == "inventory"


def test_refinement_preserves_request_and_literal_units():
    from semantic_frontend.examples import PLAIN_SALES_SOURCE, plain_sales_example
    frontend = SimpleSemanticFrontend(None, None)
    graph = frontend._build_graph("doc", PLAIN_SALES_SOURCE, plain_sales_example())
    patched, _ = frontend.prepare_patch(graph, "goal_dso", {
        "events_to_update": [{"id": "goal_dso", "modality": "request"}]})
    assert patched.event_by_id("goal_dso").modality == "request"
    assert patched.event_by_id("avg_ar").span["unit"] == "万元"


def test_structured_span_patch_and_rejection():
    from semantic_frontend.examples import DSO_SOURCE, dso_example
    frontend = SimpleSemanticFrontend(None, None)
    graph = frontend._build_graph("doc", DSO_SOURCE, dso_example())
    with pytest.raises(ValueError, match="structured span"):
        frontend.prepare_patch(graph, "avg_ar", {"events_to_update": [
            {"id": "avg_ar", "span": {"rule_id": "not_allowed"}}]})


def test_attribute_only_refinement_rejects_unrelated_edges():
    from semantic_frontend.examples import DSO_SOURCE, dso_example
    frontend = SimpleSemanticFrontend(None, None)
    graph = frontend._build_graph("doc", DSO_SOURCE, dso_example())
    frontend.llm = AuthoredResponses([{
        "events_to_add": [], "events_to_update": [],
        "edges_to_add": [{"source": "avg_ar", "target": "currency", "relation": "measured_in",
                          "source_span": "本年度平均应收账款为100百万美元", "explicit": True, "confidence": .9}],
        "unresolved": []
    }])
    with pytest.raises(ValueError, match="attribute-only"):
        frontend.refine(graph, "avg_ar", {"patch_scope": "attributes_only"})
    assert graph.event_by_id("avg_ar").span["value"] == 100


def test_demo_is_separate_from_business_module():
    import semantic_frontend.frontend as business
    from tools import run_semantic_frontend_demo as demo
    assert not hasattr(business, "SemanticGapDetector")
    assert not hasattr(business, "run_case")
    assert not hasattr(business, "main")
    from tests.frontend_demo import runner
    assert runner.SimpleSemanticFrontend is business.SimpleSemanticFrontend
    assert callable(demo.main)
