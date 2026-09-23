from main_part.answer_verbalizer import PassthroughVerbalizer
from main_part.data_structure import WorkingMemoryState
from main_part.output_decoder import SemanticOutputDecoder
from main_part.output_service import SemanticAnswerService, build_runtime_output_state


def _memory_with_final_and_candidate() -> WorkingMemoryState:
    memory = WorkingMemoryState()
    memory.event_workspaces["e1"] = {
        "event_id": "e1",
        "spans": [
            {
                "span_id": "source",
                "feature_type": "avg_ar",
                "core": "average accounts receivable",
                "value": 120,
                "unit": "USD millions",
                "span_role": "observed",
            },
            {
                "span_id": "dso_result",
                "feature_type": "dso_result",
                "core": "days sales outstanding",
                "relation": "calculated as",
                "nature": "financial efficiency metric",
                "value": 45.0,
                "unit": "days",
                "span_role": "derived_result",
                "computed_by_rule_id": "dso_calc_direct",
                "evidence_span_ids": ["source"],
            },
            {
                "span_id": "candidate_bad",
                "feature_type": "dso_result",
                "core": "days sales outstanding",
                "value": 999.0,
                "unit": "days",
                "span_role": "candidate",
                "result_policy": "candidate",
            },
        ],
        "relations": [
            {
                "source_span_id": "source",
                "target_span_id": "dso_result",
                "relation_type": "input_to",
            }
        ],
        "evidence_span_ids": ["source", "dso_result", "candidate_bad"],
    }
    return memory


def test_output_projection_uses_only_control_approved_result_spans():
    memory = _memory_with_final_and_candidate()
    state, target_span_ids = build_runtime_output_state(
        memory,
        outputs=[
            {
                "operation": "output",
                "conclusion_span_ids": ["dso_result"],
                "result_spans": [{"span_id": "dso_result"}],
            }
        ],
        task_result={"status": "solved"},
    )

    assert target_span_ids == ["dso_result"]
    decoded = SemanticOutputDecoder().decode(state, target_span_ids=target_span_ids)
    assert [span.span_id for span in decoded.target_spans] == ["dso_result"]
    assert decoded.target_spans[0].value == 45.0
    assert decoded.target_spans[0].semantic_text == "days sales outstanding"
    assert "candidate_bad" not in {span.span_id for span in decoded.supporting_spans}


def test_runtime_answer_is_deterministic_when_no_verbalizer_is_desired():
    memory = _memory_with_final_and_candidate()
    service = SemanticAnswerService(
        decoder=SemanticOutputDecoder(),
        verbalizer=PassthroughVerbalizer(),
    )

    answer = service.answer_runtime(
        memory,
        outputs=[{"operation": "output", "conclusion_span_ids": ["dso_result"]}],
        task_result={"status": "solved"},
    )

    assert answer.structured["status"] == "ok"
    assert "45.0 days" in answer.answer


def test_output_feature_type_selects_an_existing_non_numeric_result_span():
    memory = _memory_with_final_and_candidate()
    memory.event_workspaces["e1"]["spans"].append(
        {
            "span_id": "inventory_capital_tied_up",
            "feature_type": "inventory_capital_tied_up",
            "core": "inventory capital tied up",
            "relation": "derived as",
            "nature": "inventory capital tied up",
            "derived_by_rule_id": "supply_rule",
        }
    )
    service = SemanticAnswerService(
        decoder=SemanticOutputDecoder(),
        verbalizer=PassthroughVerbalizer(),
    )

    answer = service.answer_runtime(
        memory,
        outputs=[
            {
                "operation": "output",
                "result_feature_type": "inventory_capital_tied_up",
            }
        ],
        task_result={"status": "solved"},
    )

    assert [item["span_id"] for item in answer.structured["target_spans"]] == [
        "inventory_capital_tied_up"
    ]


def test_control_and_goal_bookkeeping_are_not_user_facing_support():
    memory = _memory_with_final_and_candidate()
    memory.event_workspaces["e1"]["spans"].append(
        {
            "span_id": "control_progress",
            "feature_type": "control_progress",
            "core": "control progress state",
            "span_role": "control_state",
        }
    )
    memory.event_workspaces["e1"]["spans"][1]["evidence_span_ids"].append(
        "control_progress"
    )
    service = SemanticAnswerService(
        decoder=SemanticOutputDecoder(),
        verbalizer=PassthroughVerbalizer(),
    )

    answer = service.answer_runtime(
        memory,
        outputs=[{"operation": "output", "conclusion_span_ids": ["dso_result"]}],
        task_result={"status": "solved"},
    )

    assert "control_progress" not in {
        item["span_id"] for item in answer.structured["supporting_spans"]
    }


def test_non_solved_control_result_stays_unresolved():
    memory = _memory_with_final_and_candidate()
    state, target_span_ids = build_runtime_output_state(
        memory,
        outputs=[],
        task_result={"status": "blocked", "reason": "sales basis is missing"},
    )

    decoded = SemanticOutputDecoder().decode(state, target_span_ids=target_span_ids)
    assert decoded.status == "unresolved"
    assert "sales basis is missing" in decoded.unresolved
