from main_part.answer_verbalizer import (
    NumericPreservationGuard,
    OpenAICompatibleVerbalizer,
    PassthroughVerbalizer,
)
from main_part.output_decoder import (
    SemanticOutputDecoder,
    SemanticPrototype,
    SemanticPrototypeIndex,
)
from main_part.output_service import SemanticAnswerService


def _dso_graph():
    return {
        "spans": [
            {
                "span_id": "avg_ar",
                "feature_type": "avg_ar",
                "core": "average accounts receivable",
                "relation": "measured as",
                "nature": "financial balance",
                "value": 100,
                "unit": "CNY ten-thousands",
                "span_role": "observed",
                "source": "document",
            },
            {
                "span_id": "sales",
                "feature_type": "sales",
                "core": "sales revenue",
                "relation": "measured as",
                "nature": "financial flow",
                "value": 1200,
                "unit": "CNY ten-thousands",
                "span_role": "observed",
                "source": "document",
            },
            {
                "span_id": "days",
                "feature_type": "days",
                "core": "days in measurement period",
                "relation": "measured as",
                "nature": "time quantity",
                "value": 365,
                "unit": "days",
                "span_role": "observed",
                "source": "document",
            },
            {
                "span_id": "goal_dso",
                "feature_type": "goal_dso",
                "core": "days sales outstanding",
                "nature": "calculation goal",
                "value": None,
                "unit": "days",
                "span_role": "goal",
                "source": "document",
            },
            {
                "span_id": "dso_result",
                "feature_type": "dso",
                "core": "days sales outstanding",
                "relation": "calculated as",
                "nature": "financial efficiency metric",
                "value": 30.4167,
                "unit": "days",
                "span_role": "result",
                "derived_by_rule_id": "rule_dso",
                "runtime_value_resolution": {
                    "evidence_span_ids": ["avg_ar", "sales", "days"]
                },
            },
        ],
        "events": [
            {
                "event_id": "dso_mainline",
                "evidence_span_ids": [
                    "avg_ar",
                    "sales",
                    "days",
                    "goal_dso",
                    "dso_result",
                ],
                "relations": [
                    {
                        "source_span_id": "avg_ar",
                        "target_span_id": "dso_result",
                        "relation_type": "input_to",
                    },
                    {
                        "source_span_id": "sales",
                        "target_span_id": "dso_result",
                        "relation_type": "input_to",
                    },
                    {
                        "source_span_id": "days",
                        "target_span_id": "dso_result",
                        "relation_type": "input_to",
                    },
                ],
            }
        ],
    }


def test_dso_runtime_graph():
    service = SemanticAnswerService(
        decoder=SemanticOutputDecoder(),
        verbalizer=PassthroughVerbalizer(),
    )
    result = service.answer(
        _dso_graph(),
        target_feature_types=["dso"],
        user_query="请计算DSO。",
    )
    assert "30.4167 days" in result.answer
    assert result.structured["target_spans"][0]["feature_type"] == "dso"
    assert len(result.structured["supporting_spans"]) == 3


def test_requested_goal_and_result_keeps_only_completed_value():
    decoded = SemanticOutputDecoder().decode(
        _dso_graph(),
        target_feature_types=["dso", "goal_dso"],
    )
    assert [x.feature_type for x in decoded.target_spans] == ["dso"]
    assert decoded.target_spans[0].value == 30.4167


def test_runtime_resolution_evidence_is_used():
    decoded = SemanticOutputDecoder().decode(
        _dso_graph(),
        target_feature_types=["dso"],
    )
    assert [x.span_id for x in decoded.supporting_spans] == [
        "avg_ar",
        "sales",
        "days",
    ]


def test_vector_fallback_core():
    index = SemanticPrototypeIndex(
        [
            SemanticPrototype(
                "days sales outstanding",
                (1.0, 0.0),
                "core",
                "dso",
            )
        ],
        min_score=0.4,
    )
    graph = {
        "spans": [
            {
                "span_id": "result",
                "feature_type": "dso",
                "core": "",
                "span_core_vector": [0.99, 0.01],
                "value": 25.0,
                "unit": "days",
                "span_role": "result",
                "derived_by_rule_id": "rule_dso",
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder(index).decode(
        graph,
        target_feature_types=["dso"],
    )
    assert decoded.target_spans[0].semantic_text == "days sales outstanding"
    assert decoded.target_spans[0].semantic_source == "vector_prototype"


def test_generic_core_vector_is_not_used_as_relation_text():
    index = SemanticPrototypeIndex(
        [
            SemanticPrototype(
                "average accounts receivable",
                (1.0, 0.0),
                "semantic",
                "avg_ar",
            )
        ],
        min_score=0.1,
    )
    graph = {
        "spans": [
            {
                "span_id": "x",
                "feature_type": "avg_ar",
                "core": "average accounts receivable",
                "relation": "",
                "relation_vector": [1.0, 0.0],
                "value": 10,
                "unit": "USD millions",
                "span_role": "result",
            }
        ]
    }
    decoded = SemanticOutputDecoder(index).decode(
        graph,
        target_span_ids=["x"],
    )
    assert decoded.target_spans[0].relation_text == ""


def test_final_conclusion_is_preserved():
    state = {
        **_dso_graph(),
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "DSO is 30.4167 days.",
                "status": "final",
                "supporting_evidence": ["avg_ar", "sales", "days"],
                "supporting_rules": ["rule_dso"],
                "unresolved_conditions": [],
                "reasoning_chain": ["avg_ar/sales*days"],
                "confidence": 0.97,
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(
        state,
        target_feature_types=["dso"],
    )
    assert decoded.conclusions[0].answer == "DSO is 30.4167 days."
    assert "FINAL_CONCLUSIONS:" in decoded.rough_text


def test_unresolved_conclusion_sets_status():
    state = {
        "spans": [],
        "events": [],
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "Insufficient information.",
                "status": "insufficient_evidence",
                "unresolved_conditions": ["sales missing"],
                "confidence": 0.2,
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(state)
    assert decoded.status == "unresolved"
    assert "sales missing" in decoded.unresolved


def test_numeric_guard_accepts_small_rounding():
    guard = NumericPreservationGuard(relative_tolerance=1e-3)
    structured = {
        "target_spans": [{"value": 30.4167, "unit": "days"}],
    }
    assert guard.check(structured, "结果约为 30.42 天。").ok


def test_numeric_guard_rejects_changed_value():
    guard = NumericPreservationGuard(relative_tolerance=1e-3)
    structured = {
        "target_spans": [{"value": 30.4167, "unit": "days"}],
    }
    check = guard.check(structured, "结果约为 35 天。")
    assert not check.ok


def test_chat_url_accepts_base_or_full_endpoint():
    a = OpenAICompatibleVerbalizer(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-8B",
    )
    b = OpenAICompatibleVerbalizer(
        base_url="http://localhost:8000/v1/chat/completions",
        model="Qwen/Qwen3-8B",
    )
    assert a.chat_url == "http://localhost:8000/v1/chat/completions"
    assert b.chat_url == "http://localhost:8000/v1/chat/completions"


def test_candidate_span_is_not_exposed_as_final_output():
    graph = _dso_graph()
    graph["spans"].append(
        {
            "span_id": "dso_candidate_bad",
            "feature_type": "dso",
            "core": "days sales outstanding",
            "value": 999,
            "unit": "days",
            "span_role": "candidate",
            "result_policy": "candidate",
            "derived_by_rule_id": "candidate_rule",
        }
    )
    decoded = SemanticOutputDecoder().decode(
        graph, target_feature_types=["dso"]
    )
    assert [x.value for x in decoded.target_spans] == [30.4167]


def test_reasoning_result_graph_key_is_supported():
    state = {
        "graph": _dso_graph(),
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "DSO is 30.4167 days.",
                "status": "final",
                "supporting_evidence": ["avg_ar", "sales", "days"],
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(
        state,
        target_feature_types=["dso"],
    )
    assert decoded.target_spans[0].value == 30.4167
    assert [x.span_id for x in decoded.supporting_spans] == [
        "avg_ar", "sales", "days"
    ]


def test_nested_payload_final_conclusion_is_supported():
    state = {
        "graph": _dso_graph(),
        "payload": {
            "final_conclusions": [
                {
                    "question_id": "q1",
                    "answer": "DSO is 30.4167 days.",
                    "status": "final",
                    "supporting_evidence": ["avg_ar", "sales", "days"],
                }
            ]
        },
    }
    decoded = SemanticOutputDecoder().decode(state, target_feature_types=["dso"])
    assert len(decoded.conclusions) == 1
    assert decoded.conclusions[0].answer == "DSO is 30.4167 days."


def test_final_conclusion_evidence_drives_support():
    graph = _dso_graph()
    # remove direct provenance from result so only final_conclusions link evidence
    for span in graph["spans"]:
        if span["span_id"] == "dso_result":
            span.pop("runtime_value_resolution", None)
            span.pop("evidence_span_ids", None)
    graph["events"][0]["relations"] = []

    state = {
        "graph": graph,
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "DSO is 30.4167 days.",
                "status": "final",
                "supporting_evidence": ["avg_ar", "sales", "days"],
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(state, target_feature_types=["dso"])
    assert [x.span_id for x in decoded.supporting_spans] == [
        "avg_ar", "sales", "days"
    ]


def test_candidate_support_span_is_filtered():
    graph = _dso_graph()
    graph["spans"].append(
        {
            "span_id": "candidate_support",
            "feature_type": "candidate_fact",
            "core": "unpromoted candidate evidence",
            "value": 999,
            "unit": "x",
            "span_role": "candidate",
            "result_policy": "candidate",
        }
    )
    for span in graph["spans"]:
        if span["span_id"] == "dso_result":
            span["evidence_span_ids"] = ["candidate_support", "avg_ar"]
    decoded = SemanticOutputDecoder().decode(graph, target_feature_types=["dso"])
    assert "candidate_support" not in {
        x.span_id for x in decoded.supporting_spans
    }


def test_irrelevant_relations_are_not_exposed():
    graph = _dso_graph()
    graph["spans"].extend(
        [
            {
                "span_id": "noise_a",
                "feature_type": "noise",
                "core": "noise a",
                "value": None,
                "span_role": "observed",
            },
            {
                "span_id": "noise_b",
                "feature_type": "noise",
                "core": "noise b",
                "value": None,
                "span_role": "observed",
            },
        ]
    )
    graph["events"][0]["relations"].append(
        {
            "source_span_id": "noise_a",
            "target_span_id": "noise_b",
            "relation_type": "irrelevant",
        }
    )
    decoded = SemanticOutputDecoder().decode(graph, target_feature_types=["dso"])
    assert all(r.relation_type != "irrelevant" for r in decoded.relations)


def test_prefer_final_conclusions_false_does_not_force_unresolved():
    state = {
        **_dso_graph(),
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "Insufficient information.",
                "status": "insufficient_evidence",
                "unresolved_conditions": ["stale old conclusion"],
            }
        ],
    }
    decoded = SemanticOutputDecoder(
        prefer_final_conclusions=False
    ).decode(state, target_feature_types=["dso"])
    assert decoded.status == "ok"
    assert "stale old conclusion" not in decoded.unresolved


def test_numeric_guard_rejects_extra_hallucinated_number():
    guard = NumericPreservationGuard(relative_tolerance=1e-3)
    structured = {
        "rough_text": "RESULT_SPANS:\n- DSO = 30.4167 days.",
        "target_spans": [{"value": 30.4167, "unit": "days"}],
        "conclusions": [],
    }
    check = guard.check(
        structured,
        "DSO is 30.4167 days, and the recommended threshold is 99 days.",
    )
    assert not check.ok
    assert check.reason.startswith("unexpected_numeric_value_added:")


def test_numeric_guard_checks_final_conclusion_number():
    guard = NumericPreservationGuard(relative_tolerance=1e-3)
    structured = {
        "rough_text": "FINAL_CONCLUSIONS:\n- DSO is 30.4167 days.",
        "target_spans": [],
        "conclusions": [
            {"answer": "DSO is 30.4167 days.", "status": "final"}
        ],
    }
    assert guard.check(structured, "DSO is 30.42 days.").ok
    assert not guard.check(structured, "DSO is 35 days.").ok


def test_numeric_guard_supports_scientific_notation():
    guard = NumericPreservationGuard(relative_tolerance=1e-6)
    structured = {
        "rough_text": "RESULT_SPANS:\n- score = 1e-5.",
        "target_spans": [{"value": 1e-5, "unit": ""}],
        "conclusions": [],
    }
    assert guard.check(structured, "The score is 1e-5.").ok


def test_provisional_final_conclusion_is_not_promoted_to_ok():
    state = {
        "spans": [],
        "events": [],
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "Current result is provisional.",
                "status": "provisional",
                "confidence": 0.55,
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(state)
    assert decoded.status == "provisional"


def test_prefer_final_conclusions_false_hides_them_from_structured_output():
    state = {
        **_dso_graph(),
        "final_conclusions": [
            {
                "question_id": "q1",
                "answer": "Old answer is 99 days.",
                "status": "final",
            }
        ],
    }
    decoded = SemanticOutputDecoder(
        prefer_final_conclusions=False
    ).decode(state, target_feature_types=["dso"])
    assert decoded.conclusions == []
    assert "99" not in decoded.rough_text


def test_graph_level_frontend_unresolved_is_preserved():
    state = {
        "graph": {
            **_dso_graph(),
            "frontend_translation": {
                "status": "unresolved",
                "unresolved": [{"reason": "missing_time_scope"}],
            },
        }
    }
    decoded = SemanticOutputDecoder().decode(
        state,
        target_feature_types=["dso"],
    )
    assert decoded.status == "unresolved"
    assert "missing_time_scope" in decoded.unresolved


def test_live_top_level_graph_beats_stale_snapshot():
    live = _dso_graph()
    stale = {
        "spans": [
            {
                "span_id": "stale",
                "feature_type": "dso",
                "core": "stale dso",
                "value": 999,
                "unit": "days",
                "span_role": "result",
            }
        ],
        "events": [],
    }
    state = {**live, "world_state_snapshot": stale}
    decoded = SemanticOutputDecoder().decode(
        state,
        target_feature_types=["dso"],
    )
    assert decoded.target_spans[0].value == 30.4167


def test_partial_result_span_is_not_final_target_without_promotion():
    graph = {
        "spans": [
            {
                "span_id": "partial_dso",
                "feature_type": "dso",
                "core": "days sales outstanding",
                "value": 31.0,
                "unit": "days",
                "span_role": "result",
                "status": "partial",
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(
        graph,
        target_feature_types=["dso"],
    )
    assert decoded.target_spans == []
    assert decoded.status == "no_target"


def test_partial_result_can_be_promoted_explicitly():
    graph = {
        "spans": [
            {
                "span_id": "partial_dso",
                "feature_type": "dso",
                "core": "days sales outstanding",
                "value": 31.0,
                "unit": "days",
                "span_role": "result",
                "status": "partial",
                "output_eligible": True,
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(
        graph,
        target_feature_types=["dso"],
    )
    assert decoded.target_spans[0].value == 31.0


def test_event_local_evidence_is_preferred_over_unrelated_observed_spans():
    graph = {
        "spans": [
            {
                "span_id": "input_a",
                "feature_type": "x",
                "core": "input a",
                "value": 1,
                "span_role": "observed",
                "source": "input",
            },
            {
                "span_id": "result",
                "feature_type": "score",
                "core": "score",
                "value": 2,
                "span_role": "result",
            },
            {
                "span_id": "unrelated",
                "feature_type": "noise",
                "core": "unrelated",
                "value": 999,
                "span_role": "observed",
                "source": "input",
            },
        ],
        "events": [
            {
                "event_id": "E1",
                "evidence_span_ids": ["input_a", "result"],
                "relations": [],
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(
        graph,
        target_feature_types=["score"],
    )
    assert [x.span_id for x in decoded.supporting_spans] == ["input_a"]


def test_goal_placeholder_alone_is_not_a_successful_target():
    graph = {
        "spans": [
            {
                "span_id": "goal",
                "feature_type": "goal_dso",
                "core": "days sales outstanding",
                "value": None,
                "unit": "days",
                "span_role": "goal",
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(
        graph, target_feature_types=["dso", "goal_dso"]
    )
    assert decoded.status == "no_target"
    assert decoded.target_spans == []


def test_multi_target_does_not_drop_unrelated_unvalued_output():
    graph = {
        "spans": [
            {
                "span_id": "a",
                "feature_type": "a",
                "core": "a",
                "value": 1,
                "span_role": "result",
            },
            {
                "span_id": "b",
                "feature_type": "b",
                "core": "b",
                "value": None,
                "span_role": "result",
            },
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(
        graph, target_feature_types=["a", "b"]
    )
    assert {(x.feature_type, x.value) for x in decoded.target_spans} == {
        ("a", 1),
        ("b", None),
    }


def test_string_target_feature_type_is_not_split_into_characters():
    decoded = SemanticOutputDecoder().decode(
        _dso_graph(), target_feature_types="dso"
    )
    assert decoded.target_spans[0].feature_type == "dso"


def test_zero_support_limit_returns_no_support():
    decoded = SemanticOutputDecoder().decode(
        _dso_graph(),
        target_feature_types=["dso"],
        max_supporting_spans=0,
    )
    assert decoded.supporting_spans == []


def test_outer_frontend_ok_does_not_mask_inner_unresolved():
    state = {
        "frontend_translation": {"status": "ok", "unresolved": []},
        "graph": {
            **_dso_graph(),
            "frontend_translation": {
                "status": "unresolved",
                "unresolved": [{"reason": "graph gap"}],
            },
        },
    }
    decoded = SemanticOutputDecoder().decode(
        state, target_feature_types=["dso"]
    )
    assert decoded.status == "unresolved"
    assert "graph gap" in decoded.unresolved


def test_frontend_unresolved_status_without_reason_is_preserved():
    state = {
        "graph": {
            **_dso_graph(),
            "frontend_translation": {
                "status": "unresolved",
                "unresolved": [],
            },
        }
    }
    decoded = SemanticOutputDecoder().decode(
        state, target_feature_types=["dso"]
    )
    assert decoded.status == "unresolved"
    assert "frontend_translation:unresolved" in decoded.unresolved


def test_unknown_failed_conclusion_status_is_not_ok():
    state = {
        "spans": [],
        "events": [],
        "final_conclusions": [
            {
                "question_id": "q",
                "answer": "failed",
                "status": "failed",
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(state)
    assert decoded.status == "unresolved"


def test_status_only_insufficient_conclusion_is_not_dropped():
    state = {
        "spans": [],
        "events": [],
        "final_conclusions": [
            {
                "question_id": "q",
                "answer": "",
                "status": "insufficient_evidence",
                "unresolved_conditions": ["missing x"],
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(state)
    assert len(decoded.conclusions) == 1
    assert decoded.status == "unresolved"
    assert "missing x" in decoded.unresolved


def test_guard_rejects_categorical_option_flip():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": "FINAL_CONCLUSIONS:\n- [final] A",
        "target_spans": [],
        "conclusions": [{"answer": "A", "status": "final"}],
    }
    check = guard.check(structured, "The answer is B.")
    assert not check.ok
    assert check.reason.startswith("categorical_answer_changed:")


def test_guard_accepts_preserved_categorical_option():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": "FINAL_CONCLUSIONS:\n- [final] A",
        "target_spans": [],
        "conclusions": [{"answer": "A", "status": "final"}],
    }
    assert guard.check(structured, "答案是 A。").ok


def test_guard_rejects_unit_change():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": "RESULT_SPANS:\n- DSO = 30.4 days.",
        "target_spans": [{"value": 30.4, "unit": "days"}],
        "conclusions": [],
    }
    check = guard.check(structured, "DSO is 30.4 months.")
    assert not check.ok
    assert check.reason.startswith("required_unit_missing_or_changed:")


def test_guard_accepts_translated_days_unit():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": "RESULT_SPANS:\n- DSO = 30.4 days.",
        "target_spans": [{"value": 30.4, "unit": "days"}],
        "conclusions": [],
    }
    assert guard.check(structured, "DSO 约为 30.4 天。").ok


def test_guard_rejects_empty_output_even_without_numbers():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": "FINAL_CONCLUSIONS:\n- [final] approved",
        "target_spans": [],
        "conclusions": [{"answer": "approved", "status": "final"}],
    }
    assert not guard.check(structured, "").ok


def test_final_conclusion_numbers_are_authoritative_over_stale_target_numbers():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": (
            "FINAL_CONCLUSIONS:\n- [final] DSO is 30.4 days.\n"
            "RESULT_SPANS:\n- DSO = 31.0 days."
        ),
        "target_spans": [{"value": 31.0, "unit": "days"}],
        "conclusions": [
            {"answer": "DSO is 30.4 days.", "status": "final"}
        ],
    }
    assert guard.check(structured, "DSO is 30.4 days.").ok


def test_verbalizer_falls_back_on_http_error(monkeypatch):
    import requests
    from main_part.answer_verbalizer import OpenAICompatibleVerbalizer

    def boom(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(
        "main_part.answer_verbalizer.requests.post",
        boom,
    )
    verbalizer = OpenAICompatibleVerbalizer(
        base_url="http://localhost:1/v1",
        model="Qwen/Qwen3-8B",
        fallback_on_error=True,
    )
    structured = {
        "rough_text": "deterministic fallback",
        "status": "ok",
        "target_spans": [{"value": 45, "unit": "days"}],
        "conclusions": [],
    }
    assert (
        verbalizer.verbalize(structured)
        == "deterministic fallback"
    )


def test_verbalizer_rejects_empty_constructor_fields():
    import pytest
    from main_part.answer_verbalizer import OpenAICompatibleVerbalizer

    with pytest.raises(ValueError):
        OpenAICompatibleVerbalizer(base_url="", model="Qwen/Qwen3-8B")
    with pytest.raises(ValueError):
        OpenAICompatibleVerbalizer(
            base_url="http://localhost/v1",
            model="",
        )


def test_prototype_index_accepts_frontend_feature_vector():
    index = SemanticPrototypeIndex.from_records(
        [
            {
                "feature_type": "dso",
                "core": "days sales outstanding",
                "feature": [1.0, 0.0],
            }
        ]
    )
    assert index.nearest([1.0, 0.0], kind="core", feature_type="dso") is not None


def test_vector_fallback_beats_raw_debug_text():
    index = SemanticPrototypeIndex(
        [
            SemanticPrototype(
                text="days sales outstanding",
                vector=(1.0, 0.0),
                kind="core",
                feature_type="dso",
            )
        ]
    )
    graph = {
        "spans": [
            {
                "span_id": "r",
                "feature_type": "dso",
                "core": "",
                "text": "raw source sentence that should not block vector recovery",
                "span_core_vector": [0.99, 0.01],
                "value": 30.4,
                "unit": "days",
                "span_role": "result",
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder(index).decode(
        graph, target_feature_types=["dso"]
    )
    assert decoded.target_spans[0].semantic_text == "days sales outstanding"
    assert decoded.target_spans[0].semantic_source == "vector_prototype"


def test_automatic_target_selection_does_not_emit_derived_intermediate():
    graph = {
        "spans": [
            {
                "span_id": "avg",
                "feature_type": "avg_ar",
                "core": "average accounts receivable",
                "value": 100,
                "unit": "CNY ten-thousands",
                "span_role": "derived",
                "derived_by_rule_id": "avg_rule",
            },
            {
                "span_id": "dso",
                "feature_type": "dso",
                "core": "days sales outstanding",
                "value": 30.4,
                "unit": "days",
                "span_role": "result",
                "derived_by_rule_id": "dso_rule",
            },
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(graph)
    assert [x.span_id for x in decoded.target_spans] == ["dso"]


def test_automatic_target_selection_allows_text_only_result():
    graph = {
        "spans": [
            {
                "span_id": "decision",
                "feature_type": "decision",
                "core": "candidate A is supported",
                "value": None,
                "span_role": "result",
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(graph)
    assert [x.span_id for x in decoded.target_spans] == ["decision"]


def test_derived_intermediate_can_be_explicitly_promoted():
    graph = {
        "spans": [
            {
                "span_id": "mid",
                "feature_type": "intermediate_score",
                "core": "intermediate score",
                "value": 0.8,
                "span_role": "derived",
                "output_eligible": True,
            }
        ],
        "events": [],
    }
    decoded = SemanticOutputDecoder().decode(graph)
    assert [x.span_id for x in decoded.target_spans] == ["mid"]


def test_event_local_support_order_is_deterministic():
    graph = {
        "spans": [
            {"span_id": "s2", "feature_type": "x", "core": "s2", "span_role": "observed"},
            {"span_id": "s1", "feature_type": "x", "core": "s1", "span_role": "observed"},
            {"span_id": "r", "feature_type": "r", "core": "r", "value": 1, "span_role": "result"},
        ],
        "events": [
            {
                "event_id": "e",
                "evidence_span_ids": ["s2", "s1", "r"],
                "relations": [],
            }
        ],
    }
    decoded = SemanticOutputDecoder().decode(graph)
    assert [x.span_id for x in decoded.supporting_spans] == ["s2", "s1"]


def test_final_conclusion_unit_change_is_rejected_without_matching_target_value():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": (
            "FINAL_CONCLUSIONS:\n- [final] DSO is 30.4 days.\n"
            "RESULT_SPANS:\n- DSO = 31.0 days."
        ),
        "target_spans": [{"value": 31.0, "unit": "days"}],
        "conclusions": [
            {"answer": "DSO is 30.4 days.", "status": "final"}
        ],
    }
    check = guard.check(structured, "DSO is 30.4 months.")
    assert not check.ok
    assert check.reason.startswith("conclusion_unit_missing_or_changed:")


def test_unit_guard_does_not_match_day_inside_today():
    guard = NumericPreservationGuard()
    structured = {
        "rough_text": "RESULT_SPANS:\n- DSO = 30.4 days.",
        "target_spans": [{"value": 30.4, "unit": "days"}],
        "conclusions": [],
    }
    check = guard.check(
        structured,
        "DSO is 30.4 months today.",
    )
    assert not check.ok
    assert check.reason.startswith("required_unit_missing_or_changed:")
