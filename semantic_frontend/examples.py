"""Prompt examples based on installed INPUT shapes, never expected answers."""
import json

DSO_SOURCE = (
    "本报告金额均以百万美元表示，采用年度360天口径。"
    "本年度平均应收账款为100百万美元，净赊销收入为800百万美元。"
    "请计算应收账款周转天数。"
)

# The names/core/relation/nature follow the installed DSO input contract.
# Values and source text are a separate illustrative example, not a holdout answer.
ROWS = [
    ("currency", "measurement currency", "identifies", "amount comparability attribute",
     "measurement_context", "USD", None, "金额均以百万美元表示"),
    ("scale", "measurement scale", "identifies", "amount normalization attribute",
     "measurement_context", "millions", None, "金额均以百万美元表示"),
    ("period", "reporting period", "identifies", "time comparability attribute",
     "measurement_context", "banking annual convention", None, "采用年度360天口径"),
    ("days", "days in measurement period", "reported_value", "period length",
     "input_evidence", 360.0, "days", "采用年度360天口径"),
    ("goal_dso", "days sales outstanding objective", "requests", "finance calculation goal",
     "reasoning_goal", None, None, "请计算应收账款周转天数"),
    ("avg_ar", "average accounts receivable", "reported_value", "period average balance",
     "input_evidence", 100.0, "百万美元", "本年度平均应收账款为100百万美元"),
    ("sales", "net credit sales", "reported_value", "period flow amount",
     "input_evidence", 800.0, "百万美元", "净赊销收入为800百万美元"),
]


def dso_example():
    events = []
    for name, core, relation, nature, role, value, unit, source in ROWS:
        span = dict(core=core, relation=relation, nature=nature, feature_type=name, span_role=role)
        if value is not None:
            span["value"] = value
        if unit is not None:
            span["unit"] = unit
        if name == "avg_ar":
            span["semantic_relations"] = [{"core": "accounts_receivable", "relation": "aggregation",
                                           "nature": "average", "source_span": source}]
        elif name == "sales":
            span["semantic_relations"] = [{"core": "sales_revenue", "relation": "sales_basis",
                                           "nature": "net_credit", "source_span": source}]
        elif name == "days":
            span["semantic_relations"] = [{"core": "reporting_period", "relation": "temporal_measure",
                                           "nature": "days", "source_span": source}]
        events.append(dict(id=name, canonical_text=source, mainline_id="M1", level=0,
                           parent_id=None, modality="request" if name == "goal_dso" else "fact", polarity="positive",
                           source_span=source, span=span, confidence=0.95))
    return {"mainlines": [{"id": "M1", "title": "finance_dso_reasoning", "summary": "报告口径与应收账款周转天数计算请求",
                           "event_ids": [e["id"] for e in events]}], "events": events,
            "edges": [
                {"source": "days", "target": "period", "relation": "measures",
                 "source_span": "采用年度360天口径", "explicit": True, "confidence": 0.95},
                {"source": "avg_ar", "target": "currency", "relation": "measured_in",
                 "source_span": "本年度平均应收账款为100百万美元", "explicit": True, "confidence": 0.95},
                {"source": "sales", "target": "currency", "relation": "measured_in",
                 "source_span": "净赊销收入为800百万美元", "explicit": True, "confidence": 0.95}]}


PLAIN_SALES_SOURCE = "公司本年度平均应收账款为100万元，全年销售收入为1200万元，按365天计算DSO。"


def plain_sales_example():
    graph = dso_example()
    keep = {"period", "days", "goal_dso", "avg_ar", "sales", "scale"}
    graph["events"] = [e for e in graph["events"] if e["id"] in keep]
    replacements = {
        "period": ("本年度", "本年度", None),
        "days": ("按365天", 365.0, "days"),
        "goal_dso": ("按365天计算DSO", None, None),
        "avg_ar": ("公司本年度平均应收账款为100万元", 100.0, "万元"),
        "sales": ("全年销售收入为1200万元", 1200.0, "万元"),
        "scale": ("100万元", "万", None),
    }
    for event in graph["events"]:
        source, value, unit = replacements[event["id"]]
        event.update(source_span=source, canonical_text=source)
        event["span"].pop("value", None)
        event["span"].pop("unit", None)
        if value is not None:
            event["span"]["value"] = value
        if unit is not None:
            event["span"]["unit"] = unit
        if event["id"] == "sales":
            event["id"] = "total_sales"
            event["span"].update(core="total sales revenue", feature_type="total_sales",
                                  semantic_relations=[{"core": "sales_revenue", "relation": "sales_scope",
                                                       "nature": "total", "source_span": source}])
        elif event["id"] == "avg_ar":
            event["span"]["semantic_relations"] = [{"core": "accounts_receivable", "relation": "aggregation",
                                                       "nature": "average", "source_span": source}]
        elif event["id"] == "days":
            event["span"]["semantic_relations"] = [{"core": "reporting_period", "relation": "temporal_measure",
                                                       "nature": "days", "source_span": source}]
    graph["mainlines"][0]["event_ids"] = [e["id"] for e in graph["events"]]
    # No unsupported currency identification or goal-to-input inference edges.
    graph["edges"] = []
    return graph


SUPPLY_SOURCE = (
    "公司客户需求波动较大，安全库存水平偏低。"
    "请判断是否存在缺货暴露。"
)

SUPPLY_SINGLE_SOURCE = (
    "某公司库存周转速度明显变慢。"
    "请判断是否会形成库存资金占用。"
)


def supply_chain_example():
    """A translation-only supply-chain example with an explicit requested goal."""

    rows = [
        {
            "id": "demand_volatility_high",
            "canonical_text": "客户需求波动较大",
            "source_span": "客户需求波动较大",
            "modality": "fact",
            "span": {
                "core": "volatility",
                "relation": "observed for",
                "nature": "customer demand changes sharply",
                "feature_type": "demand_volatility_high",
                "span_role": "input_evidence",
                "semantic_relations": [{
                    "core": "customer demand",
                    "relation": "is adjacent context for",
                    "nature": "volatility",
                    "source_span": "客户需求波动较大",
                }],
            },
        },
        {
            "id": "safety_stock_low",
            "canonical_text": "安全库存水平偏低",
            "source_span": "安全库存水平偏低",
            "modality": "fact",
            "span": {
                "core": "safety stock",
                "relation": "configured for",
                "nature": "buffer is insufficient",
                "feature_type": "safety_stock_low",
                "span_role": "input_evidence",
                "semantic_relations": [{
                    "core": "shortage protection",
                    "relation": "is adjacent context for",
                    "nature": "safety stock",
                    "source_span": "安全库存水平偏低",
                }],
            },
        },
        {
            "id": "goal_stockout_exposure",
            "canonical_text": "判断是否存在缺货暴露",
            "source_span": "请判断是否存在缺货暴露",
            "modality": "request",
            "span": {
                "core": "stockout exposure objective",
                "relation": "requests",
                "nature": "supply chain assessment goal",
                "feature_type": "goal_stockout_exposure",
                "span_role": "reasoning_goal",
            },
        },
    ]
    events = [
        {
            **row,
            "mainline_id": "M1",
            "level": 0,
            "parent_id": None,
            "polarity": "positive",
            "confidence": 0.95,
        }
        for row in rows
    ]
    return {
        "mainlines": [{
            "id": "M1",
            "title": "supply_chain_assessment",
            "summary": "需求波动、安全库存与明确的缺货暴露判断请求",
            "event_ids": [row["id"] for row in rows],
        }],
        "events": events,
        "edges": [],
    }


def supply_chain_single_example():
    """Show that an ambiguous surface core needs its adjacent context core."""

    rows = [
        {
            "id": "inventory_turnover_slow",
            "canonical_text": "库存周转速度明显变慢",
            "source_span": "某公司库存周转速度明显变慢",
            "modality": "fact",
            "span": {
                "core": "turnover",
                "relation": "measured for",
                "nature": "inventory moves more slowly",
                "feature_type": "inventory_turnover_slow",
                "span_role": "input_evidence",
                "semantic_relations": [{
                    "core": "inventory",
                    "relation": "is adjacent context for",
                    "nature": "turnover",
                    "source_span": "库存周转速度明显变慢",
                }],
            },
        },
        {
            "id": "goal_inventory_capital_tied_up",
            "canonical_text": "判断是否形成库存资金占用",
            "source_span": "请判断是否会形成库存资金占用",
            "modality": "request",
            "span": {
                "core": "inventory capital tied up objective",
                "relation": "requests",
                "nature": "supply chain assessment goal",
                "feature_type": "goal_inventory_capital_tied_up",
                "span_role": "reasoning_goal",
            },
        },
    ]
    events = [{
        **row,
        "mainline_id": "M1",
        "level": 0,
        "parent_id": None,
        "polarity": "positive",
        "confidence": 0.95,
    } for row in rows]
    return {
        "mainlines": [{
            "id": "M1",
            "title": "inventory_turnover_assessment",
            "summary": "库存周转事实与明确的库存资金占用判断请求",
            "event_ids": [row["id"] for row in rows],
        }],
        "events": events,
        "edges": [],
    }


def format_examples():
    return """STRUCTURE EXAMPLE (not facts about the current document):
The public frontend schema keeps mainlines/events/edges. Each small event may carry
a `span` object; Python maps event.id to span_id and mainline to an event workspace.
Use core/relation/nature/feature_type/span_role as the existing graph vocabulary.
Put literal numeric values in value, not only in canonical_text or magnitude.
Do not calculate a derived value or convert scales. Preserve numeric value and unit together.
Reuse field labels ONLY when their meaning is explicitly present in the current source.
Never copy example values, units, periods, currency, goals or relations into another document.
Ordinary revenue is NOT automatically net credit sales; a year does NOT imply 360 days.
An endpoint balance is NOT an average. Unknown fields stay absent, not guessed.
For multiple companies/periods, use distinct IDs and preserve their local grouping.
Coarse parsing may omit narrative detail, but must not merge explicit measurement
fields and the requested goal into one unstructured sentence.

SOURCE:
""" + DSO_SOURCE + "\nEXPECTED INPUT GRAPH (no answer or rule path):\n" + json.dumps(
        dso_example(), ensure_ascii=False, indent=2) + "\nCONTRAST SOURCE:\n" + PLAIN_SALES_SOURCE + \
        "\nCONTRAST INPUT GRAPH:\n" + json.dumps(plain_sales_example(), ensure_ascii=False, indent=2) + \
        "\nSUPPLY-CHAIN STRUCTURE EXAMPLE (translation only; no answer):\nSOURCE:\n" + SUPPLY_SOURCE + \
        "\nEXPECTED INPUT GRAPH:\n" + json.dumps(supply_chain_example(), ensure_ascii=False, indent=2) + \
        "\nAMBIGUOUS-CORE CONTEXT EXAMPLE (translation only; no answer):\nSOURCE:\n" + SUPPLY_SINGLE_SOURCE + \
        "\nEXPECTED INPUT GRAPH:\n" + json.dumps(supply_chain_single_example(), ensure_ascii=False, indent=2) + """

BOUNDARY EXAMPLES:
- "收入为800万元" -> value=800, unit="万元"; do not label it sales/net credit sales
  unless credit-sales meaning is stated. Do not inherit USD/millions from the example.
- "尚未披露平均应收账款" -> preserve negative polarity; do not produce avg_ar value=0.
- "预计下半年下降" -> retain prediction/possibility and future time, not a completed fact.
- "A发生，随后B发生" -> before/after may be extracted; do not add cause.
- A missing detail request whose answer is absent -> empty patch plus unresolved.
- "预测误差较高" alone does NOT state high safety stock or excess inventory risk.
- A requested outcome such as "判断是否存在缺货暴露" is a request goal, not a fact
  that stockout exposure already exists.
"""


def format_patch_examples():
    return '''LOCAL PATCH EXAMPLE, not facts about the current document:
Source: 本年度平均应收账款为100万元。
Existing target id: avg_ar, with an unclear concept label.
If the source supports the correction, return only changed fields:
{"events_to_add": [], "events_to_update": [{"id": "avg_ar", "span": {
"core": "average accounts receivable", "relation": "reported_value",
"nature": "period average balance", "feature_type": "avg_ar",
"span_role": "input_evidence", "value": 100, "unit": "万元"}}],
"edges_to_add": [], "unresolved": []}
Never copy example values, IDs or units. Preserve source values. net credit sales
is only appropriate if credit sales is stated, not for ordinary revenue.
If the needed detail is absent, leave all edit arrays empty and return unresolved
with reason not_in_source or ambiguous_in_source. Do not invent links to other events.
'''
