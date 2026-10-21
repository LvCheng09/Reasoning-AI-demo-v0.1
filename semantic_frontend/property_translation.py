"""Second translation pass: source-local properties, not semantic judging.

The model cannot rewrite values, scopes, goals or facts through this boundary.
Only source-translated properties with literal evidence are serialized.
"""
from copy import deepcopy
import json
from jsonschema import Draft202012Validator

PROPERTY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["properties"],
    "properties": {"properties": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["fact_id", "status", "semantic_relations", "note"],
        "properties": {
            "fact_id": {"type": "string", "minLength": 1},
            "status": {"enum": ["extracted", "not_stated", "uncertain"]},
            "note": {"type": "string"},
            "semantic_relations": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["concept", "core", "relation", "nature", "polarity", "source_span"],
                "properties": {**{k: {"type": "string", "minLength": 1}
                               for k in ("concept", "core", "relation", "nature", "source_span")},
                               "polarity": {"enum": ["positive", "negative", "unknown"]}}}},
        }}}}}

PROPERTY_INSTRUCTIONS = """第二步任务：fact_properties。逐条翻译原文中这条事实的额外性质。
不是审查上一份输出，也不是推理、解题或判断是否获准。不得新增事实、改变数值或范围。
只返回根对象properties，数组覆盖输入每一个fact_id一次，不得省略、重复、修改编号。
每项固定字段：fact_id、status、semantic_relations、note。
status三选一：extracted（原文明示额外性质）；not_stated（原文未说明额外性质）；uncertain（不能确定）。
主概念、数值、单位、主体、期间、决策、政策、币种已经在第一步提取，不要重复作为额外性质。
重点阅读每条source_text中的描述、限定和否定；不能只看主概念或金额，不能忽略额外性质。
semantic_relations每项为{concept,core,relation,nature,polarity,source_span}。
concept是本条原文明示性质的简短中文名称，不是事实的主概念。core/relation/nature使用简洁规范中文。
否定单独用polarity=negative，概念和nature写被否定的肯定性质。比如明确“非木制”写木制性质、negative，不写木制positive。
source_span逐字引用本条source_text中的性质短句。不能引用其他事实，不需要Knowledge或Rule定义。
semantic_relations必须输出，没有就[]。not_stated数组为空；uncertain数组为空且note说明不确定点。
extracted至少有一个性质。某事实的性质不能复制给其他事实。只输出JSON实例，不回传输入或字段说明。"""


def fact_records(graph):
    return [{"fact_id": s["span_id"],
             "main_concept": s["semantic_concept"],
             "source_text": s["text"]} for s in graph["spans"][1:-1]]


def render_property_request(graph, correction=None):
    prompt = PROPERTY_INSTRUCTIONS
    prompt += '\n空性质记录的格式示范（不是本次事实，实际fact_id必须从本次输入复制）：{"properties":[{"fact_id":"example_only","status":"not_stated","semantic_relations":[],"note":""}]}\n不能对所有记录机械套用not_stated；明示性质必须提取。'
    if correction is not None:
        prompt += "\n上次输出被拒绝：" + correction["validation_error"] + "\n从本次未变化的事实原文重新提取，不续写旧回复。"
    return (prompt + "\n<fact_sources>\n" +
            json.dumps(fact_records(graph), ensure_ascii=False, separators=(",", ":")) +
            "\n</fact_sources>\n只返回properties根字段的JSON。")


def merge_property_reply(graph, data):
    """Structural/evidence validation only; no semantic inference or fill-in."""
    errors = list(Draft202012Validator(PROPERTY_SCHEMA).iter_errors(data))
    if errors:
        raise ValueError("Property schema: " + "; ".join(
            "/".join(map(str, e.absolute_path)) + ": " + e.message[:200] for e in errors[:8]))
    result = deepcopy(graph)
    targets = {s["span_id"]: s for s in result["spans"][1:-1]}
    rows = data["properties"]
    ids = [r["fact_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(targets):
        raise ValueError("Property rows must cover every supplied fact_id exactly once; missing=" +
                         str(sorted(set(targets) - set(ids))) + "; extra=" + str(sorted(set(ids) - set(targets))))
    for row in rows:
        status = row["status"]
        items = row["semantic_relations"]
        if status == "extracted" and not items:
            raise ValueError(row["fact_id"] + ": extracted requires a property")
        if status != "extracted" and items:
            raise ValueError(row["fact_id"] + ": non-extracted status cannot contain asserted properties")
        if status == "uncertain" and not row["note"].strip():
            raise ValueError(row["fact_id"] + ": uncertain requires a note")
        span = targets[row["fact_id"]]
        evidence = []
        seen = set()
        for item in items:
            source = item["source_span"]
            if not source.strip() or source not in span["text"]:
                raise ValueError(row["fact_id"] + ": property source_span must be a literal substring of its own fact source")
            triple = {k: item[k] for k in ("core", "relation", "nature")}
            if any(not s.strip() for s in triple.values()):
                raise ValueError("Property triple fields must be nonempty")
            key = (*[triple[k] for k in ("core", "relation", "nature")], item["polarity"])
            if key in seen:
                raise ValueError(row["fact_id"] + ": duplicate property triple")
            seen.add(key)
            relation = {**triple, "polarity": item["polarity"], "semantic_concept": item["concept"], "source_span": source}
            span["semantic_relations"].append(relation)
            evidence.append({"triple": triple, "polarity": item["polarity"], "source_span": source})
        span["metadata"]["qualifier_evidence"] = evidence
        span["metadata"]["property_translation"] = {"status": status, "note": row["note"]}
    return result
