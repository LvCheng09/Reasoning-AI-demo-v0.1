"""Actual Launcher/Scheduler regression from authored graphs, never a live LLM test."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from main_part.model_launcher import MainPartModelLauncher
from main_part.rule_vector_encoder import RuleVectorEncoder
from main_part.output_decoder import SemanticOutputDecoder
from main_part.output_service import SemanticAnswerService
from main_part.answer_verbalizer import PassthroughVerbalizer
from semantic_frontend.adapter import runtime_graph
from semantic_frontend.frontend import SimpleSemanticFrontend
from semantic_frontend.examples import (
    DSO_SOURCE, dso_example, SUPPLY_SOURCE, supply_chain_example,
    SUPPLY_SINGLE_SOURCE, supply_chain_single_example,
)


def seed_hashes():
    root = Path(__file__).resolve().parents[1] / "data/demo/db"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.sqlite")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default=os.getenv("MAIN_PART_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new output path; previous evidence is never overwritten")
    os.environ["MAIN_PART_FRONTEND_IMPLEMENTATION"] = "semantic_v1"
    encoder = RuleVectorEncoder(encoder_name_or_path=args.embedding_model, local_files_only=True, load_encoder=True)
    original = seed_hashes()
    report = {"status": "running", "scope": "Authored release graphs with real cached Qwen vectors and Launcher/Scheduler; no external LLM, training or blind generalization claim.", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    save()
    try:
        return validate_cases(args, encoder, original, report, save)
    except Exception as exc:
        report.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


def validate_cases(args, encoder, original, report, save) -> int:
    def forbidden_frontend(*args, **kwargs):
        raise RuntimeError("external frontend calls forbidden in release graph regression")
    # A second authored goal over the same observation exercises multi-step
    # composition. This is a graph fixture, not a claimed LLM extraction.
    cash_flow = deepcopy(supply_chain_single_example())
    cash_text = SUPPLY_SINGLE_SOURCE.replace("库存资金占用", "经营现金流压力")
    goal = cash_flow["events"][-1]
    goal.update(id="goal_operating_cash_flow_pressure", canonical_text="判断经营现金流压力",
                source_span="请判断是否会形成经营现金流压力")
    goal["span"].update(core="operating cash flow pressure objective", feature_type="goal_operating_cash_flow_pressure")
    cash_flow["mainlines"][0]["event_ids"][-1] = goal["id"]
    for name, text, authored, expected_feature in (
        ("dso", DSO_SOURCE, dso_example(), "dso_result"),
        ("stockout", SUPPLY_SOURCE, supply_chain_example(), "stockout_exposure"),
        ("inventory", SUPPLY_SINGLE_SOURCE, supply_chain_single_example(), "inventory_capital_tied_up"),
        ("cash_flow", cash_text, cash_flow, "operating_cash_flow_pressure"),
    ):
        started = time.perf_counter()
        graph = runtime_graph(SimpleSemanticFrontend(None, None)._build_graph(name, text, authored))
        launcher = MainPartModelLauncher(vector_encoder=encoder, session_id=f"release_{name}")
        launcher.scheduler.frontend = forbidden_frontend
        result = launcher.submit_event_graph(text, graph)
        answer = launcher.render_final_answer(SemanticAnswerService(
            decoder=SemanticOutputDecoder(), verbalizer=PassthroughVerbalizer()))
        targets = answer.structured.get("target_spans", [])
        found = [s for s in targets if s.get("feature_type") == expected_feature]
        runtime_ok = not any(x.get("operation") == "runtime_delivery_failure" for x in result["outputs"])
        solved = launcher.runtime_summary()["task_result"].get("status") == "solved"
        numeric_ok = True
        if name == "dso":
            source = {s["span_id"]: s for s in graph["spans"]}
            expected = source["avg_ar"]["value"] / source["sales"]["value"] * source["days"]["value"]
            numeric_ok = bool(found) and abs(float(found[0].get("value", -1)) - expected) < 1e-6 and found[0].get("unit") == "days"
        passed = runtime_ok and solved and bool(found) and numeric_ok and seed_hashes() == original
        report["cases"].append({"case": name, "status": "PASS" if passed else "FAIL",
                                "wall_seconds": time.perf_counter() - started,
                                "runtime_db_paths": launcher.db_paths, "result": result,
                                "answer": answer.structured, "answer_text": answer.answer})
        report["seed_databases_unchanged"] = seed_hashes() == original
        if not passed:
            report["status"] = "FAIL"
            save()
            return 1
        save()
        print(f"{name}: PASS", flush=True)
    report["status"] = "PASS"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
