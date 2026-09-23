from __future__ import annotations

"""Demo-only needs and field coverage; not semantic accuracy or production matching."""
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from semantic_frontend.graph import Event, EventGraph
from semantic_frontend.embedding import SentenceTransformerEmbedder
from semantic_frontend.frontend import SimpleSemanticFrontend

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    if len(a) != len(b) or not all(math.isfinite(v) for v in a + b):
        raise ValueError("incompatible vectors")
    denominator = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    if denominator == 0:
        raise ValueError("zero vector")
    return sum(x * y for x, y in zip(a, b)) / denominator


# ============================================================
# 8. Minimal semantic-gap demo
# ============================================================

@dataclass
class SemanticNeed:
    target_semantics: str
    required_fields: list[str] = field(default_factory=list)
    incoming_relation: Optional[str] = None
    outgoing_relation: Optional[str] = None
    similarity_threshold: float = 0.55


class SemanticGapDetector:
    """
    Prototype only.

    In the real reasoning system this should be replaced by:
        Matcher blocking reason -> frontend_patch request
    """

    def __init__(self, embedder: SentenceTransformerEmbedder):
        self.embedder = embedder

    def locate_target(
        self,
        graph: EventGraph,
        target_semantics: str,
        threshold: float,
    ) -> tuple[Optional[Event], float]:
        query = self.embedder.encode([target_semantics])[0]

        best_event: Optional[Event] = None
        best_score = -1.0

        for event in graph.events:
            if event.feature is None:
                continue
            score = cosine(query, event.feature)
            if score > best_score:
                best_event = event
                best_score = score

        if best_score < threshold:
            return None, best_score

        return best_event, best_score

    def inspect(
        self,
        graph: EventGraph,
        need: SemanticNeed,
    ) -> dict[str, Any]:
        event, similarity = self.locate_target(
            graph,
            need.target_semantics,
            need.similarity_threshold,
        )

        if event is None:
            return {
                "satisfied": False,
                "target_event_id": None,
                "similarity": similarity,
                "missing": ["target_event"],
            }

        missing: list[str] = []

        for field_name in need.required_fields:
            if not getattr(event, field_name, None):
                missing.append(f"field:{field_name}")

        if need.incoming_relation:
            found = any(
                edge.target == event.id
                and edge.relation == need.incoming_relation
                for edge in graph.edges
            )
            if not found:
                missing.append(
                    f"incoming:{need.incoming_relation}"
                )

        if need.outgoing_relation:
            found = any(
                edge.source == event.id
                and edge.relation == need.outgoing_relation
                for edge in graph.edges
            )
            if not found:
                missing.append(
                    f"outgoing:{need.outgoing_relation}"
                )

        return {
            "satisfied": not missing,
            "target_event_id": event.id,
            "target_event_text": event.canonical_text,
            "similarity": similarity,
            "missing": missing,
        }


# ============================================================
# 9. Demo runner
# ============================================================

def run_case(
    frontend: SimpleSemanticFrontend,
    case: dict[str, Any],
    max_refine_rounds: int = 2,
) -> dict[str, Any]:
    graph = frontend.initial_parse(
        doc_id=case["id"],
        text=case["text"],
    )

    detector = SemanticGapDetector(frontend.embedder)
    trace: list[dict[str, Any]] = []

    for raw_need in case.get("needs", []):
        need = SemanticNeed(**raw_need)

        for round_index in range(max_refine_rounds + 1):
            result = detector.inspect(graph, need)

            trace.append({
                "need": asdict(need),
                "round": round_index,
                **result,
            })

            if result["satisfied"]:
                break

            if round_index == max_refine_rounds:
                break

            target_event_id = result["target_event_id"]

            if target_event_id is None:
                # No arbitrary first-event fallback into another mainline.
                break

            graph = frontend.refine(
                graph,
                target_event_id=target_event_id,
                need={
                    "target_semantics": need.target_semantics,
                    "missing": result["missing"],
                    "instruction": (
                        "Expose only the missing semantic detail "
                        "supported by the original source."
                    ),
                },
            )

    final_checks = [
        detector.inspect(graph, SemanticNeed(**raw_need))
        for raw_need in case.get("needs", [])
    ]

    success = (
        sum(1 for item in final_checks if item["satisfied"])
        / max(1, len(final_checks))
    )

    return {
        "case_id": case["id"],
        "field_coverage": success,
        "semantic_accuracy": "UNVERIFIED",
        "trace": trace,
        "final_checks": final_checks,
        "graph": graph.to_dict(),
    }
