from __future__ import annotations

"""Semantic output decoder for the explicit reasoning model.

This module turns the model's explicit result state into rough but semantically
faithful text. It does not perform new reasoning or arithmetic.

Priority order:
1. Existing symbolic/text semantics already stored on the span/conclusion.
2. Structured attributes such as feature_type/value/unit/provenance.
3. Vector -> nearest semantic prototype only as a fallback.

The decoder is intentionally backend-agnostic: it accepts dict-like runtime
objects and imports no scheduler/executor/frontend code.
"""

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Iterable, Mapping, Optional, Sequence


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _normalize_string_sequence(value: Any) -> list[str]:
    """Accept either one string or a sequence of strings without splitting strings into chars."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple, set)):
        raise TypeError("expected a string or a sequence of strings")
    out: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _finite_vector(value: Any) -> Optional[list[float]]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    if not items:
        return None
    out: list[float] = []
    try:
        for item in items:
            number = float(item)
            if number != number or number in (float("inf"), float("-inf")):
                return None
            out.append(number)
    except (TypeError, ValueError):
        return None
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / (na * nb)


@dataclass(frozen=True)
class SemanticPrototype:
    text: str
    vector: tuple[float, ...]
    kind: str = "semantic"
    feature_type: str = ""


@dataclass(frozen=True)
class PrototypeMatch:
    text: str
    score: float
    kind: str
    feature_type: str = ""


class SemanticPrototypeIndex:
    """Small in-memory index for approximate vector decoding.

    Core/relation/nature live in different semantic spaces in the current model,
    so relation and nature lookup are intentionally strict: a generic/core
    prototype is never used as a relation or nature phrase.
    """

    VALID_KINDS = {"core", "relation", "nature", "semantic"}

    def __init__(
        self,
        prototypes: Iterable[SemanticPrototype] = (),
        *,
        min_score: float = 0.45,
    ):
        self.prototypes = [p for p in prototypes if p.kind in self.VALID_KINDS]
        self.min_score = float(min_score)
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        min_score: float = 0.45,
    ) -> "SemanticPrototypeIndex":
        prototypes: list[SemanticPrototype] = []
        for record in records:
            feature_type = _first_text(
                record.get("feature_type"),
                record.get("feature_schema_id"),
            )
            for text_key, vector_key in (
                ("core", "span_core_vector"),
                ("relation", "relation_vector"),
                ("nature", "nature_vector"),
            ):
                text = _clean_text(record.get(text_key))
                vector = _finite_vector(record.get(vector_key))
                if text and vector:
                    prototypes.append(
                        SemanticPrototype(
                            text=text,
                            vector=tuple(vector),
                            kind=text_key,
                            feature_type=feature_type,
                        )
                    )

            # Generic vectors are allowed as a fallback for core semantics only.
            generic_text = _first_text(
                record.get("core"),
                record.get("text"),
                feature_type,
            )
            generic_vector = (
                _finite_vector(record.get("vector"))
                or _finite_vector(record.get("feature"))
            )
            if generic_text and generic_vector:
                prototypes.append(
                    SemanticPrototype(
                        text=generic_text,
                        vector=tuple(generic_vector),
                        kind="semantic",
                        feature_type=feature_type,
                    )
                )
        return cls(prototypes, min_score=min_score)

    def nearest(
        self,
        vector: Any,
        *,
        kind: Optional[str] = None,
        feature_type: str = "",
    ) -> Optional[PrototypeMatch]:
        query = _finite_vector(vector)
        if query is None:
            return None

        best: Optional[PrototypeMatch] = None
        best_same_feature = False

        for prototype in self.prototypes:
            if kind in {"relation", "nature"}:
                if prototype.kind != kind:
                    continue
            elif kind == "core":
                if prototype.kind not in {"core", "semantic"}:
                    continue
            elif kind and prototype.kind != kind:
                continue

            score = _cosine(query, prototype.vector)
            same_feature = bool(
                feature_type
                and prototype.feature_type
                and prototype.feature_type == feature_type
            )

            # Feature type is used only as a tie-breaker. It must not inflate
            # cosine above 1.0 or change the meaning of min_score.
            if best is None:
                choose = True
            elif score > best.score + 1e-12:
                choose = True
            elif abs(score - best.score) <= 1e-12 and same_feature and not best_same_feature:
                choose = True
            else:
                choose = False

            if choose:
                best = PrototypeMatch(
                    text=prototype.text,
                    score=score,
                    kind=prototype.kind,
                    feature_type=prototype.feature_type,
                )
                best_same_feature = same_feature

        if best is None or best.score < self.min_score:
            return None
        return best


@dataclass
class DecodedSpan:
    span_id: str
    feature_type: str
    semantic_text: str
    relation_text: str
    nature_text: str
    value: Any = None
    unit: str = ""
    role: str = ""
    source: str = ""
    derived_by_rule_id: str = ""
    evidence_span_ids: tuple[str, ...] = ()
    confidence: Optional[float] = None
    semantic_source: str = "stored_text"

    def compact_value(self) -> str:
        if self.value is None:
            return ""
        return f"{self.value} {self.unit}".strip()

    def rough_sentence(self) -> str:
        label = self.semantic_text or self.feature_type or self.span_id
        value = self.compact_value()
        if value:
            return f"{label} = {value}."
        if self.relation_text and self.nature_text:
            return f"{label}: {self.relation_text}; {self.nature_text}."
        if self.nature_text:
            return f"{label}: {self.nature_text}."
        return f"{label}."


@dataclass
class DecodedRelation:
    source_id: str
    target_id: str
    relation_type: str
    source_text: str = ""
    target_text: str = ""
    confidence: Optional[float] = None

    def rough_sentence(self) -> str:
        source = self.source_text or self.source_id
        target = self.target_text or self.target_id
        relation = self.relation_type or "related_to"
        return f"{source} --{relation}--> {target}."


@dataclass
class DecodedConclusion:
    question_id: str
    answer: str
    status: str
    confidence: Optional[float] = None
    supporting_evidence: tuple[str, ...] = ()
    supporting_rules: tuple[str, ...] = ()
    unresolved_conditions: tuple[str, ...] = ()
    reasoning_chain: tuple[str, ...] = ()


@dataclass
class StructuredAnswer:
    status: str
    conclusions: list[DecodedConclusion]
    target_spans: list[DecodedSpan]
    supporting_spans: list[DecodedSpan]
    relations: list[DecodedRelation]
    unresolved: list[str]
    rough_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "conclusions": [asdict(x) for x in self.conclusions],
            "target_spans": [asdict(x) for x in self.target_spans],
            "supporting_spans": [asdict(x) for x in self.supporting_spans],
            "relations": [asdict(x) for x in self.relations],
            "unresolved": list(self.unresolved),
            "rough_text": self.rough_text,
        }


class SemanticOutputDecoder:
    """Decode current model output without importing backend code.

    The input may be either:
    - the current runtime Event Graph: {"spans": ..., "events": ...}
    - ReasoningResult.to_dict(), which may additionally contain
      final_conclusions / conclusions / world_state_snapshot.
    """

    RESULT_ROLES = {
        "answer",
        "result",
        "output",
        "conclusion",
        "target",
    }

    def __init__(
        self,
        prototype_index: Optional[SemanticPrototypeIndex] = None,
        *,
        prefer_final_conclusions: bool = True,
    ):
        self.prototype_index = prototype_index
        self.prefer_final_conclusions = bool(prefer_final_conclusions)

    def decode(
        self,
        runtime_state: Mapping[str, Any],
        *,
        target_feature_types: Optional[Sequence[str]] = None,
        target_span_ids: Optional[Sequence[str]] = None,
        include_supporting_spans: bool = True,
        max_supporting_spans: int = 12,
    ) -> StructuredAnswer:
        if not isinstance(runtime_state, Mapping):
            raise TypeError("runtime_state must be a mapping")
        try:
            support_limit = max(0, int(max_supporting_spans))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_supporting_spans must be an integer") from exc

        graph = self._select_graph(runtime_state)
        spans = self._collect_spans(graph)
        span_by_id = {
            _first_text(s.get("span_id"), s.get("id")): s
            for s in spans
            if _first_text(s.get("span_id"), s.get("id"))
        }
        decoded_by_id = {
            span_id: self._decode_span(span)
            for span_id, span in span_by_id.items()
        }

        conclusions = self._collect_final_conclusions(runtime_state)
        active_conclusions = conclusions if self.prefer_final_conclusions else []
        targets = self._select_targets(
            span_by_id,
            decoded_by_id,
            target_feature_types=target_feature_types,
            target_span_ids=target_span_ids,
        )
        relations = self._collect_relations(graph, decoded_by_id)
        support_ids = self._supporting_ids(
            targets,
            relations,
            span_by_id,
            graph=graph,
            conclusions=active_conclusions,
        )
        target_ids = {item.span_id for item in targets}

        supporting: list[DecodedSpan] = []
        supporting_signatures: set[tuple[str, str, str, str, str, str]] = set()
        if include_supporting_spans and support_limit > 0:
            for span_id in support_ids:
                if span_id in target_ids:
                    continue
                item = decoded_by_id.get(span_id)
                raw_span = span_by_id.get(span_id)
                if (
                    item is not None
                    and raw_span is not None
                    and self._is_output_eligible(raw_span)
                ):
                    signature = (
                        item.feature_type,
                        item.semantic_text,
                        item.relation_text,
                        item.nature_text,
                        repr(item.value),
                        item.unit,
                    )
                    if signature in supporting_signatures:
                        continue
                    supporting_signatures.add(signature)
                    supporting.append(item)
                if len(supporting) >= support_limit:
                    break

        relevant_ids = target_ids | {item.span_id for item in supporting}
        if relevant_ids:
            relations = [
                rel for rel in relations
                if rel.source_id in relevant_ids and rel.target_id in relevant_ids
            ]
        else:
            relations = []

        unresolved = self._collect_unresolved(
            runtime_state,
            active_conclusions,
            graph=graph,
        )

        if unresolved:
            status = "unresolved"
        elif active_conclusions:
            status = self._conclusion_status(active_conclusions)
        elif targets:
            status = "ok"
        else:
            status = "no_target"

        rough_text = self._render(
            conclusions=conclusions,
            targets=targets,
            supporting=supporting,
            relations=relations,
            unresolved=unresolved,
            status=status,
            prefer_final_conclusions=self.prefer_final_conclusions,
        )

        return StructuredAnswer(
            status=status,
            conclusions=active_conclusions,
            target_spans=targets,
            supporting_spans=supporting,
            relations=relations,
            unresolved=unresolved,
            rough_text=rough_text,
        )

    # ------------------------------------------------------------------
    # Span decoding
    # ------------------------------------------------------------------

    def _decode_span(self, span: Mapping[str, Any]) -> DecodedSpan:
        metadata = _mapping(span.get("metadata"))
        semantic_event = _mapping(metadata.get("semantic_event"))

        span_id = _first_text(span.get("span_id"), span.get("id"))
        feature_type = _first_text(
            span.get("feature_type"),
            span.get("feature_schema_id"),
            semantic_event.get("event_type"),
        )

        semantic_text = _first_text(
            span.get("core"),
            semantic_event.get("canonical_text"),
        )
        # ``derived`` describes provenance, not the business concept.  The
        # executor may prefix a readable core with it, but exposing it as part
        # of a user-facing metric name produces wording such as “derived DSO”.
        # Keep provenance in ``derived_by_rule_id`` and render the concept
        # itself here.  This is a generic normalization, not domain mapping.
        if semantic_text.lower().startswith("derived "):
            semantic_text = semantic_text[8:].strip()
        semantic_source = "stored_text" if semantic_text else ""
        relation_text = _clean_text(span.get("relation"))
        nature_text = _clean_text(span.get("nature"))

        if self.prototype_index is not None:
            if not semantic_text:
                recovered = self._recover_text(span, "core", feature_type)
                if recovered:
                    semantic_text = recovered
                    semantic_source = "vector_prototype"
            if not relation_text:
                relation_text = self._recover_text(span, "relation", feature_type)
            if not nature_text:
                nature_text = self._recover_text(span, "nature", feature_type)

        if not semantic_text:
            generic_feature = feature_type.lower() in {"", "event", "span", "unknown"}
            if feature_type and not generic_feature:
                semantic_text = feature_type
                semantic_source = "schema_fallback"
            else:
                semantic_text = _first_text(span.get("text"), feature_type)
                semantic_source = "raw_text_fallback" if span.get("text") else "schema_fallback"

        evidence_span_ids = self._extract_evidence_span_ids(span)

        confidence = span.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        return DecodedSpan(
            span_id=span_id,
            feature_type=feature_type,
            semantic_text=semantic_text,
            relation_text=relation_text,
            nature_text=nature_text,
            value=span.get("value"),
            unit=_clean_text(span.get("unit")),
            role=_first_text(
                span.get("span_role"),
                span.get("role"),
                metadata.get("span_role"),
                metadata.get("role"),
            ),
            source=_first_text(span.get("source"), metadata.get("origin")),
            derived_by_rule_id=_first_text(
                span.get("derived_by_rule_id"),
                metadata.get("derived_by_rule_id"),
            ),
            evidence_span_ids=tuple(evidence_span_ids),
            confidence=confidence,
            semantic_source=semantic_source,
        )

    def _recover_text(
        self,
        span: Mapping[str, Any],
        kind: str,
        feature_type: str,
    ) -> str:
        if self.prototype_index is None:
            return ""
        keys = {
            "core": ("span_core_vector", "vector", "feature"),
            "relation": ("relation_vector",),
            "nature": ("nature_vector",),
        }[kind]
        for key in keys:
            match = self.prototype_index.nearest(
                span.get(key),
                kind=kind,
                feature_type=feature_type,
            )
            if match is not None:
                return match.text
        return ""

    @staticmethod
    def _extract_evidence_span_ids(span: Mapping[str, Any]) -> list[str]:
        metadata = _mapping(span.get("metadata"))
        found: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str):
                value = value.strip()
                if value and value not in found:
                    found.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    add(item)

        add(span.get("evidence_span_ids"))
        add(metadata.get("evidence_span_ids"))

        runtime_resolution = _mapping(span.get("runtime_value_resolution"))
        add(runtime_resolution.get("evidence_span_ids"))

        # Structured expressions/calculations may carry explicit span references.
        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    key_text = str(key).lower()
                    if key_text in {
                        "span_id",
                        "source_span_id",
                        "evidence_span_id",
                    } or key_text.endswith("_span_ids"):
                        add(child)
                    else:
                        walk(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    walk(child)

        walk(span.get("expression"))
        walk(span.get("calculation"))
        return found

    # ------------------------------------------------------------------
    # Graph selection and collection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_graph(runtime_state: Mapping[str, Any]) -> Mapping[str, Any]:
        # Explicit current/result graph wrappers have highest priority.
        for key in (
            "current_event_graph",
            "result_graph",
            "event_graph",
            "graph",
        ):
            candidate = runtime_state.get(key)
            if isinstance(candidate, Mapping) and isinstance(
                candidate.get("spans"), (list, tuple)
            ):
                return candidate

        # A live top-level runtime graph should beat stale snapshots/workspaces.
        if isinstance(runtime_state.get("spans"), (list, tuple)):
            return runtime_state

        for key in ("world_state_snapshot", "workspace"):
            candidate = runtime_state.get(key)
            if isinstance(candidate, Mapping) and isinstance(
                candidate.get("spans"), (list, tuple)
            ):
                return candidate

        return runtime_state

    @staticmethod
    def _collect_spans(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        # Top-level spans are authoritative. Event-local spans only fill missing IDs.
        by_id: dict[str, Mapping[str, Any]] = {}
        anonymous: list[Mapping[str, Any]] = []

        def add(items: Any, *, overwrite: bool) -> None:
            for span in _list_of_mappings(items):
                span_id = _first_text(span.get("span_id"), span.get("id"))
                if not span_id:
                    anonymous.append(span)
                    continue
                if overwrite or span_id not in by_id:
                    by_id[span_id] = span

        add(graph.get("spans"), overwrite=True)
        for event in _list_of_mappings(graph.get("events")):
            add(event.get("spans"), overwrite=False)

        return list(by_id.values()) + anonymous

    @staticmethod
    def _collect_relations(
        graph: Mapping[str, Any],
        decoded: Mapping[str, DecodedSpan],
    ) -> list[DecodedRelation]:
        raw: list[Mapping[str, Any]] = []
        raw.extend(_list_of_mappings(graph.get("relations")))
        raw.extend(_list_of_mappings(graph.get("edges")))
        for event in _list_of_mappings(graph.get("events")):
            raw.extend(_list_of_mappings(event.get("relations")))

        out: list[DecodedRelation] = []
        seen: set[tuple[str, str, str]] = set()

        for relation in raw:
            source = _first_text(
                relation.get("source_span_id"),
                relation.get("source"),
            )
            target = _first_text(
                relation.get("target_span_id"),
                relation.get("target"),
            )
            relation_type = _first_text(
                relation.get("relation_type"),
                relation.get("relation"),
                relation.get("core"),
            )
            key = (source, target, relation_type)
            if not source or not target or key in seen:
                continue
            seen.add(key)

            confidence = relation.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None

            out.append(
                DecodedRelation(
                    source_id=source,
                    target_id=target,
                    relation_type=relation_type,
                    source_text=(
                        decoded[source].semantic_text if source in decoded else ""
                    ),
                    target_text=(
                        decoded[target].semantic_text if target in decoded else ""
                    ),
                    confidence=confidence,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Conclusion and target selection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_final_conclusions(
        runtime_state: Mapping[str, Any],
    ) -> list[DecodedConclusion]:
        raw: list[Mapping[str, Any]] = []
        raw.extend(_list_of_mappings(runtime_state.get("final_conclusions")))

        # Some controller/output wrappers store the reasoning result in `payload`
        # or `reasoning_result` rather than exposing final_conclusions top-level.
        for container_key in ("payload", "reasoning_result"):
            container = _mapping(runtime_state.get(container_key))
            raw.extend(_list_of_mappings(container.get("final_conclusions")))

        # Stable de-duplication before decoding.
        deduped_raw: list[Mapping[str, Any]] = []
        seen_raw: set[tuple[str, str, str]] = set()
        for item in raw:
            key = (
                _clean_text(item.get("question_id")),
                _clean_text(item.get("status")),
                _clean_text(item.get("answer")),
            )
            if key in seen_raw:
                continue
            seen_raw.add(key)
            deduped_raw.append(item)

        out: list[DecodedConclusion] = []
        for item in deduped_raw:
            answer = _clean_text(item.get("answer"))
            status = _clean_text(item.get("status"))
            unresolved_raw = item.get("unresolved_conditions") or []
            if not answer and not status and not unresolved_raw:
                continue
            confidence = item.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            out.append(
                DecodedConclusion(
                    question_id=_clean_text(item.get("question_id")),
                    answer=answer,
                    status=status,
                    confidence=confidence,
                    supporting_evidence=tuple(
                        str(x) for x in (item.get("supporting_evidence") or [])
                    ),
                    supporting_rules=tuple(
                        str(x) for x in (item.get("supporting_rules") or [])
                    ),
                    unresolved_conditions=tuple(
                        str(x) for x in (item.get("unresolved_conditions") or [])
                    ),
                    reasoning_chain=tuple(
                        str(x) for x in (item.get("reasoning_chain") or [])
                    ),
                )
            )
        return out

    def _select_targets(
        self,
        span_by_id: Mapping[str, Mapping[str, Any]],
        decoded_by_id: Mapping[str, DecodedSpan],
        *,
        target_feature_types: Optional[Sequence[str]],
        target_span_ids: Optional[Sequence[str]],
    ) -> list[DecodedSpan]:
        wanted_ids = _normalize_string_sequence(target_span_ids)
        wanted_features = {
            x.lower() for x in _normalize_string_sequence(target_feature_types)
        }

        if wanted_ids:
            selected = [
                decoded_by_id[x]
                for x in wanted_ids
                if x in decoded_by_id
                and self._is_target_output_eligible(span_by_id[x])
                and not self._is_unresolved_goal_placeholder(
                    span_by_id[x], decoded_by_id[x]
                )
            ]
            return self._prefer_completed_targets(selected)

        if wanted_features:
            selected = [
                item
                for span_id, item in decoded_by_id.items()
                if item.feature_type.lower() in wanted_features
                and self._is_target_output_eligible(span_by_id[span_id])
                and not self._is_unresolved_goal_placeholder(
                    span_by_id[span_id], item
                )
            ]
            return self._prefer_completed_targets(selected)

        selected: list[DecodedSpan] = []
        for span_id, raw in span_by_id.items():
            if not self._is_target_output_eligible(raw):
                continue
            item = decoded_by_id[span_id]
            metadata = _mapping(raw.get("metadata"))
            role = item.role.lower()
            explicitly_promoted = (
                raw.get("output_eligible") is True
                or metadata.get("output_eligible") is True
            )
            is_result_role = role in self.RESULT_ROLES
            is_goal_with_value = (
                item.feature_type.lower().startswith("goal_")
                and item.value is not None
            )

            # Conservative automatic output selection:
            # - explicit result/output/conclusion roles may be text or numeric;
            # - completed goal spans may be output;
            # - otherwise a span must be explicitly promoted by Control.
            #
            # Merely being derived is not enough: intermediate derived spans are
            # common in the Rule Path and must not silently become final answers.
            if is_result_role or is_goal_with_value or explicitly_promoted:
                selected.append(item)

        return self._prefer_completed_targets(selected)


    @staticmethod
    def _is_output_eligible(span: Mapping[str, Any]) -> bool:
        """Keep provisional/candidate/shadow spans out of user-facing output.

        An explicit `output_eligible=True` may override the conservative filters
        after Control has promoted a candidate.
        """
        metadata = _mapping(span.get("metadata"))
        if span.get("output_eligible") is True or metadata.get("output_eligible") is True:
            return True

        result_policy = _first_text(
            span.get("result_policy"),
            metadata.get("result_policy"),
        ).lower()
        role = _first_text(
            span.get("span_role"),
            span.get("role"),
            metadata.get("span_role"),
            metadata.get("role"),
        ).lower()
        status = _first_text(
            span.get("status"),
            metadata.get("status"),
            _mapping(metadata.get("lifecycle")).get("status"),
        ).lower()

        if result_policy in {"candidate", "shadow", "proposal"}:
            return False
        if role in {"candidate", "shadow", "proposal"}:
            return False
        if status in {
            "candidate",
            "shadow",
            "proposal",
            "rejected",
            "reject",
            "archived",
            "invalid",
        }:
            return False
        return True

    @classmethod
    def _is_presentation_support_eligible(cls, span: Mapping[str, Any]) -> bool:
        """Exclude runtime-control and goal bookkeeping from user evidence.

        Those nodes remain in the graph and in the audit trail.  They are not
        source facts supporting the answer, and presenting them invites the
        verbalizer to explain its own scheduling mechanics instead of the
        domain result.
        """
        if not cls._is_output_eligible(span):
            return False
        metadata = _mapping(span.get("metadata"))
        role = _first_text(
            span.get("span_role"), span.get("role"),
            metadata.get("span_role"), metadata.get("role"),
        ).lower()
        feature_type = _first_text(
            span.get("feature_type"), span.get("feature_schema_id"),
        ).lower()
        core = _clean_text(span.get("core")).lower()
        if role in {
            "control_state", "control", "strategy", "reasoning_goal",
            "goal", "objective", "planning", "scheduler_state",
        }:
            return False
        if feature_type.startswith(("control_", "goal_", "strategy_")):
            return False
        return not core.startswith(("control ", "strategy "))

    @classmethod
    def _is_target_output_eligible(cls, span: Mapping[str, Any]) -> bool:
        """Stricter gate for user-facing result spans."""
        metadata = _mapping(span.get("metadata"))
        if span.get("output_eligible") is True or metadata.get("output_eligible") is True:
            return True
        if not cls._is_output_eligible(span):
            return False
        status = _first_text(
            span.get("status"),
            metadata.get("status"),
            _mapping(metadata.get("lifecycle")).get("status"),
        ).lower()
        return status not in {
            "provisional",
            "partial",
            "pending",
            "unknown",
            "unvalidated",
        }


    @staticmethod
    def _is_unresolved_goal_placeholder(
        raw: Mapping[str, Any],
        item: DecodedSpan,
    ) -> bool:
        metadata = _mapping(raw.get("metadata"))
        promoted = (
            raw.get("output_eligible") is True
            or metadata.get("output_eligible") is True
        )
        return (
            item.feature_type.lower().startswith("goal_")
            and item.value is None
            and not promoted
        )


    @staticmethod
    def _prefer_completed_targets(
        items: Sequence[DecodedSpan],
    ) -> list[DecodedSpan]:
        # Suppress only an unresolved goal alias when its completed base result
        # exists (goal_dso -> dso). Do not globally discard unrelated
        # unvalued requested outputs.
        valued_features = {
            item.feature_type.lower()
            for item in items
            if item.value is not None and item.feature_type
        }
        pool: list[DecodedSpan] = []
        for item in items:
            feature = item.feature_type.lower()
            if (
                item.value is None
                and feature.startswith("goal_")
                and feature[5:] in valued_features
            ):
                continue
            pool.append(item)

        seen: set[str] = set()
        out: list[DecodedSpan] = []
        for item in sorted(
            pool,
            key=lambda x: (
                not bool(x.derived_by_rule_id),
                x.value is None,
                x.span_id,
            ),
        ):
            if item.span_id in seen:
                continue
            seen.add(item.span_id)
            out.append(item)
        return out

    @staticmethod
    def _conclusion_status(conclusions: Sequence[DecodedConclusion]) -> str:
        statuses = {_clean_text(item.status).lower() for item in conclusions}

        unresolved_statuses = {
            "insufficient_evidence", "unresolved", "reject", "rejected",
            "no_decision", "blocked", "stop_path", "invalid", "failed",
            "failure", "error", "cancelled", "aborted",
        }
        provisional_statuses = {
            "provisional", "candidate", "partial", "pending", "unknown",
            "unvalidated",
        }
        success_statuses = {
            "final", "resolved", "complete", "completed", "accepted",
            "success", "succeeded", "ok",
        }

        if statuses & unresolved_statuses:
            return "unresolved"
        if statuses & provisional_statuses:
            return "provisional"
        if not statuses or statuses == {""}:
            return "provisional"
        if statuses <= success_statuses:
            return "ok"

        # Unknown statuses must never be promoted to a confident final answer.
        return "provisional"

    # ------------------------------------------------------------------
    # Support and unresolved state
    # ------------------------------------------------------------------

    @staticmethod
    def _supporting_ids(
        targets: Sequence[DecodedSpan],
        relations: Sequence[DecodedRelation],
        span_by_id: Mapping[str, Mapping[str, Any]],
        *,
        graph: Mapping[str, Any],
        conclusions: Sequence[DecodedConclusion] = (),
    ) -> list[str]:
        target_ids = {item.span_id for item in targets}
        support: list[str] = []

        def add(span_id: str) -> None:
            if (
                span_id
                and span_id in span_by_id
                and span_id not in target_ids
                and span_id not in support
                and SemanticOutputDecoder._is_presentation_support_eligible(
                    span_by_id[span_id]
                )
            ):
                support.append(span_id)

        for target in targets:
            for span_id in target.evidence_span_ids:
                add(span_id)

        # final_conclusions are authoritative in the current model; preserve
        # their explicit evidence links when those IDs refer to graph spans.
        for conclusion in conclusions:
            for span_id in conclusion.supporting_evidence:
                add(span_id)

        for relation in relations:
            if relation.source_id in target_ids:
                add(relation.target_id)
            if relation.target_id in target_ids:
                add(relation.source_id)

        # Prefer the local Event evidence bundle that actually contains a target.
        if not support and target_ids:
            for event in _list_of_mappings(graph.get("events")):
                event_span_ids: list[str] = []

                for value in event.get("evidence_span_ids") or []:
                    span_id = _clean_text(value)
                    if span_id and span_id not in event_span_ids:
                        event_span_ids.append(span_id)

                for local_span in _list_of_mappings(event.get("spans")):
                    span_id = _first_text(
                        local_span.get("span_id"),
                        local_span.get("id"),
                    )
                    if span_id and span_id not in event_span_ids:
                        event_span_ids.append(span_id)

                if target_ids & set(event_span_ids):
                    for span_id in event_span_ids:
                        add(span_id)

        # Last-resort compatibility fallback: all observed inputs.
        if not support:
            for span_id, span in span_by_id.items():
                role = _first_text(
                    span.get("span_role"),
                    span.get("role"),
                ).lower()
                source = _clean_text(span.get("source")).lower()
                origin = _clean_text(
                    _mapping(span.get("metadata")).get("origin")
                ).lower()
                if (
                    role == "observed"
                    or source in {"document", "input"}
                    or origin.startswith("semantic_frontend")
                ):
                    add(span_id)
        return support

    @staticmethod
    def _collect_unresolved(
        runtime_state: Mapping[str, Any],
        conclusions: Sequence[DecodedConclusion],
        *,
        graph: Mapping[str, Any],
    ) -> list[str]:
        values: list[str] = []

        for source in (runtime_state, graph):
            translation = _mapping(source.get("frontend_translation"))
            translation_items = translation.get("unresolved", []) or []
            for item in translation_items:
                if isinstance(item, Mapping):
                    values.append(
                        _first_text(
                            item.get("reason"),
                            item.get("missing"),
                            item,
                        )
                    )
                else:
                    values.append(str(item))
            translation_status = _clean_text(
                translation.get("status")
            ).lower()
            if (
                translation_status in {"unresolved", "blocked", "failed", "error"}
                and not translation_items
            ):
                values.append(f"frontend_translation:{translation_status}")

            for key in (
                "unresolved",
                "unresolved_conditions",
                "missing_information",
                "missing_slots",
            ):
                value = source.get(key)
                if isinstance(value, (list, tuple)):
                    values.extend(str(x) for x in value if str(x))

        for conclusion in conclusions:
            values.extend(conclusion.unresolved_conditions)

        failure_statuses = {
            "stop_path", "blocked", "unresolved", "no_decision",
            "insufficient_evidence", "failed", "failure", "error",
            "invalid", "rejected", "repair_value_unresolved",
            "pending_rule_not_matched", "no_rules_matched",
            "no_executable_rule_produced",
        }
        for source in (runtime_state, graph):
            top_status = _clean_text(source.get("status")).lower()
            top_reason = _clean_text(source.get("reason"))
            if top_status in failure_statuses:
                values.append(top_reason or f"status:{top_status}")

        out: list[str] = []
        for value in values:
            text = _clean_text(value)
            if text and text not in out:
                out.append(text)
        return out

    # ------------------------------------------------------------------
    # Rough renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render(
        *,
        conclusions: Sequence[DecodedConclusion],
        targets: Sequence[DecodedSpan],
        supporting: Sequence[DecodedSpan],
        relations: Sequence[DecodedRelation],
        unresolved: Sequence[str],
        status: str,
        prefer_final_conclusions: bool,
    ) -> str:
        lines = [f"STATUS: {status}"]

        if conclusions and prefer_final_conclusions:
            lines.append("FINAL_CONCLUSIONS:")
            for conclusion in conclusions:
                suffix = (
                    f" (confidence={conclusion.confidence:.3f})"
                    if conclusion.confidence is not None
                    else ""
                )
                lines.append(
                    f"- [{conclusion.status or 'unknown'}]{suffix} {conclusion.answer}"
                )

        if targets:
            lines.append("RESULT_SPANS:")
            lines.extend(f"- {item.rough_sentence()}" for item in targets)

        if supporting:
            lines.append("SUPPORT:")
            lines.extend(f"- {item.rough_sentence()}" for item in supporting)

        if relations:
            lines.append("RELATIONS:")
            lines.extend(f"- {item.rough_sentence()}" for item in relations)

        if unresolved:
            lines.append("UNRESOLVED:")
            lines.extend(f"- {item}" for item in unresolved)

        if not conclusions and not targets and not unresolved:
            lines.append(
                "No explicit final conclusion or result span was identified."
            )

        return "\n".join(lines)
