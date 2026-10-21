"""Identity checks for derived semantic evidence."""
import hashlib
import json
from .knowledge_schema_contract import field_value


def fingerprint(span):
    return hashlib.sha256(json.dumps(span, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def alignment_is_current(span, spans, relations=None):
    proof = span.get("metadata", {}).get("semantic_alignment")
    if not proof:
        return True
    if span.get("_invalid_alignment"):
        return False
    source = spans.get(proof.get('source_span_id'))
    if not source or any(field_value(span, field) != field_value(source, field) for field in proof.get('preserved_fields', ['value', 'unit', 'entity_id', 'period'])):
        return False
    if relations is not None and any(edge not in relations for edge in proof.get("source_relations", [])):
        return False
    return all(key in spans and fingerprint(spans[key]) == value
               for key, value in proof.get("source_fingerprints", {}).items()) and bool(proof.get("source_fingerprints"))
