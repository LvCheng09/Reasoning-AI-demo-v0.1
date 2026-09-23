"""Matcher package.

The matcher itself performs hard required-pattern matching. Matcher rules are
control rules split into pre-match candidate preparation and post-match
temporary boundary revision.
"""

from __future__ import annotations

from typing import Any

from ..data_structure import WorkingMemoryState
from .matching import match_resolved_endpoints
from .routing import normalize_endpoint, resolve_endpoint


def match_source_to_target(
    args: dict[str, Any],
    state: Any,
    working_memory: WorkingMemoryState,
) -> dict[str, Any]:
    """Public matcher entry used by scheduler_new.

    Scheduler and pre-match matcher rules decide which candidates to test.
    Matcher resolves endpoints and performs hard matching.
    """

    source_spec = normalize_endpoint(args.get("source"))
    target_spec = normalize_endpoint(args.get("target"))
    if not source_spec or not target_spec:
        return {
            "ok": False,
            "produced": [
                {
                    "operation": "match",
                    "source": source_spec,
                    "target": target_spec,
                    "error": "source_and_target_required",
                }
            ],
            "metadata": {"source": source_spec, "target": target_spec},
        }
    source = resolve_endpoint(source_spec, args, state, working_memory, role="source")
    resolved_args = dict(args)
    if source.get("vectors"):
        resolved_args["_source_vectors"] = list(source.get("vectors") or [])
    target = resolve_endpoint(target_spec, resolved_args, state, working_memory, role="target")
    result = match_resolved_endpoints(source, target, resolved_args, state, working_memory)
    result.setdefault("metadata", {})
    result["metadata"].setdefault("source", source_spec)
    result["metadata"].setdefault("target", target_spec)
    result["metadata"].setdefault("source_kind", source["kind"])
    result["metadata"].setdefault("target_kind", target["kind"])
    return result


__all__ = ["match_source_to_target"]
