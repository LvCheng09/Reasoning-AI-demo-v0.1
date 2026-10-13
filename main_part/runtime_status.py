"""Shared terminal classification for both command-line entry points."""
from collections.abc import Mapping


def execution_outcome(result: Mapping) -> str:
    if any(isinstance(item, Mapping) and item.get("operation") == "runtime_delivery_failure"
           for item in result.get("outputs", []) or []):
        return "runtime_failed"
    summary = result.get("runtime_summary") or {}
    if summary.get("scheduler_status") != "completed":
        return "incomplete"
    status = (summary.get("task_result") or {}).get("status")
    return status if status in {"solved", "blocked"} else "unresolved"


def exit_code(outcome: str) -> int:
    return 0 if outcome in {"solved", "blocked"} else 1 if outcome == "runtime_failed" else 2
