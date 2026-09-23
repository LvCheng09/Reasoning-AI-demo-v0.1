"""Long-running runtime host for SchedulerNew.

The launcher does not reason. It watches working memory for frontend-ready
inputs, submits them to the scheduler, and lets the scheduler finish the loop.
"""

from __future__ import annotations

import time
import logging
from copy import deepcopy
from threading import Lock
from typing import Any

from .data_structure import WorkingMemoryState
from .scheduler_new import SchedulerNew


class RuntimeLauncher:
    def __init__(
        self,
        scheduler: SchedulerNew,
        *,
        working_memory: WorkingMemoryState | None = None,
        poll_interval: float = 0.2,
        max_attempts: int = 3,
    ) -> None:
        self.scheduler = scheduler
        self.working_memory = working_memory or scheduler.working_memory
        self.scheduler.working_memory = self.working_memory
        self.poll_interval = max(0.01, float(poll_interval))
        self.processed_entry_ids: set[str] = set()
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = int(max_attempts)
        self.retry_counts: dict[str, int] = {}
        self.dead_letters: dict[str, dict[str, Any]] = {}
        self._run_lock = Lock()

    def run_once(self) -> list[dict[str, Any]]:
        # Prevent two polling callers from delivering the same input together.
        with self._run_lock:
            return self._run_once()

    def _run_once(self) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        entries = self.working_memory.query_entries(
            zone="input_workspace",
            payload_type="frontend_ready_input",
            reverse=False,
        )
        for entry in entries:
            delivery = entry.payload.get("_runtime_delivery", {})
            if entry.entry_id in self.processed_entry_ids or delivery.get("status") in {"processed", "dead_letter"}:
                continue
            # A retry restarts from pre-delivery in-memory state, never from a
            # half-executed graph. Persistent/tool effects remain at-least-once.
            fields = ("entries", "zone_entries", "event_workspaces", "_event_claim_reservations", "_event_claim_consumption")
            checkpoint = deepcopy((self.scheduler.state, {name: getattr(self.working_memory, name) for name in fields}))
            attempt = int(delivery.get("attempts", 0)) + 1
            try:
                self.scheduler.clear_transient_state(clear_outputs=True, clear_errors=True, clear_runtime_workspaces=False)
                payload = dict(entry.payload)
                payload.pop("_runtime_delivery", None)
                self.scheduler.receive_input(payload, source="runtime_launcher")
                self.scheduler.run()
                if self.scheduler.state.status != "completed":
                    raise RuntimeError(f"scheduler did not complete: {self.scheduler.state.status}; {self.scheduler.state.blocking_reason}")
            except Exception as exc:
                logging.getLogger(__name__).exception("Runtime delivery failed for %s, attempt %s", entry.entry_id, attempt)
                self.scheduler.state = checkpoint[0]
                for name, value in checkpoint[1].items():
                    setattr(self.working_memory, name, value)
                live_entry = next(item for item in self.working_memory.entries if item.entry_id == entry.entry_id)
                status = "dead_letter" if attempt >= self.max_attempts else "retry_pending"
                failure = {"status": status, "entry_id": entry.entry_id, "attempts": attempt,
                           "retry_count": attempt, "error_type": type(exc).__name__, "error": str(exc)}
                live_entry.payload["_runtime_delivery"] = failure
                self.retry_counts[entry.entry_id] = attempt
                if status == "dead_letter":
                    self.dead_letters[entry.entry_id] = deepcopy(failure)
                outputs.append({"operation": "runtime_delivery_failure", **failure})
                # Keep FIFO: do not let later inputs supersede this retry.
                break
            self.processed_entry_ids.add(entry.entry_id)
            entry.payload["_runtime_delivery"] = {"status": "processed", "attempts": attempt}
            self.retry_counts.pop(entry.entry_id, None)
            output = self.scheduler.read_latest_output()
            if output:
                outputs.append(output)
        return outputs

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.poll_interval)


__all__ = ["RuntimeLauncher"]
