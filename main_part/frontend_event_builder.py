"""Public boundary for frontend LLM event-graph builders."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class FrontendEventBuilder(Protocol):
    """Build and revise event graphs without exposing frontend internals."""

    def build_event_graph(self, game_input: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return an ``event_graph.v1`` candidate for one input."""

    def patch_event_graph(
        self,
        *,
        game_input: Mapping[str, Any],
        current_event_graph: Mapping[str, Any],
        patches: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        """Return a revised ``event_graph.v1`` candidate."""


__all__ = ["FrontendEventBuilder"]
