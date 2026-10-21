"""Canonical rule layer names."""

from __future__ import annotations

from typing import Any


ALLOWED_RULE_LAYERS = {
    "auto_learning",
    "check_policy",
    "common_sense",
    "control",
    "domain",
    "matcher_rule",
    "meta",
    "structure",
}

CONTENT_LAYERS = {"common_sense", "domain", "check_policy"}
DEPRECATED_RULE_LAYER_ALIASES = {"common", "commonsense"}


def normalize_rule_layer(value: Any, *, default: str = "domain") -> str:
    layer = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "common": "common_sense",
        "commonsense": "common_sense",
        "matcher": "matcher_rule",
        "match_control": "matcher_rule",
        "matcher_control": "matcher_rule",
        "structural": "structure",
    }
    layer = aliases.get(layer, layer)
    if layer in ALLOWED_RULE_LAYERS:
        return layer
    if not layer and default:
        return normalize_rule_layer(default, default="domain")
    if not layer:
        return ""
    raise ValueError(f"unsupported rule_layer: {layer!r}")
