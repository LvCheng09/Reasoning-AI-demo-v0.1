from __future__ import annotations

import os
import re
from typing import Any


DEFAULT_CHINESE_INSTRUCT_MODEL = "Qwen3-14B-MLX-4bit"
DEFAULT_ENGLISH_INSTRUCT_MODEL = "Qwen3-14B-MLX-4bit"


def detect_text_language(text: str) -> str:
    """Return zh/en/mixed/unknown using cheap deterministic signals."""
    text = str(text or "")
    if not text.strip():
        return "unknown"
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if cjk_count == 0 and latin_count == 0:
        return "unknown"
    if cjk_count >= max(3, latin_count * 0.35):
        return "zh"
    if latin_count >= max(8, cjk_count * 2.5):
        return "en"
    return "mixed"


def select_instruction_model(extractor: Any, text: str, *, task_prefix: str = "llm") -> str:
    """Route all local instruction tasks to the configured Qwen route by default."""
    language = detect_text_language(text)
    chinese_model = (
        str(getattr(extractor, f"{task_prefix}_chinese_model", "") or "")
        or os.getenv(f"MAIN_PART_{task_prefix.upper()}_CHINESE_MODEL", "")
        or os.getenv("MAIN_PART_LLM_CHINESE_MODEL", "")
        or DEFAULT_CHINESE_INSTRUCT_MODEL
    )
    english_model = (
        str(getattr(extractor, f"{task_prefix}_english_model", "") or "")
        or os.getenv(f"MAIN_PART_{task_prefix.upper()}_ENGLISH_MODEL", "")
        or os.getenv("MAIN_PART_LLM_ENGLISH_MODEL", "")
        or DEFAULT_ENGLISH_INSTRUCT_MODEL
    )
    fallback_model = (
        str(getattr(extractor, f"{task_prefix}_model", "") or "")
        or os.getenv(f"MAIN_PART_{task_prefix.upper()}_MODEL", "")
        or chinese_model
    )
    if language == "en":
        return english_model
    if language in {"zh", "mixed"}:
        return chinese_model
    return fallback_model
