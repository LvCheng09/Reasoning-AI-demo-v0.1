"""Composition only: build the optional frontend without mixing it into reasoning."""
import os

REQUIRED_FRONTEND_MODEL = "Qwen/Qwen3-8B"


def validate_frontend_selection():
    implementation = os.getenv("MAIN_PART_FRONTEND_IMPLEMENTATION", "semantic_v1").strip()
    if implementation != "semantic_v1":
        raise ValueError("only MAIN_PART_FRONTEND_IMPLEMENTATION=semantic_v1 is supported")
    model = os.getenv("LLM_MODEL", "").strip()
    if model and model != REQUIRED_FRONTEND_MODEL:
        raise ValueError(f"only LLM_MODEL={REQUIRED_FRONTEND_MODEL} is supported")


def frontend_from_env():
    validate_frontend_selection()
    from semantic_frontend import SemanticFrontendAdapter, SimpleSemanticFrontend, SingleSmallLLM
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError("semantic_v1 requires LLM_BASE_URL and LLM_MODEL")
    llm = SingleSmallLLM(
        base_url, model, os.getenv("LLM_API_KEY", "local"),
        timeout=int(os.getenv("LLM_READ_TIMEOUT_SECONDS", "180")),
        max_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
    )
    return SemanticFrontendAdapter(SimpleSemanticFrontend(llm, embedder=None))
