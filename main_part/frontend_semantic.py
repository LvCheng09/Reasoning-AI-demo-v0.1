"""Composition only: build the optional frontend without mixing it into reasoning."""
import os


def frontend_from_env():
    implementation = os.getenv("MAIN_PART_FRONTEND_IMPLEMENTATION", "legacy").strip()
    if implementation == "legacy":
        return None
    if implementation != "semantic_v1":
        raise ValueError("unknown MAIN_PART_FRONTEND_IMPLEMENTATION")
    from semantic_frontend import SemanticFrontendAdapter, SimpleSemanticFrontend, SingleSmallLLM
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError("semantic_v1 requires LLM_BASE_URL and LLM_MODEL")
    llm = SingleSmallLLM(base_url, model, os.getenv("LLM_API_KEY", "local"))
    return SemanticFrontendAdapter(SimpleSemanticFrontend(llm, embedder=None))
