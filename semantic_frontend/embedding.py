from __future__ import annotations

"""Optional standalone embedding implementation."""
from typing import Any, Optional

# ============================================================
# 5. Python embedding = second normalization
# ============================================================

class SentenceTransformerEmbedder:
    """
    canonical_text -> normalized semantic feature vector

    The vector is supplementary.
    Explicit graph fields and relations remain the primary precise structure.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Install sentence-transformers first: "
                "pip install sentence-transformers"
            ) from e

        self.model_name = model_name_or_path
        kwargs: dict[str, Any] = {"trust_remote_code": False}
        if device:
            kwargs["device"] = device

        self.model = SentenceTransformer(
            model_name_or_path,
            **kwargs,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
