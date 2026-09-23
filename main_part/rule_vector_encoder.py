"""Small encoder helper for rule import/upsert scripts."""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Any, Iterable

from .vector_contracts import canonical_embedding_model


class RuleVectorEncoder:
    def __init__(
        self,
        *,
        encoder_name_or_path: str = "",
        max_length: int = 256,
        local_files_only: bool = False,
        load_encoder: bool = True,
        cache_size: int = 8192,
        batch_size: int = 32,
        device: str = "",
    ) -> None:
        self.encoder_name_or_path = str(encoder_name_or_path or "Qwen/Qwen3-Embedding-0.6B")
        self.embedding_model = canonical_embedding_model(self.encoder_name_or_path)
        self.vector_version = "v1"
        self.max_length = int(max_length)
        self.local_files_only = bool(local_files_only)
        self.encoder = None
        self.tokenizer = None
        self._torch = None
        self.cache_size = max(0, int(cache_size))
        self.batch_size = max(1, int(batch_size))
        self.requested_device = str(
            device or os.getenv("MAIN_PART_EMBEDDING_DEVICE", "auto")
        ).strip().lower()
        self.device = "cpu"
        self._vector_cache: OrderedDict[str, list[float] | None] = OrderedDict()
        if load_encoder:
            self._load_encoder()

    def _load_encoder(self) -> None:
        if self.encoder is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Rule vector encoder requires torch and transformers.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_name_or_path,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self.device = _resolve_torch_device(torch, self.requested_device)
        self.encoder.to(self.device)
        self.encoder.eval()
        self._torch = torch

    def _encode_text_vector(self, text: str) -> list[float] | None:
        text = str(text or "").strip()
        if not text:
            return None
        cached = self._cache_get(text)
        if cached is not _CACHE_MISS:
            return list(cached) if cached else None
        result = self.encode_text_vectors([text])[0]
        return list(result) if result else None

    def encode_text_vectors(self, texts: Iterable[str]) -> list[list[float] | None]:
        return list(self.encode_text_vectors_status(texts)["vectors"])

    def encode_text_vectors_status(self, texts: Iterable[str]) -> dict[str, Any]:
        normalized = [str(text or "").strip() for text in texts]
        results: list[list[float] | None] = [None] * len(normalized)
        errors: list[dict[str, Any]] = []
        missing_positions: dict[str, list[int]] = {}
        for index, text in enumerate(normalized):
            if not text:
                errors.append({"index": index, "status": "empty_text"})
                continue
            cached = self._cache_get(text)
            if cached is not _CACHE_MISS:
                results[index] = list(cached) if cached else None
                continue
            missing_positions.setdefault(text, []).append(index)
        if not missing_positions:
            return _encoding_status(results, errors)
        if self.encoder is None or self.tokenizer is None:
            try:
                self._load_encoder()
            except Exception as exc:
                errors.extend(
                    {
                        "index": index,
                        "status": "encoder_load_failed",
                        "error": repr(exc),
                    }
                    for positions in missing_positions.values()
                    for index in positions
                )
                return _encoding_status(results, errors)
        unique_texts = list(missing_positions)
        for start in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[start : start + self.batch_size]
            batch_result = self._encode_batch_status(batch)
            vectors = batch_result["vectors"]
            batch_errors = batch_result["errors"]
            for error in batch_errors:
                text_index = int(error.get("batch_index", -1))
                if 0 <= text_index < len(batch):
                    for original_index in missing_positions[batch[text_index]]:
                        copied = dict(error)
                        copied["index"] = original_index
                        copied.pop("batch_index", None)
                        errors.append(copied)
                else:
                    for text in batch:
                        for original_index in missing_positions[text]:
                            copied = dict(error)
                            copied["index"] = original_index
                            copied.setdefault("status", copied.get("type") or "batch_encode_failed")
                            copied.setdefault("type", copied.get("status") or "batch_encode_failed")
                            errors.append(copied)
            for text, vector in zip(batch, vectors, strict=False):
                self._cache_put(text, vector)
                for index in missing_positions[text]:
                    results[index] = list(vector) if vector else None
                    if not vector and not any(error.get("index") == index for error in errors):
                        errors.append({"index": index, "status": "empty_vector"})
        return _encoding_status(results, errors)

    def _encode_batch(self, texts: list[str]) -> list[list[float] | None]:
        return list(self._encode_batch_status(texts)["vectors"])

    def _encode_batch_status(self, texts: list[str]) -> dict[str, Any]:
        if type(self)._encode_batch is not RuleVectorEncoder._encode_batch:
            try:
                vectors = self._encode_batch(texts)
            except Exception as exc:
                return {
                    "ok": False,
                    "status": "batch_encode_failed",
                    "vectors": [None] * len(texts),
                    "errors": [
                        {
                            "batch_index": index,
                            "status": "batch_encode_failed",
                            "type": "batch_encode_failed",
                            "error": repr(exc),
                        }
                        for index, _text in enumerate(texts)
                    ],
                }
            return {
                "ok": all(bool(vector) for vector in vectors),
                "status": "ok" if all(bool(vector) for vector in vectors) else "partial",
                "vectors": list(vectors),
                "errors": [
                    {
                        "batch_index": index,
                        "status": "empty_vector",
                        "type": "empty_vector",
                    }
                    for index, vector in enumerate(vectors)
                    if not vector
                ],
            }
        try:
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(8, self.max_length),
            )
            encoded = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            with self._torch.no_grad():
                output = self.encoder(**encoded)
                hidden = output.last_hidden_state
                mask = encoded.get("attention_mask")
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors = [
                [float(value) for value in row]
                for row in pooled.detach().cpu().tolist()
            ]
            return {
                "ok": True,
                "status": "ok",
                "vectors": vectors,
                "errors": [],
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "batch_encode_failed",
                "vectors": [None] * len(texts),
                "errors": [
                    {
                        "batch_index": index,
                        "status": "batch_encode_failed",
                        "type": "batch_encode_failed",
                        "error": repr(exc),
                    }
                    for index, _text in enumerate(texts)
                ],
            }

    def _cache_get(self, text: str) -> list[float] | None | object:
        if text not in self._vector_cache:
            return _CACHE_MISS
        value = self._vector_cache.pop(text)
        self._vector_cache[text] = value
        return value

    def _cache_put(self, text: str, vector: list[float] | None) -> None:
        if self.cache_size <= 0:
            return
        self._vector_cache[text] = list(vector) if vector else None
        self._vector_cache.move_to_end(text)
        while len(self._vector_cache) > self.cache_size:
            self._vector_cache.popitem(last=False)


_CACHE_MISS = object()


def _resolve_torch_device(torch: Any, requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        if bool(getattr(getattr(torch, "backends", None), "mps", None)) and torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if value == "mps":
        if not bool(getattr(getattr(torch, "backends", None), "mps", None)) or not torch.backends.mps.is_available():
            raise RuntimeError("requested embedding device mps is unavailable")
        return "mps"
    if value == "cuda":
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            raise RuntimeError("requested embedding device cuda is unavailable")
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"unsupported embedding device: {value}")


def _encoding_status(vectors: list[list[float] | None], errors: list[dict[str, Any]]) -> dict[str, Any]:
    encoded_count = sum(1 for vector in vectors if vector)
    if encoded_count == len(vectors):
        status = "ok"
    elif encoded_count > 0:
        status = "partial"
    elif vectors:
        status = "failed"
    else:
        status = "empty"
    return {
        "ok": status == "ok",
        "status": status,
        "vectors": vectors,
        "encoded_count": encoded_count,
        "requested_count": len(vectors),
        "errors": errors,
    }


__all__ = ["RuleVectorEncoder"]
