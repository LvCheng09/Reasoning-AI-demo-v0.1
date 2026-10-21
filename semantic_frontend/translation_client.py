"""Fixed authorized translation transport.

Raw responses are retained before validation; only Qwen3-8B is authorized.
"""
from copy import deepcopy
import os
import time

from .llm_client import SingleSmallLLM

MODEL = "Qwen/Qwen3-8B"
ENDPOINT = "https://api.siliconflow.cn/v1"
class FixedTranslationLLM(SingleSmallLLM):
    """One authorized model; retain raw replies before parsing/validation."""
    def __init__(self, on_response=None, *, timeout_seconds=900, max_retries=1):
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError("max_retries must be an integer from 0 to 2")
        if os.getenv("LLM_MODEL", MODEL) != MODEL or os.getenv("LLM_BASE_URL", ENDPOINT).rstrip("/") != ENDPOINT:
            raise ValueError("Only SiliconFlow Qwen/Qwen3-8B is authorized")
        key = os.getenv("LLM_API_KEY", "").strip()
        if not key:
            raise ValueError("LLM_API_KEY is required")
        super().__init__(ENDPOINT, MODEL, key, timeout=timeout_seconds,
                         max_tokens=8192, max_retries=max_retries)
        self.records = []
        self.on_response = on_response

    def _payload(self, user_prompt):
        result = super()._payload(user_prompt)
        from .source_contract import SYSTEM as source_system
        result["messages"][0]["content"] = source_system
        result["response_format"] = {"type": "json_object"}
        return result

    def _post(self, payload):
        if payload["model"] != MODEL:
            raise RuntimeError("Unauthorized model request")
        start = time.perf_counter()
        body = super()._post(payload)
        data = body.json()
        record = {"requested_model": MODEL, "returned_model": data.get("model"),
                  "elapsed_seconds": time.perf_counter() - start, "usage": data.get("usage"),
                  "choices": data.get("choices"), "request_messages": deepcopy(payload["messages"]),
                  "response_format": deepcopy(payload["response_format"])}
        self.records.append(record)
        if self.on_response is not None:
            self.on_response(record)
        if data.get("model") not in (None, MODEL):
            raise RuntimeError("Provider returned a different model")
        if data.get("choices", [{}])[0].get("finish_reason") == "length":
            raise ValueError("Response truncated at the output limit; re-extract without repeated facts. Partial JSON is not a valid extraction")
        return body
