from __future__ import annotations

"""One configured LLM endpoint; no routing or graph manipulation.

Designed for OpenAI-compatible chat/completions APIs, including SiliconFlow.
The frontend LLM is used as a semantic translator, so thinking is disabled by
default and the response length is bounded.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .prompts import SYSTEM_PROMPT


@dataclass
class LLMUsage:
    calls: int = 0
    input_chars: int = 0
    output_chars: int = 0


class SingleSmallLLM:
    """A single fixed model for every frontend call.

    Compatible with an OpenAI-style endpoint:
        POST {base_url}/chat/completions

    Defaults are intentionally suitable for semantic extraction:
    - thinking disabled
    - deterministic decoding
    - bounded output length
    - retry on timeout / temporary server overload

    There is intentionally:
    - no model router
    - no small/medium/large cascade
    - no task-based model selection
    """

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout: int = 180,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
    ):
        self.base_url = self._normalize_base_url(base_url)
        self.model = str(model).strip()
        self.api_key = str(api_key or "").strip()
        self.timeout = int(timeout)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.usage = LLMUsage()

        if not self.model:
            raise ValueError("model must be a non-empty string")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        value = str(base_url or "").strip()

        # Be forgiving if a Markdown link was accidentally pasted into an env var:
        # [https://api.example.com/v1](https://api.example.com/v1)
        md = re.fullmatch(r"\[(https?://[^\]]+)\]\((https?://[^)]+)\)", value)
        if md:
            value = md.group(1)

        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                "base_url must start with http:// or https://; "
                f"got {value!r}"
            )
        return value

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = str(text or "").strip()

        if not text:
            raise ValueError("LLM returned empty content")

        # Defensive cleanup in case a provider/model still emits a thinking block.
        text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(
                    "LLM response did not contain a JSON object. "
                    f"Response prefix: {text[:300]!r}"
                )
            data = json.loads(text[start : end + 1])

        if not isinstance(data, dict):
            raise ValueError(
                f"LLM JSON response must be an object, got {type(data).__name__}"
            )
        return data

    def _payload(self, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0,
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    # Shorter connect timeout, configurable read timeout.
                    timeout=(20, self.timeout),
                )

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_seconds * (2**attempt))
                        continue

                response.raise_for_status()
                return response

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
            except requests.HTTPError:
                # Non-retryable 4xx errors should surface immediately.
                raise

        raise RuntimeError(
            f"LLM request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def chat_json(self, user_prompt: str) -> dict[str, Any]:
        user_prompt = str(user_prompt)
        response = self._post(self._payload(user_prompt))

        try:
            body = response.json()
        except ValueError as exc:
            raise ValueError(
                f"LLM API returned non-JSON HTTP response: {response.text[:500]!r}"
            ) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "LLM API response missing choices[0].message.content: "
                f"{json.dumps(body, ensure_ascii=False)[:1000]}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM API returned empty choices[0].message.content")

        self.usage.calls += 1
        self.usage.input_chars += len(SYSTEM_PROMPT) + len(user_prompt)
        self.usage.output_chars += len(content)

        return self._extract_json(content)
