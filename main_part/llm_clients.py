from __future__ import annotations

import gc
import os
import re
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from .llm_routing import DEFAULT_CHINESE_INSTRUCT_MODEL, DEFAULT_ENGLISH_INSTRUCT_MODEL


HF_MODEL_ALIASES = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-14B-MLX-4bit": "Qwen/Qwen3-14B",
    "Qwen3-14B-MLX-6bit": "Qwen/Qwen3-14B",
    DEFAULT_CHINESE_INSTRUCT_MODEL: "Qwen/Qwen3-14B",
    DEFAULT_ENGLISH_INSTRUCT_MODEL: "Qwen/Qwen3-14B",
}

MLX_MODEL_ALIASES = {
    "Qwen3-14B": "mlx-community/Qwen3-14B-4bit",
    "Qwen3-14B-MLX-4bit": "mlx-community/Qwen3-14B-4bit",
    "Qwen3-14B-MLX-6bit": "mlx-community/Qwen3-14B-6bit",
    DEFAULT_CHINESE_INSTRUCT_MODEL: "mlx-community/Qwen3-14B-4bit",
    DEFAULT_ENGLISH_INSTRUCT_MODEL: "mlx-community/Qwen3-14B-4bit",
}


def resolve_hf_model_id(model: str) -> str:
    """Resolve short routing names to HuggingFace repository ids."""
    model = str(model or "").strip()
    return os.getenv("MAIN_PART_QWEN3_INSTRUCT_MODEL_ID", HF_MODEL_ALIASES.get(model, model))


def resolve_mlx_model_id(model: str) -> str:
    """Resolve short routing names to MLX model repository ids.

    The exact MLX repo can be overridden because local users may keep a
    downloaded/symlinked model path instead of a HuggingFace repo id.
    """
    model = str(model or "").strip()
    override = os.getenv("MAIN_PART_MLX_MODEL_ID", "").strip()
    if override:
        expanded = Path(override).expanduser()
        is_path_like = override.startswith(("/", "~", ".")) or expanded.exists()
        if is_path_like:
            if not expanded.exists():
                raise FileNotFoundError(
                    f"MAIN_PART_MLX_MODEL_ID points to a local path that does not exist: {expanded}. "
                    "Use the repo id mlx-community/Qwen3-14B-4bit, or replace snapshots/<hash> with the real hash."
                )
            return str(expanded)
        return override
    if model == "Qwen3-14B-MLX-6bit":
        return os.getenv("MAIN_PART_QWEN3_14B_MLX_6BIT_MODEL_ID", MLX_MODEL_ALIASES[model])
    if model in {"Qwen3-14B", "Qwen3-14B-MLX-4bit", DEFAULT_CHINESE_INSTRUCT_MODEL, DEFAULT_ENGLISH_INSTRUCT_MODEL}:
        return os.getenv("MAIN_PART_QWEN3_14B_MLX_4BIT_MODEL_ID", MLX_MODEL_ALIASES.get(model, model))
    return MLX_MODEL_ALIASES.get(model, model)


class TransformersInstructionClient:
    """Small JSON-instruction client backed by local HuggingFace Transformers.

    This is intentionally lazy: local models are only loaded when the first span
    or event request reaches that language route. In Colab/local runs, set
    MAIN_PART_LLM_BACKEND=transformers to enable it.
    """

    def __init__(
        self,
        *,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.device_map = device_map or os.getenv("MAIN_PART_HF_DEVICE_MAP", "auto")
        self.torch_dtype = torch_dtype or os.getenv("MAIN_PART_HF_TORCH_DTYPE", "auto")
        self.max_new_tokens = int(max_new_tokens or os.getenv("MAIN_PART_HF_MAX_NEW_TOKENS", "2048"))
        self.temperature = float(temperature if temperature is not None else os.getenv("MAIN_PART_HF_TEMPERATURE", "0"))
        self._models: dict[str, tuple[Any, Any]] = {}

    def extract_spans(self, *, text: str, prompt: str, model: str) -> str:
        return self.complete(prompt=prompt, text=text, model=model)

    def extract_events(self, *, text: str, prompt: str, model: str) -> str:
        return self.complete(prompt=prompt, text=text, model=model)

    def complete(self, *, prompt: str, text: str = "", model: str, max_new_tokens: int | None = None) -> str:
        tokenizer, causal_model = self._load_model(model)
        messages = [
            {
                "role": "system",
                "content": "你只输出可解析 JSON。禁止 Markdown、解释、代码块、<think>。第一字符必须是 { 或 [。",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(prompt, return_tensors="pt").input_ids
        device = getattr(causal_model, "device", None)
        if device is not None and hasattr(encoded, "to"):
            encoded = encoded.to(device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        try:
            output = causal_model.generate(encoded, **generation_kwargs)
            generated = output[0][encoded.shape[-1]:] if hasattr(encoded, "shape") else output[0]
            text_out = tokenizer.decode(generated, skip_special_tokens=True)
        finally:
            clear_torch_cuda_cache()
        return _extract_json_like_text(text_out)

    def _load_model(self, model: str) -> tuple[Any, Any]:
        model_id = resolve_hf_model_id(model)
        cached = self._models.get(model_id)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Transformers LLM backend requires torch and transformers. "
                "Install them or use an OpenAI-compatible endpoint."
            ) from exc

        dtype = self._resolve_torch_dtype(torch)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        causal_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        causal_model.eval()
        self._models[model_id] = (tokenizer, causal_model)
        return tokenizer, causal_model

    def _resolve_torch_dtype(self, torch_module: Any) -> Any | None:
        dtype = str(self.torch_dtype or "").strip().lower()
        if not dtype or dtype == "auto":
            return None
        if dtype in {"fp16", "float16", "half"}:
            return torch_module.float16
        if dtype in {"bf16", "bfloat16"}:
            return torch_module.bfloat16
        if dtype in {"fp32", "float32"}:
            return torch_module.float32
        return None


class MLXInstructionClient:
    """Local instruction client backed by mlx-lm.

    This is the intended MacBook path. It uses one cached Qwen 14B model for
    span/event prompts and returns only the JSON-looking part of the
    generation.
    """

    def __init__(
        self,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.max_new_tokens = int(max_new_tokens or os.getenv("MAIN_PART_MLX_MAX_NEW_TOKENS", "768"))
        self.temperature = float(temperature if temperature is not None else os.getenv("MAIN_PART_MLX_TEMPERATURE", "0"))
        self.clear_cache_after_call = os.getenv("MAIN_PART_MLX_CLEAR_CACHE_AFTER_CALL", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self._models: dict[str, tuple[Any, Any, Any]] = {}

    def extract_spans(self, *, text: str, prompt: str, model: str) -> str:
        return self.complete(prompt=prompt, text=text, model=model)

    def extract_events(self, *, text: str, prompt: str, model: str) -> str:
        return self.complete(prompt=prompt, text=text, model=model)

    def complete(self, *, prompt: str, text: str = "", model: str, max_new_tokens: int | None = None) -> str:
        del text
        model_obj, tokenizer, generate_fn = self._load_model(model)
        clear_mlx_cache()
        messages = [
            {
                "role": "system",
                "content": "你只输出严格 JSON。禁止 Markdown、解释、代码块、<think>。第一字符必须是 { 或 [。",
            },
            {"role": "user", "content": f"{prompt}\n\n/no_think"},
        ]
        max_tokens = int(max_new_tokens or self.max_new_tokens)
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"
        try:
            try:
                output = generate_fn(
                    model_obj,
                    tokenizer,
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temp=self.temperature,
                    verbose=False,
                )
            except TypeError:
                try:
                    output = generate_fn(
                        model_obj,
                        tokenizer,
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                    )
                except TypeError:
                    output = generate_fn(
                        model_obj,
                        tokenizer,
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                    )
        finally:
            if self.clear_cache_after_call:
                clear_mlx_cache()
        return _extract_json_like_text(str(output))

    def _load_model(self, model: str) -> tuple[Any, Any, Any]:
        model_id = resolve_mlx_model_id(model)
        cached = self._models.get(model_id)
        if cached is not None:
            return cached
        try:
            from mlx_lm import generate, load
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "MLX LLM backend requires mlx-lm. Install it with `pip install mlx-lm`, "
                "or set MAIN_PART_LLM_BACKEND=transformers/openai-compatible."
            ) from exc
        model_obj, tokenizer = load(model_id)
        cached = (model_obj, tokenizer, generate)
        self._models[model_id] = cached
        return cached


def _role_env(role: str, name: str, default: str = "") -> str:
    role = role.upper()
    for key in (
        f"MAIN_PART_{role}_{name}",
        f"MAIN_PART_REMOTE_{role}_{name}",
        f"MAIN_PART_REMOTE_API_{name}",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return os.getenv(f"MAIN_PART_REMOTE_{name}", default).strip() or default


class RemoteInstructionClient:
    """OpenAI-compatible remote JSON client for one LLM role.

    The local process keeps DB/rules/memory/private trace. This client sends
    only the task prompt it receives for its role to a configured remote
    /chat/completions endpoint.
    """

    def __init__(self, *, role: str) -> None:
        self.role = str(role or "event").strip().lower()
        self.base_url = _role_env(self.role, "API_BASE", "").rstrip("/")
        self.api_key = _role_env(self.role, "API_KEY", "")
        self.model = _role_env(self.role, "MODEL", "")
        self.temperature = float(_role_env(self.role, "TEMPERATURE", os.getenv("MAIN_PART_REMOTE_TEMPERATURE", "0")) or 0)
        self.timeout = float(_role_env(self.role, "TIMEOUT", os.getenv("MAIN_PART_LLM_TIMEOUT", "120")) or 120)
        self.response_format = os.getenv("MAIN_PART_REMOTE_RESPONSE_FORMAT", "json_object").strip()
        self.enable_thinking = _optional_bool(_role_env(self.role, "ENABLE_THINKING", ""))
        self.system_prompt = _role_env(
            self.role,
            "SYSTEM_PROMPT",
            "你只输出可解析 JSON。禁止 Markdown、解释、代码块、<think>。第一字符必须是 { 或 [。",
        )

    def extract_spans(self, *, text: str, prompt: str, model: str) -> str:
        return self.complete(prompt=prompt, text=text, model=model)

    def extract_events(self, *, text: str, prompt: str, model: str) -> str:
        return self.complete(prompt=prompt, text=text, model=model)

    def complete(self, *, prompt: str, text: str = "", model: str, max_new_tokens: int | None = None) -> str:
        del text
        if not self.api_key:
            raise RuntimeError(
                f"Remote {self.role} LLM API key is not configured. "
                f"Set MAIN_PART_{self.role.upper()}_API_KEY or MAIN_PART_REMOTE_API_KEY."
            )
        if not self.base_url:
            raise RuntimeError(
                f"Remote {self.role} LLM API base is not configured. "
                f"Set MAIN_PART_{self.role.upper()}_API_BASE or MAIN_PART_REMOTE_API_BASE."
            )
        body: dict[str, Any] = {
            "model": self.model or model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self._max_tokens(max_new_tokens),
        }
        if self.enable_thinking is not None:
            body["enable_thinking"] = self.enable_thinking
        if self.response_format:
            body["response_format"] = {"type": self.response_format}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            raise RuntimeError(f"Remote {self.role} LLM HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Remote {self.role} LLM request failed: {type(exc).__name__}: {exc}") from exc
        content = (
            (payload.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _extract_json_like_text(str(content))

    def _max_tokens(self, max_new_tokens: int | None) -> int:
        value = _role_env(self.role, "MAX_TOKENS", "")
        if value:
            return int(value)
        return int(max_new_tokens or os.getenv("MAIN_PART_REMOTE_MAX_TOKENS", "1024"))


def maybe_build_remote_instruction_clients_from_env() -> dict[str, RemoteInstructionClient] | None:
    backend = os.getenv("MAIN_PART_LLM_BACKEND", "").strip().lower()
    if backend not in {"remote", "remote_api", "api", "openai_compatible", "openai-compatible"}:
        return None
    return {
        "span": RemoteInstructionClient(role="span"),
        "event": RemoteInstructionClient(role="event"),
    }


def _optional_bool(value: str) -> bool | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {value!r}")


def clear_mlx_cache() -> dict[str, Any]:
    """Release MLX temporary buffers and report cleanup failures."""
    errors: list[dict[str, str]] = []
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        gc.collect()
        return {"ok": False, "status": "not_available", "errors": [{"operation": "import_mlx", "error": repr(exc)}]}
    except Exception as exc:
        gc.collect()
        return {"ok": False, "status": "import_failed", "errors": [{"operation": "import_mlx", "error": repr(exc)}]}
    for operation, callback in (("synchronize", mx.synchronize), ("clear_cache", mx.clear_cache)):
        try:
            callback()
        except Exception as exc:
            errors.append({"operation": operation, "error": repr(exc)})
    gc.collect()
    return {"ok": not errors, "status": "completed" if not errors else "failed", "errors": errors}


def clear_torch_cuda_cache() -> dict[str, Any]:
    """Release PyTorch CUDA temporary buffers and report cleanup failures."""
    errors: list[dict[str, str]] = []
    try:
        import torch
    except ModuleNotFoundError as exc:
        gc.collect()
        return {"ok": False, "status": "not_available", "errors": [{"operation": "import_torch", "error": repr(exc)}]}
    except Exception as exc:
        gc.collect()
        return {"ok": False, "status": "import_failed", "errors": [{"operation": "import_torch", "error": repr(exc)}]}
    if not torch.cuda.is_available():
        gc.collect()
        return {"ok": True, "status": "not_required", "errors": []}
    for operation, callback in (
        ("synchronize", torch.cuda.synchronize),
        ("empty_cache", torch.cuda.empty_cache),
        ("ipc_collect", torch.cuda.ipc_collect),
    ):
        try:
            callback()
        except Exception as exc:
            errors.append({"operation": operation, "error": repr(exc)})
    gc.collect()
    return {"ok": not errors, "status": "completed" if not errors else "failed", "errors": errors}


def maybe_build_instruction_client_from_env() -> TransformersInstructionClient | MLXInstructionClient | RemoteInstructionClient | None:
    backend = os.getenv("MAIN_PART_LLM_BACKEND", "").strip().lower()
    if backend in {"mlx", "mlx-lm", "mlx_lm", "local_mlx"}:
        return MLXInstructionClient()
    if backend in {"transformers", "hf", "huggingface", "local"}:
        return TransformersInstructionClient()
    if backend in {"remote", "remote_api", "api", "openai_compatible", "openai-compatible"}:
        # Backward-compatible fallback for callers that expect one client.
        return RemoteInstructionClient(role="event")
    return None


def _extract_json_like_text(text: str) -> str:
    text = _strip_thinking_text(str(text or "")).strip()
    if not text:
        return text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            candidate = ""
    for candidate in reversed(_balanced_json_candidates(text)):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    candidates = _balanced_json_candidates(text)
    if candidates:
        return candidates[-1].strip()
    return text


def _strip_thinking_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if re.search(r"<think>", text, flags=re.I):
        starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
        if starts:
            return text[min(starts):]
    return re.sub(r"</?think>", "", text, flags=re.I)


def _balanced_json_candidates(text: str) -> list[str]:
    """Return balanced JSON-looking object/array substrings.

    Local instruction models often echo prompt examples before the final JSON.
    A first-brace-to-last-brace slice joins unrelated JSON blocks and becomes
    invalid.  Balanced scanning keeps each candidate separate and lets callers
    prefer the last valid block.
    """

    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    closing = {"}", "]"}
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in pairs:
            if not stack:
                start = idx
            stack.append(pairs[ch])
            continue
        if ch in closing and stack:
            expected = stack.pop()
            if ch != expected:
                stack.clear()
                start = None
                continue
            if not stack and start is not None:
                candidates.append(text[start : idx + 1].strip())
                start = None
    return candidates
