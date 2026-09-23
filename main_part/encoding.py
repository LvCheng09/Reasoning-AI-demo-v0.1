from __future__ import annotations

from typing import Any, Sequence


def encode_context_window(
    extractor: Any,
    text: str,
    *,
    device: str,
    torch_mod: Any,
) -> dict[str, Any]:
    try:
        encoded = extractor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=extractor.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
    except (NotImplementedError, TypeError, ValueError):
        encoded = extractor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=extractor.max_length,
        )
        offsets = None

    device_inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch_mod.no_grad():
        outputs = extractor.encoder(
            **device_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        task_hidden = compute_task_hidden(
            extractor=extractor,
            hidden_states=outputs.hidden_states,
            attention_mask=device_inputs["attention_mask"],
            torch_mod=torch_mod,
        )
        span_logits = (
            extractor.span_classifier(task_hidden)
            if extractor.span_classifier is not None
            else None
        )
        head_logits = (
            extractor.head_classifier(task_hidden)
            if extractor.head_classifier is not None
            else None
        )

    return {
        "hidden": task_hidden[0].detach().cpu(),
        "offsets": offsets,
        "attention_mask": device_inputs["attention_mask"][0].detach().cpu(),
        "span_logits": span_logits[0].detach().cpu() if span_logits is not None else None,
        "head_logits": head_logits[0].detach().cpu() if head_logits is not None else None,
    }


def compute_task_hidden(
    extractor: Any,
    hidden_states: Sequence[Any],
    attention_mask: Any,
    *,
    torch_mod: Any,
) -> Any:
    available_depth = min(extractor.layer_fusion_depth, len(hidden_states))
    selected_states = list(hidden_states[-available_depth:])
    if extractor.scalar_mix is not None and available_depth == extractor.scalar_mix.num_layers:
        mixed_hidden = extractor.scalar_mix(selected_states)
    else:
        mixed_hidden = torch_mod.stack(selected_states, dim=0).mean(dim=0)

    if extractor.sequence_encoder is not None:
        mixed_hidden = extractor.sequence_encoder(mixed_hidden, attention_mask)
    return mixed_hidden


def pool_context_vector(
    extractor: Any,
    state: dict[str, Any],
    start_char: int,
    end_char: int,
    fallback_text: str,
    *,
    device: str,
    torch_mod: Any,
    bucket_span_width_fn: Any,
) -> Any:
    offsets = state["offsets"]
    if offsets is not None:
        token_indices = extract_token_indices_from_offsets(offsets, start_char, end_char)
        if token_indices:
            representation = build_span_representation(
                extractor=extractor,
                state=state,
                token_indices=token_indices,
                return_cpu=True,
                device=device,
                torch_mod=torch_mod,
                bucket_span_width_fn=bucket_span_width_fn,
            )
            if representation is not None:
                return representation
    return encode_text_vector(
        extractor=extractor,
        text=fallback_text,
        device=device,
        torch_mod=torch_mod,
    )


def build_span_representation(
    extractor: Any,
    state: dict[str, Any],
    token_indices: list[int],
    *,
    return_cpu: bool = True,
    device: str,
    torch_mod: Any,
    bucket_span_width_fn: Any,
) -> Any | None:
    if not token_indices:
        return None

    hidden = state["hidden"]
    if extractor.span_pooler is None or extractor.span_projector is None:
        pooled = hidden[token_indices].mean(dim=0)
        if return_cpu:
            return pooled
        return pooled.unsqueeze(0).to(device)

    token_states = hidden[token_indices].to(device)
    span_batch = token_states.unsqueeze(0)
    span_mask = torch_mod.ones((1, len(token_indices)), dtype=torch_mod.bool, device=device)
    start_vec = token_states[0].unsqueeze(0)
    end_vec = token_states[-1].unsqueeze(0)
    width_bucket = torch_mod.tensor(
        [bucket_span_width_fn(len(token_indices), extractor.width_bucket_count)],
        dtype=torch_mod.long,
        device=device,
    )
    with torch_mod.no_grad():
        pooled = extractor.span_pooler(span_batch, span_mask)
        projected = extractor.span_projector(start_vec, end_vec, pooled, width_bucket)
    if return_cpu:
        return projected[0].detach().cpu()
    return projected


def extract_token_indices_from_offsets(
    offsets: list[list[int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
        and not (token_end <= start_char or token_start >= end_char)
    ]


def encode_text_vector(
    extractor: Any,
    text: str,
    *,
    device: str,
    torch_mod: Any,
) -> Any:
    encoded = extractor.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=min(extractor.max_length, 128),
    )
    device_inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch_mod.no_grad():
        outputs = extractor.encoder(
            **device_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        task_hidden = compute_task_hidden(
            extractor=extractor,
            hidden_states=outputs.hidden_states,
            attention_mask=device_inputs["attention_mask"],
            torch_mod=torch_mod,
        )[0].detach().cpu()

    attention_mask = encoded["attention_mask"][0].detach().cpu().bool()
    valid_hidden = task_hidden[attention_mask]
    if valid_hidden.shape[0] > 2:
        valid_hidden = valid_hidden[1:-1]
    if valid_hidden.shape[0] == 0:
        return task_hidden.mean(dim=0)
    return valid_hidden.mean(dim=0)

