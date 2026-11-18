"""Token and pooling directional attribution and contrast-pair collision diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

try:
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling
except ImportError:
    from sentence_transformers.models import Normalize, Pooling

from embedxray.utils.model import (
    encode_sentences,
    model_module,
    preserve_training_mode,
    to_tensor,
)
from embedxray.utils.tensor_ops import safe_l2_normalize

DEFAULT_BOILERPLATE_PATTERN = (
    r"^(\[.*\]|\{.*\}|<.*>|#+|\*+|-+|_+|[0-9]{2,4}[-/:][0-9]{2}.*|[\W_]+)$"
)


def _model_tokenizer(model: Any) -> Any:
    """Resolves a tokenizer exposed directly or through a model wrapper."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return getattr(model_module(model), "tokenizer", None)


def _extract_tokens(model: Any, text: str, record: Mapping[str, Any]) -> list[str]:
    """Resolves token labels from the encoded record or the model tokenizer."""
    tokens = record.get("tokens")
    if isinstance(tokens, (tuple, list)):
        return [str(token) for token in tokens]

    tokenizer = _model_tokenizer(model)
    ids = record.get("input_ids")
    if ids is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(to_tensor(ids).reshape(-1).tolist())

    tokenize = getattr(tokenizer, "tokenize", None)
    if callable(tokenize):
        return list(tokenize(text))
    if callable(tokenizer) and hasattr(tokenizer, "convert_ids_to_tokens"):
        encoded = tokenizer(text, add_special_tokens=True)
        if isinstance(encoded, Mapping) and "input_ids" in encoded:
            ids = to_tensor(encoded["input_ids"]).reshape(-1).tolist()
            return tokenizer.convert_ids_to_tokens(ids)
    return text.split()


def _feature_records(model: Any, texts: list[str], batch_size: int) -> list[dict[str, Any]]:
    """Reads per-sample features from supported encode output layouts."""
    try:
        raw = model.encode(
            texts,
            batch_size=batch_size,
            output_value=None,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
    except (TypeError, ValueError, NotImplementedError):
        raw = None

    if isinstance(raw, list) and len(raw) == len(texts):
        return [dict(item) if isinstance(item, Mapping) else {} for item in raw]
    if isinstance(raw, Mapping) and raw.get("token_embeddings") is not None:
        tokens = to_tensor(raw["token_embeddings"])
        if len(texts) == 1 and tokens.ndim == 2:
            return [dict(raw)]
        if tokens.ndim == 3 and tokens.size(0) == len(texts):
            batched = {}
            for key in ("sentence_embedding", "attention_mask", "input_ids", "tokens"):
                value = raw.get(key)
                if isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) > 0:
                    if len(value) == len(texts):
                        batched[key] = value
            return [
                {
                    **{key: value[index] for key, value in batched.items()},
                    "token_embeddings": tokens[index],
                }
                for index in range(len(texts))
            ]
    return [{} for _ in texts]


def _hook_record(model: Any, text: str) -> dict[str, Any]:
    """Captures real token features for one input, independent of encode sorting."""
    root = model_module(model)
    if root is None:
        return {}

    named: dict[str, Any] = {}
    unnamed: dict[str, Any] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, Mapping):
            tensor = output.get("token_embeddings", output.get("last_hidden_state"))
        elif isinstance(output, (list, tuple)) and output:
            tensor = output[0]
        else:
            tensor = output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.size(0) != 1:
            return

        record: dict[str, Any] = {"token_embeddings": tensor[0].detach()}
        input_features = inputs[0] if inputs and isinstance(inputs[0], Mapping) else {}
        output_features = output if isinstance(output, Mapping) else {}
        for key in ("attention_mask", "input_ids"):
            value = output_features.get(key, input_features.get(key))
            if isinstance(value, torch.Tensor):
                record[key] = value.reshape(-1).detach()
        captured = named if isinstance(output, Mapping) and "token_embeddings" in output else unnamed
        captured.clear()
        captured.update(record)

    try:
        for module in root.modules():
            handles.append(module.register_forward_hook(capture))
        pooled = encode_sentences(model, [text], batch_size=1)[0]
    finally:
        for handle in handles:
            handle.remove()

    captured = named or unnamed
    if captured:
        captured["sentence_embedding"] = pooled
    return captured


def _projection_direction(
    model: Any, sentence: torch.Tensor, token_dim: int, eps: float
) -> torch.Tensor | None:
    """Maps a pooled direction into the token space for prefix and concatenated pooling."""
    full_dim = token_dim
    root = model_module(model)
    if root is not None:
        layers = list(root.children())
        pool = next((layer for layer in reversed(layers) if isinstance(layer, Pooling)), None)
        if pool is not None:
            pool_dim = getattr(
                pool, "embedding_dimension", getattr(pool, "word_embedding_dimension", None)
            )
            if pool_dim != token_dim or any(
                not isinstance(layer, Normalize) for layer in layers[layers.index(pool) + 1 :]
            ):
                return None
            full_dim = pool.pooling_output_dimension

    sent_dim = sentence.numel()
    truncate_dim = getattr(model, "truncate_dim", getattr(root, "truncate_dim", None))
    if sent_dim != full_dim and not (0 < sent_dim < full_dim and sent_dim == truncate_dim):
        return None

    unit = sentence / (torch.linalg.vector_norm(sentence) + eps)
    if full_dim == token_dim:
        return unit
    padded = torch.nn.functional.pad(unit, (0, full_dim - sent_dim))
    return padded.reshape(-1, token_dim).sum(dim=0)


def _score_tokens(
    model: Any,
    text: str,
    record: Mapping[str, Any],
    boilerplate: re.Pattern[str],
    special_tokens: set[str],
    eps: float,
) -> dict[str, Any]:
    """Scores valid token features and records why a sample cannot be measured."""
    sample: dict[str, Any] = {
        "text": text,
        "boilerplate_hijack_score": None,
        "max_token_weight": None,
        "tokens": [],
    }
    if record.get("token_embeddings") is None or record.get("sentence_embedding") is None:
        return {**sample, "unavailable_reason": "Token or sentence embeddings are unavailable."}

    embeddings = to_tensor(record["token_embeddings"]).to(torch.float32)
    if embeddings.ndim == 3 and embeddings.size(0) == 1:
        embeddings = embeddings.squeeze(0)
    if embeddings.ndim != 2 or embeddings.size(1) == 0:
        return {**sample, "unavailable_reason": "Token embedding shape is unsupported."}

    raw_mask = record.get("attention_mask")
    mask = (
        to_tensor(raw_mask).reshape(-1).to(device=embeddings.device) == 1
        if raw_mask is not None
        else torch.ones(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
    )
    if mask.numel() != embeddings.size(0):
        reason = "Attention mask does not align with token embeddings."
        return {**sample, "unavailable_reason": reason}
    positions = mask.nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return {**sample, "unavailable_reason": "No tokens are valid under the attention mask."}

    tokens = _extract_tokens(model, text, record)
    if len(tokens) == embeddings.size(0):
        tokens = [tokens[index] for index in positions.tolist()]
    elif len(tokens) != positions.numel():
        return {**sample, "unavailable_reason": "Token labels do not align with token embeddings."}

    sentence = to_tensor(record["sentence_embedding"]).reshape(-1).to(embeddings)
    direction = _projection_direction(model, sentence, embeddings.size(1), eps)
    if direction is None:
        reason = "Token and sentence embedding spaces cannot be aligned."
        return {**sample, "unavailable_reason": reason}

    projections = torch.mv(embeddings[positions, : direction.numel()], direction)
    positive = projections.clamp(min=0.0)
    weights = positive / (positive.sum() + eps)

    token_entries: list[dict[str, Any]] = []
    hijack_score = 0.0
    for token, projection, weight in zip(tokens, projections.tolist(), weights.tolist()):
        cleaned_token = token.lstrip("Ġ▁")
        is_boilerplate = bool(
            token not in special_tokens
            and (boilerplate.match(token) or boilerplate.match(cleaned_token))
        )
        if is_boilerplate:
            hijack_score += weight
        token_entries.append(
            {
                "token": token,
                "directional_projection": projection,
                "weight": weight,
                "is_boilerplate": is_boilerplate,
            }
        )
    return {
        **sample,
        "boilerplate_hijack_score": min(hijack_score, 1.0),
        "max_token_weight": weights.max().item(),
        "tokens": token_entries,
    }


def compute_token_attribution(
    model: Any,
    texts: list[str],
    batch_size: int = 32,
    boilerplate_pattern: str = DEFAULT_BOILERPLATE_PATTERN,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Computes directional token attribution on samples whose features can be aligned."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    boilerplate = re.compile(boilerplate_pattern)
    tokenizer = _model_tokenizer(model)
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    samples: list[dict[str, Any]] = []
    with preserve_training_mode(model), torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for text, record in zip(batch, _feature_records(model, batch, batch_size)):
                if (
                    record.get("token_embeddings") is None
                    or record.get("sentence_embedding") is None
                ):
                    record = _hook_record(model, text)
                sample = _score_tokens(model, text, record, boilerplate, special_tokens, eps)
                samples.append(sample)

    evaluated = [sample for sample in samples if sample["boilerplate_hijack_score"] is not None]
    count = len(evaluated)
    return {
        "boilerplate_hijack_score": (
            sum(sample["boilerplate_hijack_score"] for sample in evaluated) / count
            if count else None
        ),
        "max_token_dominance": (
            sum(sample["max_token_weight"] for sample in evaluated) / count if count else None
        ),
        "evaluated_samples": count,
        "samples": samples,
    }


def compute_pair_collisions(
    model: Any,
    contrast_pairs: list[tuple[str, str]],
    batch_size: int = 32,
    collision_threshold: float = 0.90,
) -> dict[str, Any]:
    """Computes cosine similarity and collision rate across contrasting sentence pairs."""
    if not contrast_pairs:
        return {
            "mean_contrast_similarity": 0.0,
            "collision_rate": 0.0,
            "offending_pair_indices": [],
            "pair_similarities": [],
            "offending_pairs": [],
        }

    left_texts = [pair[0] for pair in contrast_pairs]
    right_texts = [pair[1] for pair in contrast_pairs]
    with preserve_training_mode(model), torch.no_grad():
        left_emb = safe_l2_normalize(encode_sentences(model, left_texts, batch_size))
        right_emb = safe_l2_normalize(encode_sentences(model, right_texts, batch_size))
    if left_emb.shape != right_emb.shape:
        raise ValueError("Contrast pairs must have matching embedding dimensions.")
    sim_list = (left_emb * right_emb).sum(dim=-1).clamp(-1.0, 1.0).tolist()
    offending_indices = [index for index, sim in enumerate(sim_list) if sim > collision_threshold]
    offending_pairs = [
        {
            "index": index,
            "left": contrast_pairs[index][0],
            "right": contrast_pairs[index][1],
            "similarity": sim_list[index],
        }
        for index in offending_indices
    ]

    return {
        "mean_contrast_similarity": sum(sim_list) / len(sim_list),
        "collision_rate": len(offending_indices) / len(sim_list),
        "offending_pair_indices": offending_indices,
        "pair_similarities": sim_list,
        "offending_pairs": offending_pairs,
    }
