"""Read-only access to model inference and sentence embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


def model_module(model: Any) -> nn.Module | None:
    """Finds the PyTorch model behind a compatible encode wrapper."""
    if isinstance(model, nn.Module):
        return model
    for name in ("model", "module"):
        wrapped = getattr(model, name, None)
        if isinstance(wrapped, nn.Module):
            return wrapped
    return None


@contextmanager
def preserve_training_mode(model: Any) -> Iterator[None]:
    """Runs inference in eval mode and restores each module's original flag."""
    root = model_module(model)
    states = [(module, module.training) for module in root.modules()] if root is not None else []
    try:
        if root is not None:
            root.eval()
        yield
    finally:
        for module, training in states:
            module.training = training


def to_tensor(value: Any) -> torch.Tensor:
    """Converts a model output to a detached tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return torch.stack([item.detach() for item in value])
    return torch.as_tensor(value)


def encode_sentences(model: Any, texts: list[str], batch_size: int) -> torch.Tensor:
    """Encodes nonempty text batches as matrices of sentence embeddings."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    try:
        raw = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
    except TypeError:
        raw = model.encode(texts)

    tensor = to_tensor(raw).to(torch.float32)
    if tensor.ndim == 1 and len(texts) == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.size(0) != len(texts) or tensor.size(1) == 0:
        raise ValueError("model.encode must return one nonempty embedding vector per text.")
    return tensor
