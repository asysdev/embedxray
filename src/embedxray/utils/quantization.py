"""Vectorized FP16, INT8, INT4, and 1-bit Binary quantization simulation utilities."""

from __future__ import annotations

import torch

from embedxray.utils.tensor_ops import _chunked_topk_dot


def simulate_fp16(x: torch.Tensor) -> torch.Tensor:
    """Simulates FP16 precision by casting to float16 and back to float32."""
    return x.to(torch.float16).to(torch.float32)


def simulate_int8(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Simulates per-dimension asymmetric min-max INT8 quantization across the batch."""
    x_f32 = x.to(torch.float32)
    if x_f32.numel() == 0:
        return x_f32
    x_min = x_f32.amin(dim=0, keepdim=True)
    x_max = x_f32.amax(dim=0, keepdim=True)
    scale = torch.clamp((x_max - x_min) / 255.0, min=eps)
    q = torch.clamp(torch.round((x_f32 - x_min) / scale - 128.0), min=-128.0, max=127.0)
    return (q + 128.0) * scale + x_min


def simulate_int4(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Simulates per-dimension asymmetric min-max INT4 quantization across the batch."""
    x_f32 = x.to(torch.float32)
    if x_f32.numel() == 0:
        return x_f32
    x_min = x_f32.amin(dim=0, keepdim=True)
    x_max = x_f32.amax(dim=0, keepdim=True)
    scale = torch.clamp((x_max - x_min) / 15.0, min=eps)
    q = torch.clamp(torch.round((x_f32 - x_min) / scale - 8.0), min=-8.0, max=7.0)
    return (q + 8.0) * scale + x_min


def binarize_embeddings(x: torch.Tensor) -> torch.Tensor:
    """Applies sign thresholding (x > 0) to produce {0.0, 1.0} float32 binary embeddings."""
    return (x > 0).to(torch.float32)


def chunked_topk_binary(
    queries_bin: torch.Tensor,
    corpus_bin: torch.Tensor,
    k: int,
    chunk_size: int = 2048,
    exclude_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieves by Hamming distance, breaking ties by ascending corpus index."""
    queries = 2.0 * queries_bin.to(torch.float32) - 1.0
    corpus = queries if queries_bin is corpus_bin else 2.0 * corpus_bin.to(torch.float32) - 1.0
    return _chunked_topk_dot(queries, corpus, k, chunk_size, exclude_self)
