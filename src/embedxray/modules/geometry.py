"""High-dimensional geometry and index pathology diagnostics for embedding spaces."""

from __future__ import annotations

import math
from typing import Any

import torch

from embedxray.utils.tensor_ops import (
    chunked_topk_cosine,
    safe_l2_normalize,
    safe_svdvals,
)


def compute_anisotropy(
    embeddings: torch.Tensor,
    max_pairs: int = 50000,
    seed: int = 42,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Computes sampled mean pairwise cosine similarity and normalized spectral entropy."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor of shape (N, D).")
    if max_pairs < 0:
        raise ValueError("max_pairs must be nonnegative.")

    num_samples, dim = embeddings.shape
    if num_samples < 2 or dim < 1:
        return {
            "mean_pairwise_cosine": 0.0,
            "spectral_entropy": 0.0,
        }

    norm_emb = safe_l2_normalize(embeddings, eps=eps)

    total_ordered_pairs = num_samples * (num_samples - 1)
    pair_count = min(max_pairs, total_ordered_pairs)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    idx_i = torch.randint(0, num_samples, (pair_count,), generator=gen, dtype=torch.int64)
    offsets = torch.randint(1, num_samples, (pair_count,), generator=gen, dtype=torch.int64)
    idx_j = (idx_i + offsets) % num_samples

    idx_i = idx_i.to(norm_emb.device)
    idx_j = idx_j.to(norm_emb.device)

    chunk_size = 2048
    dot_sum = 0.0
    for start in range(0, pair_count, chunk_size):
        end = min(start + chunk_size, pair_count)
        vecs_i = norm_emb[idx_i[start:end]]
        vecs_j = norm_emb[idx_j[start:end]]
        dot_sum += float((vecs_i * vecs_j).sum().item())

    mean_pairwise_cosine = dot_sum / pair_count if pair_count else 0.0

    if dim <= 1:
        spectral_entropy = 0.0
    else:
        sigma = safe_svdvals(norm_emb)
        sigma_sum = sigma.sum()
        if float(sigma_sum.item()) <= eps:
            spectral_entropy = 0.0
        else:
            probs = sigma / (sigma_sum + eps)
            raw_entropy = -torch.sum(probs * torch.log(probs + eps)).item()
            norm_entropy = raw_entropy / math.log(float(dim))
            spectral_entropy = float(min(max(norm_entropy, 0.0), 1.0))

    return {
        "mean_pairwise_cosine": float(mean_pairwise_cosine),
        "spectral_entropy": float(spectral_entropy),
    }


def compute_dimension_health(
    embeddings: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Identifies collapsed or dead dimensions with variance < 1 / (100 * D)."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor of shape (N, D).")

    num_samples, dim = embeddings.shape
    if num_samples == 0 or dim == 0:
        return {
            "dead_dimension_count": 0,
            "dead_dimension_ratio": 0.0,
            "dead_dimension_indices": [],
        }

    norm_emb = safe_l2_normalize(embeddings, eps=eps)
    variances = torch.var(norm_emb, dim=0, unbiased=False)
    threshold = 1.0 / (100.0 * float(dim))

    dead_mask = variances < threshold
    dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(-1).tolist()
    if isinstance(dead_indices, int):
        dead_indices = [dead_indices]

    dead_count = len(dead_indices)
    dead_ratio = float(dead_count) / float(dim)

    return {
        "dead_dimension_count": int(dead_count),
        "dead_dimension_ratio": float(dead_ratio),
        "dead_dimension_indices": [int(i) for i in dead_indices],
    }


def compute_hubness(
    corpus_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor | None = None,
    k: int = 10,
    chunk_size: int = 2048,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Computes k-occurrence distribution skewness, Robin Hood index, and top hub indices."""
    if corpus_embeddings.ndim != 2:
        raise ValueError("corpus_embeddings must be a 2D tensor of shape (N, D).")

    num_corpus = corpus_embeddings.size(0)
    if query_embeddings is None:
        queries = corpus_embeddings
        exclude_self = True
    else:
        queries = query_embeddings
        exclude_self = False

    topk_indices = chunked_topk_cosine(
        queries=queries,
        corpus=corpus_embeddings,
        k=k,
        chunk_size=chunk_size,
        exclude_self=exclude_self,
    )[1]

    flat_indices = topk_indices.reshape(-1)
    if flat_indices.numel() == 0:
        return {
            "hubness_skewness": 0.0,
            "robin_hood_index": 0.0,
            "top_hub_indices": [],
            "top_hub_counts": [],
            "top_hubs": [],
            "max_k_occurrence": 0,
            "mean_k_occurrence": 0.0,
            "num_retrieved": 0,
        }
    try:
        counts = torch.bincount(flat_indices, minlength=num_corpus).to(torch.float32)
    except (RuntimeError, NotImplementedError):
        if flat_indices.device.type != "mps":
            raise
        counts = torch.bincount(flat_indices.cpu(), minlength=num_corpus).to(torch.float32)

    mu = counts.mean()
    centered = counts - mu
    sigma = torch.sqrt(torch.mean(centered**2))
    third_moment = torch.mean(centered**3)
    hubness_skewness = float((third_moment / (sigma**3 + eps)).item())

    total_occurrences = counts.sum()
    robin_hood_index = float(
        (torch.sum(torch.abs(centered)) / (2.0 * total_occurrences + eps)).item()
    )

    num_worst = min(num_corpus, max(5, math.ceil(0.001 * float(num_corpus))))
    top_counts_tensor, top_indices_tensor = torch.topk(
        counts, k=num_worst, largest=True, sorted=True
    )

    top_hub_indices = [int(i) for i in top_indices_tensor.tolist()]
    top_hub_counts = [int(round(c)) for c in top_counts_tensor.tolist()]
    top_hubs = [
        {"index": idx, "k_occurrences": cnt}
        for idx, cnt in zip(top_hub_indices, top_hub_counts)
    ]

    return {
        "hubness_skewness": hubness_skewness,
        "robin_hood_index": robin_hood_index,
        "top_hub_indices": top_hub_indices,
        "top_hub_counts": top_hub_counts,
        "top_hubs": top_hubs,
        "max_k_occurrence": int(top_hub_counts[0]) if top_hub_counts else 0,
        "mean_k_occurrence": float(mu.item()),
        "num_retrieved": flat_indices.numel(),
    }
