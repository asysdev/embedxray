"""Memory-safe tensor operations for embedding diagnostics."""

from __future__ import annotations

import torch


def safe_l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalizes a tensor along its last dimension with numerical stability."""
    x_f32 = x.to(torch.float32)
    norm = torch.linalg.vector_norm(x_f32, ord=2, dim=-1, keepdim=True)
    return x_f32 / torch.clamp(norm, min=eps)


def safe_svdvals(x: torch.Tensor) -> torch.Tensor:
    """Computes singular values of mean-centered embeddings with CPU fallback for MPS."""
    x_f32 = x.to(torch.float32)
    x_centered = x_f32 - x_f32.mean(dim=0, keepdim=True)
    try:
        return torch.linalg.svdvals(x_centered)
    except (RuntimeError, NotImplementedError):
        return torch.linalg.svdvals(x_centered.cpu()).to(x.device)


def chunked_topk_cosine(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    chunk_size: int = 2048,
    exclude_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieves by cosine similarity, breaking ties by ascending corpus index."""
    q_norm = safe_l2_normalize(queries)
    c_norm = q_norm if queries is corpus else safe_l2_normalize(corpus)
    return _chunked_topk_normalized_cosine(q_norm, c_norm, k, chunk_size, exclude_self)


def _chunked_topk_normalized_cosine(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    chunk_size: int,
    exclude_self: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores normalized vectors with FP64 accumulation on CPU to reduce rounding differences."""
    accumulation_dtype = torch.float64 if queries.device.type == "cpu" else torch.float32
    return _chunked_topk_dot(
        queries, corpus, k, chunk_size, exclude_self, accumulation_dtype=accumulation_dtype
    )


def _chunked_topk_dot(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    chunk_size: int,
    exclude_self: bool,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ranks FP32 scores from bounded query and corpus blocks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if k < 0:
        raise ValueError("k must be nonnegative.")
    if queries.ndim != 2 or corpus.ndim != 2 or queries.size(1) != corpus.size(1):
        raise ValueError("queries and corpus must be 2D with matching embedding dimensions.")
    if queries.device != corpus.device:
        raise ValueError("queries and corpus must be on the same device.")

    num_queries = queries.size(0)
    num_corpus = corpus.size(0)
    if exclude_self and num_queries != num_corpus:
        raise ValueError("exclude_self requires matching query and corpus row counts.")
    max_possible_k = num_corpus - (1 if exclude_self else 0)
    actual_k = min(k, max(0, max_possible_k))
    values = queries.new_empty((num_queries, actual_k))
    indices = queries.new_empty((num_queries, actual_k), dtype=torch.int64)
    if indices.numel() == 0:
        return values, indices

    corpus_chunk_size = chunk_size
    if num_queries == num_corpus and 1 < num_corpus <= chunk_size:
        corpus_chunk_size = (num_corpus + 1) // 2

    for q_start in range(0, num_queries, chunk_size):
        q_end = min(q_start + chunk_size, num_queries)
        q_chunk = queries[q_start:q_end].to(accumulation_dtype)
        running_vals = values[q_start:q_end, :0]
        running_idx = indices[q_start:q_end, :0]

        if exclude_self:
            q_global_idx = torch.arange(q_start, q_end, device=queries.device).unsqueeze(1)

        for c_start in range(0, num_corpus, corpus_chunk_size):
            c_end = min(c_start + corpus_chunk_size, num_corpus)
            c_chunk = corpus[c_start:c_end].to(accumulation_dtype)
            sim_chunk = torch.matmul(q_chunk, c_chunk.transpose(0, 1)).to(torch.float32)

            if exclude_self:
                c_global_idx = torch.arange(c_start, c_end, device=queries.device).unsqueeze(0)
                self_mask = q_global_idx == c_global_idx
                sim_chunk = sim_chunk.masked_fill(self_mask, -float("inf"))

            chunk_local_idx = _stable_topk_indices(sim_chunk, actual_k)
            chunk_vals = sim_chunk.gather(1, chunk_local_idx)
            chunk_global_idx = chunk_local_idx + c_start

            merged_vals = torch.cat([running_vals, chunk_vals], dim=-1)
            merged_idx = torch.cat([running_idx, chunk_global_idx], dim=-1)

            top_pos = _stable_topk_indices(merged_vals, actual_k)
            running_vals = merged_vals.gather(1, top_pos)
            running_idx = merged_idx.gather(1, top_pos)

        values[q_start:q_end] = running_vals
        indices[q_start:q_end] = running_idx

    return values, indices


def _stable_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Selects by descending score while preserving input order within ties."""
    try:
        return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    except (RuntimeError, NotImplementedError):
        if scores.device.type != "mps":
            raise
        ranked = torch.argsort(scores.cpu(), dim=1, descending=True, stable=True)
        return ranked[:, :k].to(scores.device)
