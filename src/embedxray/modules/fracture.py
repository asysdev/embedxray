"""Quantization and Matryoshka dimension truncation fracture simulator."""

from __future__ import annotations

from typing import Any

import torch

from embedxray.utils.quantization import (
    binarize_embeddings,
    chunked_topk_binary,
    simulate_fp16,
    simulate_int4,
    simulate_int8,
)
from embedxray.utils.tensor_ops import (
    _chunked_topk_normalized_cosine,
    chunked_topk_cosine,
    safe_l2_normalize,
)

BYTES_PER_DIM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "binary": 0.125,
}
_QUANTIZERS = {"fp16": simulate_fp16, "int8": simulate_int8, "int4": simulate_int4}


def _resolve_matryoshka_dims(full_dim: int, matryoshka_dims: list[int] | None) -> list[int]:
    """Resolves, filters, deduplicates, and sorts valid Matryoshka dimensions descending."""
    if matryoshka_dims is None:
        candidates = [
            d
            for d in (full_dim, full_dim // 2, full_dim // 4, full_dim // 8, 64)
            if 8 <= d <= full_dim
        ]
    else:
        candidates = [d for d in matryoshka_dims if 0 < d <= full_dim]

    if not candidates:
        candidates = [full_dim]

    return sorted(set(candidates), reverse=True)


def _compute_overlap_at_k(
    base_indices: torch.Tensor, candidate_indices: torch.Tensor, chunk_size: int
) -> float | None:
    """Computes mean set intersection ratio |I_base(q) intersect I_cand(q)| / actual_k."""
    if base_indices.numel() == 0:
        return None
    if base_indices is candidate_indices:
        return 1.0
    matches = 0
    for base, candidate in zip(
        base_indices.split(chunk_size), candidate_indices.split(chunk_size)
    ):
        combined = torch.cat((base, candidate), dim=1).sort(dim=1).values
        matches += (combined[:, 1:] == combined[:, :-1]).sum().item()
    return matches / base_indices.numel()


def _fp32_self_similarity(normalized: torch.Tensor) -> float | None:
    """Uses the same zero-vector convention as cosine retrieval."""
    if normalized.size(0) == 0:
        return None
    return (normalized != 0).any(dim=1).float().mean().item()


def evaluate_fracture(
    corpus_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor | None = None,
    matryoshka_dims: list[int] | None = None,
    quantization_modes: tuple[str, ...] = ("fp16", "int8", "int4", "binary"),
    k: int = 10,
    chunk_size: int = 2048,
    cliff_threshold: float = 0.95,
) -> dict[str, Any]:
    """Evaluates retrieval overlap and self-similarity drift across Matryoshka dims and precisions."""
    if corpus_embeddings.ndim != 2:
        raise ValueError("corpus_embeddings must be a 2D tensor of shape (N, D).")

    full_dim = corpus_embeddings.size(1)
    if full_dim == 0:
        raise ValueError("corpus_embeddings must have a positive embedding dimension.")
    if query_embeddings is None:
        queries = corpus_embeddings
        exclude_self = True
    else:
        queries = query_embeddings
        exclude_self = False

    dims = _resolve_matryoshka_dims(full_dim, matryoshka_dims)
    ordered_modes: list[str] = ["fp32"]
    for mode in quantization_modes:
        mode_lower = mode.lower()
        if mode_lower not in BYTES_PER_DIM:
            raise ValueError(f"Unsupported quantization mode: {mode}")
        if mode_lower not in ordered_modes:
            ordered_modes.append(mode_lower)

    base_corpus = safe_l2_normalize(corpus_embeddings)
    base_queries = base_corpus if exclude_self else safe_l2_normalize(queries)
    base_indices = _chunked_topk_normalized_cosine(
        base_queries, base_corpus, k, chunk_size, exclude_self
    )[1]
    overlap_available = base_indices.numel() > 0

    grid: list[dict[str, Any]] = []
    results_by_config: dict[str, dict[str, Any]] = {}
    overlap_matrix: dict[int, dict[str, float | None]] = {}
    drift_matrix: dict[int, dict[str, float | None]] = {}

    for dim in dims:
        overlap_matrix[dim] = {}
        drift_matrix[dim] = {}

        if dim == full_dim:
            c_slice, q_slice = base_corpus, base_queries
        else:
            c_slice = safe_l2_normalize(corpus_embeddings[:, :dim])
            q_slice = c_slice if exclude_self else safe_l2_normalize(queries[:, :dim])

        for mode in ordered_modes:
            cand_indices = base_indices
            if mode == "fp32":
                if dim != full_dim and overlap_available:
                    cand_indices = _chunked_topk_normalized_cosine(
                        q_slice, c_slice, k, chunk_size, exclude_self
                    )[1]
                c_quant = c_slice
            elif mode == "binary":
                c_bin = binarize_embeddings(c_slice)
                c_quant = 2.0 * c_bin - 1.0
                if overlap_available:
                    q_bin = c_bin if exclude_self else binarize_embeddings(q_slice)
                    cand_indices = chunked_topk_binary(
                        q_bin, c_bin, k, chunk_size, exclude_self
                    )[1]
            else:
                quantize = _QUANTIZERS[mode]
                c_quant = quantize(c_slice)
                if overlap_available:
                    q_quant = c_quant if exclude_self else quantize(q_slice)
                    cand_indices = chunked_topk_cosine(
                        q_quant, c_quant, k, chunk_size, exclude_self
                    )[1]

            overlap = _compute_overlap_at_k(base_indices, cand_indices, chunk_size)
            if mode == "fp32":
                self_sim = _fp32_self_similarity(c_slice)
            elif c_slice.size(0) == 0:
                self_sim = None
            else:
                similarities = (c_slice * safe_l2_normalize(c_quant)).sum(dim=-1)
                self_sim = similarities.clamp(-1.0, 1.0).mean().item()
            bytes_per_vec = float(BYTES_PER_DIM[mode] * float(dim))

            entry = {
                "dim": int(dim),
                "mode": mode,
                "bytes_per_vector": bytes_per_vec,
                "overlap_at_k": overlap,
                "self_similarity_drift": self_sim,
            }
            grid.append(entry)
            results_by_config[f"{dim}_{mode}"] = entry
            overlap_matrix[dim][mode] = overlap
            drift_matrix[dim][mode] = self_sim

    full_fp32_bytes = float(4.0 * float(full_dim))
    qualifying = [
        item for item in grid
        if item["overlap_at_k"] is not None and item["overlap_at_k"] >= cliff_threshold
    ]
    if qualifying:
        cliff_entry = min(
            qualifying,
            key=lambda x: (x["bytes_per_vector"], -x["overlap_at_k"]),
        )
        fracture_cliff_bytes = float(cliff_entry["bytes_per_vector"])
    else:
        cliff_entry = {
            "dim": int(full_dim),
            "mode": "fp32",
            "bytes_per_vector": full_fp32_bytes,
            "overlap_at_k": 1.0 if overlap_available else None,
            "self_similarity_drift": _fp32_self_similarity(base_corpus),
        }
        fracture_cliff_bytes = full_fp32_bytes

    return {
        "actual_k": base_indices.size(1),
        "num_queries": base_indices.size(0),
        "matryoshka_dims": dims,
        "quantization_modes": ordered_modes,
        "grid": grid,
        "results_by_config": results_by_config,
        "overlap_matrix": overlap_matrix,
        "drift_matrix": drift_matrix,
        "fracture_cliff_bytes": fracture_cliff_bytes,
        "fracture_cliff_config": cliff_entry,
        "full_fp32_bytes": full_fp32_bytes,
        "compression_ratio": float(full_fp32_bytes / max(fracture_cliff_bytes, 1e-12)),
    }
