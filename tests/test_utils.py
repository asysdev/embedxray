"""Unit tests for Phase 1 low-level tensor and quantization utilities."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from embedxray.utils.quantization import (
    binarize_embeddings,
    chunked_topk_binary,
    simulate_fp16,
    simulate_int4,
    simulate_int8,
)
from embedxray.utils.tensor_ops import (
    chunked_topk_cosine,
    safe_l2_normalize,
    safe_svdvals,
)


def test_safe_l2_normalize_unit_norm_and_zero_vector() -> None:
    gen = torch.Generator().manual_seed(11)
    x = torch.randn((16, 32), generator=gen) * 5.0
    x[0] = 0.0

    normed = safe_l2_normalize(x)
    assert normed.shape == (16, 32)
    assert torch.isfinite(normed).all()
    assert torch.allclose(normed[0], torch.zeros(32))

    lengths = torch.linalg.vector_norm(normed[1:], ord=2, dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-6)


def test_safe_svdvals_standard_and_cpu_fallback(isotropic_embeddings: torch.Tensor) -> None:
    svals = safe_svdvals(isotropic_embeddings)
    assert svals.ndim == 1
    assert svals.size(0) == min(isotropic_embeddings.shape)
    assert (svals >= 0).all()

    real_svdvals = torch.linalg.svdvals
    call_count = {"n": 0}

    def flaky_svdvals(tensor: torch.Tensor) -> torch.Tensor:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise NotImplementedError("Simulated MPS SVD fallback")
        return real_svdvals(tensor)

    with patch("torch.linalg.svdvals", side_effect=flaky_svdvals):
        fallback_svals = safe_svdvals(isotropic_embeddings)

    assert call_count["n"] == 2
    assert torch.allclose(svals, fallback_svals, atol=1e-6)


def test_chunked_topk_cosine_exactness_and_self_exclusion(
    isotropic_embeddings: torch.Tensor,
) -> None:
    queries = isotropic_embeddings[:40]
    corpus = isotropic_embeddings
    k = 7

    vals_chunked, idx_chunked = chunked_topk_cosine(
        queries=queries,
        corpus=corpus,
        k=k,
        chunk_size=13,
        exclude_self=False,
    )

    q_norm = safe_l2_normalize(queries)
    c_norm = safe_l2_normalize(corpus)
    exact_sim = q_norm @ c_norm.T
    vals_exact, idx_exact = torch.topk(exact_sim, k=k, dim=-1)

    assert torch.allclose(vals_chunked, vals_exact, atol=1e-6)
    assert torch.equal(idx_chunked, idx_exact)

    self_vals, self_idx = chunked_topk_cosine(
        queries=corpus,
        corpus=corpus,
        k=k,
        chunk_size=15,
        exclude_self=True,
    )
    row_indices = torch.arange(corpus.size(0)).unsqueeze(1)
    assert not (self_idx == row_indices).any()

    exact_self_sim = c_norm @ c_norm.T
    exact_self_sim.fill_diagonal_(-float("inf"))
    expected_self_vals, expected_self_idx = torch.topk(exact_self_sim, k=k, dim=-1)
    assert torch.allclose(self_vals, expected_self_vals, atol=1e-6)
    assert torch.equal(self_idx, expected_self_idx)


def test_chunked_topk_cosine_clamps_k() -> None:
    gen = torch.Generator().manual_seed(99)
    emb = safe_l2_normalize(torch.randn((5, 8), generator=gen))
    vals, idx = chunked_topk_cosine(emb, emb, k=20, chunk_size=2, exclude_self=True)
    assert vals.shape == (5, 4)
    assert idx.shape == (5, 4)


def test_quantization_simulators_fidelity_hierarchy(
    isotropic_embeddings: torch.Tensor,
) -> None:
    fp16_emb = simulate_fp16(isotropic_embeddings)
    int8_emb = simulate_int8(isotropic_embeddings)
    int4_emb = simulate_int4(isotropic_embeddings)
    bin_emb = binarize_embeddings(isotropic_embeddings)

    assert fp16_emb.dtype == torch.float32
    assert int8_emb.dtype == torch.float32
    assert int4_emb.dtype == torch.float32
    assert bin_emb.dtype == torch.float32

    unique_bins = torch.unique(bin_emb)
    assert set(unique_bins.tolist()).issubset({0.0, 1.0})

    err_fp16 = torch.mean((isotropic_embeddings - fp16_emb) ** 2).item()
    err_int8 = torch.mean((isotropic_embeddings - int8_emb) ** 2).item()
    err_int4 = torch.mean((isotropic_embeddings - int4_emb) ** 2).item()

    assert err_fp16 < err_int8 < err_int4


def test_chunked_topk_binary_matches_exact_hamming(
    isotropic_embeddings: torch.Tensor,
) -> None:
    bin_emb = binarize_embeddings(isotropic_embeddings)
    k = 6
    scores, idx = chunked_topk_binary(
        queries_bin=bin_emb,
        corpus_bin=bin_emb,
        k=k,
        chunk_size=19,
        exclude_self=True,
    )

    dim = bin_emb.size(1)
    xor_dist = torch.cdist(bin_emb, bin_emb, p=1.0)
    expected_inner = dim - 2.0 * xor_dist
    expected_inner.fill_diagonal_(-float("inf"))
    exp_scores, exp_idx = torch.topk(expected_inner, k=k, dim=-1)

    assert torch.allclose(scores, exp_scores, atol=1e-5)
    gathered_scores = torch.gather(expected_inner, dim=1, index=idx)
    assert torch.allclose(gathered_scores, exp_scores, atol=1e-5)
    row_indices = torch.arange(bin_emb.size(0)).unsqueeze(1)
    assert not (idx == row_indices).any()


@pytest.mark.parametrize("retrieve", [chunked_topk_cosine, chunked_topk_binary])
def test_topk_ties_and_similarity_block_sizes(
    retrieve: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    embeddings = torch.ones((6, 8))
    expected = [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1], [0, 1]]
    for chunk_size in (1, 2, 5, 2048):
        with patch("torch.matmul", wraps=torch.matmul) as matmul:
            scores, indices = retrieve(embeddings, embeddings, 2, chunk_size, True)
        assert indices.tolist() == expected
        assert torch.allclose(scores, torch.full_like(scores, scores[0, 0].item()))
        for call in matmul.call_args_list:
            left, right = call.args
            shape = (left.size(0), right.size(1))
            assert max(shape) <= chunk_size
            assert shape != (6, 6)
