"""Unit tests for Phase 2 high-dimensional geometry and hubness diagnostics."""

from __future__ import annotations

import torch

from embedxray.modules.geometry import (
    compute_anisotropy,
    compute_dimension_health,
    compute_hubness,
)


def test_compute_anisotropy_contrasts_isotropic_and_anisotropic(
    isotropic_embeddings: torch.Tensor,
    anisotropic_embeddings: torch.Tensor,
) -> None:
    iso_stats = compute_anisotropy(isotropic_embeddings, max_pairs=10000, seed=42)
    aniso_stats = compute_anisotropy(anisotropic_embeddings, max_pairs=10000, seed=42)

    assert abs(iso_stats["mean_pairwise_cosine"]) < 0.05
    assert aniso_stats["mean_pairwise_cosine"] > 0.85

    assert 0.80 <= iso_stats["spectral_entropy"] <= 1.0
    assert 0.0 <= aniso_stats["spectral_entropy"] < iso_stats["spectral_entropy"]


def test_compute_dimension_health_detects_dead_dimensions(
    isotropic_embeddings: torch.Tensor,
    anisotropic_embeddings: torch.Tensor,
) -> None:
    iso_health = compute_dimension_health(isotropic_embeddings)
    assert iso_health["dead_dimension_count"] == 0
    assert iso_health["dead_dimension_ratio"] == 0.0
    assert iso_health["dead_dimension_indices"] == []

    aniso_health = compute_dimension_health(anisotropic_embeddings)
    assert aniso_health["dead_dimension_count"] >= 8
    assert set(range(56, 64)).issubset(set(aniso_health["dead_dimension_indices"]))
    assert abs(aniso_health["dead_dimension_ratio"] - (aniso_health["dead_dimension_count"] / 64.0)) < 1e-6


def test_compute_hubness_identifies_artificial_hubs(
    isotropic_embeddings: torch.Tensor,
    hub_embeddings: torch.Tensor,
) -> None:
    iso_hub = compute_hubness(isotropic_embeddings, k=5, chunk_size=32)
    bad_hub = compute_hubness(hub_embeddings, k=5, chunk_size=32)

    assert bad_hub["hubness_skewness"] > iso_hub["hubness_skewness"]
    assert bad_hub["robin_hood_index"] > iso_hub["robin_hood_index"]
    assert 0 in bad_hub["top_hub_indices"][:2]
    assert len(bad_hub["top_hub_indices"]) == 5


def test_compute_hubness_with_explicit_queries(
    isotropic_embeddings: torch.Tensor,
) -> None:
    queries = isotropic_embeddings[:24]
    corpus = isotropic_embeddings[24:]
    res = compute_hubness(corpus_embeddings=corpus, query_embeddings=queries, k=4, chunk_size=16)
    assert res["mean_k_occurrence"] > 0.0
    assert len(res["top_hub_indices"]) == 5


def test_zero_pair_budget_still_computes_spectral_entropy(
    isotropic_embeddings: torch.Tensor,
) -> None:
    stats = compute_anisotropy(isotropic_embeddings, max_pairs=0)
    assert stats["mean_pairwise_cosine"] == 0.0
    assert stats["spectral_entropy"] == compute_anisotropy(isotropic_embeddings)["spectral_entropy"]
