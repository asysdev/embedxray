"""Unit tests for Phase 2 Matryoshka and quantization fracture simulation."""

from __future__ import annotations

import torch

from embedxray.modules.fracture import _compute_overlap_at_k, evaluate_fracture


def test_evaluate_fracture_grid_and_monotonic_bytes(
    isotropic_embeddings: torch.Tensor,
) -> None:
    result = evaluate_fracture(
        corpus_embeddings=isotropic_embeddings,
        matryoshka_dims=[64, 32, 16],
        quantization_modes=("fp16", "int8", "int4", "binary"),
        k=5,
        chunk_size=32,
        cliff_threshold=0.90,
    )

    assert result["matryoshka_dims"] == [64, 32, 16]
    assert result["quantization_modes"] == ["fp32", "fp16", "int8", "int4", "binary"]
    assert len(result["grid"]) == 3 * 5

    base_entry = result["results_by_config"]["64_fp32"]
    assert abs(base_entry["overlap_at_k"] - 1.0) < 1e-6
    assert abs(base_entry["self_similarity_drift"] - 1.0) < 1e-6
    assert base_entry["bytes_per_vector"] == 256.0

    fp16_64 = result["results_by_config"]["64_fp16"]
    int4_64 = result["results_by_config"]["64_int4"]
    bin_64 = result["results_by_config"]["64_binary"]

    assert fp16_64["bytes_per_vector"] == 128.0
    assert int4_64["bytes_per_vector"] == 32.0
    assert bin_64["bytes_per_vector"] == 8.0

    assert fp16_64["overlap_at_k"] >= 0.95
    assert fp16_64["self_similarity_drift"] > int4_64["self_similarity_drift"] > bin_64["self_similarity_drift"]
    assert result["fracture_cliff_bytes"] <= 128.0


def test_evaluate_fracture_default_dims_and_impossible_cliff(
    isotropic_embeddings: torch.Tensor,
) -> None:
    res = evaluate_fracture(
        corpus_embeddings=isotropic_embeddings,
        matryoshka_dims=None,
        k=5,
        cliff_threshold=1.01,
    )
    assert res["matryoshka_dims"] == [64, 32, 16, 8]
    assert res["fracture_cliff_bytes"] == 256.0


def test_fracture_baseline_and_binary_overlap_are_stable() -> None:
    embeddings = torch.randn((8, 8), generator=torch.Generator().manual_seed(4))
    embeddings = embeddings.repeat_interleave(2, dim=0)
    results = [
        evaluate_fracture(embeddings, k=4, chunk_size=size, quantization_modes=("binary",))
        for size in (1, 3, 5, 24)
    ]
    assert all(result["grid"][0]["overlap_at_k"] == 1.0 for result in results)
    assert len({result["grid"][1]["overlap_at_k"] for result in results}) == 1

    baseline_case = torch.randn((16, 8), generator=torch.Generator().manual_seed(0))
    baseline_case = baseline_case.repeat_interleave(2, dim=0)
    result = evaluate_fracture(baseline_case, k=4, chunk_size=7, quantization_modes=())
    assert result["grid"][0]["overlap_at_k"] == 1.0


def test_batched_overlap_matches_set_intersections() -> None:
    baseline = torch.tensor([[2, 0, 5], [4, 6, 8], [1, 0, 9], [4, 7, 8]])
    candidates = torch.tensor([[5, 3, 2], [0, 1, 2], [9, 0, 1], [0, 4, 1]])
    assert _compute_overlap_at_k(baseline, candidates, chunk_size=3) == 0.5


def test_no_eligible_neighbors_cannot_qualify_for_compression() -> None:
    corpus = torch.ones(1, 8)
    for options in ({}, {"query_embeddings": corpus[:0]}, {"k": 0}):
        result = evaluate_fracture(corpus, **options)
        assert all(row["overlap_at_k"] is None for row in result["grid"])
        assert result["fracture_cliff_bytes"] == 32.0
        assert result["compression_ratio"] == 1.0

    empty = evaluate_fracture(corpus[:0])
    assert all(row["self_similarity_drift"] is None for row in empty["grid"])
    zeros = evaluate_fracture(torch.zeros(3, 8), quantization_modes=("fp16",))
    assert all(row["self_similarity_drift"] == 0.0 for row in zeros["grid"])
