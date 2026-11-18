"""Unit tests for Phase 3 token attribution and contrast pair collision diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from embedxray.modules.attribution import (
    compute_pair_collisions,
    compute_token_attribution,
)


def test_compute_token_attribution_detects_boilerplate_hijack(
    mock_model: Any,
) -> None:
    clean_texts = [
        "neural networks optimize gradient descent across batches",
        "semantic search indexes dense vector representations",
    ]
    hijacked_texts = [
        "[DOC_ID:9981] ### 2026-11-17 *** neural networks",
        "<HTML> [HEADER] ___ 2026-11-17 ### semantic",
    ]

    clean_res = compute_token_attribution(mock_model, clean_texts, batch_size=2)
    hijack_res = compute_token_attribution(mock_model, hijacked_texts, batch_size=2)

    assert clean_res["boilerplate_hijack_score"] == 0.0
    assert hijack_res["boilerplate_hijack_score"] > 0.50
    assert len(hijack_res["samples"]) == 2

    for sample in hijack_res["samples"]:
        weight_sum = sum(t["weight"] for t in sample["tokens"])
        assert abs(weight_sum - 1.0) < 1e-5
        assert any(t["is_boilerplate"] for t in sample["tokens"])


def test_compute_token_attribution_fallback_hooks(
    hook_only_model: Any,
) -> None:
    texts = [
        "[CONFIDENTIAL] ### quarterly revenue forecast",
        "standard product description without boilerplate",
    ]
    res = compute_token_attribution(hook_only_model, texts, batch_size=2)
    assert len(res["samples"]) == 2
    assert 0.0 <= res["boilerplate_hijack_score"] <= 1.0
    assert len(res["samples"][0]["tokens"]) == 5

    class WrappedModel:
        def __init__(self, model: Any) -> None:
            self.model = model

        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            return self.model.encode(texts, **kwargs)

    wrapped = compute_token_attribution(WrappedModel(hook_only_model), texts, batch_size=2)
    assert wrapped == res


def test_compute_pair_collisions_identifies_high_similarity_pairs(
    mock_model: Any,
) -> None:
    contrast_pairs = [
        (
            "[BOILERPLATE_A] [BOILERPLATE_B] [BOILERPLATE_C] #1 #2 #3 patient is healthy",
            "[BOILERPLATE_A] [BOILERPLATE_B] [BOILERPLATE_C] #1 #2 #3 patient is critical",
        ),
        (
            "cat sits on mat quietly",
            "quantum compiler optimizes tensor graph",
        ),
    ]

    result = compute_pair_collisions(
        mock_model,
        contrast_pairs=contrast_pairs,
        batch_size=2,
        collision_threshold=0.90,
    )

    assert 0 in result["offending_pair_indices"]
    assert 1 not in result["offending_pair_indices"]
    assert abs(result["collision_rate"] - 0.5) < 1e-6
    assert result["pair_similarities"][0] > 0.90
    assert result["pair_similarities"][1] < 0.90


def test_pooled_only_output_does_not_fabricate_token_scores() -> None:
    class PooledOnly(torch.nn.Module):
        def encode(self, texts: list[str]) -> torch.Tensor:
            return torch.tensor([[1.0, 2.0]] * len(texts))

    model = PooledOnly()
    result = compute_token_attribution(model, ["### body"])
    assert result["boilerplate_hijack_score"] is None
    assert result["evaluated_samples"] == 0
    assert result["samples"][0]["tokens"] == []
    assert model.training
    assert not model._forward_hooks


def test_attention_mask_and_token_labels_use_batched_features() -> None:
    class BatchedModel:
        def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {
                "token_embeddings": torch.tensor([[[2.0, 0.0], [1.0, 0.0]]] * 2),
                "sentence_embedding": torch.tensor([[1.0, 0.0]] * 2),
                "attention_mask": torch.tensor([[0, 0], [0, 1]]),
                "tokens": [["###", "body"], ["###", "body"]],
            }

    result = compute_token_attribution(BatchedModel(), ["all masked", "partially masked"])
    masked, measured = result["samples"]
    assert masked["tokens"] == []
    assert masked["boilerplate_hijack_score"] is None
    assert [token["token"] for token in measured["tokens"]] == ["body"]
    assert result["evaluated_samples"] == 1
    assert result["boilerplate_hijack_score"] == 0.0
