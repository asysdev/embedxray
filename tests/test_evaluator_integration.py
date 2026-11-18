"""End-to-end integration tests for Phase 4 EmbeddingXRayEvaluator and XRayReport."""

from __future__ import annotations

import csv
import json
from inspect import signature
from pathlib import Path
from typing import Any

import pytest
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertConfig, BertModel, BertTokenizer

try:
    from sentence_transformers.base.modules import Transformer
    from sentence_transformers.sentence_transformer.modules import Pooling
except ImportError:
    from sentence_transformers.models import Pooling, Transformer

from conftest import OfflineMockSentenceTransformer
from embedxray import EmbeddingXRayEvaluator, XRayReport


def test_evaluator_end_to_end_metrics_and_artifacts(
    mock_model: OfflineMockSentenceTransformer,
    tmp_path: Path,
) -> None:
    corpus_texts = [
        f"document number {i} discusses distributed vector indexing and search"
        for i in range(24)
    ]
    query_texts = [
        "distributed vector search query",
        "[HEADER] ### 2026-11-17 vector indexing",
    ]
    contrast_pairs = [
        ("system status online healthy", "quantum compiler graph optimization"),
        ("[TAG_1] [TAG_2] #1 #2 #3 status online", "[TAG_1] [TAG_2] #1 #2 #3 status offline"),
    ]

    params_before = [p.detach().clone() for p in mock_model.parameters()]

    evaluator = EmbeddingXRayEvaluator(
        corpus_texts=corpus_texts,
        query_texts=query_texts,
        contrast_pairs=contrast_pairs,
        matryoshka_dims=[64, 32, 16],
        k=4,
        batch_size=8,
        chunk_size=10,
        name="val",
        write_csv=True,
        write_report=True,
    )

    assert evaluator.primary_metric == "val_xray_health_score"
    assert evaluator.greater_is_better is True

    metrics = evaluator(mock_model, output_path=str(tmp_path), epoch=1, steps=100)
    assert isinstance(metrics, dict)
    assert evaluator.primary_metric in metrics
    assert 0.0 <= metrics["val_xray_health_score"] <= 100.0
    assert "val_mean_pairwise_cosine" in metrics
    assert "val_spectral_entropy" in metrics
    assert "val_dead_dimension_ratio" in metrics
    assert "val_hubness_skewness" in metrics
    assert "val_robin_hood_index" in metrics
    assert "val_boilerplate_hijack_score" in metrics
    assert "val_collision_rate" in metrics
    assert "val_fracture_cliff_bytes" in metrics
    assert "val_overlap_at_4_d64_fp32" in metrics
    assert "val_overlap_at_4_d32_int8" in metrics

    params_after = [p.detach().clone() for p in mock_model.parameters()]
    for p_before, p_after in zip(params_before, params_after):
        assert torch.equal(p_before, p_after)

    assert isinstance(evaluator.last_report, XRayReport)

    csv_path = tmp_path / "xray_evaluation_val_results.csv"
    json_path = tmp_path / "xray_report_val.json"
    md_path = tmp_path / "xray_report_val.md"

    assert csv_path.exists()
    assert json_path.exists()
    assert md_path.exists()

    evaluator(mock_model, output_path=str(tmp_path), epoch=2, steps=200)
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3

    loaded_json: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    assert "health_score" in loaded_json
    assert loaded_json["metadata"]["embedding_dim"] == 64

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Embedding X-Ray Diagnostic Report" in md_text
    assert "| Dimension | Mode | Bytes / Vec | Overlap@k | Self-Similarity Drift |" in md_text


def test_evaluator_penalizes_pathological_model() -> None:
    healthy_model = OfflineMockSentenceTransformer(embedding_dim=64, boilerplate_boost=0.0)
    hijacked_model = OfflineMockSentenceTransformer(embedding_dim=64, boilerplate_boost=15.0)

    texts = [
        f"[BOILER_{i % 2}] ### 2026-11-17 unique payload token_{i} semantic context_{i * 3}"
        for i in range(20)
    ]
    contrast_pairs = [
        (
            "[BOILER_1] ### 2026-11-17 patient recovered completely",
            "[BOILER_1] ### 2026-11-17 patient deteriorated rapidly",
        )
    ]

    evaluator = EmbeddingXRayEvaluator(
        corpus_texts=texts,
        contrast_pairs=contrast_pairs,
        k=4,
        write_csv=False,
        write_report=False,
    )

    healthy_metrics = evaluator(healthy_model)
    hijacked_metrics = evaluator(hijacked_model)

    assert healthy_metrics["xray_health_score"] > hijacked_metrics["xray_health_score"]
    assert hijacked_metrics["boilerplate_hijack_score"] > healthy_metrics["boilerplate_hijack_score"]


def test_evaluator_accepts_corpus_and_queries_kwargs(
    mock_model: OfflineMockSentenceTransformer,
) -> None:
    corpus = [
        "### HEADER: 2026-11-17 | System database connection pool exhausted on primary node.",
        "### HEADER: 2026-11-17 | Vector index HNSW graph construction completed in 42ms.",
        "Query planner selected sequential scan instead of index scan due to stale statistics.",
    ] * 4
    queries = [
        "Why is the database connection failing?",
        "How fast did the HNSW vector index build?",
    ]
    contrast_pairs = [
        ("Grant administrator access.", "Revoke administrator access."),
    ]

    evaluator = EmbeddingXRayEvaluator(
        corpus=corpus,
        queries=queries,
        contrast_pairs=contrast_pairs,
        matryoshka_dims=[64, 32],
        write_csv=False,
        write_report=False,
    )
    metrics = evaluator(mock_model)
    assert "xray_health_score" in metrics
    assert evaluator.corpus_texts == corpus
    assert evaluator.query_texts == queries


@pytest.fixture
def local_transformer(tmp_path: Path) -> Transformer:
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "#"]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    if "vocab" in signature(BertTokenizer).parameters:
        tokenizer = BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})
    else:
        tokenizer = BertTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(tmp_path)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(55)
        BertModel(
            BertConfig(
                vocab_size=len(vocab), hidden_size=12, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16,
            )
        ).save_pretrained(tmp_path)
    return Transformer(str(tmp_path))


@pytest.mark.parametrize("pooling_modes, truncate_dim", [(False, 8), (True, None), (True, 16)])
def test_real_model_attribution_respects_embedding_layout_and_mode(
    local_transformer: Transformer, pooling_modes: bool, truncate_dim: int | None,
) -> None:
    if pooling_modes and "embedding_dimension" in signature(Pooling).parameters:
        pooling = Pooling(12, ("mean", "max"))
    elif pooling_modes:
        pooling = Pooling(12, pooling_mode_mean_tokens=True, pooling_mode_max_tokens=True)
    else:
        pooling = Pooling(12)
    model = SentenceTransformer(modules=[local_transformer, pooling], device="cpu", truncate_dim=truncate_dim)
    model.train()
    local_transformer.eval()
    original_modes = [layer.training for layer in model.modules()]

    evaluator = EmbeddingXRayEvaluator(corpus_texts=["hello world", "hello #"], k=1)
    report = evaluator.evaluate_report(model)

    assert [layer.training for layer in model.modules()] == original_modes
    assert report.metadata["embedding_dim"] == (truncate_dim or 24)
    assert report.attribution["evaluated_samples"] == 2
    assert "unavailable_reason" not in report.attribution["samples"][0]
    tokens = report.attribution["samples"][0]["tokens"]
    assert [token["token"] for token in tokens] == ["[CLS]", "hello", "world", "[SEP]"]
    feature = model.encode(["hello world"], output_value=None)[0]
    sentence = feature["sentence_embedding"]
    unit = sentence / (torch.linalg.vector_norm(sentence) + 1e-12)
    token_embeddings = feature["token_embeddings"]
    lifted = token_embeddings.repeat(1, 2) if pooling_modes else token_embeddings
    projections = lifted[:, : unit.numel()] @ unit
    assert [token["directional_projection"] for token in tokens] == pytest.approx(projections.tolist())


def test_model_mode_is_restored_on_failure(mock_model: OfflineMockSentenceTransformer) -> None:
    mock_model.train()
    mock_model.encoder_layer.eval()
    evaluator = EmbeddingXRayEvaluator(corpus_texts=["hello"], quantization_modes=("unknown",))
    with pytest.raises(ValueError, match="Unsupported quantization mode"):
        evaluator(mock_model)
    assert mock_model.training
    assert not mock_model.encoder_layer.training


def test_empty_queries_report_unavailable_overlap(
    mock_model: OfflineMockSentenceTransformer, tmp_path: Path,
) -> None:
    evaluator = EmbeddingXRayEvaluator(
        corpus_texts=["hello", "world"], query_texts=[], attribution_texts=[], k=1,
    )
    report = evaluator.evaluate_report(mock_model, str(tmp_path))
    assert report.metadata["num_queries"] == 0
    assert report.hubness["num_retrieved"] == 0
    assert report.fracture["compression_ratio"] == 1.0
    assert not any(key.startswith("overlap_at_") for key in report.metrics)
    assert "boilerplate_hijack_score" not in report.metrics
    assert "Unavailable" in (tmp_path / "xray_report.md").read_text(encoding="utf-8")
    loaded = json.loads((tmp_path / "xray_report.json").read_text(encoding="utf-8"))
    assert loaded["fracture"]["grid"][0]["overlap_at_k"] is None


def test_csv_preserves_metrics_across_schema_changes(
    mock_model: OfflineMockSentenceTransformer, tmp_path: Path,
) -> None:
    corpus = ["first input", "second input", "third input"]
    common = dict(corpus_texts=corpus, k=1, quantization_modes=(), write_report=False)
    first = EmbeddingXRayEvaluator(**common, matryoshka_dims=[64])
    second = EmbeddingXRayEvaluator(**common, matryoshka_dims=[32, 16])
    expected = [evaluator(mock_model, str(tmp_path)) for evaluator in (first, second, first)]
    with (tmp_path / first.csv_file).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(expected)
    for row, metrics in zip(rows, expected):
        assert None not in row
        reported = {key: float(value) for key, value in row.items() if key in metrics and value}
        assert reported == metrics
        assert all(row[key] == "" for key in row.keys() - metrics.keys() - {"epoch", "steps"})
