"""SentenceEvaluator integration for embedxray."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
except ImportError:
    try:
        from sentence_transformers.evaluation import SentenceEvaluator
    except ImportError:
        from sentence_transformers.base.evaluation import BaseEvaluator as SentenceEvaluator

from embedxray.modules.attribution import (
    DEFAULT_BOILERPLATE_PATTERN,
    compute_pair_collisions,
    compute_token_attribution,
)
from embedxray.modules.fracture import evaluate_fracture
from embedxray.modules.geometry import (
    compute_anisotropy,
    compute_dimension_health,
    compute_hubness,
)
from embedxray.report import XRayReport, compute_composite_health_score
from embedxray.utils.model import encode_sentences, preserve_training_mode


class EmbeddingXRayEvaluator(SentenceEvaluator):
    """Read-only diagnostic evaluator for sentence-transformers models."""

    def __init__(
        self,
        corpus_texts: list[str] | None = None,
        query_texts: list[str] | None = None,
        contrast_pairs: list[tuple[str, str]] | None = None,
        attribution_texts: list[str] | None = None,
        matryoshka_dims: list[int] | None = None,
        quantization_modes: tuple[str, ...] = ("fp16", "int8", "int4", "binary"),
        k: int = 10,
        batch_size: int = 32,
        chunk_size: int = 2048,
        max_anisotropy_pairs: int = 50000,
        cliff_threshold: float = 0.95,
        collision_threshold: float = 0.90,
        boilerplate_pattern: str = DEFAULT_BOILERPLATE_PATTERN,
        name: str = "",
        write_csv: bool = True,
        write_report: bool = True,
        seed: int = 42,
        *,
        corpus: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> None:
        super().__init__()
        resolved_corpus = corpus_texts if corpus_texts is not None else corpus
        resolved_queries = query_texts if query_texts is not None else queries

        if not resolved_corpus:
            raise ValueError("corpus_texts (or corpus) must contain at least one text sample.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.corpus_texts = list(resolved_corpus)
        self.corpus = self.corpus_texts
        self.query_texts = list(resolved_queries) if resolved_queries is not None else None
        self.queries = self.query_texts
        self.contrast_pairs = list(contrast_pairs) if contrast_pairs is not None else None
        self.attribution_texts = (
            list(attribution_texts)
            if attribution_texts is not None
            else self.corpus_texts[:64]
        )
        self.matryoshka_dims = list(matryoshka_dims) if matryoshka_dims is not None else None
        self.quantization_modes = tuple(quantization_modes)
        self.k = int(k)
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.max_anisotropy_pairs = int(max_anisotropy_pairs)
        self.cliff_threshold = float(cliff_threshold)
        self.collision_threshold = float(collision_threshold)
        self.boilerplate_pattern = boilerplate_pattern
        self.name = name
        self.write_csv = bool(write_csv)
        self.write_report = bool(write_report)
        self.seed = int(seed)

        self.greater_is_better = True
        self.primary_metric = (
            f"{self.name}_xray_health_score" if self.name else "xray_health_score"
        )
        self.csv_file = (
            f"xray_evaluation_{self.name}_results.csv"
            if self.name
            else "xray_evaluation_results.csv"
        )
        self.last_report: XRayReport | None = None

    def _prefix_metrics(self, base_metrics: dict[str, float]) -> dict[str, float]:
        """Applies evaluator name prefix using SentenceEvaluator conventions."""
        if hasattr(self, "prefix_name_to_metrics") and callable(self.prefix_name_to_metrics):
            return self.prefix_name_to_metrics(base_metrics, self.name)
        if not self.name:
            return dict(base_metrics)
        return {f"{self.name}_{key}": val for key, val in base_metrics.items()}

    def evaluate_report(
        self,
        model: Any,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> XRayReport:
        """Runs the full read-only X-Ray diagnostic suite and returns a structured XRayReport."""
        with preserve_training_mode(model), torch.no_grad():
            corpus_emb = encode_sentences(model, self.corpus_texts, batch_size=self.batch_size)
            if self.query_texts is None:
                query_emb = None
            elif self.query_texts:
                query_emb = encode_sentences(model, self.query_texts, batch_size=self.batch_size)
            else:
                query_emb = corpus_emb.new_empty((0, corpus_emb.size(1)))

            anisotropy = compute_anisotropy(
                corpus_emb,
                max_pairs=self.max_anisotropy_pairs,
                seed=self.seed,
            )
            dim_health = compute_dimension_health(corpus_emb)
            hubness = compute_hubness(
                corpus_embeddings=corpus_emb,
                query_embeddings=query_emb,
                k=self.k,
                chunk_size=self.chunk_size,
            )
            fracture = evaluate_fracture(
                corpus_embeddings=corpus_emb,
                query_embeddings=query_emb,
                matryoshka_dims=self.matryoshka_dims,
                quantization_modes=self.quantization_modes,
                k=self.k,
                chunk_size=self.chunk_size,
                cliff_threshold=self.cliff_threshold,
            )
            attribution = compute_token_attribution(
                model=model,
                texts=self.attribution_texts,
                batch_size=self.batch_size,
                boilerplate_pattern=self.boilerplate_pattern,
            )
            collisions = compute_pair_collisions(
                model=model,
                contrast_pairs=self.contrast_pairs or [],
                batch_size=self.batch_size,
                collision_threshold=self.collision_threshold,
            )

        health_score = compute_composite_health_score(
            geometry=anisotropy,
            dimension_health=dim_health,
            hubness=hubness,
            fracture=fracture,
            attribution=attribution,
            collisions=collisions if self.contrast_pairs else None,
        )

        raw_metrics: dict[str, float] = {
            "xray_health_score": float(health_score),
            "mean_pairwise_cosine": float(anisotropy["mean_pairwise_cosine"]),
            "spectral_entropy": float(anisotropy["spectral_entropy"]),
            "dead_dimension_count": float(dim_health["dead_dimension_count"]),
            "dead_dimension_ratio": float(dim_health["dead_dimension_ratio"]),
            "hubness_skewness": float(hubness["hubness_skewness"]),
            "robin_hood_index": float(hubness["robin_hood_index"]),
            "mean_contrast_similarity": float(collisions["mean_contrast_similarity"]),
            "collision_rate": float(collisions["collision_rate"]),
            "fracture_cliff_bytes": float(fracture["fracture_cliff_bytes"]),
            "compression_ratio": float(fracture["compression_ratio"]),
        }

        for key in ("boilerplate_hijack_score", "max_token_dominance"):
            if attribution[key] is not None:
                raw_metrics[key] = float(attribution[key])

        for entry in fracture["grid"]:
            dim = entry["dim"]
            mode = entry["mode"]
            if entry["overlap_at_k"] is not None:
                raw_metrics[f"overlap_at_{self.k}_d{dim}_{mode}"] = float(entry["overlap_at_k"])
            if entry["self_similarity_drift"] is not None:
                raw_metrics[f"drift_d{dim}_{mode}"] = float(entry["self_similarity_drift"])

        prefixed_metrics = self._prefix_metrics(raw_metrics)

        report = XRayReport(
            health_score=float(health_score),
            anisotropy=anisotropy,
            dimension_health=dim_health,
            hubness=hubness,
            fracture=fracture,
            attribution=attribution,
            collisions=collisions,
            metrics=prefixed_metrics,
            metadata={
                "name": self.name,
                "epoch": epoch,
                "steps": steps,
                "num_corpus": int(corpus_emb.size(0)),
                "embedding_dim": int(corpus_emb.size(1)),
                "k": self.k,
                "actual_k": fracture["actual_k"],
                "num_queries": fracture["num_queries"],
            },
        )
        self.last_report = report

        if output_path is not None:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.write_csv:
                report.append_csv(out_dir / self.csv_file, epoch=epoch, steps=steps)
            if self.write_report:
                json_name = f"xray_report_{self.name}.json" if self.name else "xray_report.json"
                md_name = f"xray_report_{self.name}.md" if self.name else "xray_report.md"
                report.save_json(out_dir / json_name)
                report.save_markdown(out_dir / md_name)

        return report

    def __call__(
        self,
        model: Any,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Executes the X-Ray evaluation and returns a flat dictionary of metrics."""
        report = self.evaluate_report(
            model=model,
            output_path=output_path,
            epoch=epoch,
            steps=steps,
        )
        return report.metrics
