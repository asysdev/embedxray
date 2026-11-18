"""embedxray: Diagnostic and evaluation library for sentence-transformers."""

from __future__ import annotations

from embedxray.evaluator import EmbeddingXRayEvaluator
from embedxray.modules import (
    compute_anisotropy,
    compute_dimension_health,
    compute_hubness,
    compute_pair_collisions,
    compute_token_attribution,
    evaluate_fracture,
)
from embedxray.report import XRayReport, compute_composite_health_score
from embedxray.utils import (
    binarize_embeddings,
    chunked_topk_binary,
    chunked_topk_cosine,
    safe_l2_normalize,
    safe_svdvals,
    simulate_fp16,
    simulate_int4,
    simulate_int8,
)

__version__ = "0.1.0"

__all__ = [
    "EmbeddingXRayEvaluator",
    "XRayReport",
    "compute_composite_health_score",
    "compute_token_attribution",
    "compute_pair_collisions",
    "compute_anisotropy",
    "compute_dimension_health",
    "compute_hubness",
    "evaluate_fracture",
    "safe_l2_normalize",
    "safe_svdvals",
    "chunked_topk_cosine",
    "simulate_fp16",
    "simulate_int8",
    "simulate_int4",
    "binarize_embeddings",
    "chunked_topk_binary",
]
