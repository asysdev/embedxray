"""Diagnostic modules for attribution, geometry, and fracture analysis."""

from __future__ import annotations

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

__all__ = [
    "DEFAULT_BOILERPLATE_PATTERN",
    "compute_token_attribution",
    "compute_pair_collisions",
    "compute_anisotropy",
    "compute_dimension_health",
    "compute_hubness",
    "evaluate_fracture",
]
