"""Structured diagnostic report dataclasses, composite health scoring, and export utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from filelock import FileLock


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamps a floating-point value to [low, high]."""
    return float(min(max(value, low), high))


def compute_composite_health_score(
    geometry: dict[str, Any],
    dimension_health: dict[str, Any],
    hubness: dict[str, Any],
    fracture: dict[str, Any],
    attribution: dict[str, Any],
    collisions: dict[str, Any] | None = None,
) -> float:
    """Computes a composite embedding health score in [0.0, 100.0]."""
    spectral_entropy = _clamp(float(geometry.get("spectral_entropy", 0.0)))
    mean_cos = float(geometry.get("mean_pairwise_cosine", 0.0))
    cone_health = _clamp(1.0 - max(0.0, mean_cos))
    isotropy_score = 0.5 * spectral_entropy + 0.5 * cone_health

    dead_ratio = _clamp(float(dimension_health.get("dead_dimension_ratio", 0.0)))
    dim_score = 1.0 - dead_ratio

    robin_hood = _clamp(float(hubness.get("robin_hood_index", 0.0)))
    skewness = max(0.0, float(hubness.get("hubness_skewness", 0.0)))
    skew_health = 1.0 / (1.0 + skewness / 2.0)
    hub_score = 0.5 * (1.0 - robin_hood) + 0.5 * skew_health

    weighted_scores = [(0.25, isotropy_score), (0.15, dim_score)]
    if hubness.get("num_retrieved", 1) > 0:
        weighted_scores.append((0.20, hub_score))

    overlaps = [
        float(item["overlap_at_k"])
        for item in fracture.get("grid", [])
        if item.get("overlap_at_k") is not None
    ]
    if overlaps:
        weighted_scores.append((0.20, _clamp(sum(overlaps) / len(overlaps))))

    attribution_scores = []
    hijack = attribution.get("boilerplate_hijack_score")
    if hijack is not None:
        attribution_scores.append(1.0 - _clamp(float(hijack)))
    if collisions is not None and collisions.get("pair_similarities"):
        coll_rate = _clamp(float(collisions.get("collision_rate", 0.0)))
        attribution_scores.append(1.0 - coll_rate)
    if attribution_scores:
        weighted_scores.append((0.20, sum(attribution_scores) / len(attribution_scores)))

    score = sum(weight * value for weight, value in weighted_scores)
    composite = 100.0 * score / sum(weight for weight, _ in weighted_scores)
    return _clamp(composite, 0.0, 100.0)


def _format_metric(value: float | None) -> str:
    """Formats an optional metric for Markdown."""
    return "Unavailable" if value is None else f"{value:.4f}"


def _extend_csv_schema(
    target: Path, header: list[str], fieldnames: list[str], row_data: dict[str, Any]
) -> None:
    """Adds columns atomically while preserving existing rows and their column names."""
    temporary = NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=target.parent, delete=False
    )
    staging_path = Path(temporary.name)
    try:
        with temporary, target.open(newline="", encoding="utf-8") as source:
            reader = csv.reader(source)
            next(reader)
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Malformed CSV row {reader.line_num} in {target}.")
                writer.writerow(dict(zip(header, row)))
            writer.writerow(row_data)
        staging_path.chmod(target.stat().st_mode & 0o777)
        staging_path.replace(target)
    finally:
        staging_path.unlink(missing_ok=True)


@dataclass
class XRayReport:
    """Structured diagnostic report produced by EmbeddingXRayEvaluator."""

    health_score: float
    anisotropy: dict[str, float]
    dimension_health: dict[str, Any]
    hubness: dict[str, Any]
    fracture: dict[str, Any]
    attribution: dict[str, Any]
    collisions: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation of the report."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serializes the report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def save_json(self, path: str | Path) -> Path:
        """Writes the JSON report to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2), encoding="utf-8")
        return target

    def to_markdown(self) -> str:
        """Formats a concise human-readable Markdown summary of the X-Ray report."""
        hijack = _format_metric(self.attribution.get("boilerplate_hijack_score"))
        dominance = _format_metric(self.attribution.get("max_token_dominance"))
        lines = [
            "# Embedding X-Ray Diagnostic Report",
            "",
            f"**Composite Health Score**: `{self.health_score:.2f} / 100.00`",
            "",
            "## 1. High-Dimensional Geometry & Hubness",
            f"- **Mean Pairwise Cosine**: `{self.anisotropy.get('mean_pairwise_cosine', 0.0):.4f}`",
            f"- **Normalized Spectral Entropy**: `{self.anisotropy.get('spectral_entropy', 0.0):.4f}`",
            f"- **Dead Dimension Count**: `{self.dimension_health.get('dead_dimension_count', 0)}` "
            f"(`{self.dimension_health.get('dead_dimension_ratio', 0.0) * 100.0:.2f}%`)",
            f"- **Hubness Skewness**: `{self.hubness.get('hubness_skewness', 0.0):.4f}`",
            f"- **Robin Hood Index**: `{self.hubness.get('robin_hood_index', 0.0):.4f}`",
            "",
            "## 2. Token Attribution & Collisions",
            f"- **Boilerplate Hijack Score**: `{hijack}`",
            f"- **Max Token Dominance**: `{dominance}`",
            f"- **Contrast Collision Rate**: `{self.collisions.get('collision_rate', 0.0):.4f}`",
            "",
            "## 3. Matryoshka & Quantization Fracture Grid",
            f"- **Fracture Cliff Bytes**: `{self.fracture.get('fracture_cliff_bytes', 0.0):.1f} B`",
            f"- **Compression Ratio**: `{self.fracture.get('compression_ratio', 1.0):.2f}x`",
            "",
            "| Dimension | Mode | Bytes / Vec | Overlap@k | Self-Similarity Drift |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for row in self.fracture.get("grid", []):
            overlap = _format_metric(row["overlap_at_k"])
            drift = _format_metric(row["self_similarity_drift"])
            lines.append(
                f"| {row['dim']} | {row['mode']} | {row['bytes_per_vector']:.1f} "
                f"| {overlap} | {drift} |"
            )
        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str | Path) -> Path:
        """Writes the Markdown summary report to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    def append_csv(self, path: str | Path, epoch: int = -1, steps: int = -1) -> Path:
        """Adds a metric row, extending the CSV header when new metrics appear."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        row_data: dict[str, Any] = {"epoch": epoch, "steps": steps}
        row_data.update(self.metrics)

        lock_path = target.with_name(f".{target.name}.lock")
        with FileLock(str(lock_path)):
            header: list[str] = []
            if target.exists():
                with target.open(newline="", encoding="utf-8") as source:
                    header = next(csv.reader(source), [])
                if len(set(header)) != len(header):
                    raise ValueError(f"Duplicate CSV columns in {target}.")

            fieldnames = list(dict.fromkeys([*header, *row_data]))
            if header and fieldnames != header:
                _extend_csv_schema(target, header, fieldnames, row_data)
            else:
                with target.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    if not header:
                        writer.writeheader()
                    writer.writerow(row_data)
        return target
