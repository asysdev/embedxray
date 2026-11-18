# embedxray

`embedxray` is a read-only, pure-Python diagnostic and evaluation library for `sentence-transformers` (compatible with v3.x through v6.3+ `SentenceEvaluator` / `BaseEvaluator` interfaces).

## Key Capabilities

- **Token & Pooling Attribution (`embedxray.modules.attribution`)**: Measures token-level directional projection onto pooled sentence embeddings, quantifies boilerplate hijack scores, and detects contrast-pair embedding collisions.
- **High-Dimensional Geometry & Hubness (`embedxray.modules.geometry`)**: Computes pairwise cosine anisotropy, normalized spectral entropy via MPS-safe SVD, dead dimension ratios, $k$-occurrence hubness skewness, and the Robin Hood index.
- **Quantization & Matryoshka Fracture Simulation (`embedxray.modules.fracture`)**: Evaluates `Overlap@k` and self-similarity drift across Matryoshka truncation dimensions and quantization precisions (`fp32`, `fp16`, `int8`, `int4`, `binary`) to identify the minimum-byte fracture cliff.
- **Drop-In Evaluator (`embedxray.evaluator`)**: Subclasses `SentenceEvaluator` / `BaseEvaluator` to compute a composite health score (`0.0` to `100.0`), return flat metric dictionaries with `primary_metric = "xray_health_score"`, and export structured JSON, CSV, and Markdown reports.

## Installation

```bash
python -m pip install embedxray
```

## Quickstart

```python
from sentence_transformers import SentenceTransformer

from embedxray import EmbeddingXRayEvaluator

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

evaluator = EmbeddingXRayEvaluator(
    corpus_texts=[
        "Machine learning systems require numerical stability.",
        "Vector databases use quantization to reduce memory footprint.",
        "Matryoshka representation learning enables nested truncation.",
    ],
    contrast_pairs=[
        ("The service is online and healthy.", "The service is offline and failing."),
    ],
    k=2,
    write_csv=True,
    write_report=True,
)

metrics = evaluator(model, output_path="./xray_results")
print(f"X-Ray Health Score: {metrics['xray_health_score']:.2f}")
```

## Interpreting reports

- Token attribution uses the model's token embeddings and attention mask. Truncated embeddings use the corresponding token prefix. With concatenated pooling, each token is projected against the pooled blocks that were retained. Samples without aligned token features include an `unavailable_reason` in the structured report.
- Retrieval breaks ties by ascending corpus index. Omitting `query_texts` uses self-retrieval. Passing `query_texts=[]` runs diagnostics without retrieval queries. When no neighbors can be measured, overlap is `null` in JSON and the storage estimate falls back to full FP32.
- Unavailable attribution and overlap metrics are omitted from the evaluator's flat metrics. The composite score redistributes the weight of unavailable components across those with measurements. CSV exports retain previous columns and leave unavailable values blank.
