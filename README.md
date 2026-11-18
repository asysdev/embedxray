# embedxray

Diagnose what your sentence embeddings encode and how much retrieval changes when you compress them. **embedxray** evaluates a [Sentence Transformers](https://www.sbert.net/) model on your own corpus, queries, and contrast pairs. It runs inference without updating model weights and restores the model's training mode after evaluation.

| Diagnostic | What it measures |
| --- | --- |
| Token attribution | Which tokens align most strongly with the pooled embedding, and how much of their positive contribution comes from boilerplate. |
| Contrast collisions | How often text pairs you expect to distinguish exceed a cosine similarity threshold. |
| Embedding geometry | Sampled pairwise cosine, spectral entropy, and the fraction of low-variance dimensions. |
| Hubness | Whether a small set of documents appears disproportionately often in the nearest neighbors. |
| Compression sensitivity | Top-k neighbor overlap and vector fidelity across dimension truncation and FP16, INT8, INT4, and binary simulations. |

## Installation

Requires Python 3.10 or later, PyTorch, and Sentence Transformers. The test suite has passed on CPU with Sentence Transformers 3.0.0 and 6.3.0.

Install the package from PyPI once the first release is live.

```bash
python -m pip install embedxray
```

Install the current version directly from GitHub.

```bash
python -m pip install 'git+https://github.com/asysdev/embedxray.git'
```

## Quickstart

Use representative documents and queries from your retrieval workload. The first run of this example downloads the named model if it is missing from the local cache. You can also pass a local model directory to `SentenceTransformer`.

```python
from sentence_transformers import SentenceTransformer

from embedxray import EmbeddingXRayEvaluator

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

evaluator = EmbeddingXRayEvaluator(
    corpus_texts=[
        "Reset your password using the link on the sign-in page.",
        "Add a recovery phone number in your account settings.",
        "Update your payment method from the billing dashboard.",
        "Export past invoices from the billing dashboard.",
        "Contact support if your account is locked.",
    ],
    query_texts=["How do I reset my password?", "Where can I download an invoice?"],
    contrast_pairs=[
        ("Two-factor authentication is enabled.", "Two-factor authentication is disabled."),
    ],
    k=2,
    matryoshka_dims=[384, 128, 64],
    cliff_threshold=0.95,
)

report = evaluator.evaluate_report(model, output_path="xray_out")
print(f"X-Ray health score: {report.health_score:.1f}/100")
print(report.fracture["fracture_cliff_config"])
```

Call `evaluator(model)` to get a flat metric dictionary for logging or a Sentence Transformers trainer. The default primary metric is `xray_health_score`. The most recent structured report is available as `evaluator.last_report`.

## Evaluation options

| Option | Behavior |
| --- | --- |
| `query_texts` | With no queries supplied, the corpus is queried against itself and each row excludes itself. An explicit empty list skips retrieval. |
| `contrast_pairs` | Supply pairs that should be distinguishable. A collision occurs above `collision_threshold`, which defaults to `0.90`. |
| `attribution_texts` | Defaults to the first 64 corpus entries. Supply representative text for attribution, or an empty list to skip it. |
| `matryoshka_dims`, `quantization_modes` | Select prefix dimensions and simulated precisions. Each selected dimension includes an FP32 result. |
| `k`, `chunk_size` | Set the neighbor count and similarity block size. See `report.metadata["actual_k"]` for the effective count after clamping to the eligible corpus size. |

## Reading the results

- **Health score** is a 0–100 summary of geometry, dimension health, hubness, compression stability, and attribution or contrast collisions. Compare runs on the same corpus, queries, and settings. Use labeled retrieval metrics such as Recall@k or nDCG to measure relevance.
- **Overlap@k** is the average fraction of the full-dimensional FP32 neighbor set retained by a configuration. **Self-similarity drift** reports mean cosine similarity between original and simulated vectors at the same dimension. Values closer to 1 indicate better directional preservation.
- **Fracture cliff** is the lowest estimated vector payload among configurations that reach the overlap threshold. The simulation estimates vector payload from dimension and precision. Index overhead, calibration metadata, padding, and retrieval latency are outside that estimate.
- **Unavailable measurements** appear as `null` in the structured report and are omitted from flat attribution and overlap metrics. Attribution samples include an `unavailable_reason` when token features cannot be aligned. When overlap cannot be measured, the cliff uses the full FP32 storage estimate.

When `output_path` is supplied, the evaluator writes these files.

| File | Contents |
| --- | --- |
| `xray_report.md` | Human-readable diagnostic summary and compression grid. |
| `xray_report.json` | Structured scores, per-sample token evidence, and details of offending contrast pairs. This file can contain original input text. |
| `xray_evaluation_results.csv` | An appended flat metric row with epoch and step columns. |

Use `name` to label report filenames and prefix metric keys. Use `write_csv=False` or `write_report=False` to disable the corresponding exports.

## Using precomputed embeddings

Run geometry or compression diagnostics directly on PyTorch tensors shaped `(samples, dimensions)`. For example, use `evaluate_fracture(corpus_embeddings, query_embeddings, k=10)` to inspect the compression grid. The corresponding geometry entry points are `compute_anisotropy`, `compute_dimension_health`, and `compute_hubness`, all exported from `embedxray`.

Retrieval uses exact batched similarity comparisons. Runtime increases with the number of query–corpus pairs and compression configurations.

## Development

```bash
git clone https://github.com/asysdev/embedxray.git
cd embedxray
python -m pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest
```

The tests use synthetic embeddings and locally constructed models, so they run without downloading model weights.

## License

[Apache License 2.0](https://github.com/asysdev/embedxray/blob/main/LICENSE)
