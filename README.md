# Lightweight Multimodal Document Retrieval

Reproducible document-retrieval experiments comparing ColSmol-500M,
ColSmol-256M, a global visual baseline, OCR BM25, and BGE-small dense text
retrieval on DocVQA, InfoVQA, ArxivQA, and TatDQA. The project also measures
latency, VRAM, index size, and token-compression trade-offs.

No local PDFs are required. The commands download the standard ViDoRe test
datasets and model weights from Hugging Face.

## Install on the SSH server

Run these commands from the repository directory. Python 3.10-3.14 is
required.

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

After reconnecting to the server, reactivate the environment with:

```bash
source .venv/bin/activate
```

Always use `python -m pip` inside the activated environment. On older servers,
the bare `pip` command may still point to Python 2.

## Complete unattended run

From the repository directory, this single command creates/reuses `.venv`,
installs dependencies, tests the code, runs all datasets, models, text
baselines and compression experiments on physical GPU 2, then builds the final
tables and figures. It also writes a timestamped log under `artifacts/logs/`.

```bash
GPU_ID=2 bash scripts/run_all_experiments.sh
```

Run it inside tmux. Existing valid compression results are retained, and visual
embedding caches are reused, so an interrupted run can be started again.

## Environment check and tests

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 nvidia-smi
CUDA_VISIBLE_DEVICES=2 python scripts/environment_diagnostic.py --skip-model
pytest -q
CUDA_VISIBLE_DEVICES=2 python scripts/environment_diagnostic.py \
  --model vidore/colSmol-500M
```

The final command downloads the 500M model and runs a real image/query forward
pass. It writes `artifacts/results/environment_diagnostic.json`.

## ColSmol-500M: full visual retrieval

Each command evaluates every query against the complete dataset corpus. It
measures global visual retrieval and token-level MaxSim separately and caches
embeddings under `cache/embeddings/`.

### DocVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/docvqa_colsmol500m.json
```

### InfoVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/infovqa_colsmol500m.json
```

### ArxivQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/arxivqa_colsmol500m.json
```

### TatDQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/tatdqa_colsmol500m.json
```

## ColSmol-256M: model-size comparison

### DocVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/docvqa_colsmol256m.json
```

### InfoVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/infovqa_colsmol256m.json
```

### ArxivQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/arxivqa_colsmol256m.json
```

### TatDQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/tatdqa_colsmol256m.json
```

## OCR text baselines: BM25 and BGE-small

These use the ViDoRe Tesseract page OCR and join it to the exact MTEB corpus
page by image hash. A run stops instead of reporting misleading numbers if OCR
coverage is below 100%.

This is a controlled page-level comparison over the same candidate corpus.
ViDoRe's separately published chunked-OCR leaderboard baseline uses a different
indexing unit, so its score should not be presented as directly equivalent.

### DocVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/docvqa_bgesmallenv15.json
```

### InfoVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/infovqa_bgesmallenv15.json
```

### ArxivQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/arxivqa_bgesmallenv15.json
```

### TatDQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/tatdqa_bgesmallenv15.json
```

## Token-compression ablations

Run the matching 500M visual command above first. Each command reuses that
embedding cache and evaluates 100%, 75%, 50%, 25%, and 12.5% token retention.
Methods are random selection, uniform sequence sampling, local sequence mean
pooling, and k-means. Random selection is repeated with seeds 40, 41, and 42.
Mean-pool and k-means centroids are L2-normalized before MaxSim scoring;
k-means uses eight iterations per page.

### DocVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

### InfoVQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

### ArxivQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

### TatDQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

## Build final tables, plots, and failure sheet

Run this only after all commands above finish:

```bash
python scripts/build_report_artifacts.py
```

It validates that all four datasets have full-corpus results, then writes:

- `artifacts/summary/main_results.csv`
- `artifacts/summary/compression_results.csv`
- `artifacts/summary/compression_aggregate.csv` (mean and standard deviation
  across random seeds)
- `artifacts/summary/failure_analysis.csv`
- `artifacts/summary/RESULTS.md`
- `artifacts/summary/figures/*.png`

Every final JSON must contain `"schema_version": 2` and
`"benchmark_comparable": true`. Files copied from older runs at the root of
`artifacts/results/` are preserved but are not included in the new summary.
