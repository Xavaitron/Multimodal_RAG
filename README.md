# Lightweight Multimodal Document Retrieval

ColSmol-500M document retrieval with manual MaxSim, retrieval metrics, token
compression, caching, and GPU measurements.

## Install

Run from the repository directory:

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Python 3.10-3.14 is required. Do not use bare `pip` outside the activated
environment, because it may point to Python 2 on older servers.

Activate the environment again after reconnecting to the server:

```bash
source .venv/bin/activate
```

## Test

```bash
nvidia-smi
python scripts/environment_diagnostic.py --skip-model
pytest -q
python scripts/environment_diagnostic.py
```

The final command downloads `vidore/colSmol-500M` and runs one image/query
forward pass. Diagnostic results are saved to
`artifacts/results/environment_diagnostic.json`.

## Run all full benchmark evaluations

No local PDFs or images are needed. Run all four commands below for the final
results. Each command evaluates every query against the complete document
corpus and writes to a separate JSON file.

Change `CUDA_VISIBLE_DEVICES=0` if you want to use a different GPU.

### DocVQA

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --max-queries 0 \
  --max-documents 0 \
  --output artifacts/results/vidore_docvqa.json
```

### InfoVQA

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --max-queries 0 \
  --max-documents 0 \
  --output artifacts/results/vidore_infovqa.json
```

### ArxivQA

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --max-queries 0 \
  --max-documents 0 \
  --output artifacts/results/vidore_arxivqa.json
```

### TatDQA

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --max-queries 0 \
  --max-documents 0 \
  --output artifacts/results/vidore_tatdqa.json
```

A successful full result has `"benchmark_comparable": true` in its JSON.
Do not use files showing `"benchmark_comparable": false` as final results.
