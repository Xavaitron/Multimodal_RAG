# Lightweight Multimodal Document Retrieval

ColSmol-500M document retrieval with manual MaxSim, retrieval metrics, token
compression, caching, and GPU measurements.

## Install

Run from the repository directory using the default Python installation:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
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

## Run the standard dataset

No local PDFs or images are needed. This downloads and evaluates a small subset
of the standard ViDoRe DocVQA retrieval benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py --max-queries 20 --max-documents 100
```

Results are saved to `artifacts/results/vidore.json`.
Change `0` to the GPU index you want to use.

Run the complete DocVQA evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py --max-queries 0 --max-documents 0
```

Other supported ViDoRe datasets:

```bash
# Infographics
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py --dataset mteb/VidoreInfoVQARetrieval

# Scientific papers
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py --dataset mteb/VidoreArxivQARetrieval

# Financial tables
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_vidore.py --dataset mteb/VidoreTatdqaRetrieval
```

Small subsets are development checks. Use the complete dataset for final,
benchmark-comparable results.
