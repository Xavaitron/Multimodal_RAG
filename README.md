# Efficient Multimodal Document Retrieval with Lightweight VLMs

Research code for studying ColSmolVLM document retrieval on a 12 GB NVIDIA GPU. The repository keeps the scientific core independent of any convenience scorer: MaxSim, global pooling, retrieval metrics, token compression, caching, and efficiency measurements live in this project.

## What works now

- RTX/CUDA/environment diagnostics, including an optional real ColSmol-500M image/query forward pass
- manual ColBERT-style MaxSim and batched scoring
- global visual pooling baseline
- Recall@K, MRR, and nDCG@K
- random, uniform, local-mean, and k-means token compression
- embedding cache with metadata
- deterministic seeding and JSON experiment output
- unit tests for the scientific core

The initial model is `vidore/colSmol-500M`. The code uses FP16 on CUDA and never claims a batch size before measuring it.

## Run on an SSH GPU server

These commands assume Ubuntu/Linux, an NVIDIA driver visible through `nvidia-smi`, and Python 3.10-3.13. The current ColPali package does not support Python 3.15+.

```bash
ssh USER@GPU_HOST
git clone YOUR_REPOSITORY_URL
cd Multimodal_RAG

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA build recommended for the server by pytorch.org first.
# Example for CUDA 12.8; change cu128 if the server needs another wheel index.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"
nvidia-smi
python scripts/environment_diagnostic.py --skip-model
pytest -q
```

Run the real model smoke test (first run downloads model weights from Hugging Face):

```bash
hf auth login                         # only if the server requires authentication
python scripts/environment_diagnostic.py --model vidore/colSmol-500M
```

For a real local-page ranking check, put 5-20 rendered page images in a directory and
create `queries.json` as a JSON list of strings, then run:

```bash
python scripts/sanity_check.py --pages data/sanity_pages --queries queries.json
```

Run it safely across a disconnect:

```bash
mkdir -p artifacts/logs
tmux new -s colsmol
python scripts/environment_diagnostic.py --model vidore/colSmol-500M \
  2>&1 | tee artifacts/logs/diagnostic.log
# Detach with Ctrl-b d; reconnect with: tmux attach -t colsmol
```

If multiple GPUs are present, select one before launching:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/environment_diagnostic.py
```

Successful output records model parameters, CUDA properties, peak VRAM, embedding counts, embedding dimensions, and a manual MaxSim score in `artifacts/results/environment_diagnostic.json`.

## Development

```bash
pytest -q
ruff check .
```

Experiment outputs belong in `artifacts/`; downloaded weights, caches, and checkpoints are ignored by Git. See `docs/experiment_plan.md` for the milestone order and experimental contract.

## Scientific scope

The central score is

```text
S(Q, D) = sum_i max_j q_i^T d_j
```

The planned comparison is BM25 vs dense text vs global visual pooling vs multi-vector MaxSim, followed by 256M/500M capacity, fine-tuning, and token-compression ablations. RAG generation is deliberately last.
