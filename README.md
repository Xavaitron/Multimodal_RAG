# Lightweight Multimodal Document Retrieval

Research implementation of ColSmol-500M document retrieval using manual MaxSim,
retrieval baselines, token compression, caching, metrics, and GPU measurements.

## Run on the SSH GPU server

After logging into the server, run these commands in order. This uses the
server's default Python and does not create a virtual environment.

```bash
git clone https://github.com/Xavaitron/Multimodal_RAG.git
cd Multimodal_RAG
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user -e . --no-deps
nvidia-smi
python3 scripts/environment_diagnostic.py --skip-model
python3 -m pytest -q
python3 scripts/environment_diagnostic.py
python3 scripts/evaluate_vidore.py --max-queries 20 --max-documents 100
```

## Run with a standard dataset

The final command above needs no local PDFs or images. It downloads ColSmol-500M
and a 20-query/100-page subset of the standard ViDoRe DocVQA retrieval benchmark.
Results are saved to `artifacts/results/vidore.json`.

Other useful datasets:

```bash
# Infographics and visually structured pages
python3 scripts/evaluate_vidore.py --dataset mteb/VidoreInfoVQARetrieval

# Scientific papers
python3 scripts/evaluate_vidore.py --dataset mteb/VidoreArxivQARetrieval

# Financial tables
python3 scripts/evaluate_vidore.py --dataset mteb/VidoreTatdqaRetrieval
```

Use `--max-queries 0 --max-documents 0` for the complete dataset. Small subsets
are development checks, not benchmark-comparable final results.

For long SSH jobs, run inside `tmux`:

```bash
tmux new -s colsmol
python3 scripts/evaluate_vidore.py 2>&1 | tee vidore.log
# Detach: Ctrl-b d       Reconnect: tmux attach -t colsmol
```

## Project commands

```bash
python3 scripts/evaluate_bm25.py --help
python3 scripts/sanity_check.py --help
python3 -m ruff check .
python3 -m pytest -q
```

The experiment sequence and reproducibility rules are documented in
[`docs/experiment_plan.md`](docs/experiment_plan.md).
