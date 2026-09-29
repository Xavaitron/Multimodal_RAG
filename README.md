# Lightweight Multimodal Document Retrieval with ColSmolVLM

Course project for **EE798R: Intelligent Pattern Recognition**.

This repository studies the retrieval stage of a multimodal
retrieval-augmented generation (RAG) system. The central question is:

> How can a lightweight model retrieve the correct page from visually rich
> documents when meaning is carried by text, layout, tables, plots, and spatial
> position?

The project evaluates released ColSmolVLM checkpoints on four standard ViDoRe
page-retrieval datasets. It compares visual retrieval against OCR-based text
retrieval, studies whether local page-token matching is better than a single
page vector, compares 256M and 500M models, and measures how aggressively the
visual index can be compressed.

No local PDF collection is required. The evaluation datasets, OCR artifacts,
and model weights are downloaded from Hugging Face.

## What this project does

The complete experimental flow is:

```text
ViDoRe page images and questions
              |
              +--> OCR text --> BM25 / BGE-small text baselines
              |
              +--> ColSmolVLM page-token embeddings
                            |
                            +--> global pooled visual baseline
                            |
                            +--> MaxSim multi-vector retrieval
                                         |
                                         +--> token-compression experiments
                                                      |
                                                      +--> tables, plots,
                                                           failure analysis,
                                                           and project report
```

The project is specifically a **retrieval study**. RAG motivates the problem,
but an answer-generating language model is not evaluated. The output of the
system is a ranked list of relevant document pages that could later be passed
to a vision-language generator.

## Why visual retrieval is useful

An OCR-first pipeline converts a two-dimensional page into a text string:

```text
PDF page -> OCR -> text chunks -> text embedding -> retrieval
```

This can lose table structure, chart legends, small labels, logos, and spatial
relationships. The visual pipeline keeps the rendered page intact:

```text
page image -> visual token embeddings -> late interaction -> retrieval
```

ColSmolVLM represents a question with vectors

```text
Q = [q1, q2, ..., qm]
```

and a page with vectors

```text
D = [d1, d2, ..., dn].
```

The MaxSim relevance score is

```text
S(Q, D) = sum_i max_j (qi^T dj).
```

Each question token selects its strongest matching page region. The selected
similarities are then added. This lets different concepts in a question match
different locations, such as separate cells in a table.

The global visual baseline instead averages all page tokens into one vector.
Comparing these two scores isolates the value of keeping local page structure.

## Experiments

The project contains four connected experiments.

1. **Text versus visual retrieval**
   - Okapi BM25 over page OCR
   - BGE-small-en-v1.5 dense retrieval over page OCR
   - ColSmolVLM visual retrieval

2. **Global versus multi-vector representation**
   - one average vector per page
   - all page vectors scored with MaxSim

3. **Model-size comparison**
   - `vidore/colSmol-256M`
   - `vidore/colSmol-500M`

4. **Token-compression ablation**
   - retention: 100%, 75%, 50%, 25%, and 12.5%
   - methods: random, uniform, mean-pool, and k-means
   - random selection is repeated with seeds 40, 41, and 42

The main metrics are nDCG@5, mean reciprocal rank, Recall@1, Recall@5, and
Recall@10. The code also records page/query encoding time, retrieval latency,
peak VRAM, token counts, and index size.

## Datasets

| Dataset | Queries | Corpus pages | Main document type |
| --- | ---: | ---: | --- |
| DocVQA | 451 | 500 | scanned forms and documents |
| InfoVQA | 494 | 500 | infographics |
| ArxivQA | 500 | 500 | scientific plots and figures |
| TatDQA | 1,646 | 277 | financial tables and reports |

Every query is ranked against the complete corpus for its dataset. The text
and visual systems therefore solve the same page-level retrieval problem.

The OCR join records mapping coverage and non-empty OCR coverage separately.
All corpus pages are retained even if their supplied OCR text is empty. This is
important for ArxivQA, where 422 of 500 pages have non-empty OCR.

## Main findings

The 500M MaxSim system is the strongest tested method on every dataset.

| Method | DocVQA | InfoVQA | ArxivQA | TatDQA |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 34.8 | 60.9 | 18.5 | 57.4 |
| BGE-small | 27.1 | 68.5 | 32.3 | 31.4 |
| Global visual, 500M | 19.1 | 55.5 | 25.3 | 28.4 |
| MaxSim visual, 256M | 55.6 | 83.6 | 72.4 | 75.7 |
| **MaxSim visual, 500M** | **57.8** | **86.9** | **74.7** | **76.5** |

Values are nDCG@5 percentages. The important result is that changing from
global pooling to MaxSim produces a much larger improvement than changing from
the 256M model to the 500M model. Preserving local page structure matters more
than the tested increase in model size.

K-means compression retains at least 95% of the uncompressed nDCG@5 while
keeping only 12.5-25% of page tokens, reducing index memory by approximately
4-8x. The latency improvement is not consistent at these small corpus sizes,
so the strongest compression claim is memory reduction rather than speed.

## Repository structure

```text
configs/                         experiment configuration
scripts/                         evaluation and reporting entry points
src/lightweight_multimodal_retrieval/
                                 reusable retrieval implementation
tests/                           unit tests
artifacts/results/visual/        full visual evaluations
artifacts/results/text/          BM25 and BGE evaluations
artifacts/results/compression/   compression configurations and manifests
artifacts/summary/               final CSV tables, plots, and failure cases
report/report.tex                editable LaTeX project report
output/pdf/report.pdf            compiled course-project report
```

## Requirements

- Ubuntu/Linux SSH server
- Python 3.10-3.14
- NVIDIA GPU with a working driver
- `nvidia-smi` visible from the shell
- internet access for the first model and dataset download

The tested environment used Python 3.12, CUDA-enabled PyTorch, and one NVIDIA
RTX A6000. The code does not depend on a particular physical GPU number.

## Install on the SSH server

Run all commands from the repository directory.

```bash
python --version
python -m venv rag
source rag/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

After reconnecting to the server, reactivate the existing environment:

```bash
source rag/bin/activate
```

Always use `python -m pip` inside the environment. On some older servers, the
bare `pip` command may still point to Python 2.

## Select a GPU

First inspect the GPUs:

```bash
nvidia-smi
```

Set `GPU_ID` to the device you want to use. The example selects the first GPU;
replace `0` when another device is appropriate.

```bash
export GPU_ID=0
```

All commands below use that value through `CUDA_VISIBLE_DEVICES="$GPU_ID"`.
The full pipeline also accepts `GPU_ID`; if it is not set, it defaults to GPU
0.

## Complete unattended run

This is the only command needed to reproduce the complete experiment suite:

```bash
source rag/bin/activate
bash scripts/run_all_experiments.sh
```

To select a device only for this run:

```bash
GPU_ID="$GPU_ID" bash scripts/run_all_experiments.sh
```

Run the command inside `tmux`. The script:

1. creates or reuses the `rag` environment;
2. installs `requirements.txt` and the local package;
3. records the environment and performs a real model smoke test;
4. runs the unit tests;
5. evaluates both visual models on all four datasets;
6. evaluates BM25 and BGE-small on all four datasets;
7. runs every compression configuration;
8. regenerates the final tables, figures, and failure sheet; and
9. writes a timestamped log under `artifacts/logs/`.

Completed visual and text results are kept when their schema, model, dataset,
and full-corpus status are valid. Visual embedding caches and completed
compression configurations are also reused. Therefore, restarting the script
after an interruption resumes the pipeline instead of repeating every valid
experiment.

## Environment check and tests

```bash
source rag/bin/activate
CUDA_VISIBLE_DEVICES="$GPU_ID" nvidia-smi
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/environment_diagnostic.py --skip-model
pytest -q
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/environment_diagnostic.py \
  --model vidore/colSmol-500M
```

The last command downloads the 500M checkpoint and performs a real image and
query forward pass. The diagnostic is written to
`artifacts/results/environment_diagnostic.json`.

## Individual experiment commands

The following commands are the manual alternative to the full pipeline. They
are useful when only one dataset or experiment must be rerun.

### ColSmol-500M visual retrieval

Each command evaluates global pooling and MaxSim against the complete corpus
and caches the visual embeddings under `cache/embeddings/`.

#### DocVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/docvqa_colsmol500m.json
```

#### InfoVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/infovqa_colsmol500m.json
```

#### ArxivQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/arxivqa_colsmol500m.json
```

#### TatDQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --model vidore/colSmol-500M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/tatdqa_colsmol500m.json
```

### ColSmol-256M model-size comparison

#### DocVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/docvqa_colsmol256m.json
```

#### InfoVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/infovqa_colsmol256m.json
```

#### ArxivQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/arxivqa_colsmol256m.json
```

#### TatDQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_vidore.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --model vidore/colSmol-256M \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/visual/tatdqa_colsmol256m.json
```

### OCR text baselines

Each command runs BM25 and BGE-small-en-v1.5. OCR pages are joined to corpus
pages by exact filename or image digest. A source-position fallback is allowed
only after the code verifies strong same-position image agreement. The JSON
stores mapping evidence, non-empty OCR coverage, missing IDs, and per-query
ranks. Relevance labels are never used to build the OCR index.

#### DocVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/docvqa_bgesmallenv15.json
```

#### InfoVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/infovqa_bgesmallenv15.json
```

#### ArxivQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/arxivqa_bgesmallenv15.json
```

#### TatDQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_text_baselines.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --dense-model BAAI/bge-small-en-v1.5 \
  --max-queries 0 --max-documents 0 \
  --output artifacts/results/text/tatdqa_bgesmallenv15.json
```

### Token-compression ablations

Run the matching 500M visual experiment first so its embedding cache exists.
Each command tests all retention ratios and compression methods.

#### DocVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreDocVQARetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

#### InfoVQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreInfoVQARetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

#### ArxivQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreArxivQARetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

#### TatDQA

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_compression_ablation.py \
  --dataset mteb/VidoreTatdqaRetrieval \
  --model vidore/colSmol-500M \
  --ratios 1 0.75 0.5 0.25 0.125 \
  --methods random uniform mean_pool kmeans \
  --random-seeds 40 41 42
```

### Build final tables, plots, and failure analysis

Run this after the required experiment JSON files exist:

```bash
python scripts/build_report_artifacts.py
```

It validates the full-corpus results and writes:

- `artifacts/summary/main_results.csv`
- `artifacts/summary/compression_results.csv`
- `artifacts/summary/compression_aggregate.csv`
- `artifacts/summary/failure_analysis.csv`
- `artifacts/summary/RESULTS.md`
- `artifacts/summary/figures/*.png`

Only result files with the expected schema and
`"benchmark_comparable": true` are included. Older compatibility files at the
root of `artifacts/results/` are preserved but excluded from the summary.

## Result and cache behaviour

- An individual evaluation command without `--skip-existing` writes a fresh
  result to its `--output` path.
- The full pipeline uses `--skip-existing` for visual and text runs and keeps
  only results that pass its validity checks.
- Visual embeddings are cached separately from metric JSON files.
- Compression manifests record completed configurations, allowing interrupted
  ablations to resume.
- The summary builder reads the experiment JSON files; it does not rerun a
  model.

## Project report

The final EE798R report is available at:

```text
output/pdf/report.pdf
```

Its editable LaTeX source is:

```text
report/report.tex
```

The report explains the motivation, retrieval mathematics, algorithms,
implementation, datasets, quantitative results, qualitative examples,
failures, efficiency, limitations, and the path from this retriever to a
complete multimodal RAG system.

## Reproducibility notes

- Random operations use recorded seeds.
- Dataset fingerprints and model revisions are stored in result JSON files.
- Every full result records query count, corpus size, metrics, timing, and
  memory measurements.
- Efficiency values are hardware- and software-dependent; retrieval quality is
  the more portable comparison.
- ArxivQA text results must be interpreted with the recorded 84.4% non-empty
  OCR coverage.
- Released checkpoints are evaluated without task-specific fine-tuning.

## License

Project code is released under the repository license. Models and datasets
retain their original licenses and citation requirements.
