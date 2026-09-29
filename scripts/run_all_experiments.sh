#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU_ID="${GPU_ID:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/full_run_${RUN_STAMP}.log") 2>&1
trap 'echo "FAILED at line $LINENO"' ERR

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Started full experiment run at $(date -u --iso-8601=seconds)"
echo "Physical GPU selected by CUDA_VISIBLE_DEVICES: $GPU_ID"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

nvidia-smi
python scripts/environment_diagnostic.py --skip-model
pytest -q
python scripts/environment_diagnostic.py --model vidore/colSmol-500M

datasets=(
  "docvqa:mteb/VidoreDocVQARetrieval"
  "infovqa:mteb/VidoreInfoVQARetrieval"
  "arxivqa:mteb/VidoreArxivQARetrieval"
  "tatdqa:mteb/VidoreTatdqaRetrieval"
)

for entry in "${datasets[@]}"; do
  slug="${entry%%:*}"
  dataset="${entry#*:}"
  python scripts/evaluate_vidore.py \
    --dataset "$dataset" \
    --model vidore/colSmol-500M \
    --max-queries 0 \
    --max-documents 0 \
    --skip-existing \
    --output "artifacts/results/visual/${slug}_colsmol500m.json"
done

for entry in "${datasets[@]}"; do
  slug="${entry%%:*}"
  dataset="${entry#*:}"
  python scripts/evaluate_vidore.py \
    --dataset "$dataset" \
    --model vidore/colSmol-256M \
    --max-queries 0 \
    --max-documents 0 \
    --skip-existing \
    --output "artifacts/results/visual/${slug}_colsmol256m.json"
done

for entry in "${datasets[@]}"; do
  slug="${entry%%:*}"
  dataset="${entry#*:}"
  python scripts/evaluate_text_baselines.py \
    --dataset "$dataset" \
    --dense-model BAAI/bge-small-en-v1.5 \
    --max-queries 0 \
    --max-documents 0 \
    --skip-existing \
    --output "artifacts/results/text/${slug}_bgesmallenv15.json"
done

for entry in "${datasets[@]}"; do
  dataset="${entry#*:}"
  python scripts/run_compression_ablation.py \
    --dataset "$dataset" \
    --model vidore/colSmol-500M \
    --ratios 1 0.75 0.5 0.25 0.125 \
    --methods random uniform mean_pool kmeans \
    --random-seeds 40 41 42
done

python scripts/build_report_artifacts.py

echo "Completed full experiment run at $(date -u --iso-8601=seconds)"
echo "Log: $LOG_DIR/full_run_${RUN_STAMP}.log"
echo "Validated summary: artifacts/summary/RESULTS.md"
