#!/usr/bin/env python
"""Validate experiment JSON and build tables, plots, and a failure-analysis sheet."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from lightweight_multimodal_retrieval.vidore_data import DATASETS


MODELS = ("vidore/colSmol-500M", "vidore/colSmol-256M")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_columns(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "ndcg@5": metrics["ndcg@5"],
        "mrr": metrics["mrr"],
        "recall@1": metrics["recall@1"],
        "recall@5": metrics["recall@5"],
        "recall@10": metrics["recall@10"],
    }


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    rule = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, rule]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect_main(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    visual_payloads: list[dict[str, Any]] = []
    for path in sorted((results_root / "visual").glob("*.json")):
        payload = read_json(path)
        if (
            payload.get("schema_version") != 2
            or payload.get("experiment") != "vidore_colsmol_evaluation"
        ):
            continue
        visual_payloads.append(payload)
        for method, metrics in payload["metrics"].items():
            efficiency = payload["efficiency"]
            latency_key = (
                "maxsim_retrieval_ms_per_query"
                if method == "maxsim"
                else "global_retrieval_ms_per_query"
            )
            index_key = (
                "maxsim_index_size_mb" if method == "maxsim" else "global_index_size_mb"
            )
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "method": method,
                    "model": payload["model"],
                    **metric_columns(metrics),
                    "retrieval_ms_per_query": efficiency[latency_key],
                    "index_size_mb": efficiency[index_key],
                    "benchmark_comparable": payload["benchmark_comparable"],
                    "source": str(path),
                }
            )
    for path in sorted((results_root / "text").glob("*.json")):
        payload = read_json(path)
        if (
            payload.get("schema_version") != 2
            or payload.get("experiment") != "vidore_text_baselines"
        ):
            continue
        for method, values in payload["methods"].items():
            efficiency = values["efficiency"]
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "method": method,
                    "model": values.get("model", "Okapi BM25"),
                    **metric_columns(values["metrics"]),
                    "retrieval_ms_per_query": efficiency["retrieval_ms_per_query"],
                    "index_size_mb": efficiency.get("index_size_mb", ""),
                    "benchmark_comparable": payload["benchmark_comparable"],
                    "source": str(path),
                }
            )
    return rows, visual_payloads


def collect_compression(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((results_root / "compression").glob("*.json")):
        payload = read_json(path)
        if (
            payload.get("schema_version") != 2
            or payload.get("experiment") != "vidore_token_compression"
        ):
            continue
        rows.append(
            {
                "dataset": payload["dataset"],
                "model": payload["model"],
                "method": payload["compression"]["method"],
                "retention_ratio": payload["compression"]["retention_ratio"],
                "seed": payload["seed"],
                **metric_columns(payload["metrics"]),
                "retrieval_ms_per_query": payload["efficiency"]["retrieval_ms_per_query"],
                "index_size_mb": payload["efficiency"]["index_size_mb"],
                "tokens_per_page": payload["efficiency"]["tokens_per_page"]["mean"],
                "benchmark_comparable": payload["benchmark_comparable"],
                "source": str(path),
            }
        )
    return rows


def aggregate_compression(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["model"], row["method"], row["retention_ratio"])
        grouped[key].append(row)
    aggregated = []
    for (dataset, model, method, ratio), values in sorted(grouped.items()):
        quality = [row["ndcg@5"] for row in values]
        aggregated.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "retention_ratio": ratio,
                "runs": len(values),
                "ndcg@5_mean": statistics.mean(quality),
                "ndcg@5_std": statistics.stdev(quality) if len(quality) > 1 else 0.0,
                "mrr_mean": statistics.mean(row["mrr"] for row in values),
                "retrieval_ms_per_query_mean": statistics.mean(
                    row["retrieval_ms_per_query"] for row in values
                ),
                "index_size_mb_mean": statistics.mean(row["index_size_mb"] for row in values),
                "tokens_per_page_mean": statistics.mean(
                    row["tokens_per_page"] for row in values
                ),
            }
        )
    return aggregated


def verify_complete(
    main_rows: list[dict[str, Any]], compression_rows: list[dict[str, Any]]
) -> None:
    missing = []
    for dataset in DATASETS.values():
        methods = {(row["method"], row["model"]) for row in main_rows if row["dataset"] == dataset}
        expected = {("bm25", "Okapi BM25"), ("dense_text", "BAAI/bge-small-en-v1.5")}
        expected.update(
            (method, model)
            for model in MODELS
            for method in ("maxsim", "global_visual")
        )
        for item in sorted(expected - methods):
            missing.append(f"{dataset}: {item[0]} / {item[1]}")
        actual_compression = {
            (row["method"], row["retention_ratio"], row["seed"])
            for row in compression_rows
            if row["dataset"] == dataset and row["model"] == "vidore/colSmol-500M"
        }
        expected_compression = {("none", 1.0, 42)}
        for ratio in (0.75, 0.5, 0.25, 0.125):
            expected_compression.update(("random", ratio, seed) for seed in (40, 41, 42))
            expected_compression.update(
                (method, ratio, 42) for method in ("uniform", "mean_pool", "kmeans")
            )
        for method, ratio, seed in sorted(expected_compression - actual_compression):
            missing.append(f"{dataset}: compression {method} ratio={ratio} seed={seed}")
    noncomparable = [
        row["source"] for row in main_rows + compression_rows if not row["benchmark_comparable"]
    ]
    if missing or noncomparable:
        detail = "\n".join(
            [
                *(f"missing {item}" for item in missing),
                *(f"partial {item}" for item in noncomparable),
            ]
        )
        raise SystemExit(f"Results are incomplete or non-comparable:\n{detail}")


def build_failures(visual_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates_with_dataset = []
    for payload in visual_payloads:
        if payload["model"] != "vidore/colSmol-500M":
            continue
        candidates_with_dataset.extend(
            (payload["dataset"], item) for item in payload.get("per_query", [])
        )
    candidates_with_dataset.sort(
        key=lambda pair: pair[1]["maxsim_relevant_rank"] or 10**9,
        reverse=True,
    )
    failures = []
    for dataset, item in candidates_with_dataset[:30]:
        failures.append(
            {
                "dataset": dataset,
                "query_id": item["query_id"],
                "query": item["query"],
                "maxsim_relevant_rank": item["maxsim_relevant_rank"],
                "global_relevant_rank": item["global_relevant_rank"],
                "relevant_ids": " ".join(item["relevant"]),
                "retrieved_top_ids": " ".join(item["maxsim_top"][:5]),
                "category": "requires_manual_review",
                "notes": "",
            }
        )
    return failures


def make_plots(
    output_dir: Path,
    main_rows: list[dict[str, Any]],
    compression_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    dataset_labels = {value: key for key, value in DATASETS.items()}

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    methods = [
        ("bm25", "Okapi BM25", "BM25"),
        ("dense_text", "BAAI/bge-small-en-v1.5", "BGE-small"),
        ("maxsim", "vidore/colSmol-256M", "ColSmol-256M"),
        ("maxsim", "vidore/colSmol-500M", "ColSmol-500M"),
    ]
    datasets = list(DATASETS.values())
    for method_index, (method, model, label) in enumerate(methods):
        values = []
        for dataset in datasets:
            match = next(
                (
                    row
                    for row in main_rows
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["model"] == model
                ),
                None,
            )
            values.append(match["ndcg@5"] if match else 0)
        positions = [index + (method_index - 1.5) * width for index in range(len(datasets))]
        ax.bar(positions, values, width=width, label=label)
    ax.set_xticks(range(len(datasets)), [dataset_labels[item] for item in datasets])
    ax.set_ylabel("nDCG@5")
    ax.set_title("Full-corpus retrieval quality")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "main_ndcg5.png", dpi=180)
    plt.close(fig)

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in compression_rows:
        grouped[(row["dataset"], row["method"], row["retention_ratio"])].append(row)
    for dataset in datasets:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for method in ("none", "random", "uniform", "mean_pool", "kmeans"):
            points = []
            for (row_dataset, row_method, ratio), rows in grouped.items():
                if row_dataset == dataset and row_method == method:
                    points.append(
                        (
                            ratio,
                            statistics.mean(row["ndcg@5"] for row in rows),
                            statistics.mean(row["index_size_mb"] for row in rows),
                            statistics.mean(row["retrieval_ms_per_query"] for row in rows),
                        )
                    )
            if not points:
                continue
            points.sort()
            label = "uncompressed" if method == "none" else method
            axes[0].plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
            axes[1].plot([p[0] for p in points], [p[2] for p in points], marker="o", label=label)
            axes[2].plot([p[0] for p in points], [p[3] for p in points], marker="o", label=label)
        labels = ("nDCG@5", "Index size (MB)", "Retrieval ms/query")
        for axis, ylabel in zip(axes, labels, strict=True):
            axis.set_xlabel("Token retention ratio")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Compression trade-offs: {dataset_labels[dataset]}")
        fig.tight_layout()
        fig.savefig(figures / f"compression_{dataset_labels[dataset]}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--output-dir", default="artifacts/summary")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    results_root = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    main_rows, visual_payloads = collect_main(results_root)
    compression_rows = collect_compression(results_root)
    if not main_rows:
        raise SystemExit(f"No schema-v2 results found beneath {results_root}")
    if not args.allow_partial:
        verify_complete(main_rows, compression_rows)

    main_fields = [
        "dataset", "method", "model", "ndcg@5", "mrr", "recall@1", "recall@5",
        "recall@10", "retrieval_ms_per_query", "index_size_mb", "benchmark_comparable", "source",
    ]
    compression_fields = [
        "dataset", "model", "method", "retention_ratio", "seed", "ndcg@5", "mrr",
        "recall@1", "recall@5", "recall@10", "retrieval_ms_per_query", "index_size_mb",
        "tokens_per_page", "benchmark_comparable", "source",
    ]
    write_csv(output_dir / "main_results.csv", main_rows, main_fields)
    write_csv(output_dir / "compression_results.csv", compression_rows, compression_fields)
    compression_aggregate = aggregate_compression(compression_rows)
    aggregate_fields = [
        "dataset", "model", "method", "retention_ratio", "runs", "ndcg@5_mean",
        "ndcg@5_std", "mrr_mean", "retrieval_ms_per_query_mean", "index_size_mb_mean",
        "tokens_per_page_mean",
    ]
    write_csv(
        output_dir / "compression_aggregate.csv",
        compression_aggregate,
        aggregate_fields,
    )
    failures = build_failures(visual_payloads)
    failure_fields = [
        "dataset", "query_id", "query", "maxsim_relevant_rank", "global_relevant_rank",
        "relevant_ids", "retrieved_top_ids", "category", "notes",
    ]
    write_csv(output_dir / "failure_analysis.csv", failures, failure_fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_fields = ["dataset", "method", "model", "ndcg@5", "mrr", "recall@5"]
    report = (
        "# Experiment results\n\n"
        "All rows below use the complete query and document sets for their dataset.\n\n"
        + markdown_table(main_rows, report_fields)
        + "\n"
    )
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    make_plots(output_dir, main_rows, compression_rows)
    print(f"Wrote validated tables, figures, and failure sheet to {output_dir}")


if __name__ == "__main__":
    main()
