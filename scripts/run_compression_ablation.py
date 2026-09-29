#!/usr/bin/env python
"""Run the full token-retention ablation from cached ColSmol embeddings."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from lightweight_multimodal_retrieval.cache import embedding_bytes, load_embeddings
from lightweight_multimodal_retrieval.compression import (
    kmeans_tokens,
    local_mean_pool,
    random_tokens,
    uniform_tokens,
)
from lightweight_multimodal_retrieval.metrics import evaluate_run
from lightweight_multimodal_retrieval.scoring import score_documents
from lightweight_multimodal_retrieval.utils import git_commit, seed_everything, write_json
from lightweight_multimodal_retrieval.vidore_data import (
    ids_digest,
    load_vidore_bundle,
    model_slug,
)


METHODS = ("random", "uniform", "mean_pool", "kmeans")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ratio_slug(ratio: float) -> str:
    return f"r{int(round(ratio * 1000)):04d}"


def compress_document(
    tokens: Tensor,
    method: str,
    ratio: float,
    *,
    seed: int,
    kmeans_iterations: int,
) -> Tensor:
    if ratio == 1.0:
        return tokens
    functions: dict[str, Callable[[], Tensor]] = {
        "random": lambda: random_tokens(tokens, ratio, seed=seed),
        "uniform": lambda: uniform_tokens(tokens, ratio),
        "mean_pool": lambda: local_mean_pool(tokens, ratio),
        "kmeans": lambda: kmeans_tokens(
            tokens,
            ratio,
            seed=seed,
            iterations=kmeans_iterations,
        ),
    }
    compressed = functions[method]().detach()
    if method in {"mean_pool", "kmeans"}:
        compressed = F.normalize(compressed.float(), p=2, dim=-1).to(tokens.dtype)
    return compressed


def evaluate(
    queries: list[Tensor],
    documents: list[Tensor],
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: dict[str, dict[str, float]],
    *,
    document_batch_size: int,
) -> tuple[dict[str, float], float]:
    _ = score_documents(
        queries[0],
        documents,
        document_batch_size=document_batch_size,
    )
    synchronize()
    started = time.perf_counter()
    rankings: dict[str, list[str]] = {}
    for index, (query_id, query) in enumerate(zip(query_ids, queries, strict=True), 1):
        scores = score_documents(
            query,
            documents,
            document_batch_size=document_batch_size,
        )
        order = torch.argsort(scores, descending=True).cpu().tolist()
        rankings[query_id] = [corpus_ids[position] for position in order]
        if index % 100 == 0 or index == len(queries):
            print(f"Scored queries: {index}/{len(queries)}", flush=True)
    synchronize()
    elapsed = time.perf_counter() - started
    return evaluate_run(rankings, qrels), 1000 * elapsed / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mteb/VidoreDocVQARetrieval")
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--output-dir", default="artifacts/results/compression")
    parser.add_argument("--ratios", nargs="+", type=float, default=[1, 0.75, 0.5, 0.25, 0.125])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[40, 41, 42])
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--score-document-batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all")
    parser.add_argument("--max-documents", type=int, default=0, help="0 means all")
    parser.add_argument("--force", action="store_true", help="replace completed result files")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Compression evaluation requires a CUDA GPU")
    if any(not 0 < ratio <= 1 for ratio in args.ratios):
        raise SystemExit("all ratios must be in (0, 1]")
    if args.kmeans_iterations < 1 or args.score_document_batch_size < 1:
        raise SystemExit("iteration and batch counts must be positive")

    seed_everything(42)
    bundle = load_vidore_bundle(
        args.dataset,
        max_queries=args.max_queries,
        max_documents=args.max_documents,
    )
    cache_root = Path(args.cache_dir) / bundle.slug / model_slug(args.model)
    document_path = cache_root / "documents.pt"
    query_path = cache_root / "queries.pt"
    if not document_path.exists() or not query_path.exists():
        raise SystemExit(
            "Missing visual embedding cache. Run scripts/evaluate_vidore.py for this "
            "dataset and model first."
        )
    documents_by_id, document_metadata = load_embeddings(document_path)
    queries_by_id, query_metadata = load_embeddings(query_path)
    expected = {
        "cache_schema_version": 2,
        "dataset": args.dataset,
        "model": args.model,
        "document_ids_digest": ids_digest(bundle.corpus_ids),
        "query_ids_digest": ids_digest(bundle.query_ids),
    }
    actual = {
        "cache_schema_version": document_metadata.get("cache_schema_version"),
        "dataset": document_metadata.get("dataset"),
        "model": document_metadata.get("model"),
        "document_ids_digest": document_metadata.get("ids_digest"),
        "query_ids_digest": query_metadata.get("ids_digest"),
    }
    if actual != expected:
        raise SystemExit(
            f"Embedding cache does not match this run: expected {expected}, got {actual}"
        )

    source_documents = [documents_by_id[item].to("cuda") for item in bundle.corpus_ids]
    queries = [queries_by_id[item].to("cuda") for item in bundle.query_ids]
    output_dir = Path(args.output_dir)
    completed: list[str] = []

    configurations: list[tuple[str, float, int]] = [("none", 1.0, 42)]
    for ratio in sorted(set(args.ratios), reverse=True):
        if ratio == 1.0:
            continue
        for method in args.methods:
            seeds = args.random_seeds if method == "random" else [42]
            configurations.extend((method, ratio, seed) for seed in seeds)

    for method, ratio, seed in configurations:
        seed_suffix = f"_seed{seed}" if method == "random" else ""
        filename = (
            f"{bundle.slug}_{model_slug(args.model)}_{method}_{ratio_slug(ratio)}"
            f"{seed_suffix}.json"
        )
        output = output_dir / filename
        if output.exists() and not args.force:
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                compression = existing.get("compression", {})
                if (
                    existing.get("schema_version") == 2
                    and existing.get("benchmark_comparable")
                    and existing.get("dataset") == args.dataset
                    and existing.get("model") == args.model
                    and existing.get("seed") == seed
                    and compression.get("method") == method
                    and compression.get("retention_ratio") == ratio
                ):
                    print(f"Keeping completed result {output}", flush=True)
                    completed.append(str(output))
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        print(f"Compressing method={method} ratio={ratio} seed={seed}", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()
        compressed = []
        for index, document in enumerate(source_documents, 1):
            compressed.append(
                compress_document(
                    document,
                    method,
                    ratio,
                    seed=seed + index,
                    kmeans_iterations=args.kmeans_iterations,
                )
            )
            if index % 100 == 0 or index == len(source_documents):
                print(f"Compressed pages: {index}/{len(source_documents)}", flush=True)
        synchronize()
        compression_seconds = time.perf_counter() - started
        metrics, retrieval_ms = evaluate(
            queries,
            compressed,
            bundle.query_ids,
            bundle.corpus_ids,
            bundle.qrels,
            document_batch_size=args.score_document_batch_size,
        )
        token_counts = [item.shape[0] for item in compressed]
        result = {
            "schema_version": 2,
            "experiment": "vidore_token_compression",
            "dataset": args.dataset,
            "dataset_fingerprints": bundle.fingerprints,
            "model": args.model,
            "model_revision": document_metadata.get("model_revision"),
            "git_commit": git_commit(),
            "seed": seed,
            "benchmark_comparable": bundle.is_full,
            "queries": len(bundle.query_ids),
            "documents": len(bundle.corpus_ids),
            "compression": {
                "method": method,
                "retention_ratio": ratio,
                "sequence_aware": method in {"uniform", "mean_pool"},
                "spatial_layout_aware": False,
                "centroids_l2_normalized": method in {"mean_pool", "kmeans"},
                "kmeans_iterations": args.kmeans_iterations if method == "kmeans" else None,
            },
            "metrics": metrics,
            "efficiency": {
                "compression_ms_per_page": 1000
                * compression_seconds
                / len(source_documents),
                "retrieval_ms_per_query": retrieval_ms,
                "index_size_mb": embedding_bytes(compressed) / 1024**2,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "tokens_per_page": {
                    "min": min(token_counts),
                    "mean": statistics.mean(token_counts),
                    "max": max(token_counts),
                },
            },
        }
        write_json(output, result)
        completed.append(str(output))
        print(f"Wrote {output}", flush=True)
        del compressed
        torch.cuda.empty_cache()

    manifest = output_dir / f"{bundle.slug}_{model_slug(args.model)}_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": 2,
            "experiment": "vidore_token_compression_manifest",
            "dataset": args.dataset,
            "model": args.model,
            "git_commit": git_commit(),
            "results": completed,
        },
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
