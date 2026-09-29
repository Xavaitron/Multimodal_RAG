#!/usr/bin/env python
"""Evaluate one ColSmol model on one ViDoRe dataset with reusable caches."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lightweight_multimodal_retrieval.cache import (
    embedding_bytes,
    load_embeddings,
    save_embeddings,
    validate_cache_metadata,
)
from lightweight_multimodal_retrieval.colsmol import ColSmolEncoder
from lightweight_multimodal_retrieval.metrics import evaluate_run
from lightweight_multimodal_retrieval.scoring import global_embedding, score_documents
from lightweight_multimodal_retrieval.utils import git_commit, seed_everything, write_json
from lightweight_multimodal_retrieval.vidore_data import (
    ids_digest,
    load_vidore_bundle,
    model_slug,
)


def batched(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rank_from_score_matrix(scores: Tensor, corpus_ids: list[str]) -> dict[int, list[str]]:
    orders = torch.argsort(scores, dim=1, descending=True).detach().cpu().tolist()
    return {index: [corpus_ids[item] for item in order] for index, order in enumerate(orders)}


def relevant_rank(ranking: list[str], judgments: dict[str, float]) -> int | None:
    relevant = {corpus_id for corpus_id, gain in judgments.items() if gain > 0}
    return next((rank for rank, corpus_id in enumerate(ranking, 1) if corpus_id in relevant), None)


def cache_paths(cache_root: str, dataset_slug: str, model_name: str) -> tuple[Path, Path]:
    directory = Path(cache_root) / dataset_slug / model_slug(model_name)
    return directory / "documents.pt", directory / "queries.pt"


def load_cached_pair(
    document_path: Path,
    query_path: Path,
    document_expected: dict[str, Any],
    query_expected: dict[str, Any],
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]] | None:
    if not document_path.exists() or not query_path.exists():
        return None
    try:
        documents, document_metadata = load_embeddings(document_path)
        queries, query_metadata = load_embeddings(query_path)
        validate_cache_metadata(document_metadata, document_expected)
        validate_cache_metadata(query_metadata, query_expected)
    except (EOFError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Ignoring invalid embedding cache: {error}", flush=True)
        return None
    shared_keys = {
        "embedding_dimension",
        "embedding_dtype",
        "input_image_sizes",
        "model_revision",
        "model_parameters",
        "page_encoding_ms_per_page",
        "query_encoding_ms_per_query",
        "encoding_peak_vram_mb",
    }
    shared = {key: document_metadata.get(key) for key in shared_keys}
    shared["query_encoding_ms_per_query"] = query_metadata.get("query_encoding_ms_per_query")
    return documents, queries, shared


def encode_and_cache(
    model_name: str,
    bundle: Any,
    query_batch_size: int,
    document_path: Path,
    query_path: Path,
    document_expected: dict[str, Any],
    query_expected: dict[str, Any],
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    encoder = ColSmolEncoder(model_name)

    images = bundle.images()
    synchronize()
    started = time.perf_counter()
    document_values: list[Tensor] = []
    for index, image in enumerate(images, 1):
        document_values.append(encoder.encode_images([image])[0])
        if index % 50 == 0 or index == len(images):
            print(f"Encoded pages: {index}/{len(images)}", flush=True)
    synchronize()
    document_seconds = time.perf_counter() - started

    synchronize()
    started = time.perf_counter()
    query_values: list[Tensor] = []
    for query_batch in batched(bundle.query_texts, query_batch_size):
        query_values.extend(encoder.encode_queries(query_batch))
    synchronize()
    query_seconds = time.perf_counter() - started

    shared = {
        "embedding_dimension": document_values[0].shape[-1],
        "embedding_dtype": str(document_values[0].dtype).removeprefix("torch."),
        "input_image_sizes": sorted({f"{image.width}x{image.height}" for image in images}),
        "model_revision": encoder.model_revision,
        "model_parameters": encoder.parameter_count,
        "page_encoding_ms_per_page": 1000 * document_seconds / len(document_values),
        "query_encoding_ms_per_query": 1000 * query_seconds / len(query_values),
        "encoding_peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    documents = dict(zip(bundle.corpus_ids, document_values, strict=True))
    queries = dict(zip(bundle.query_ids, query_values, strict=True))
    save_embeddings(document_path, documents, document_expected | shared)
    save_embeddings(query_path, queries, query_expected | shared)
    return documents, queries, shared


def default_output(bundle: Any, model_name: str) -> str:
    return f"artifacts/results/visual/{bundle.slug}_{model_slug(model_name)}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mteb/VidoreDocVQARetrieval")
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all")
    parser.add_argument("--max-documents", type=int, default=0, help="0 means all")
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--score-document-batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--top-k-details", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.query_batch_size < 1 or args.score_document_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    if args.top_k_details < 1:
        raise SystemExit("--top-k-details must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("This evaluation requires a CUDA GPU")

    seed_everything(42)
    bundle = load_vidore_bundle(
        args.dataset,
        max_queries=args.max_queries,
        max_documents=args.max_documents,
    )
    output_path = args.output or default_output(bundle, args.model)
    if args.skip_existing and Path(output_path).exists():
        try:
            existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
            if (
                existing.get("schema_version") == 2
                and existing.get("benchmark_comparable")
                and existing.get("dataset") == args.dataset
                and existing.get("model") == args.model
            ):
                print(f"Keeping completed result {output_path}")
                return
        except (OSError, json.JSONDecodeError):
            pass
    document_path, query_path = cache_paths(args.cache_dir, bundle.slug, args.model)
    common = {
        "cache_schema_version": 2,
        "dataset": args.dataset,
        "model": args.model,
        "seed": 42,
        "dataset_fingerprints": bundle.fingerprints,
    }
    document_expected = common | {
        "kind": "documents",
        "ids_digest": ids_digest(bundle.corpus_ids),
        "count": len(bundle.corpus_ids),
    }
    query_expected = common | {
        "kind": "queries",
        "ids_digest": ids_digest(bundle.query_ids),
        "count": len(bundle.query_ids),
    }

    cached = None
    if not args.force_reencode:
        cached = load_cached_pair(
            document_path,
            query_path,
            document_expected,
            query_expected,
        )
    cache_hit = cached is not None
    if cached is None:
        documents_by_id, queries_by_id, encoding = encode_and_cache(
            args.model,
            bundle,
            args.query_batch_size,
            document_path,
            query_path,
            document_expected,
            query_expected,
        )
    else:
        documents_by_id, queries_by_id, encoding = cached
        print(f"Loaded embeddings from {document_path.parent}", flush=True)

    documents = [documents_by_id[corpus_id].to("cuda") for corpus_id in bundle.corpus_ids]
    queries = [queries_by_id[query_id].to("cuda") for query_id in bundle.query_ids]
    token_counts = [tensor.shape[0] for tensor in documents]

    torch.cuda.reset_peak_memory_stats()
    query_global = torch.stack([global_embedding(tensor) for tensor in queries])
    document_global = torch.stack([global_embedding(tensor) for tensor in documents])
    _ = query_global[:1] @ document_global.transpose(0, 1)
    synchronize()
    started = time.perf_counter()
    global_scores = query_global @ document_global.transpose(0, 1)
    global_orders = rank_from_score_matrix(global_scores, bundle.corpus_ids)
    synchronize()
    global_seconds = time.perf_counter() - started

    _ = score_documents(
        queries[0],
        documents,
        document_batch_size=args.score_document_batch_size,
    )
    synchronize()
    started = time.perf_counter()
    maxsim_rankings: dict[str, list[str]] = {}
    for index, (query_id, query) in enumerate(zip(bundle.query_ids, queries, strict=True), 1):
        scores = score_documents(
            query,
            documents,
            document_batch_size=args.score_document_batch_size,
        )
        order = torch.argsort(scores, descending=True).detach().cpu().tolist()
        maxsim_rankings[query_id] = [bundle.corpus_ids[item] for item in order]
        if index % 100 == 0 or index == len(queries):
            print(f"Scored queries: {index}/{len(queries)}", flush=True)
    synchronize()
    maxsim_seconds = time.perf_counter() - started
    global_rankings = {
        query_id: global_orders[index] for index, query_id in enumerate(bundle.query_ids)
    }

    details = []
    for query_id, query_text in zip(bundle.query_ids, bundle.query_texts, strict=True):
        maxsim_ranking = maxsim_rankings[query_id]
        global_ranking = global_rankings[query_id]
        details.append(
            {
                "query_id": query_id,
                "query": query_text,
                "relevant": bundle.qrels[query_id],
                "maxsim_relevant_rank": relevant_rank(maxsim_ranking, bundle.qrels[query_id]),
                "global_relevant_rank": relevant_rank(global_ranking, bundle.qrels[query_id]),
                "maxsim_top": maxsim_ranking[: args.top_k_details],
                "global_top": global_ranking[: args.top_k_details],
            }
        )

    result = {
        "schema_version": 2,
        "experiment": "vidore_colsmol_evaluation",
        "dataset": args.dataset,
        "dataset_fingerprints": bundle.fingerprints,
        "model": args.model,
        "model_revision": encoding["model_revision"],
        "precision": encoding["embedding_dtype"],
        "embedding_dimension": encoding["embedding_dimension"],
        "input_image_sizes": encoding["input_image_sizes"],
        "git_commit": git_commit(),
        "seed": 42,
        "benchmark_comparable": bundle.is_full,
        "queries": len(bundle.query_ids),
        "documents": len(bundle.corpus_ids),
        "metrics": {
            "maxsim": evaluate_run(maxsim_rankings, bundle.qrels),
            "global_visual": evaluate_run(global_rankings, bundle.qrels),
        },
        "efficiency": {
            "cache_hit": cache_hit,
            "page_encoding_ms_per_page": encoding["page_encoding_ms_per_page"],
            "query_encoding_ms_per_query": encoding["query_encoding_ms_per_query"],
            "maxsim_retrieval_ms_per_query": 1000 * maxsim_seconds / len(queries),
            "global_retrieval_ms_per_query": 1000 * global_seconds / len(queries),
            "maxsim_index_size_mb": embedding_bytes(documents) / 1024**2,
            "global_index_size_mb": document_global.numel()
            * document_global.element_size()
            / 1024**2,
            "encoding_peak_vram_mb": encoding["encoding_peak_vram_mb"],
            "scoring_peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "tokens_per_page": {
                "min": min(token_counts),
                "mean": statistics.mean(token_counts),
                "max": max(token_counts),
            },
        },
        "parameters": {
            "total": encoding["model_parameters"],
            "trainable": 0,
        },
        "per_query": details,
    }
    write_json(output_path, result)
    print(f"Wrote {output_path}")
    print(json.dumps({"metrics": result["metrics"], "efficiency": result["efficiency"]}, indent=2))


if __name__ == "__main__":
    main()
