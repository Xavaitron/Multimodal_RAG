#!/usr/bin/env python
"""Evaluate ColSmol on a downloadable standard ViDoRe/MTEB retrieval dataset."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from datasets import Dataset, load_dataset

from lightweight_multimodal_retrieval.colsmol import ColSmolEncoder
from lightweight_multimodal_retrieval.metrics import evaluate_run
from lightweight_multimodal_retrieval.scoring import global_score, score_documents
from lightweight_multimodal_retrieval.utils import git_commit, seed_everything, write_json


def field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"Expected one of these fields: {names}")


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_config(dataset_name: str, config: str) -> Dataset:
    dataset = load_dataset(dataset_name, config, split="test")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset for {dataset_name}/{config}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mteb/VidoreDocVQARetrieval")
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--max-queries", type=int, default=20, help="0 means all queries")
    parser.add_argument("--max-documents", type=int, default=100, help="0 means full corpus")
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--output", default="artifacts/results/vidore.json")
    args = parser.parse_args()

    if args.max_queries < 0 or args.max_documents < 0 or args.query_batch_size < 1:
        raise SystemExit("Limits must be non-negative and query batch size must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("This evaluation requires a CUDA GPU")

    seed_everything(42)
    queries_data = load_config(args.dataset, "queries")
    qrels_data = load_config(args.dataset, "qrels")
    corpus_data = load_config(args.dataset, "corpus")

    query_count = (
        len(queries_data)
        if args.max_queries == 0
        else min(args.max_queries, len(queries_data))
    )
    query_rows = [queries_data[index] for index in range(query_count)]
    query_ids = [str(field(row, "id", "query-id", "query_id")) for row in query_rows]
    query_texts = [str(field(row, "text", "query")) for row in query_rows]
    selected_query_ids = set(query_ids)

    qrels: dict[str, dict[str, float]] = {query_id: {} for query_id in query_ids}
    for row in qrels_data:
        query_id = str(field(row, "query-id", "query_id"))
        if query_id in selected_query_ids:
            corpus_id = str(field(row, "corpus-id", "corpus_id"))
            qrels[query_id][corpus_id] = float(field(row, "score"))

    missing_qrels = [query_id for query_id, judgments in qrels.items() if not judgments]
    if missing_qrels:
        raise RuntimeError(f"Selected queries have no qrels: {missing_qrels[:5]}")

    positive_ids = {
        corpus_id
        for judgments in qrels.values()
        for corpus_id, gain in judgments.items()
        if gain > 0
    }
    all_corpus_ids = [str(value) for value in corpus_data["id"]]
    document_limit = len(all_corpus_ids) if args.max_documents == 0 else args.max_documents
    if document_limit < len(positive_ids):
        raise SystemExit(
            f"--max-documents must be at least {len(positive_ids)} to include every positive page"
        )
    candidate_ids = [corpus_id for corpus_id in all_corpus_ids if corpus_id in positive_ids]
    candidate_ids.extend(
        corpus_id
        for corpus_id in all_corpus_ids
        if corpus_id not in positive_ids and len(candidate_ids) < document_limit
    )
    candidate_ids = candidate_ids[:document_limit]
    corpus_index = {corpus_id: index for index, corpus_id in enumerate(all_corpus_ids)}
    images = [
        corpus_data[corpus_index[corpus_id]]["image"].convert("RGB")
        for corpus_id in candidate_ids
    ]

    encoder = ColSmolEncoder(args.model)
    torch.cuda.reset_peak_memory_stats()

    synchronize()
    start = time.perf_counter()
    document_embeddings = [encoder.encode_images([image])[0] for image in images]
    synchronize()
    document_seconds = time.perf_counter() - start

    synchronize()
    start = time.perf_counter()
    query_embeddings = []
    for query_batch in batched(query_texts, args.query_batch_size):
        query_embeddings.extend(encoder.encode_queries(query_batch))
    synchronize()
    query_seconds = time.perf_counter() - start

    maxsim_rankings: dict[str, list[str]] = {}
    global_rankings: dict[str, list[str]] = {}
    synchronize()
    start = time.perf_counter()
    for query_id, query_embedding in zip(query_ids, query_embeddings, strict=True):
        scores = score_documents(query_embedding, document_embeddings)
        order = torch.argsort(scores, descending=True).detach().cpu().tolist()
        maxsim_rankings[query_id] = [candidate_ids[index] for index in order]
        global_scores = torch.stack(
            [global_score(query_embedding, document) for document in document_embeddings]
        )
        global_order = torch.argsort(global_scores, descending=True).detach().cpu().tolist()
        global_rankings[query_id] = [candidate_ids[index] for index in global_order]
    synchronize()
    retrieval_seconds = time.perf_counter() - start

    is_full = query_count == len(queries_data) and len(candidate_ids) == len(corpus_data)
    result = {
        "experiment": "vidore_colsmol_evaluation",
        "dataset": args.dataset,
        "dataset_fingerprints": {
            "queries": queries_data._fingerprint,
            "qrels": qrels_data._fingerprint,
            "corpus": corpus_data._fingerprint,
        },
        "model": args.model,
        "model_revision": encoder.model_revision,
        "git_commit": git_commit(),
        "seed": 42,
        "benchmark_comparable": is_full,
        "queries": query_count,
        "documents": len(candidate_ids),
        "metrics": {
            "maxsim": evaluate_run(maxsim_rankings, qrels),
            "global_visual": evaluate_run(global_rankings, qrels),
        },
        "efficiency": {
            "page_encoding_ms_per_page": 1000 * document_seconds / len(candidate_ids),
            "query_encoding_ms_per_query": 1000 * query_seconds / query_count,
            "retrieval_ms_per_query": 1000 * retrieval_seconds / query_count,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
