#!/usr/bin/env python
"""Evaluate BM25 and a dense text retriever on ViDoRe page-level OCR."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from lightweight_multimodal_retrieval.bm25 import BM25
from lightweight_multimodal_retrieval.cache import tensor_bytes
from lightweight_multimodal_retrieval.metrics import evaluate_run
from lightweight_multimodal_retrieval.utils import git_commit, seed_everything, write_json
from lightweight_multimodal_retrieval.vidore_data import (
    OCR_DATASETS,
    corpus_position,
    field,
    image_digest,
    load_vidore_bundle,
    model_slug,
    positional_alignment_is_safe,
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def relevant_rank(ranking: list[str], judgments: dict[str, float]) -> int | None:
    relevant = {corpus_id for corpus_id, gain in judgments.items() if gain > 0}
    return next((rank for rank, item in enumerate(ranking, 1) if item in relevant), None)


def load_ocr_documents(bundle: Any) -> tuple[dict[str, str], dict[str, Any]]:
    """Join ViDoRe OCR rows to the exact MTEB corpus pages by image content."""
    ocr_name = OCR_DATASETS[bundle.slug]
    rows = load_dataset(ocr_name, split="test")
    by_digest: dict[str, str] = {}
    by_filename: dict[str, str] = {}
    indexed_rows: list[tuple[str, str]] = []
    for index, row in enumerate(rows, 1):
        text = str(field(row, "text_description", "text", "ocr_text"))
        digest = image_digest(field(row, "image"))
        indexed_rows.append((digest, text))
        # TatDQA repeats a page for multiple questions. Keep the fullest OCR copy.
        # Preserve an explicit empty OCR row as a successful page mapping.  A
        # missing mapping and a mapped page with no recognized text have very
        # different meanings and are reported separately below.
        if digest not in by_digest or len(text) > len(by_digest[digest]):
            by_digest[digest] = text
        filename = row.get("image_filename")
        if filename:
            name = Path(str(filename)).name.casefold()
            for key in (name, Path(name).stem):
                if len(text) > len(by_filename.get(key, "")):
                    by_filename[key] = text
        if index % 500 == 0:
            print(f"Indexed OCR rows: {index}/{len(rows)}", flush=True)

    corpus_images = bundle.images()
    positional_comparisons = 0
    positional_image_matches = 0
    for corpus_id, image in zip(bundle.corpus_ids, corpus_images, strict=True):
        position = corpus_position(corpus_id)
        if position is None or position >= len(indexed_rows):
            continue
        positional_comparisons += 1
        if image_digest(image) == indexed_rows[position][0]:
            positional_image_matches += 1
    positional_fallback_safe = (
        len(rows) == bundle.full_document_count
        and positional_alignment_is_safe(
            positional_image_matches,
            positional_comparisons,
        )
    )
    print(
        f"OCR positional alignment: {positional_image_matches}/"
        f"{positional_comparisons}; fallback enabled={positional_fallback_safe}",
        flush=True,
    )

    documents: dict[str, str] = {}
    missing: list[str] = []
    filename_matches = 0
    digest_matches = 0
    position_matches = 0
    for corpus_id, image in zip(bundle.corpus_ids, corpus_images, strict=True):
        name = Path(corpus_id).name.casefold()
        text = by_filename.get(name) or by_filename.get(Path(name).stem)
        if text is not None:
            filename_matches += 1
        else:
            text = by_digest.get(image_digest(image))
            if text is not None:
                digest_matches += 1
        if text is None and positional_fallback_safe:
            position = corpus_position(corpus_id)
            if position is not None and position < len(indexed_rows):
                text = indexed_rows[position][1]
                position_matches += 1
        if text is None:
            missing.append(corpus_id)
            text = ""
        documents[corpus_id] = text
    if missing:
        print(
            f"WARNING: OCR artifact matched {len(documents) - len(missing)}/"
            f"{len(bundle.corpus_ids)} pages. Using empty OCR for: {missing}",
            flush=True,
        )
    matched = len(documents) - len(missing)
    nonempty = sum(bool(text.strip()) for text in documents.values())
    return documents, {
        "dataset": ocr_name,
        "fingerprint": getattr(rows, "_fingerprint", None),
        "rows": len(rows),
        "unique_pages": len(by_digest),
        "matched_pages": matched,
        "missing_pages": len(missing),
        "missing_corpus_ids": missing,
        "mapping_coverage": matched / len(bundle.corpus_ids),
        "nonempty_ocr_pages": nonempty,
        "empty_ocr_pages": len(bundle.corpus_ids) - nonempty,
        "nonempty_ocr_coverage": nonempty / len(bundle.corpus_ids),
        "filename_matches": filename_matches,
        "image_digest_matches": digest_matches,
        "position_matches": position_matches,
        "positional_image_matches": positional_image_matches,
        "positional_comparisons": positional_comparisons,
        "positional_alignment_ratio": (
            positional_image_matches / positional_comparisons
            if positional_comparisons
            else 0.0
        ),
        "positional_fallback_enabled": positional_fallback_safe,
        "join_version": 3,
        "join": "exact filename, image digest, then verified source-row position",
        "missing_page_policy": "retain corpus page with empty OCR text",
    }


def rank_dense(scores: torch.Tensor, corpus_ids: list[str]) -> dict[int, list[str]]:
    orders = torch.argsort(scores, dim=1, descending=True).cpu().tolist()
    return {
        index: [corpus_ids[position] for position in order]
        for index, order in enumerate(orders)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mteb/VidoreDocVQARetrieval")
    parser.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all")
    parser.add_argument("--max-documents", type=int, default=0, help="0 means all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-details", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_k_details < 1:
        raise SystemExit("batch size and top-k must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("Dense baseline evaluation requires a CUDA GPU")

    seed_everything(42)
    bundle = load_vidore_bundle(
        args.dataset,
        max_queries=args.max_queries,
        max_documents=args.max_documents,
    )
    output = args.output or (
        f"artifacts/results/text/{bundle.slug}_{model_slug(args.dense_model)}.json"
    )
    if args.skip_existing and Path(output).exists():
        try:
            existing = json.loads(Path(output).read_text(encoding="utf-8"))
            dense = existing.get("methods", {}).get("dense_text", {})
            if (
                existing.get("schema_version") == 2
                and existing.get("benchmark_comparable")
                and existing.get("dataset") == args.dataset
                and dense.get("model") == args.dense_model
                and existing.get("ocr", {}).get("join_version") == 3
            ):
                print(f"Keeping completed result {output}")
                return
        except (OSError, json.JSONDecodeError):
            pass
    documents, ocr = load_ocr_documents(bundle)

    bm25_started = time.perf_counter()
    bm25 = BM25(documents)
    bm25_build_seconds = time.perf_counter() - bm25_started
    bm25_started = time.perf_counter()
    bm25_rankings = {
        query_id: [item[0] for item in bm25.rank(query)]
        for query_id, query in zip(bundle.query_ids, bundle.query_texts, strict=True)
    }
    bm25_seconds = time.perf_counter() - bm25_started

    from sentence_transformers import SentenceTransformer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(args.dense_model, device="cuda")
    encode_documents = getattr(model, "encode_document", model.encode)
    encode_queries = getattr(model, "encode_query", model.encode)
    synchronize()
    started = time.perf_counter()
    document_embeddings = encode_documents(
        list(documents.values()),
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    synchronize()
    document_seconds = time.perf_counter() - started
    started = time.perf_counter()
    query_embeddings = encode_queries(
        bundle.query_texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    synchronize()
    query_seconds = time.perf_counter() - started
    _ = query_embeddings[:1] @ document_embeddings.T
    synchronize()
    started = time.perf_counter()
    dense_scores = query_embeddings @ document_embeddings.T
    dense_orders = rank_dense(dense_scores, bundle.corpus_ids)
    synchronize()
    dense_seconds = time.perf_counter() - started
    dense_rankings = {
        query_id: dense_orders[index] for index, query_id in enumerate(bundle.query_ids)
    }

    details = []
    for query_id, query in zip(bundle.query_ids, bundle.query_texts, strict=True):
        details.append(
            {
                "query_id": query_id,
                "query": query,
                "relevant": bundle.qrels[query_id],
                "bm25_relevant_rank": relevant_rank(
                    bm25_rankings[query_id], bundle.qrels[query_id]
                ),
                "dense_relevant_rank": relevant_rank(
                    dense_rankings[query_id], bundle.qrels[query_id]
                ),
                "bm25_top": bm25_rankings[query_id][: args.top_k_details],
                "dense_top": dense_rankings[query_id][: args.top_k_details],
            }
        )

    revision = getattr(getattr(model, "_model_card_vars", {}), "get", lambda *_: None)(
        "model_revision"
    )
    result = {
        "schema_version": 2,
        "experiment": "vidore_text_baselines",
        "dataset": args.dataset,
        "dataset_fingerprints": bundle.fingerprints,
        "ocr": ocr,
        "git_commit": git_commit(),
        "seed": 42,
        "benchmark_comparable": bundle.is_full,
        "queries": len(bundle.query_ids),
        "documents": len(bundle.corpus_ids),
        "methods": {
            "bm25": {
                "metrics": evaluate_run(bm25_rankings, bundle.qrels),
                "parameters": {"k1": bm25.k1, "b": bm25.b},
                "efficiency": {
                    "index_build_seconds": bm25_build_seconds,
                    "retrieval_ms_per_query": 1000 * bm25_seconds / len(bundle.query_ids),
                },
            },
            "dense_text": {
                "model": args.dense_model,
                "model_revision": revision,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "precision": str(document_embeddings.dtype).removeprefix("torch."),
                "metrics": evaluate_run(dense_rankings, bundle.qrels),
                "efficiency": {
                    "document_encoding_ms_per_page": 1000
                    * document_seconds
                    / len(bundle.corpus_ids),
                    "query_encoding_ms_per_query": 1000 * query_seconds / len(bundle.query_ids),
                    "retrieval_ms_per_query": 1000 * dense_seconds / len(bundle.query_ids),
                    "index_size_mb": tensor_bytes(document_embeddings) / 1024**2,
                    "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
                },
            },
        },
        "per_query": details,
    }
    write_json(output, result)
    print(f"Wrote {output}")
    metric_summary = {name: value["metrics"] for name, value in result["methods"].items()}
    print(json.dumps(metric_summary, indent=2))


if __name__ == "__main__":
    main()
