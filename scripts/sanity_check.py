#!/usr/bin/env python
"""Encode local page images, rank them with manual MaxSim, and persist evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from lightweight_multimodal_retrieval.cache import save_embeddings
from lightweight_multimodal_retrieval.colsmol import ColSmolEncoder
from lightweight_multimodal_retrieval.scoring import global_score, score_documents
from lightweight_multimodal_retrieval.utils import git_commit, seed_everything, write_json


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", required=True, help="Directory containing PNG/JPG page images")
    parser.add_argument("--queries", required=True, help="JSON list of query strings")
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--output", default="artifacts/results/sanity_check.json")
    parser.add_argument("--cache", default="cache/sanity_pages.pt")
    args = parser.parse_args()

    seed_everything(42)
    extensions = {".png", ".jpg", ".jpeg"}
    page_paths = sorted(
        path for path in Path(args.pages).iterdir() if path.suffix.casefold() in extensions
    )
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    if not 1 <= len(page_paths) <= 100:
        raise SystemExit("Sanity check expects 1-100 page images")
    valid_queries = (
        isinstance(queries, list)
        and queries
        and all(isinstance(item, str) for item in queries)
    )
    if not valid_queries:
        raise SystemExit("--queries must contain a non-empty JSON list of strings")

    images = [Image.open(path).convert("RGB") for path in page_paths]
    encoder = ColSmolEncoder(args.model)
    torch.cuda.reset_peak_memory_stats()

    synchronize()
    started = time.perf_counter()
    documents = [encoder.encode_images([image])[0] for image in images]
    synchronize()
    page_seconds = time.perf_counter() - started

    synchronize()
    started = time.perf_counter()
    query_embeddings = encoder.encode_queries(queries)
    synchronize()
    query_seconds = time.perf_counter() - started

    page_ids = [path.stem for path in page_paths]
    cache_metadata = {
        "model": args.model,
        "model_revision": encoder.model_revision,
        "dtype": str(documents[0].dtype),
        "page_ids": page_ids,
        "seed": 42,
        "git_commit": git_commit(),
    }
    save_embeddings(args.cache, dict(zip(page_ids, documents, strict=True)), cache_metadata)

    rankings = []
    for query_text, query_embedding in zip(queries, query_embeddings, strict=True):
        scores = score_documents(query_embedding, documents).detach().cpu().tolist()
        global_scores = [
            float(global_score(query_embedding, document).item()) for document in documents
        ]
        order = sorted(range(len(page_ids)), key=lambda index: scores[index], reverse=True)
        rankings.append(
            {
                "query": query_text,
                "maxsim": [{"page_id": page_ids[i], "score": scores[i]} for i in order],
                "global_visual": [
                    {"page_id": page_ids[i], "score": global_scores[i]}
                    for i in sorted(
                        range(len(page_ids)),
                        key=lambda index: global_scores[index],
                        reverse=True,
                    )
                ],
            }
        )

    result = {
        "experiment": "colsmol_sanity_check",
        "model": args.model,
        "model_revision": encoder.model_revision,
        "git_commit": git_commit(),
        "seed": 42,
        "pages": len(documents),
        "queries": len(query_embeddings),
        "tokens_per_page": [embedding.shape[0] for embedding in documents],
        "embedding_dimension": documents[0].shape[-1],
        "efficiency": {
            "page_encoding_ms_per_page": 1000 * page_seconds / len(documents),
            "query_encoding_ms_per_query": 1000 * query_seconds / len(query_embeddings),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
        "rankings": rankings,
    }
    write_json(args.output, result)
    print(f"Wrote {args.output} and {args.cache}")


if __name__ == "__main__":
    main()
