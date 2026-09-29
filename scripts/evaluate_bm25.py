#!/usr/bin/env python
"""Evaluate BM25 from OCR, query, and qrels JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightweight_multimodal_retrieval.bm25 import BM25
from lightweight_multimodal_retrieval.metrics import evaluate_run
from lightweight_multimodal_retrieval.utils import git_commit, write_json


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True, help='JSON object: {"page_id": "OCR text"}')
    parser.add_argument("--queries", required=True, help='JSON object: {"query_id": "query text"}')
    parser.add_argument(
        "--qrels", required=True, help='JSON object: {"query_id": {"page_id": gain}}'
    )
    parser.add_argument("--output", default="artifacts/results/bm25.json")
    args = parser.parse_args()

    documents = read_json(args.documents)
    queries = read_json(args.queries)
    qrels = read_json(args.qrels)
    retriever = BM25(documents)
    rankings = {
        query_id: [item[0] for item in retriever.rank(query)]
        for query_id, query in queries.items()
    }
    result = {
        "experiment": "bm25",
        "method": "okapi_bm25",
        "git_commit": git_commit(),
        "queries": len(queries),
        "documents": len(documents),
        "metrics": evaluate_run(rankings, qrels),
        "rankings": rankings,
    }
    write_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
