"""Shared ViDoRe dataset loading and identity helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


DATASETS = {
    "docvqa": "mteb/VidoreDocVQARetrieval",
    "infovqa": "mteb/VidoreInfoVQARetrieval",
    "arxivqa": "mteb/VidoreArxivQARetrieval",
    "tatdqa": "mteb/VidoreTatdqaRetrieval",
}

OCR_DATASETS = {
    "docvqa": "vidore/docvqa_test_subsampled_tesseract",
    "infovqa": "vidore/infovqa_test_subsampled_tesseract",
    "arxivqa": "vidore/arxivqa_test_subsampled_tesseract",
    "tatdqa": "vidore/tatdqa_test_tesseract",
}


@dataclass
class VidoreBundle:
    dataset_name: str
    slug: str
    query_ids: list[str]
    query_texts: list[str]
    corpus_ids: list[str]
    corpus: Any
    corpus_indices: list[int]
    qrels: dict[str, dict[str, float]]
    fingerprints: dict[str, str]
    full_query_count: int
    full_document_count: int

    @property
    def is_full(self) -> bool:
        return (
            len(self.query_ids) == self.full_query_count
            and len(self.corpus_ids) == self.full_document_count
        )

    def images(self) -> list[Any]:
        return [self.corpus[index]["image"].convert("RGB") for index in self.corpus_indices]


def field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"Expected one of these fields: {names}")


def dataset_slug(dataset_name: str) -> str:
    dataset_id = dataset_name.rsplit("/", 1)[-1]
    known = {value: key for key, value in DATASETS.items()}
    if dataset_name in known:
        return known[dataset_name]
    dataset_id = dataset_id.removeprefix("Vidore").removesuffix("Retrieval")
    slug = re.sub(r"(?<!^)(?=[A-Z])", "_", dataset_id).lower()
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def model_slug(model_name: str) -> str:
    tail = model_name.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "", tail)


def ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def image_digest(image: Any) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def corpus_position(corpus_id: str) -> int | None:
    """Return N from generated MTEB IDs such as ``corpus-test-N``."""
    match = re.fullmatch(r"corpus-test-(\d+)", corpus_id)
    return int(match.group(1)) if match else None


def positional_alignment_is_safe(
    matching_images: int,
    comparable_images: int,
    *,
    minimum_images: int = 50,
    minimum_ratio: float = 0.8,
) -> bool:
    """Require strong image evidence before using source-row position as a join key."""
    if matching_images < 0 or comparable_images < 0 or matching_images > comparable_images:
        raise ValueError("invalid positional alignment counts")
    if minimum_images < 1 or not 0 < minimum_ratio <= 1:
        raise ValueError("invalid positional alignment thresholds")
    return (
        comparable_images >= minimum_images
        and matching_images / comparable_images >= minimum_ratio
    )


def load_vidore_bundle(
    dataset_name: str,
    *,
    max_queries: int = 0,
    max_documents: int = 0,
) -> VidoreBundle:
    """Load MTEB corpus/queries/qrels with a deterministic candidate subset."""
    from datasets import Dataset, load_dataset

    if max_queries < 0 or max_documents < 0:
        raise ValueError("dataset limits must be non-negative")

    def load_config(config: str) -> Dataset:
        dataset = load_dataset(dataset_name, config, split="test")
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset for {dataset_name}/{config}")
        return dataset

    queries_data = load_config("queries")
    qrels_data = load_config("qrels")
    corpus_data = load_config("corpus")

    query_count = len(queries_data) if max_queries == 0 else min(max_queries, len(queries_data))
    query_rows = [queries_data[index] for index in range(query_count)]
    query_ids = [str(field(row, "id", "query-id", "query_id")) for row in query_rows]
    query_texts = [str(field(row, "text", "query")) for row in query_rows]
    selected_queries = set(query_ids)

    qrels: dict[str, dict[str, float]] = {query_id: {} for query_id in query_ids}
    for row in qrels_data:
        query_id = str(field(row, "query-id", "query_id"))
        if query_id in selected_queries:
            corpus_id = str(field(row, "corpus-id", "corpus_id"))
            qrels[query_id][corpus_id] = float(field(row, "score"))
    missing = [query_id for query_id, judgments in qrels.items() if not judgments]
    if missing:
        raise RuntimeError(f"Selected queries have no qrels: {missing[:5]}")

    positive_ids = {
        corpus_id
        for judgments in qrels.values()
        for corpus_id, gain in judgments.items()
        if gain > 0
    }
    all_corpus_ids = [str(value) for value in corpus_data["id"]]
    document_limit = len(all_corpus_ids) if max_documents == 0 else max_documents
    if document_limit < len(positive_ids):
        raise ValueError(
            f"max_documents must be at least {len(positive_ids)} to include all positives"
        )

    candidate_ids = [item for item in all_corpus_ids if item in positive_ids]
    candidate_ids.extend(
        item
        for item in all_corpus_ids
        if item not in positive_ids and len(candidate_ids) < document_limit
    )
    candidate_ids = candidate_ids[:document_limit]
    corpus_lookup = {corpus_id: index for index, corpus_id in enumerate(all_corpus_ids)}
    corpus_indices = [corpus_lookup[corpus_id] for corpus_id in candidate_ids]

    return VidoreBundle(
        dataset_name=dataset_name,
        slug=dataset_slug(dataset_name),
        query_ids=query_ids,
        query_texts=query_texts,
        corpus_ids=candidate_ids,
        corpus=corpus_data,
        corpus_indices=corpus_indices,
        qrels=qrels,
        fingerprints={
            "queries": queries_data._fingerprint,
            "qrels": qrels_data._fingerprint,
            "corpus": corpus_data._fingerprint,
        },
        full_query_count=len(queries_data),
        full_document_count=len(corpus_data),
    )
