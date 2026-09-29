"""Small auditable Okapi BM25 baseline for OCR text."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25:
    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.k1 = k1
        self.b = b
        self.document_ids = list(documents)
        self.terms = {doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()}
        self.lengths = {doc_id: sum(counts.values()) for doc_id, counts in self.terms.items()}
        self.average_length = sum(self.lengths.values()) / len(self.lengths)
        document_frequency: Counter[str] = Counter()
        for counts in self.terms.values():
            document_frequency.update(counts.keys())
        n_docs = len(documents)
        self.idf = {
            term: math.log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str, document_id: str) -> float:
        counts = self.terms[document_id]
        length_norm = 1.0 - self.b + self.b * self.lengths[document_id] / self.average_length
        score = 0.0
        for term in tokenize(query):
            frequency = counts.get(term, 0)
            if frequency:
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
                )
        return score

    def rank(self, query: str, *, top_k: int | None = None) -> Sequence[tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.document_ids]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
