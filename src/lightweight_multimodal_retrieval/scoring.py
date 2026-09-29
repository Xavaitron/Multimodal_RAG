"""Auditable reference retrieval scorers independent of library convenience APIs."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_tokens(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [tokens, dim], got {tuple(value.shape)}")
    if value.numel() == 0 or value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one token")


def maxsim(query: Tensor, document: Tensor, *, normalize: bool = False) -> Tensor:
    """Return sum_i max_j(q_i dot d_j) for one query-document pair."""
    _validate_tokens("query", query)
    _validate_tokens("document", document)
    if query.shape[-1] != document.shape[-1]:
        raise ValueError("query and document embedding dimensions must match")
    if normalize:
        query = F.normalize(query, p=2, dim=-1)
        document = F.normalize(document, p=2, dim=-1)
    similarities = query @ document.transpose(0, 1)
    return similarities.max(dim=1).values.sum()


def score_documents(
    query: Tensor,
    documents: list[Tensor],
    *,
    normalize: bool = False,
    document_batch_size: int = 16,
) -> Tensor:
    """Score a query against variable-length documents in bounded-size chunks."""
    if document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    scores: list[Tensor] = []
    for start in range(0, len(documents), document_batch_size):
        chunk = documents[start : start + document_batch_size]
        scores.extend(maxsim(query, doc, normalize=normalize) for doc in chunk)
    return torch.stack(scores) if scores else torch.empty(0, device=query.device)


def global_embedding(tokens: Tensor) -> Tensor:
    """Mean-pool token embeddings and L2-normalize the result."""
    _validate_tokens("tokens", tokens)
    return F.normalize(tokens.mean(dim=0), p=2, dim=0)


def global_score(query: Tensor, document: Tensor) -> Tensor:
    """Cosine score between mean-pooled query and document representations."""
    return global_embedding(query) @ global_embedding(document)
