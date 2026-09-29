"""Token-index compression algorithms used in the ablation study."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def target_token_count(n_tokens: int, ratio: float) -> int:
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    return max(1, min(n_tokens, round(n_tokens * ratio)))


def random_tokens(tokens: Tensor, ratio: float, *, seed: int) -> Tensor:
    k = target_token_count(tokens.shape[0], ratio)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(tokens.shape[0], generator=generator)[:k].sort().values
    return tokens[indices.to(tokens.device)]


def uniform_tokens(tokens: Tensor, ratio: float) -> Tensor:
    """Retain approximately uniform positions while returning exactly K tokens."""
    k = target_token_count(tokens.shape[0], ratio)
    indices = torch.linspace(0, tokens.shape[0] - 1, steps=k, device=tokens.device).round().long()
    return tokens[indices]


def local_mean_pool(tokens: Tensor, ratio: float) -> Tensor:
    """Pool contiguous token regions to exactly K representations."""
    k = target_token_count(tokens.shape[0], ratio)
    boundaries = torch.linspace(0, tokens.shape[0], steps=k + 1).round().long()
    return torch.stack([tokens[boundaries[i] : boundaries[i + 1]].mean(dim=0) for i in range(k)])


def kmeans_tokens(tokens: Tensor, ratio: float, *, seed: int = 42, iterations: int = 25) -> Tensor:
    """Compress tokens to K centroids using deterministic k-means initialization."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    k = target_token_count(tokens.shape[0], ratio)
    if k == tokens.shape[0]:
        return tokens.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(tokens.shape[0], generator=generator)[:k].to(tokens.device)
    centroids = tokens[initial].clone()
    for _ in range(iterations):
        assignments = torch.cdist(tokens.float(), centroids.float()).argmin(dim=1)
        updated = []
        for cluster in range(k):
            members = tokens[assignments == cluster]
            updated.append(members.mean(dim=0) if len(members) else centroids[cluster])
        next_centroids = torch.stack(updated)
        if torch.allclose(next_centroids, centroids, rtol=1e-4, atol=1e-5):
            centroids = next_centroids
            break
        centroids = next_centroids
    if math.isnan(float(centroids.float().sum())):
        raise RuntimeError("k-means produced NaN centroids")
    return centroids
