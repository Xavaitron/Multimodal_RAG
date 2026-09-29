"""Versioned embedding cache with human-readable metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def save_embeddings(
    path: str | Path, embeddings: dict[str, Tensor], metadata: dict[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpu_embeddings = {key: value.detach().cpu() for key, value in embeddings.items()}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"embeddings": cpu_embeddings, "metadata": metadata}, temporary)
    temporary.replace(destination)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_temporary.replace(metadata_path)


def load_embeddings(path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if set(payload) != {"embeddings", "metadata"}:
        raise ValueError("unrecognized cache format")
    return payload["embeddings"], payload["metadata"]


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def embedding_bytes(embeddings: dict[str, Tensor] | list[Tensor]) -> int:
    values = embeddings.values() if isinstance(embeddings, dict) else embeddings
    return sum(tensor_bytes(tensor) for tensor in values)


def validate_cache_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"embedding cache metadata mismatch: {mismatches}")
