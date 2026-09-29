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
    torch.save({"embeddings": cpu_embeddings, "metadata": metadata}, destination)
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_embeddings(path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if set(payload) != {"embeddings", "metadata"}:
        raise ValueError("unrecognized cache format")
    return payload["embeddings"], payload["metadata"]
