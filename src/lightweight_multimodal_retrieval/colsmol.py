"""Thin adapter around the current ColSmol implementation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from torch import Tensor


class ColSmolEncoder:
    """Load ColSmol through ColIdefics3 while exposing stable project methods."""

    def __init__(self, model_name: str = "vidore/colSmol-500M", device: str = "cuda") -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor

        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def model_revision(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    def encode_images(self, images: Sequence[Image.Image]) -> list[Tensor]:
        batch = self.processor.process_images(list(images)).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(**batch)
        return [item.detach() for item in torch.unbind(embeddings)]

    def encode_queries(self, queries: Sequence[str]) -> list[Tensor]:
        batch = self.processor.process_queries(list(queries)).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(**batch)
        return [item.detach() for item in torch.unbind(embeddings)]
