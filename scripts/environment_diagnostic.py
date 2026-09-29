#!/usr/bin/env python
"""Inspect the GPU and optionally run one real ColSmol page/query forward pass."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from lightweight_multimodal_retrieval.colsmol import ColSmolEncoder
from lightweight_multimodal_retrieval.scoring import maxsim
from lightweight_multimodal_retrieval.utils import git_commit, write_json


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def attention_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu/eager"
    capability = torch.cuda.get_device_capability()
    return "pytorch-sdpa (FlashAttention2 not assumed)" if capability >= (7, 5) else "eager"


def system_report() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "colpali_engine": package_version("colpali-engine"),
        "transformers": package_version("transformers"),
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "attention_backend": attention_backend(),
        "git_commit": git_commit(),
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        report.update(
            gpu_name=props.name,
            compute_capability=".".join(map(str, torch.cuda.get_device_capability(0))),
            total_vram_gb=round(props.total_memory / 1024**3, 3),
            fp16_supported=torch.cuda.get_device_capability(0) >= (5, 3),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return report


def sample_page() -> Image.Image:
    image = Image.new("RGB", (768, 1024), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 80), "Annual results", fill="black")
    draw.text((60, 180), "Revenue 2023: 42 million", fill="black")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vidore/colSmol-500M")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--output", default="artifacts/results/environment_diagnostic.json")
    args = parser.parse_args()

    report = system_report()
    print(report)
    if not args.skip_model:
        if not torch.cuda.is_available():
            raise SystemExit(
                "A CUDA GPU is required for the model smoke test; "
                "use --skip-model for diagnostics only"
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        encoder = ColSmolEncoder(args.model)
        report["model"] = args.model
        report["model_revision"] = encoder.model_revision
        report["model_parameters"] = encoder.parameter_count
        report["vram_after_model_load_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)

        document = encoder.encode_images([sample_page()])[0]
        torch.cuda.synchronize()
        report["vram_after_page_forward_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        query = encoder.encode_queries(["What was revenue in 2023?"])[0]
        torch.cuda.synchronize()
        report["vram_after_query_forward_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        report["visual_embeddings"] = document.shape[0]
        report["query_embeddings"] = query.shape[0]
        report["embedding_dimension"] = document.shape[-1]
        report["manual_maxsim"] = float(maxsim(query, document).item())

    write_json(Path(args.output), report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
