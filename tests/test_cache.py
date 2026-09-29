import pytest
import torch

from lightweight_multimodal_retrieval.cache import (
    embedding_bytes,
    load_embeddings,
    save_embeddings,
    validate_cache_metadata,
)


def test_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "embeddings.pt"
    expected = {"page-1": torch.tensor([[1.0, 2.0]])}
    metadata = {"model": "example", "revision": "abc", "dtype": "float32"}
    save_embeddings(path, expected, metadata)
    actual, actual_metadata = load_embeddings(path)
    torch.testing.assert_close(actual["page-1"], expected["page-1"])
    assert actual_metadata == metadata
    assert not (tmp_path / "embeddings.pt.tmp").exists()


def test_embedding_bytes() -> None:
    embeddings = [torch.zeros(2, 3, dtype=torch.float16), torch.zeros(1, 3)]
    assert embedding_bytes(embeddings) == 24


def test_cache_metadata_validation() -> None:
    validate_cache_metadata({"model": "a", "seed": 42}, {"model": "a"})
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_cache_metadata({"model": "a"}, {"model": "b"})
