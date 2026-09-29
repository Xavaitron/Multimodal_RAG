import torch

from lightweight_multimodal_retrieval.cache import load_embeddings, save_embeddings


def test_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "embeddings.pt"
    expected = {"page-1": torch.tensor([[1.0, 2.0]])}
    metadata = {"model": "example", "revision": "abc", "dtype": "float32"}
    save_embeddings(path, expected, metadata)
    actual, actual_metadata = load_embeddings(path)
    torch.testing.assert_close(actual["page-1"], expected["page-1"])
    assert actual_metadata == metadata
