import pytest
import torch

from lightweight_multimodal_retrieval.compression import (
    kmeans_tokens,
    local_mean_pool,
    random_tokens,
    target_token_count,
    uniform_tokens,
)


@pytest.mark.parametrize("ratio, expected", [(1.0, 8), (0.5, 4), (0.25, 2), (0.125, 1)])
def test_compressors_return_exact_k(ratio: float, expected: int) -> None:
    tokens = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    assert target_token_count(8, ratio) == expected
    assert random_tokens(tokens, ratio, seed=42).shape == (expected, 3)
    assert uniform_tokens(tokens, ratio).shape == (expected, 3)
    assert local_mean_pool(tokens, ratio).shape == (expected, 3)
    assert kmeans_tokens(tokens, ratio, seed=42).shape == (expected, 3)


def test_random_compression_is_reproducible() -> None:
    tokens = torch.randn(20, 4)
    first = random_tokens(tokens, 0.5, seed=7)
    second = random_tokens(tokens, 0.5, seed=7)
    torch.testing.assert_close(first, second)
