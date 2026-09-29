import torch

from lightweight_multimodal_retrieval.scoring import global_embedding, global_score, maxsim


def test_maxsim_matches_manual_value() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    document = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]])
    assert maxsim(query, document).item() == 3.0


def test_global_embedding_is_normalized() -> None:
    pooled = global_embedding(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    torch.testing.assert_close(torch.linalg.vector_norm(pooled), torch.tensor(1.0))


def test_global_score_is_cosine_of_means() -> None:
    tokens = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert global_score(tokens, tokens).item() == 1.0
