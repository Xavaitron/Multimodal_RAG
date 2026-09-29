from lightweight_multimodal_retrieval.bm25 import BM25


def test_bm25_ranks_exact_match_first() -> None:
    retriever = BM25({"revenue": "Revenue in 2023 was 42 million", "weather": "Rain tomorrow"})
    assert retriever.rank("2023 revenue")[0][0] == "revenue"
