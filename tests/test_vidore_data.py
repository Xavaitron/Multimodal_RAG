import pytest

from lightweight_multimodal_retrieval.vidore_data import (
    corpus_position,
    dataset_slug,
    ids_digest,
    model_slug,
    positional_alignment_is_safe,
)


def test_known_dataset_slugs() -> None:
    assert dataset_slug("mteb/VidoreDocVQARetrieval") == "docvqa"
    assert dataset_slug("mteb/VidoreInfoVQARetrieval") == "infovqa"


def test_model_slug_is_path_safe() -> None:
    assert model_slug("vidore/colSmol-500M") == "colsmol500m"


def test_ids_digest_is_stable_and_order_sensitive() -> None:
    assert ids_digest(["a", "b"]) == ids_digest(["a", "b"])
    assert ids_digest(["a", "b"]) != ids_digest(["b", "a"])


def test_corpus_position_only_accepts_generated_mteb_ids() -> None:
    assert corpus_position("corpus-test-78") == 78
    assert corpus_position("document-78") is None


def test_positional_alignment_requires_strong_evidence() -> None:
    assert positional_alignment_is_safe(422, 500)
    assert not positional_alignment_is_safe(399, 500)
    assert not positional_alignment_is_safe(40, 40)
    with pytest.raises(ValueError, match="invalid positional alignment counts"):
        positional_alignment_is_safe(51, 50)
