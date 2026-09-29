from lightweight_multimodal_retrieval.vidore_data import dataset_slug, ids_digest, model_slug


def test_known_dataset_slugs() -> None:
    assert dataset_slug("mteb/VidoreDocVQARetrieval") == "docvqa"
    assert dataset_slug("mteb/VidoreInfoVQARetrieval") == "infovqa"


def test_model_slug_is_path_safe() -> None:
    assert model_slug("vidore/colSmol-500M") == "colsmol500m"


def test_ids_digest_is_stable_and_order_sensitive() -> None:
    assert ids_digest(["a", "b"]) == ids_digest(["a", "b"])
    assert ids_digest(["a", "b"]) != ids_digest(["b", "a"])
