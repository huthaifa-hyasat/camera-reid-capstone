from typing import List

import numpy as np


EXPECTED_DIM = 512
EXPECTED_DTYPE = np.float32
NORM_TOLERANCE = 1e-2


def _validate_embedding(
    embedding: np.ndarray,
    name: str = "embedding"
) -> None:
    if embedding is None:
        raise ValueError(f"{name} is None.")

    if not isinstance(embedding, np.ndarray):
        raise ValueError(
            f"{name} must be a numpy array, got {type(embedding)}."
        )

    if embedding.shape != (EXPECTED_DIM,):
        raise ValueError(
            f"{name} must have shape ({EXPECTED_DIM},), "
            f"got {embedding.shape}."
        )

    if embedding.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"{name} must have dtype {EXPECTED_DTYPE}, "
            f"got {embedding.dtype}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"{name} contains non-finite values (NaN/Inf)."
        )

    norm = float(np.linalg.norm(embedding))

    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(
            f"{name} is not L2-normalized "
            f"(norm={norm:.6f}, expected ~1.0 "
            f"within tolerance {NORM_TOLERANCE})."
        )


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray
) -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.

    Because both embeddings are already L2-normalized, cosine similarity
    is equivalent to their dot product.
    """

    _validate_embedding(
        embedding_a,
        name="embedding_a"
    )

    _validate_embedding(
        embedding_b,
        name="embedding_b"
    )

    return float(
        np.dot(embedding_a, embedding_b)
    )


def compare_to_gallery(
    query_embedding: np.ndarray,
    gallery_embeddings: List[np.ndarray],
) -> List[float]:
    """
    Compare one query embedding against every embedding in a gallery.

    Returns:
        A list of similarity scores in the same order as the gallery.

    This function does NOT:
        - rank results
        - select the best match
        - apply thresholds
        - decide known/new/uncertain
    """

    _validate_embedding(
        query_embedding,
        name="query_embedding"
    )

    if gallery_embeddings is None:
        raise ValueError(
            "gallery_embeddings is None."
        )

    if len(gallery_embeddings) == 0:
        raise ValueError(
            "gallery_embeddings is empty; "
            "nothing to compare against."
        )

    scores: List[float] = []

    for i, gallery_embedding in enumerate(
        gallery_embeddings
    ):
        _validate_embedding(
            gallery_embedding,
            name=f"gallery_embeddings[{i}]"
        )

        score = float(
            np.dot(
                query_embedding,
                gallery_embedding
            )
        )

        scores.append(score)

    return scores