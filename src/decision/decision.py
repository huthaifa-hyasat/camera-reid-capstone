import os
import sys
from typing import List, NamedTuple, Optional

import numpy as np


# Allow importing matcher.py from src/matching
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "matching"
    )
)

from matcher import (
    compare_to_gallery,
    _validate_embedding,
)


class DecisionResult(NamedTuple):
    """
    Result of a gallery matching decision.
    """

    status: str
    person_id: Optional[int]
    similarity: Optional[float]


def decide(
    query_embedding: np.ndarray,
    gallery_person_ids: List[int],
    gallery_embeddings: List[np.ndarray],
    match_threshold: float,
    reject_threshold: float,
) -> DecisionResult:
    """
    Classify one query embedding against the gallery.

    Decision rules:

        similarity >= match_threshold
            -> KNOWN

        reject_threshold <= similarity < match_threshold
            -> UNCERTAIN

        similarity < reject_threshold
            -> NEW

    Empty gallery:
        -> NEW with person_id=None and similarity=None

    Args:
        query_embedding:
            L2-normalized 512-dimensional float32 embedding.

        gallery_person_ids:
            Person IDs corresponding to gallery_embeddings.

        gallery_embeddings:
            L2-normalized 512-dimensional float32 embeddings.

        match_threshold:
            Minimum similarity required for KNOWN.

        reject_threshold:
            Similarity below this value means NEW.

    Returns:
        DecisionResult(status, person_id, similarity)

    Raises:
        ValueError:
            If the query embedding is invalid,
            gallery lengths do not match,
            or thresholds are invalid.
    """

    # Validate query embedding
    _validate_embedding(
        query_embedding,
        name="query_embedding"
    )

    # Validate parallel lists
    if len(gallery_person_ids) != len(gallery_embeddings):
        raise ValueError(
            f"gallery_person_ids (len={len(gallery_person_ids)}) "
            f"and gallery_embeddings (len={len(gallery_embeddings)}) "
            f"must have the same length."
        )

    # Validate threshold ordering
    if reject_threshold > match_threshold:
        raise ValueError(
            f"reject_threshold ({reject_threshold}) must be <= "
            f"match_threshold ({match_threshold})."
        )

    # Explicit empty-gallery case
    if len(gallery_embeddings) == 0:
        return DecisionResult(
            status="NEW",
            person_id=None,
            similarity=None
        )

    # Compare query against all gallery embeddings
    scores = compare_to_gallery(
        query_embedding,
        gallery_embeddings
    )

    # Find best similarity
    best_idx = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[best_idx]
    )

    best_person_id = gallery_person_ids[
        best_idx
    ]

    # Decision logic
    if best_score >= match_threshold:
        return DecisionResult(
            status="KNOWN",
            person_id=best_person_id,
            similarity=best_score
        )

    elif best_score >= reject_threshold:
        return DecisionResult(
            status="UNCERTAIN",
            person_id=best_person_id,
            similarity=best_score
        )

    else:
        return DecisionResult(
            status="NEW",
            person_id=None,
            similarity=best_score
        )