"""
Minimal manual test for src/decision/decision.py ONLY.

Uses synthetic L2-normalized vectors.

No detector, OSNet, SQLite, video pipeline,
or real gallery persistence is involved.
"""

import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

import numpy as np

from decision.decision import decide


MATCH_THRESHOLD = 0.75
REJECT_THRESHOLD = 0.55


def make_unit_vector(
    seed: int,
    dim: int = 512
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    vector = rng.standard_normal(
        dim
    ).astype(np.float32)

    vector = vector / np.linalg.norm(vector)

    return vector


def blend(
    a: np.ndarray,
    b: np.ndarray,
    weight_a: float
) -> np.ndarray:
    """
    Blend two vectors and normalize the result.
    """

    vector = (
        weight_a * a
        + (1 - weight_a) * b
    )

    vector = vector / np.linalg.norm(vector)

    return vector.astype(np.float32)


def main():

    base = make_unit_vector(
        seed=1
    )

    unrelated = make_unit_vector(
        seed=99
    )

    gallery_ids = [1, 2, 3]

    gallery_embeddings = [
        base,
        make_unit_vector(seed=2),
        make_unit_vector(seed=3),
    ]

    # ---------------------------------------------------------
    # 1. Empty gallery
    # ---------------------------------------------------------
    result_empty = decide(
        base,
        [],
        [],
        MATCH_THRESHOLD,
        REJECT_THRESHOLD
    )

    print(
        "Empty gallery: "
        f"status={result_empty.status}, "
        f"person_id={result_empty.person_id}, "
        f"similarity={result_empty.similarity}"
    )

    # ---------------------------------------------------------
    # 2. KNOWN case
    # ---------------------------------------------------------
    result_known = decide(
        base,
        gallery_ids,
        gallery_embeddings,
        MATCH_THRESHOLD,
        REJECT_THRESHOLD
    )

    print(
        "Identical query: "
        f"status={result_known.status}, "
        f"person_id={result_known.person_id}, "
        f"similarity={result_known.similarity:.4f}"
    )

    # ---------------------------------------------------------
    # 3. NEW case
    # ---------------------------------------------------------
    result_new = decide(
        unrelated,
        gallery_ids,
        gallery_embeddings,
        MATCH_THRESHOLD,
        REJECT_THRESHOLD
    )

    print(
        "Unrelated query: "
        f"status={result_new.status}, "
        f"person_id={result_new.person_id}, "
        f"similarity={result_new.similarity:.4f}"
    )

    # ---------------------------------------------------------
    # 4. UNCERTAIN case
    # ---------------------------------------------------------
    mid_query = blend(
        base,
        unrelated,
        weight_a=0.5
    )

    result_uncertain = decide(
        mid_query,
        gallery_ids,
        gallery_embeddings,
        MATCH_THRESHOLD,
        REJECT_THRESHOLD
    )

    print(
        "Blended query: "
        f"status={result_uncertain.status}, "
        f"person_id={result_uncertain.person_id}, "
        f"similarity={result_uncertain.similarity:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Mismatched list lengths
    # ---------------------------------------------------------
    try:

        decide(
            base,
            [1, 2],
            [base],
            MATCH_THRESHOLD,
            REJECT_THRESHOLD
        )

        print(
            "[FAIL] Expected ValueError for "
            "mismatched lengths, none raised."
        )

    except ValueError as e:

        print(
            "[OK] Correctly raised ValueError "
            f"for mismatched lengths: {e}"
        )

    # ---------------------------------------------------------
    # 6. Invalid threshold ordering
    # ---------------------------------------------------------
    try:

        decide(
            base,
            gallery_ids,
            gallery_embeddings,
            match_threshold=0.5,
            reject_threshold=0.9
        )

        print(
            "[FAIL] Expected ValueError for "
            "inverted thresholds, none raised."
        )

    except ValueError as e:

        print(
            "[OK] Correctly raised ValueError "
            f"for inverted thresholds: {e}"
        )

    # ---------------------------------------------------------
    # 7. Invalid query embedding
    # ---------------------------------------------------------
    try:

        invalid_query = np.zeros(
            128,
            dtype=np.float32
        )

        decide(
            invalid_query,
            gallery_ids,
            gallery_embeddings,
            MATCH_THRESHOLD,
            REJECT_THRESHOLD
        )

        print(
            "[FAIL] Expected ValueError for "
            "invalid query embedding, none raised."
        )

    except ValueError as e:

        print(
            "[OK] Correctly raised ValueError "
            f"for invalid query embedding: {e}"
        )


if __name__ == "__main__":
    main()