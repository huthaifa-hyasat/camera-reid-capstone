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

from matching.matcher import (
    cosine_similarity,
    compare_to_gallery,
)


def make_unit_vector(
    seed: int,
    dim: int = 512
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    vector = rng.standard_normal(dim).astype(
        np.float32
    )

    vector = vector / np.linalg.norm(vector)

    return vector


def main():
    identical = make_unit_vector(seed=1)

    similar_seed_based = make_unit_vector(
        seed=1
    )

    different = make_unit_vector(
        seed=2
    )

    # Test 1: identical vectors
    score_identical = cosine_similarity(
        identical,
        similar_seed_based
    )

    print(
        "Identical-vector similarity "
        f"(~1.0 expected): {score_identical:.6f}"
    )

    # Test 2: different vectors
    score_different = cosine_similarity(
        identical,
        different
    )

    print(
        "Different-vector similarity "
        f"(near 0 expected): {score_different:.6f}"
    )

    # Test 3: gallery comparison
    gallery = [
        identical,
        different,
        make_unit_vector(seed=3),
    ]

    scores = compare_to_gallery(
        identical,
        gallery
    )

    formatted_scores = [
        f"{score:.4f}"
        for score in scores
    ]

    print(
        f"Gallery scores "
        f"(query=identical): {formatted_scores}"
    )

    # Test 4: wrong shape
    bad_shape = np.zeros(
        128,
        dtype=np.float32
    )

    try:
        cosine_similarity(
            identical,
            bad_shape
        )

        print(
            "[FAIL] Expected ValueError "
            "for wrong shape, none raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for wrong shape: {e}"
        )

    # Test 5: wrong dtype
    bad_dtype = identical.astype(
        np.float64
    )

    try:
        cosine_similarity(
            identical,
            bad_dtype
        )

        print(
            "[FAIL] Expected ValueError "
            "for wrong dtype, none raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for wrong dtype: {e}"
        )

    # Test 6: non-normalized vector
    not_normalized = (
        identical * 5.0
    ).astype(np.float32)

    try:
        cosine_similarity(
            identical,
            not_normalized
        )

        print(
            "[FAIL] Expected ValueError "
            "for non-normalized vector, none raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for non-normalized vector: {e}"
        )

    # Test 7: NaN
    nan_vector = identical.copy()

    nan_vector[0] = np.nan

    try:
        cosine_similarity(
            identical,
            nan_vector
        )

        print(
            "[FAIL] Expected ValueError "
            "for NaN vector, none raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for NaN vector: {e}"
        )

    # Test 8: empty gallery
    try:
        compare_to_gallery(
            identical,
            []
        )

        print(
            "[FAIL] Expected ValueError "
            "for empty gallery, none raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for empty gallery: {e}"
        )


if __name__ == "__main__":
    main()