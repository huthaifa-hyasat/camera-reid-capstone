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

from gallery.gallery import InMemoryGallery


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


def main():

    gallery = InMemoryGallery()

    # 1. Empty gallery
    print(
        "is_empty() on fresh gallery "
        f"(True expected): {gallery.is_empty()}"
    )

    # 2. Add persons
    embedding_1 = make_unit_vector(seed=1)
    embedding_2 = make_unit_vector(seed=2)
    embedding_3 = make_unit_vector(seed=3)

    id_1 = gallery.add_person(
        embedding_1
    )

    id_2 = gallery.add_person(
        embedding_2
    )

    id_3 = gallery.add_person(
        embedding_3
    )

    print(
        "Assigned person_ids "
        f"(expected 1, 2, 3): "
        f"{id_1}, {id_2}, {id_3}"
    )

    print(
        "is_empty() after adding "
        f"(False expected): {gallery.is_empty()}"
    )

    print(
        "len(gallery) "
        f"(3 expected): {len(gallery)}"
    )

    # 3. get_all()
    person_ids, embeddings = gallery.get_all()

    print(
        f"get_all() person_ids: {person_ids}"
    )

    print(
        "get_all() embeddings count: "
        f"{len(embeddings)}"
    )

    # 4. Retrieve by ID
    retrieved = gallery.get_embedding(
        id_2
    )

    matches_original = np.array_equal(
        retrieved,
        embedding_2
    )

    print(
        "get_embedding(id_2) matches original "
        f"(True expected): {matches_original}"
    )

    # 5. Unknown ID
    try:
        gallery.get_embedding(999)

        print(
            "[FAIL] Expected KeyError for "
            "unknown person_id, none was raised."
        )

    except KeyError as e:
        print(
            "[OK] Correctly raised KeyError "
            f"for unknown person_id: {e}"
        )

    # 6. Wrong shape
    bad_shape = np.zeros(
        128,
        dtype=np.float32
    )

    try:
        gallery.add_person(
            bad_shape
        )

        print(
            "[FAIL] Expected ValueError for "
            "wrong shape, none was raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for wrong shape: {e}"
        )

    # 7. Not normalized
    not_normalized = (
        embedding_1 * 5.0
    ).astype(np.float32)

    try:
        gallery.add_person(
            not_normalized
        )

        print(
            "[FAIL] Expected ValueError for "
            "non-normalized vector, none was raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for non-normalized vector: {e}"
        )


if __name__ == "__main__":
    main()