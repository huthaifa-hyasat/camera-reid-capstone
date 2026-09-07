"""
Minimal manual test for src/persistence/db.py ONLY.

Uses synthetic L2-normalized vectors and a real SQLite file on disk.

Tests:
- fresh database
- adding gallery entries
- retrieving all entries
- retrieving one embedding
- invalid person_id
- invalid embedding
- persistence after closing and reopening the database
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

import numpy as np

from persistence.db import GalleryDB


TEST_DB_PATH = "test_gallery.db"


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
    # Remove old test database if it exists
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    # ---------------------------------------------------------
    # 1. Fresh database should be empty
    # ---------------------------------------------------------
    db = GalleryDB(TEST_DB_PATH)

    print(
        "is_empty() on fresh db "
        f"(True expected): {db.is_empty()}"
    )

    # ---------------------------------------------------------
    # 2. Add two people
    # ---------------------------------------------------------
    embedding_1 = make_unit_vector(seed=1)
    embedding_2 = make_unit_vector(seed=2)

    id_1 = db.add_person(
        embedding_1,
        label="test-person-1"
    )

    id_2 = db.add_person(
        embedding_2
    )

    print(
        f"Assigned person_ids: {id_1}, {id_2}"
    )

    print(
        "is_empty() after adding "
        f"(False expected): {db.is_empty()}"
    )

    # ---------------------------------------------------------
    # 3. Retrieve all entries
    # ---------------------------------------------------------
    all_entries = db.get_all()

    print(
        f"get_all() returned {len(all_entries)} entries."
    )

    for (
        person_id,
        embedding,
        first_seen,
        label
    ) in all_entries:

        print(
            f"  person_id={person_id} "
            f"label={label} "
            f"first_seen={first_seen} "
            f"embedding_shape={embedding.shape} "
            f"dtype={embedding.dtype}"
        )

    # ---------------------------------------------------------
    # 4. Retrieve one embedding and compare with original
    # ---------------------------------------------------------
    retrieved = db.get_embedding(id_1)

    matches_original = np.array_equal(
        retrieved,
        embedding_1
    )

    print(
        "get_embedding(id_1) matches original "
        f"(True expected): {matches_original}"
    )

    # ---------------------------------------------------------
    # 5. Unknown person_id
    # ---------------------------------------------------------
    try:
        db.get_embedding(9999)

        print(
            "[FAIL] Expected KeyError for "
            "unknown person_id, none raised."
        )

    except KeyError as e:
        print(
            "[OK] Correctly raised KeyError "
            f"for unknown person_id: {e}"
        )

    # ---------------------------------------------------------
    # 6. Invalid embedding shape
    # ---------------------------------------------------------
    bad_shape = np.zeros(
        128,
        dtype=np.float32
    )

    try:
        db.add_person(
            bad_shape
        )

        print(
            "[FAIL] Expected ValueError for "
            "wrong shape, none raised."
        )

    except ValueError as e:
        print(
            "[OK] Correctly raised ValueError "
            f"for wrong shape: {e}"
        )

    # Close first connection
    db.close()

    # ---------------------------------------------------------
    # 7. Reopen database to verify persistence
    # ---------------------------------------------------------
    db2 = GalleryDB(TEST_DB_PATH)

    print(
        "\nAfter reopening database:"
    )

    print(
        "is_empty() "
        f"(False expected — data should persist): "
        f"{db2.is_empty()}"
    )

    reloaded = db2.get_embedding(
        id_2
    )

    persisted_correctly = np.array_equal(
        reloaded,
        embedding_2
    )

    print(
        "Reloaded embedding matches original "
        f"(True expected): {persisted_correctly}"
    )

    db2.close()

    # ---------------------------------------------------------
    # 8. Cleanup
    # ---------------------------------------------------------
    os.remove(TEST_DB_PATH)

    print(
        "\nTest database file removed."
    )


if __name__ == "__main__":
    main()