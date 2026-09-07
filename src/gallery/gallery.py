from typing import Dict, List, Tuple

import numpy as np

import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "matching"
    )
)

from matcher import _validate_embedding


class InMemoryGallery:
    def __init__(self):
        self._entries: Dict[int, np.ndarray] = {}
        self._next_id: int = 1

    def add_person(self, embedding: np.ndarray) -> int:
        """
        Add a new person to the gallery with a freshly assigned person_id.
        """

        _validate_embedding(
            embedding,
            name="embedding"
        )

        person_id = self._next_id

        self._entries[person_id] = embedding.copy()

        self._next_id += 1

        return person_id

    def get_all(
        self
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Retrieve all gallery entries.

        Returns:
            person_ids and embeddings in insertion order.
        """

        person_ids = list(
            self._entries.keys()
        )

        embeddings = [
            self._entries[person_id]
            for person_id in person_ids
        ]

        return person_ids, embeddings

    def is_empty(self) -> bool:
        """
        Return True if the gallery contains no entries.
        """

        return len(self._entries) == 0

    def get_embedding(
        self,
        person_id: int
    ) -> np.ndarray:
        """
        Retrieve the embedding for a specific person_id.
        """

        if person_id not in self._entries:
            raise KeyError(
                f"No gallery entry found for person_id={person_id}."
            )

        return self._entries[person_id]

    def __len__(self) -> int:
        return len(self._entries)