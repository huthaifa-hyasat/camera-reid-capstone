import os
import sqlite3
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "matching"))
from matcher import _validate_embedding

EXPECTED_DIM = 512
EXPECTED_DTYPE = np.float32
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class GalleryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self._init_schema()
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not initialize database: {e}") from e

    def _init_schema(self):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            schema = file.read()
        self.conn.executescript(schema)
        self.conn.commit()

    def add_person(self, embedding: np.ndarray, label: Optional[str] = None) -> int:
        _validate_embedding(embedding, name="embedding")
        cursor = self.conn.execute(
            "INSERT INTO gallery (embedding, first_seen, label) VALUES (?, ?, ?);",
            (embedding.tobytes(), datetime.now().isoformat(), label)
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get_all(self) -> List[Tuple[int, np.ndarray, str, Optional[str]]]:
        rows = self.conn.execute(
            "SELECT person_id, embedding, first_seen, label FROM gallery ORDER BY person_id ASC;"
        ).fetchall()
        return [
            (int(person_id), self._deserialize_embedding(blob), first_seen, label)
            for person_id, blob, first_seen, label in rows
        ]

    def get_embedding(self, person_id: int) -> np.ndarray:
        row = self.conn.execute(
            "SELECT embedding FROM gallery WHERE person_id = ?;",
            (person_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Person ID {person_id} was not found.")
        return self._deserialize_embedding(row[0])

    def is_empty(self) -> bool:
        row = self.conn.execute("SELECT COUNT(*) FROM gallery;").fetchone()
        return row[0] == 0

    def create_session(self) -> int:
        started_at = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?);",
            (started_at,)
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def end_session(self, session_id: int):
        ended_at = datetime.now().isoformat()
        cursor = self.conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?;",
            (ended_at, session_id)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Session ID {session_id} was not found.")
        self.conn.commit()

    def add_visit_event(
        self,
        session_id: int,
        person_id: Optional[int],
        status: str,
        similarity: Optional[float] = None
    ) -> int:
        timestamp = datetime.now().isoformat()
        cursor = self.conn.execute(
            """
            INSERT INTO visit_events
                (session_id, person_id, timestamp, status, similarity)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                session_id,
                person_id,
                timestamp,
                status,
                similarity
            )
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        embedding = np.frombuffer(blob, dtype=EXPECTED_DTYPE)
        if embedding.shape != (EXPECTED_DIM,):
            raise RuntimeError("Invalid embedding stored in database.")
        return embedding.copy()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()