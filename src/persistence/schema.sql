CREATE TABLE IF NOT EXISTS gallery (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding BLOB NOT NULL,
    first_seen TEXT NOT NULL,
    label TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS visit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    person_id INTEGER,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    similarity REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (person_id) REFERENCES gallery(person_id)
);