import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from persistence.db import GalleryDB


TEST_DB_PATH = "test_sessions.db"


def main():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    db = GalleryDB(TEST_DB_PATH)

    session_id = db.create_session()
    print(f"Created session: {session_id}")

    event_id = db.add_visit_event(
        session_id=session_id,
        person_id=None,
        status="NEW",
        similarity=0.42
    )
    print(f"Created visit event: {event_id}")

    db.end_session(session_id)
    print("Session ended successfully.")

    db.close()

    db = GalleryDB(TEST_DB_PATH)

    sessions = db.conn.execute(
        "SELECT session_id, started_at, ended_at FROM sessions;"
    ).fetchall()

    events = db.conn.execute(
        "SELECT event_id, session_id, person_id, status, similarity FROM visit_events;"
    ).fetchall()

    print(f"Sessions stored: {len(sessions)}")
    print(f"Visit events stored: {len(events)}")

    db.close()
    os.remove(TEST_DB_PATH)

    print("Test database removed.")


if __name__ == "__main__":
    main()