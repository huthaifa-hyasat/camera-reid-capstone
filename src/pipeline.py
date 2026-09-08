import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detection"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "embedding"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "persistence"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "decision"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from person_detector import PersonDetector
from reid_embedder import ReIDEmbedder
from db import GalleryDB
from decision import decide
import config

def select_primary_person(detections, frame_shape):
    if not detections:
        return None

    height, width = frame_shape[:2]
    center = np.array([width / 2.0, height / 2.0])

    def area(d):
        x1, y1, x2, y2 = d["bbox"]
        return (x2 - x1) * (y2 - y1)

    def distance(d):
        x1, y1, x2, y2 = d["bbox"]
        person_center = np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ])
        return float(np.linalg.norm(person_center - center))

    max_area = max(area(d) for d in detections)
    candidates = [d for d in detections if area(d) == max_area]

    return candidates[0] if len(candidates) == 1 else min(
        candidates,
        key=distance
    )


def annotate_frame(frame, status, person_id=None, similarity=None, bbox=None):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines = [f"status: {status}"]

    if person_id is not None:
        lines.append(f"person_id: {person_id}")

    if similarity is not None:
        lines.append(f"similarity: {similarity:.4f}")

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame


def run_pipeline(
    input_video_path: str,
    output_video_path: str,
    db_path: str,
    match_threshold: float = config.MATCH_THRESHOLD,
    reject_threshold: float = config.REJECT_THRESHOLD,
    frame_skip: int = 1
):
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1.")

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: '{input_video_path}'."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open video writer for '{output_video_path}'."
        )

    detector = PersonDetector(
        weights_path=config.YOLO_WEIGHTS,
        device=config.DEVICE
    )

    embedder = ReIDEmbedder(
        model_name=config.OSNET_MODEL_NAME,
        device=config.DEVICE
    )

    db = GalleryDB(db_path)
    session_id = db.create_session()

    person_ids = []
    embeddings = []

    for person_id, embedding, _, _ in db.get_all():
        person_ids.append(person_id)
        embeddings.append(embedding)

    counts = {
        "KNOWN": 0,
        "UNCERTAIN": 0,
        "NEW": 0,
        "NO_PERSON": 0,
        "EMBED_ERROR": 0,
        "DETECT_ERROR": 0
    }

    frames_read = 0
    frames_processed = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            frames_read += 1

            if (frames_read - 1) % frame_skip != 0:
                writer.write(frame)
                continue

            frames_processed += 1

            try:
                detections = detector.detect(frame)
            except (ValueError, RuntimeError):
                counts["DETECT_ERROR"] += 1
                db.add_visit_event(
                    session_id,
                    None,
                    "DETECT_ERROR"
                )
                annotate_frame(frame, "DETECT_ERROR")
                writer.write(frame)
                continue

            primary = select_primary_person(
                detections,
                frame.shape
            )

            if primary is None:
                counts["NO_PERSON"] += 1
                db.add_visit_event(
                    session_id,
                    None,
                    "NO_PERSON"
                )
                annotate_frame(frame, "NO_PERSON")
                writer.write(frame)
                continue

            x1, y1, x2, y2 = primary["bbox"]
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                counts["EMBED_ERROR"] += 1
                db.add_visit_event(
                    session_id,
                    None,
                    "EMBED_ERROR"
                )
                annotate_frame(
                    frame,
                    "EMBED_ERROR",
                    bbox=primary["bbox"]
                )
                writer.write(frame)
                continue

            try:
                query_embedding = embedder.embed(crop)
            except (ValueError, RuntimeError):
                counts["EMBED_ERROR"] += 1
                db.add_visit_event(
                    session_id,
                    None,
                    "EMBED_ERROR"
                )
                annotate_frame(
                    frame,
                    "EMBED_ERROR",
                    bbox=primary["bbox"]
                )
                writer.write(frame)
                continue

            result = decide(
                query_embedding,
                person_ids,
                embeddings,
                match_threshold,
                reject_threshold
            )

            if result.status == "NEW":
                display_person_id = db.add_person(query_embedding)
                person_ids.append(display_person_id)
                embeddings.append(query_embedding)
            else:
                display_person_id = result.person_id

            counts[result.status] += 1

            db.add_visit_event(
                session_id=session_id,
                person_id=display_person_id,
                status=result.status,
                similarity=result.similarity
            )

            annotate_frame(
                frame,
                result.status,
                display_person_id,
                result.similarity,
                primary["bbox"]
            )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        db.end_session(session_id)
        db.close()

    print("\n--- Pipeline Run Summary ---")
    print(f"Frames read      : {frames_read}")
    print(f"Frames processed : {frames_processed}")

    for key, value in counts.items():
        print(f"{key:12s}: {value}")

    return counts


if __name__ == "__main__":
    run_pipeline(
        input_video_path=os.path.join(
            "demo",
            "input_video.mp4"
        ),
        output_video_path=os.path.join(
            "demo",
            "output_annotated.mp4"
        ),
        db_path=os.path.join(
            "data",
            "gallery.db"
        ),
        frame_skip=1
    )