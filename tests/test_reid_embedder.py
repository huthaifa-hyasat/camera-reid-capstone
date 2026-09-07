import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from detection.person_detector import PersonDetector
from embedding.reid_embedder import ReIDEmbedder


def main():
    image_path = "sample.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[FATAL] Could not load test image at '{image_path}'.")
        return

    # Use the already verified detector only to obtain a real person crop.
    detector = PersonDetector(
        weights_path="yolov8n.pt",
        device="cpu"
    )

    detections = detector.detect(frame)

    if not detections:
        print("[FATAL] No person detected; cannot test embedding module.")
        return

    # Select the largest detected person.
    def area(detection):
        x1, y1, x2, y2 = detection["bbox"]
        return (x2 - x1) * (y2 - y1)

    best = max(detections, key=area)

    x1, y1, x2, y2 = best["bbox"]

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        print("[FATAL] Selected crop is empty.")
        return

    # Create the Re-ID embedder.
    embedder = ReIDEmbedder(
        model_name="osnet_x0_25",
        device="cpu"
    )

    # Generate embedding.
    embedding = embedder.embed(crop)

    print(f"Embedding shape  : {embedding.shape}")
    print(f"Embedding dtype  : {embedding.dtype}")
    print(f"Embedding L2 norm: {np.linalg.norm(embedding):.6f}")
    print(f"All finite       : {np.isfinite(embedding).all()}")

    # Invalid input test: None
    try:
        embedder.embed(None)
        print("[FAIL] Expected ValueError for None crop, but none was raised.")
    except ValueError as e:
        print(f"[OK] Correctly raised ValueError for None crop: {e}")

    # Invalid input test: empty crop
    try:
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        embedder.embed(empty_crop)
        print("[FAIL] Expected ValueError for empty crop, but none was raised.")
    except ValueError as e:
        print(f"[OK] Correctly raised ValueError for empty crop: {e}")


if __name__ == "__main__":
    main()