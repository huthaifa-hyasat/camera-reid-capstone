import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from detection.person_detector import PersonDetector


def main():
    image_path = "sample.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[FATAL] Could not load test image at '{image_path}'.")
        return

    detector = PersonDetector(
        weights_path="yolov8n.pt",
        device="cpu"
    )

    detections = detector.detect(frame)

    print(f"Detected {len(detections)} person(s).")

    for i, det in enumerate(detections):
        print(
            f"  [{i}] "
            f"bbox={det['bbox']} "
            f"confidence={det['confidence']:.4f} "
            f"class={det['class_name']}"
        )

    # Invalid-input handling check
    try:
        detector.detect(None)
        print("[FAIL] Expected ValueError for None frame, but none was raised.")
    except ValueError as e:
        print(f"[OK] Correctly raised ValueError for invalid frame: {e}")


if __name__ == "__main__":
    main()