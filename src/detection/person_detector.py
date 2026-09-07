from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cpu"
    ):
        """
        Load the YOLOv8n model once.
        """
        self.device = device

        try:
            self._model = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO weights from '{weights_path}': {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run person detection on a single frame.

        Returns:
            A list of dictionaries containing:
            - bbox
            - confidence
            - class_id
            - class_name
        """
        self._validate_frame(frame)

        results = self._model.predict(
            source=frame,
            device=self.device,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        boxes = results[0].boxes
        detections: List[Dict[str, Any]] = []

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, cls_id in zip(
            xyxy, confs, cls_ids
        ):
            detections.append({
                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ),
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": "person",
            })

        return detections

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """
        Validate OpenCV-style BGR frame.
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"Input frame must be a numpy array, got {type(frame)}."
            )

        if frame.size == 0:
            raise ValueError("Input frame is empty.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Input frame must have shape (H, W, 3), "
                f"got shape {frame.shape}."
            )