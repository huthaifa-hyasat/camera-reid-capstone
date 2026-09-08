"""
Single source of truth for model names, paths, and thresholds.
"""

# --- Models ---

YOLO_WEIGHTS = "yolov8n.pt"
OSNET_MODEL_NAME = "osnet_x0_25"
DEVICE = "cpu"

# --- Decision thresholds ---

MATCH_THRESHOLD = 0.5800
REJECT_THRESHOLD = 0.5310