"""
Single source of truth for model names, paths, and thresholds.
"""

# --- Models ---

YOLO_WEIGHTS = "yolov8n.pt"
OSNET_MODEL_NAME = "osnet_x0_25"
DEVICE = "cpu"


# --- Decision thresholds ---

# PROVISIONAL / UNCALIBRATED VALUES.
# These are placeholders only.
# They were NOT derived from Market-1501 yet.
# Do not report them as validated thresholds.

MATCH_THRESHOLD = 0.75
REJECT_THRESHOLD = 0.55