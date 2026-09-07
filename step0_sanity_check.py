import time
import sys

import numpy as np
import cv2
import torch
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_IMAGE_PATH = "sample.jpg"
DEVICE = "cpu"
YOLO_WEIGHTS = "yolov8n.pt"
OSNET_MODEL_NAME = "osnet_x0_25"
OSNET_INPUT_HEIGHT = 256   # standard re-id input size
OSNET_INPUT_WIDTH = 128
# ImageNet normalization stats — appropriate here because the loaded
# weights are the ImageNet-pretrained backbone (see note above).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    # -----------------------------------------------------------------
    # Step 1: Load the image
    # -----------------------------------------------------------------
    frame = cv2.imread(SAMPLE_IMAGE_PATH)
    if frame is None:
        print(f"[FATAL] Could not load image at '{SAMPLE_IMAGE_PATH}'.")
        sys.exit(1)

    print(f"[OK] Loaded image: {SAMPLE_IMAGE_PATH}  shape={frame.shape}")

    # -----------------------------------------------------------------
    # Step 2: Run YOLOv8n person detection
    # -----------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FATAL] 'ultralytics' is not installed in this environment.")
        sys.exit(1)

    detector = YOLO(YOLO_WEIGHTS)

    t0 = time.perf_counter()
    results = detector.predict(source=frame, device=DEVICE, classes=[0], verbose=False)
    detection_time = time.perf_counter() - t0

    boxes = results[0].boxes
    person_count = 0 if boxes is None else len(boxes)

    print(f"[OK] Detection inference time: {detection_time:.4f} s")
    print(f"[OK] Detected person count: {person_count}")

    if person_count == 0:
        print("[FATAL] No person detected in the sample image.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Step 3: Select the largest-area detected person and crop
    # -----------------------------------------------------------------
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = xyxy[best_idx].astype(int)

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop_bgr = frame[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        print("[FATAL] Selected crop is empty after clamping to frame bounds.")
        sys.exit(1)

    print(f"[OK] Crop dimensions (H, W, C): {crop_bgr.shape}")

    # -----------------------------------------------------------------
    # Step 4: Build OSNet via torchreid.models.build_model (0.2.5 API)
    # -----------------------------------------------------------------
    try:
        import torchreid
    except ImportError:
        print("[FATAL] 'torchreid' is not installed in this environment.")
        sys.exit(1)

    try:
        # num_classes is required to construct the model's classifier head,
        # but is irrelevant to the embedding output: torchreid classification
        # -style models (including OSNet) return pooled *features*, not
        # classifier logits, when the model is in eval() mode. The value
        # below is an unused placeholder for that reason.
        model = torchreid.models.build_model(
            name=OSNET_MODEL_NAME,
            num_classes=1000,
            pretrained=True,
        )
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"[FATAL] Failed to build/load OSNet model '{OSNET_MODEL_NAME}': {e}")
        sys.exit(1)

    print(f"[OK] Built OSNet model: {OSNET_MODEL_NAME} on device={DEVICE} (eval mode)")
    print("[NOTE] Weights source: ImageNet-pretrained backbone (build_model pretrained=True).")
    print("       NOT the Market-1501-trained checkpoint planned for the final pipeline.")

    # -----------------------------------------------------------------
    # Step 5: Preprocess crop for OSNet input
    # -----------------------------------------------------------------
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((OSNET_INPUT_HEIGHT, OSNET_INPUT_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = preprocess(crop_rgb).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    # -----------------------------------------------------------------
    # Step 6: Run inference to produce the embedding
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    embedding_time = time.perf_counter() - t0

    print(f"[OK] Embedding inference time: {embedding_time:.4f} s")

    if isinstance(output, (tuple, list)):
        # Defensive: some model modes can return (features, logits); take features.
        print("[NOTE] Model returned multiple outputs; using the first as the embedding.")
        output = output[0]

    embedding_np = output.detach().cpu().numpy()
    embedding_vec = embedding_np[0]

    # -----------------------------------------------------------------
    # Step 7: Report embedding properties
    # -----------------------------------------------------------------
    l2_norm = float(np.linalg.norm(embedding_vec))
    is_finite = bool(np.isfinite(embedding_vec).all())

    print("\n--- Embedding Report ---")
    print(f"Embedding shape          : {embedding_np.shape}")
    print(f"Embedding dtype          : {embedding_np.dtype}")
    print(f"Embedding L2 norm        : {l2_norm:.6f}")
    print(f"Explicit L2 normalization applied: NO")
    print("  (This script reports the RAW model output. No normalization")
    print("   step was applied here. Whether/where to L2-normalize is a")
    print("   deliberate design decision for the real embedding module,")
    print("   not assumed in this sanity check.)")
    print(f"All values finite         : {is_finite}")

    print("\n--- Summary ---")
    print(f"Detected person count    : {person_count}")
    print(f"Crop dimensions          : {crop_bgr.shape}")
    print(f"Embedding shape          : {embedding_np.shape}")
    print(f"Embedding dtype          : {embedding_np.dtype}")
    print(f"Embedding L2 norm        : {l2_norm:.6f}")
    print(f"Detection inference time : {detection_time:.4f} s")
    print(f"Embedding inference time : {embedding_time:.4f} s")
    print(f"Finite check passed      : {is_finite}")
    print(f"Weights used             : ImageNet-pretrained backbone (not Market-1501)")

    if not is_finite:
        print("\n[FATAL] Embedding contains non-finite values (NaN/Inf). Stop and diagnose.")
        sys.exit(1)

    print("\n[STEP 0 SANITY CHECK COMPLETE]")


if __name__ == "__main__":
    main()