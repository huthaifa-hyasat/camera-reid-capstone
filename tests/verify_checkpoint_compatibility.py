"""
Standalone validation for the OSNet x0_25 Market-1501 checkpoint.

This script:
1. Builds OSNet x0_25 without ImageNet pretrained weights.
2. Loads the downloaded Market-1501 checkpoint.
3. Compares checkpoint keys against model keys.
4. Reports matched and mismatched parameters.
5. Rejects unexpected backbone mismatches.
6. Runs one real person crop through the loaded model.
7. Verifies that a finite 512-dimensional embedding is produced.

This script does NOT modify:
- ReIDEmbedder
- matcher.py
- benchmark code
- pipeline.py
"""

import os
import sys

import cv2
import numpy as np
import torch
import torchreid


# ---------------------------------------------------------------------------
# Import the already-tested PersonDetector
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "detection"
    )
)

from person_detector import PersonDetector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(
    "models",
    "osnet_x0_25_market1501.pth"
)

SAMPLE_IMAGE_PATH = "sample.jpg"

MODEL_NAME = "osnet_x0_25"

DEVICE = "cpu"

OSNET_INPUT_HEIGHT = 256
OSNET_INPUT_WIDTH = 128

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]

# The classifier is expected to mismatch because the model is
# constructed with a placeholder num_classes value.
EXPECTED_MISMATCH_PREFIXES = (
    "classifier.",
)


def main():

    # -----------------------------------------------------------------------
    # Step 1: Build OSNet without ImageNet pretrained weights
    # -----------------------------------------------------------------------

    try:
        model = torchreid.models.build_model(
            name=MODEL_NAME,
            num_classes=1000,
            pretrained=False,
        )

        model = model.to(DEVICE)
        model.eval()

    except Exception as e:

        print(
            f"[FATAL] Failed to build {MODEL_NAME}: {e}"
        )

        return

    model_state = model.state_dict()

    print(
        f"[OK] Built {MODEL_NAME} with pretrained=False."
    )

    print(
        "Model state_dict entries: "
        f"{len(model_state)}"
    )

    print(
        "Total model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # -----------------------------------------------------------------------
    # Step 2: Verify checkpoint file exists
    # -----------------------------------------------------------------------

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        print(
            f"[FATAL] Checkpoint not found at "
            f"'{CHECKPOINT_PATH}'."
        )

        return

    print(
        f"[OK] Checkpoint found at "
        f"'{CHECKPOINT_PATH}'."
    )

    # -----------------------------------------------------------------------
    # Step 3: Load raw checkpoint
    # -----------------------------------------------------------------------

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

    except Exception as e:

        print(
            f"[FATAL] Could not load checkpoint with "
            f"torch.load(): {e}"
        )

        return

    # -----------------------------------------------------------------------
    # Step 4: Extract state dictionary
    # -----------------------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        checkpoint_state = checkpoint[
            "state_dict"
        ]

        print(
            "[OK] Found checkpoint['state_dict']."
        )

    elif isinstance(
        checkpoint,
        dict
    ):

        checkpoint_state = checkpoint

        print(
            "[OK] Checkpoint appears to be a raw state dict."
        )

    else:

        print(
            "[FATAL] Unexpected checkpoint object type: "
            f"{type(checkpoint)}"
        )

        return

    # -----------------------------------------------------------------------
    # Step 5: Normalize checkpoint keys
    # -----------------------------------------------------------------------

    def normalize_key(key):
        """
        Remove a possible DataParallel 'module.' prefix.
        """

        if key.startswith("module."):
            return key[7:]

        return key

    checkpoint_keys = {
        normalize_key(key): value
        for key, value in checkpoint_state.items()
    }

    # -----------------------------------------------------------------------
    # Step 6: Compare model and checkpoint keys
    # -----------------------------------------------------------------------

    matched = []

    shape_mismatched = []

    for key, checkpoint_tensor in checkpoint_keys.items():

        if key in model_state:

            if (
                model_state[key].shape
                ==
                checkpoint_tensor.shape
            ):

                matched.append(
                    key
                )

            else:

                shape_mismatched.append(
                    (
                        key,
                        tuple(model_state[key].shape),
                        tuple(checkpoint_tensor.shape)
                    )
                )

    missing_in_checkpoint = [
        key
        for key in model_state.keys()
        if key not in checkpoint_keys
    ]

    total_model_keys = len(
        model_state
    )

    match_percentage = (
        100.0 * len(matched) / total_model_keys
        if total_model_keys > 0
        else 0.0
    )

    print(
        "\n--- Key-Level Compatibility Report ---"
    )

    print(
        "Total model state_dict keys : "
        f"{total_model_keys}"
    )

    print(
        "Matched (name + shape)      : "
        f"{len(matched)} "
        f"({match_percentage:.1f}%)"
    )

    print(
        "Shape-mismatched keys       : "
        f"{len(shape_mismatched)}"
    )

    print(
        "Missing from checkpoint     : "
        f"{len(missing_in_checkpoint)}"
    )

    # -----------------------------------------------------------------------
    # Step 7: Print shape mismatches
    # -----------------------------------------------------------------------

    if shape_mismatched:

        print(
            "\nShape-mismatched keys:"
        )

        for (
            key,
            model_shape,
            checkpoint_shape
        ) in shape_mismatched:

            print(
                f"  {key}: "
                f"model={model_shape} "
                f"vs checkpoint={checkpoint_shape}"
            )

    # -----------------------------------------------------------------------
    # Step 8: Print missing keys
    # -----------------------------------------------------------------------

    if missing_in_checkpoint:

        print(
            "\nModel keys with no checkpoint counterpart:"
        )

        for key in missing_in_checkpoint:

            print(
                f"  {key}"
            )

    # -----------------------------------------------------------------------
    # Step 9: Identify unexpected mismatches
    # -----------------------------------------------------------------------

    all_problem_keys = (
        [
            key
            for key, _, _
            in shape_mismatched
        ]
        +
        missing_in_checkpoint
    )

    unexpected_problems = [
        key
        for key in all_problem_keys
        if not key.startswith(
            EXPECTED_MISMATCH_PREFIXES
        )
    ]

    # -----------------------------------------------------------------------
    # Step 10: Stop if nothing matches
    # -----------------------------------------------------------------------

    if len(matched) == 0:

        print(
            "\n[FATAL] Zero model keys matched "
            "the checkpoint."
        )

        print(
            "The checkpoint is incompatible "
            "with this model build."
        )

        return

    # -----------------------------------------------------------------------
    # Step 11: Stop if unexpected backbone mismatches exist
    # -----------------------------------------------------------------------

    if unexpected_problems:

        print(
            "\n[FATAL] Unexpected mismatches found "
            "outside the classifier head:"
        )

        for key in unexpected_problems:

            print(
                f"  {key}"
            )

        print(
            "\nThe checkpoint should not be trusted "
            "with this model build."
        )

        return

    # -----------------------------------------------------------------------
    # Step 12: Compatibility passed
    # -----------------------------------------------------------------------

    print(
        "\n[OK] All mismatches are confined to "
        "the expected classifier head."
    )

    print(
        "The OSNet backbone appears compatible "
        "with the checkpoint."
    )

    # -----------------------------------------------------------------------
    # Step 13: Actually load Market-1501 weights
    # -----------------------------------------------------------------------

    try:

        torchreid.utils.load_pretrained_weights(
            model,
            CHECKPOINT_PATH
        )

        model.eval()

    except Exception as e:

        print(
            "\n[FATAL] Failed to apply checkpoint "
            f"using load_pretrained_weights(): {e}"
        )

        return

    print(
        "\n[OK] Market-1501 checkpoint loaded "
        "successfully into OSNet."
    )

    # -----------------------------------------------------------------------
    # Step 14: Load sample image
    # -----------------------------------------------------------------------

    frame = cv2.imread(
        SAMPLE_IMAGE_PATH
    )

    if frame is None:

        print(
            f"[FATAL] Could not load "
            f"'{SAMPLE_IMAGE_PATH}'."
        )

        return

    print(
        f"[OK] Loaded sample image: "
        f"{SAMPLE_IMAGE_PATH}"
    )

    # -----------------------------------------------------------------------
    # Step 15: Detect a person using the tested detector
    # -----------------------------------------------------------------------

    try:

        detector = PersonDetector(
            weights_path="yolov8n.pt",
            device=DEVICE
        )

        detections = detector.detect(
            frame
        )

    except Exception as e:

        print(
            f"[FATAL] Person detection failed: {e}"
        )

        return

    if not detections:

        print(
            "[FATAL] No person detected "
            "in sample image."
        )

        return

    # -----------------------------------------------------------------------
    # Step 16: Select largest detected person
    # -----------------------------------------------------------------------

    def area(detection):

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        return (
            (x2 - x1)
            *
            (y2 - y1)
        )

    best_detection = max(
        detections,
        key=area
    )

    x1, y1, x2, y2 = (
        best_detection["bbox"]
    )

    crop_bgr = frame[
        y1:y2,
        x1:x2
    ]

    if crop_bgr.size == 0:

        print(
            "[FATAL] Selected person crop is empty."
        )

        return

    print(
        f"[OK] Person crop shape: "
        f"{crop_bgr.shape}"
    )

    # -----------------------------------------------------------------------
    # Step 17: Preprocess crop
    # -----------------------------------------------------------------------

    import torchvision.transforms as T

    preprocess = T.Compose([
        T.ToPILImage(),

        T.Resize(
            (
                OSNET_INPUT_HEIGHT,
                OSNET_INPUT_WIDTH
            )
        ),

        T.ToTensor(),

        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    crop_rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = preprocess(
        crop_rgb
    ).unsqueeze(0).to(
        DEVICE
    )

    # -----------------------------------------------------------------------
    # Step 18: Forward pass
    # -----------------------------------------------------------------------

    try:

        with torch.no_grad():

            output = model(
                input_tensor
            )

    except Exception as e:

        print(
            f"[FATAL] OSNet forward pass failed: {e}"
        )

        return

    # -----------------------------------------------------------------------
    # Step 19: Handle possible multiple outputs
    # -----------------------------------------------------------------------

    if isinstance(
        output,
        (tuple, list)
    ):

        output = output[0]

    # -----------------------------------------------------------------------
    # Step 20: Convert embedding to numpy
    # -----------------------------------------------------------------------

    embedding = (
        output
        .detach()
        .cpu()
        .numpy()[0]
    )

    is_finite = bool(
        np.isfinite(
            embedding
        ).all()
    )

    # -----------------------------------------------------------------------
    # Step 21: Final verification
    # -----------------------------------------------------------------------

    print(
        "\n--- Sanity Forward Pass ---"
    )

    print(
        f"Embedding shape : "
        f"{embedding.shape}"
    )

    print(
        f"Embedding dtype : "
        f"{embedding.dtype}"
    )

    print(
        f"All finite      : "
        f"{is_finite}"
    )

    if embedding.shape != (
        512,
    ):

        print(
            "[FAIL] Expected embedding shape "
            "(512)."
        )

        return

    if not is_finite:

        print(
            "[FAIL] Embedding contains "
            "NaN or Inf values."
        )

        return

    print(
        "\n[CHECKPOINT COMPATIBILITY "
        "VALIDATION COMPLETE — PASSED]"
    )


if __name__ == "__main__":
    main()