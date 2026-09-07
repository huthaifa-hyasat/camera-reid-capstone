"""
Standalone verification for the downloaded osnet_x0_25 Market-1501
checkpoint.

This script does NOT modify ReIDEmbedder or build any model wiring.
It only verifies:
- file existence
- plausible file size
- that the file is not an HTML page
- that torch.load() can parse it
"""

import os
import torch


CHECKPOINT_PATH = os.path.join(
    "models",
    "osnet_x0_25_market1501.pth"
)

MIN_EXPECTED_SIZE_BYTES = 100_000


def main():

    # ---------------------------------------------------------
    # 1. File existence
    # ---------------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        print(
            f"[FAIL] File does not exist at "
            f"'{CHECKPOINT_PATH}'."
        )
        return

    print(
        f"[OK] File exists at "
        f"'{CHECKPOINT_PATH}'."
    )

    # ---------------------------------------------------------
    # 2. File size
    # ---------------------------------------------------------

    size_bytes = os.path.getsize(
        CHECKPOINT_PATH
    )

    print(
        f"File size: {size_bytes:,} bytes"
    )

    if size_bytes < MIN_EXPECTED_SIZE_BYTES:
        print(
            f"[FAIL] File is suspiciously small "
            f"(< {MIN_EXPECTED_SIZE_BYTES:,} bytes)."
        )
        print(
            "This is likely an HTML error page or "
            "a failed/truncated download."
        )
        return

    print(
        "[OK] File size looks plausible "
        "for a checkpoint."
    )

    # ---------------------------------------------------------
    # 3. Check for HTML content
    # ---------------------------------------------------------

    with open(
        CHECKPOINT_PATH,
        "rb"
    ) as file:

        header_bytes = file.read(256)

    header_lower = header_bytes.lower()

    if (
        b"<!doctype html" in header_lower
        or b"<html" in header_lower
    ):

        print(
            "[FAIL] File content looks like an HTML page, "
            "not a binary checkpoint."
        )

        print(
            "Re-download the checkpoint from the "
            "official source."
        )

        return

    print(
        "[OK] File does not look like an HTML page."
    )

    # ---------------------------------------------------------
    # 4. Try loading checkpoint with PyTorch
    # ---------------------------------------------------------

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu"
        )

    except Exception as e:

        print(
            "[FAIL] torch.load() could not parse "
            f"this file as a checkpoint: {e}"
        )

        return

    print(
        "[OK] torch.load() successfully parsed "
        "the file."
    )

    # ---------------------------------------------------------
    # 5. Inspect checkpoint structure
    # ---------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        print(
            f"Checkpoint is a dict with "
            f"{len(checkpoint)} top-level key(s)."
        )

        if "state_dict" in checkpoint:

            print(
                "Found 'state_dict' key with "
                f"{len(checkpoint['state_dict'])} entries."
            )

        else:

            print(
                "No 'state_dict' key found."
            )

            print(
                "Checkpoint may be a raw state dict itself."
            )

    else:

        print(
            f"[NOTE] Checkpoint object type: "
            f"{type(checkpoint)}"
        )

    print(
        "\n[CHECKPOINT VERIFICATION COMPLETE]"
    )


if __name__ == "__main__":
    main()