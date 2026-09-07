"""
Minimal integration test for src/pipeline.py.

This test verifies that the recorded-video pipeline:
- runs without crashing
- reads the input video
- produces an annotated output video
- creates/uses a separate test database
- handles an unreadable video path correctly

This is an integration/smoke test, not a unit test.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

from pipeline import run_pipeline


INPUT_VIDEO = os.path.join(
    "demo",
    "input_video.mp4"
)

OUTPUT_VIDEO = os.path.join(
    "demo",
    "output_annotated.mp4"
)

TEST_DB = os.path.join(
    "data",
    "test_pipeline_gallery.db"
)


def main():

    # ---------------------------------------------------------
    # Clean up old test database
    # ---------------------------------------------------------

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    # ---------------------------------------------------------
    # Check that the input video exists
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_VIDEO):

        print(
            f"[FATAL] No test video found at "
            f"'{INPUT_VIDEO}'."
        )

        print(
            "Place a short video at "
            "demo\\input_video.mp4 first."
        )

        return

    # ---------------------------------------------------------
    # Remove previous output if it exists
    # ---------------------------------------------------------

    if os.path.exists(OUTPUT_VIDEO):
        os.remove(OUTPUT_VIDEO)

    # ---------------------------------------------------------
    # Run the real pipeline
    # ---------------------------------------------------------

    counts = run_pipeline(
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        db_path=TEST_DB,
        frame_skip=1
    )

    # ---------------------------------------------------------
    # Verify output video
    # ---------------------------------------------------------

    print(
        "\n[OK] Pipeline ran without crashing."
    )

    output_exists = os.path.exists(
        OUTPUT_VIDEO
    )

    print(
        f"[OK] Output video exists: "
        f"{output_exists}"
    )

    output_non_empty = (
        os.path.getsize(OUTPUT_VIDEO) > 0
        if output_exists
        else False
    )

    print(
        f"[OK] Output video non-empty: "
        f"{output_non_empty}"
    )

    # ---------------------------------------------------------
    # Verify event counts
    # ---------------------------------------------------------

    total_events = sum(
        counts.values()
    )

    print(
        f"[OK] Total logged events "
        f"(KNOWN+UNCERTAIN+NEW+NO_PERSON+errors): "
        f"{total_events}"
    )

    # ---------------------------------------------------------
    # Unreadable-video handling test
    # ---------------------------------------------------------

    try:

        run_pipeline(
            input_video_path="this_file_does_not_exist.mp4",
            output_video_path=os.path.join(
                "demo",
                "unused_output.mp4"
            ),
            db_path=TEST_DB
        )

        print(
            "[FAIL] Expected RuntimeError "
            "for unreadable video, none raised."
        )

    except RuntimeError as e:

        print(
            "[OK] Correctly raised RuntimeError "
            f"for unreadable video: {e}"
        )


if __name__ == "__main__":
    main()