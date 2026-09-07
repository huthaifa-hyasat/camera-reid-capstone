import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import run_pipeline
import config


def main():
    input_video = os.path.join("demo", "input_video.mp4")
    output_video = os.path.join("demo", "final_output.mp4")
    db_path = os.path.join("data", "gallery.db")

    counts = run_pipeline(
        input_video_path=input_video,
        output_video_path=output_video,
        db_path=db_path,
        match_threshold=config.MATCH_THRESHOLD,
        reject_threshold=config.REJECT_THRESHOLD,
        frame_skip=1
    )

    if not os.path.exists(output_video):
        raise RuntimeError("Final output video was not created.")

    print("\nFinal pipeline test passed.")
    print(f"Output: {output_video}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()