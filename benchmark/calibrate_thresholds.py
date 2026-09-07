import os
import re

import numpy as np


RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

EMBEDDINGS_PATH = os.path.join(
    RESULTS_DIR,
    "market1501_embeddings.npz"
)

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "market1501",
    "Market-1501-v15.09.15"
)

QUERY_DIR = os.path.join(DATASET_ROOT, "query")
GALLERY_DIR = os.path.join(DATASET_ROOT, "bounding_box_test")


def parse_filename(filename):
    pattern = r"^(-?\d+)_c(\d+)s(\d+)_(\d+)_([\d]+)\.jpg$"
    match = re.match(pattern, filename)

    if match is None:
        raise ValueError(f"Invalid filename: {filename}")

    return {
        "person_id": int(match.group(1)),
        "camera_id": int(match.group(2))
    }


def load_records(directory):
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".jpg")
    )

    records = [
        parse_filename(os.path.basename(path))
        for path in paths
    ]

    return paths, records


def main():
    data = np.load(EMBEDDINGS_PATH)

    query_embeddings = data["query_embeddings"]
    gallery_embeddings = data["gallery_embeddings"]

    _, query_records = load_records(QUERY_DIR)
    _, gallery_records = load_records(GALLERY_DIR)

    genuine_scores = []
    impostor_scores = []

    gallery_embeddings = gallery_embeddings.astype(np.float32)
    query_embeddings = query_embeddings.astype(np.float32)

    gallery_ids = np.array(
        [record["person_id"] for record in gallery_records]
    )

    gallery_cams = np.array(
        [record["camera_id"] for record in gallery_records]
    )

    for i, query in enumerate(query_records):
        scores = gallery_embeddings @ query_embeddings[i]

        for j, score in enumerate(scores):
            gallery_id = gallery_ids[j]
            gallery_cam = gallery_cams[j]

            if gallery_id in {-1, 0}:
                continue

            if gallery_id == query["person_id"]:
                if gallery_cam != query["camera_id"]:
                    genuine_scores.append(float(score))
            else:
                impostor_scores.append(float(score))

    genuine_scores = np.asarray(genuine_scores, dtype=np.float32)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float32)

    thresholds = np.linspace(0.0, 1.0, 1001)

    false_reject = np.array([
        np.mean(genuine_scores < threshold)
        for threshold in thresholds
    ])

    false_accept = np.array([
        np.mean(impostor_scores >= threshold)
        for threshold in thresholds
    ])

    eer_index = int(
        np.argmin(np.abs(false_reject - false_accept))
    )

    eer_threshold = thresholds[eer_index]
    eer = (
        false_reject[eer_index] +
        false_accept[eer_index]
    ) / 2

    target_far = 0.01

    valid_thresholds = thresholds[
        false_accept <= target_far
    ]

    match_threshold = float(valid_thresholds[0])

    reject_threshold = float(
        thresholds[
            np.argmin(
                np.abs(
                    false_reject -
                    false_accept
                )
            )
        ]
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        "threshold_calibration.txt"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            f"Genuine pairs: {len(genuine_scores)}\n"
        )
        file.write(
            f"Impostor pairs: {len(impostor_scores)}\n"
        )
        file.write(
            f"EER threshold: {eer_threshold:.6f}\n"
        )
        file.write(
            f"EER: {eer:.6f}\n"
        )
        file.write(
            f"Match threshold: {match_threshold:.6f}\n"
        )
        file.write(
            f"Reject threshold: {reject_threshold:.6f}\n"
        )

    print("\n--- Threshold Calibration ---")
    print(f"Genuine pairs : {len(genuine_scores)}")
    print(f"Impostor pairs: {len(impostor_scores)}")
    print(f"EER threshold : {eer_threshold:.4f}")
    print(f"EER            : {eer:.4f}")
    print(f"Match threshold: {match_threshold:.4f}")
    print(f"Reject threshold: {reject_threshold:.4f}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()