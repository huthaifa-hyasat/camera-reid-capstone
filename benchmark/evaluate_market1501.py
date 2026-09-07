import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "embedding"))

from reid_embedder import ReIDEmbedder
from market1501_dataset import Market1501Dataset


DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "market1501",
    "Market-1501-v15.09.15"
)

QUERY_DIR = os.path.join(DATASET_ROOT, "query")
GALLERY_DIR = os.path.join(DATASET_ROOT, "bounding_box_test")
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)


def extract_embedding(embedder, image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    return embedder.embed(image)


def load_embeddings(embedder, image_paths, name):
    embeddings = []

    total = len(image_paths)

    for i, image_path in enumerate(image_paths, start=1):
        embedding = extract_embedding(embedder, image_path)
        embeddings.append(embedding)

        if i % 100 == 0 or i == total:
            print(f"{name}: {i}/{total}")

    return np.asarray(embeddings, dtype=np.float32)


def average_precision(matches):
    matches = np.asarray(matches, dtype=np.int32)

    total_positive = int(matches.sum())

    if total_positive == 0:
        return 0.0

    cumulative = np.cumsum(matches)
    positions = np.arange(1, len(matches) + 1)

    precision = cumulative / positions
    return float((precision * matches).sum() / total_positive)


def evaluate(query_records, gallery_records, query_embeddings, gallery_embeddings):
    rank1 = 0
    rank5 = 0
    average_precisions = []

    gallery_ids = np.array(
        [record["person_id"] for record in gallery_records],
        dtype=np.int32
    )

    gallery_cams = np.array(
        [record["camera_id"] for record in gallery_records],
        dtype=np.int32
    )

    for i, query in enumerate(query_records):
        scores = gallery_embeddings @ query_embeddings[i]
        order = np.argsort(-scores)

        valid_order = []

        for index in order:
            gallery_id = gallery_ids[index]
            gallery_cam = gallery_cams[index]

            if gallery_id in {-1, 0}:
                continue

            if (
                gallery_id == query["person_id"]
                and gallery_cam == query["camera_id"]
            ):
                continue

            valid_order.append(index)

        valid_order = np.asarray(valid_order, dtype=np.int32)

        matches = (
            gallery_ids[valid_order] == query["person_id"]
        ).astype(np.int32)

        if len(matches) > 0:
            if matches[0] == 1:
                rank1 += 1

            if matches[:5].sum() > 0:
                rank5 += 1

        average_precisions.append(average_precision(matches))

        if (i + 1) % 100 == 0 or i + 1 == len(query_records):
            print(f"Queries evaluated: {i + 1}/{len(query_records)}")

    total_queries = len(query_records)

    rank1_score = rank1 / total_queries
    rank5_score = rank5 / total_queries
    map_score = float(np.mean(average_precisions))

    return rank1_score, rank5_score, map_score


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = Market1501Dataset(DATASET_ROOT)
    dataset.verify_structure()

    query_paths = dataset.get_query_images()
    gallery_paths = dataset.get_gallery_images()

    query_records = dataset.get_query_records()
    gallery_records = dataset.get_gallery_records()

    print(f"Query images   : {len(query_paths)}")
    print(f"Gallery images : {len(gallery_paths)}")

    embedder = ReIDEmbedder(
        model_name="osnet_x0_25",
        device="cpu"
    )

    print("\nExtracting query embeddings...")
    query_embeddings = load_embeddings(
        embedder,
        query_paths,
        "Query embeddings"
    )

    print("\nExtracting gallery embeddings...")
    gallery_embeddings = load_embeddings(
        embedder,
        gallery_paths,
        "Gallery embeddings"
    )

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "market1501_embeddings.npz"),
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings
    )

    print("\nRunning Market-1501 evaluation...")

    rank1, rank5, map_score = evaluate(
        query_records,
        gallery_records,
        query_embeddings,
        gallery_embeddings
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "market1501_results.txt"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        file.write(f"Rank-1: {rank1:.6f}\n")
        file.write(f"Rank-5: {rank5:.6f}\n")
        file.write(f"mAP: {map_score:.6f}\n")

    print("\n--- Market-1501 Results ---")
    print(f"Rank-1 : {rank1:.4f} ({rank1 * 100:.2f}%)")
    print(f"Rank-5 : {rank5:.4f} ({rank5 * 100:.2f}%)")
    print(f"mAP    : {map_score:.4f} ({map_score * 100:.2f}%)")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()