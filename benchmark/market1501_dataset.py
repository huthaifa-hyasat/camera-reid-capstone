from pathlib import Path
import re


EXPECTED_QUERY_COUNT = 3368
EXPECTED_GALLERY_COUNT = 19732


class Market1501Dataset:
    def __init__(self, root):
        self.root = Path(root)

        self.query_dir = self.root / "query"
        self.gallery_dir = self.root / "bounding_box_test"

    def verify_structure(self):
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}"
            )

        if not self.query_dir.exists():
            raise FileNotFoundError(
                f"Query directory not found: {self.query_dir}"
            )

        if not self.gallery_dir.exists():
            raise FileNotFoundError(
                f"Gallery directory not found: {self.gallery_dir}"
            )

    @staticmethod
    def parse_filename(filename):
        pattern = r"^(-?\d+)_c(\d+)s(\d+)_(\d+)_([\d]+)\.jpg$"

        match = re.match(pattern, filename)

        if match is None:
            raise ValueError(
                f"Invalid Market-1501 filename: {filename}"
            )

        person_id = int(match.group(1))
        camera_id = int(match.group(2))
        sequence_id = int(match.group(3))
        frame_id = int(match.group(4))
        track_id = int(match.group(5))

        return {
            "person_id": person_id,
            "camera_id": camera_id,
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "track_id": track_id,
            "filename": filename,
        }

    def get_query_images(self):
        return sorted(self.query_dir.glob("*.jpg"))

    def get_gallery_images(self):
        return sorted(self.gallery_dir.glob("*.jpg"))

    def get_query_records(self):
        return [
            self.parse_filename(path.name)
            for path in self.get_query_images()
        ]

    def get_gallery_records(self):
        return [
            self.parse_filename(path.name)
            for path in self.get_gallery_images()
        ]

    def summary(self):
        query = self.get_query_records()
        gallery = self.get_gallery_records()

        return {
            "query_count": len(query),
            "gallery_count": len(gallery),
            "gallery_junk_count": sum(
                record["person_id"] in {-1, 0}
                for record in gallery
            ),
        }


if __name__ == "__main__":
    dataset = Market1501Dataset(
        "data/market1501/Market-1501-v15.09.15"
    )

    dataset.verify_structure()

    summary = dataset.summary()

    print("Market-1501 dataset verified.")
    print(f"Query images   : {summary['query_count']}")
    print(f"Gallery images : {summary['gallery_count']}")
    print(f"Junk images    : {summary['gallery_junk_count']}")