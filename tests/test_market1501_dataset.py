import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from benchmark.market1501_dataset import Market1501Dataset


DATASET_ROOT = (
    "data/market1501/Market-1501-v15.09.15"
)


def test_dataset_structure():
    dataset = Market1501Dataset(DATASET_ROOT)

    dataset.verify_structure()

    summary = dataset.summary()

    assert summary["query_count"] == 3368
    assert summary["gallery_count"] == 19732

    print("\nMarket-1501 structure test passed.")
    print(f"Query images   : {summary['query_count']}")
    print(f"Gallery images : {summary['gallery_count']}")
    print(f"Junk images    : {summary['gallery_junk_count']}")