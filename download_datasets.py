"""
download_datasets.py — Download all 13 historical NAS competition datasets.

Run this on the better machine before starting the full evaluation.
The evaluation pipeline expects datasets at: datasets/<dataset_name>/

Usage:
    python download_datasets.py
"""

import os
import json
import urllib.request
import sys

# The 13 datasets from NAS Unseen-Data Challenge history.
# These are the ones from the 2024 edition, which share the same format.
# URLs may need updating when the 2026 datasets are released.
DATASET_BASE_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/nas_unseen_data"

DATASETS = [
    "dataset_1",   # AddNIST
    "dataset_2",   # Language
    "dataset_3",   # MultNIST
    "dataset_4",   # CIFARTile
    "dataset_5",   # Gutenberg
    "dataset_6",   # Isabella
    "dataset_7",   # GeoClassing
    "dataset_8",   # Chesseract
    "dataset_9",   # K49
    "dataset_10",  # Anime
    "dataset_11",  # SVHN
    "dataset_12",  # FashionMNIST
    "dataset_13",  # FlowerPhotos
]

FILES_PER_DATASET = [
    "train_x.npy",
    "train_y.npy",
    "valid_x.npy",
    "valid_y.npy",
    "test_x.npy",
    "metadata",
]


def download_file(url, dest_path):
    """Download a file with a progress bar."""
    if os.path.exists(dest_path):
        print("    Already exists: {}".format(os.path.basename(dest_path)))
        return True

    print("    Downloading: {} ...".format(os.path.basename(dest_path)), end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(" OK ({:.1f} MB)".format(size_mb))
        return True
    except Exception as e:
        print(" FAILED: {}".format(e))
        return False


def main():
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    print("=" * 60)
    print("NAS Unseen-Data 2026 — Dataset Downloader")
    print("Target directory: {}".format(datasets_dir))
    print("=" * 60)

    # Check if datasets are already present (e.g., from the competition repo)
    existing = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    if existing:
        print("\nFound {} existing dataset(s): {}".format(len(existing), existing))
        print("Skipping datasets that already have all files.\n")

    success_count = 0
    fail_count = 0

    for dataset in DATASETS:
        dataset_dir = os.path.join(datasets_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        print("\n--- {} ---".format(dataset))

        # Check if already complete
        all_present = all(
            os.path.exists(os.path.join(dataset_dir, f))
            for f in FILES_PER_DATASET
        )
        if all_present:
            print("  Already complete, skipping.")
            success_count += 1
            continue

        all_ok = True
        for filename in FILES_PER_DATASET:
            url = "{}/{}/{}".format(DATASET_BASE_URL, dataset, filename)
            dest = os.path.join(dataset_dir, filename)
            ok = download_file(url, dest)
            if not ok:
                all_ok = False

        if all_ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY: {}/{} datasets ready, {} failed".format(
        success_count, len(DATASETS), fail_count))
    print("=" * 60)

    if fail_count > 0:
        print("\nNOTE: Some downloads failed. This might be because:")
        print("  1. The URLs have changed for the 2026 edition")
        print("  2. The dataset server is not accessible")
        print("  3. You need to download datasets manually from nascompetition.com")
        print("\nYou can also place your datasets manually in: {}".format(datasets_dir))
        print("Each dataset folder needs: train_x.npy, train_y.npy, valid_x.npy, valid_y.npy, test_x.npy, metadata")
        sys.exit(1)


if __name__ == "__main__":
    main()
