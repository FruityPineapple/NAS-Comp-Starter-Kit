"""
test_data_processor.py — Integration test for the DataProcessor.

Simulates the evaluation pipeline's data-loading flow with synthetic datasets
covering the edge cases we need to handle:
  1. Standard 4D input (CIFAR-like)
  2. 3D input (missing channel dim)
  3. Tiny spatial resolution (8×8 — Sudoku-like)
  4. Large spatial resolution (128×128)
  5. Many classes
  6. Single-channel data
"""

import sys
import os
import time
import math
import numpy as np

# Add submission dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submission'))

import torch
from data_processor import DataProcessor
from torch.utils.data import RandomSampler


# Minimal Clock replica (same as evaluation/main.py)
class Clock:
    def __init__(self, time_limit_in_hours):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + (time_limit_in_hours * 60 * 60)

    def check(self):
        return self.time_limit - time.perf_counter()


def make_synthetic_data(n_train, n_valid, n_test, shape, num_classes):
    """Generate random synthetic data matching the competition format."""
    train_x = np.random.rand(n_train, *shape).astype(np.float32)
    train_y = np.random.randint(0, num_classes, n_train)
    valid_x = np.random.rand(n_valid, *shape).astype(np.float32)
    valid_y = np.random.randint(0, num_classes, n_valid)
    test_x  = np.random.rand(n_test, *shape).astype(np.float32)

    full_shape = [n_train + n_valid + n_test] + list(shape)
    metadata = {
        'num_classes': num_classes,
        'input_shape': full_shape,
        'codename': 'SyntheticTest',
        'time_limit': 0.5,
    }
    return train_x, train_y, valid_x, valid_y, test_x, metadata


def test_case(name, shape, num_classes, n_train=200, n_valid=50, n_test=50):
    """Run a single DataProcessor test case."""
    print("\n" + "=" * 60)
    print("TEST: {}".format(name))
    print("  Shape: {}, Classes: {}, Train/Val/Test: {}/{}/{}".format(
        shape, num_classes, n_train, n_valid, n_test))
    print("=" * 60)

    train_x, train_y, valid_x, valid_y, test_x, metadata = make_synthetic_data(
        n_train, n_valid, n_test, shape, num_classes
    )
    clock = Clock(0.5)  # 30 minutes

    dp = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, clock)
    train_loader, valid_loader, test_loader = dp.process()

    # ---- Assertions (mirror main.py checks) ----
    # 1. Test loader must NOT shuffle
    assert not isinstance(test_loader.sampler, RandomSampler), \
        "FAIL: Test loader is shuffling!"

    # 2. Test loader must NOT drop last batch
    assert not test_loader.drop_last, \
        "FAIL: Test loader is dropping last batch!"

    # 3. Train loader should produce (data, label) tuples
    for batch in train_loader:
        assert len(batch) == 2, "FAIL: Train batch should be (data, label)"
        data, labels = batch
        assert data.ndim == 4, "FAIL: Train data should be 4D [B, C, H, W], got {}D".format(data.ndim)
        assert data.dtype == torch.float32, "FAIL: Data dtype should be float32"
        assert labels.dtype == torch.int64, "FAIL: Label dtype should be int64"
        break  # just check first batch

    # 4. Valid loader should produce (data, label) tuples
    for batch in valid_loader:
        assert len(batch) == 2, "FAIL: Valid batch should be (data, label)"
        break

    # 5. Test loader should produce data only (no labels)
    for batch in test_loader:
        assert isinstance(batch, torch.Tensor), \
            "FAIL: Test batch should be a single tensor, got {}".format(type(batch))
        assert batch.ndim == 4, "FAIL: Test data should be 4D"
        break

    # 6. Metadata should be enriched with data_props
    assert 'data_props' in metadata, "FAIL: metadata should contain 'data_props'"
    assert 'batch_size' in metadata, "FAIL: metadata should contain 'batch_size'"
    assert metadata['test_num_batches'] == len(test_loader), \
        "FAIL: test batch count missing or incorrect"
    assert metadata['test_num_samples'] == n_test, \
        "FAIL: test sample count missing or incorrect"

    # 7. Total test samples should be preserved (no data loss)
    total_test_samples = sum(
        batch.shape[0] for batch in test_loader
    )
    assert total_test_samples == n_test, \
        "FAIL: Test samples lost! Expected {}, got {}".format(n_test, total_test_samples)

    print("\n  [PASS] ALL ASSERTIONS PASSED")
    return True


if __name__ == '__main__':
    results = []

    # Test 1: Standard CIFAR-like input [C=3, H=32, W=32]
    results.append(test_case(
        "Standard 4D (CIFAR-like)",
        shape=(3, 32, 32), num_classes=10
    ))

    # Test 2: Missing channel dimension [H=28, W=28] (like MNIST)
    results.append(test_case(
        "3D input (missing channel dim)",
        shape=(28, 28), num_classes=10
    ))

    # Test 3: Tiny spatial resolution [C=1, H=8, W=8] (Sudoku-like)
    results.append(test_case(
        "Tiny 8x8 input (no augmentation expected)",
        shape=(1, 8, 8), num_classes=20
    ))

    # Test 4: Large spatial resolution [C=3, H=128, W=128]
    results.append(test_case(
        "Large 128x128 input (should use smaller batch)",
        shape=(3, 128, 128), num_classes=50
    ))

    # Test 5: Single-channel, medium resolution
    results.append(test_case(
        "Grayscale 48x48",
        shape=(1, 48, 48), num_classes=5
    ))

    # Test 6: Many channels (unusual domain)
    results.append(test_case(
        "Multi-channel (6 channels, 16x16)",
        shape=(6, 16, 16), num_classes=100
    ))

    # Test 7: Non-square input
    results.append(test_case(
        "Non-square (3x64x32)",
        shape=(3, 64, 32), num_classes=15
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: {}/{} tests passed".format(sum(results), len(results)))
    print("=" * 60)

    if not all(results):
        sys.exit(1)
