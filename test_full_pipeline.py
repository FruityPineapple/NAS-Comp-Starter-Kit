"""
test_full_pipeline.py — End-to-end integration test simulating the competition pipeline.

Tests all three tiers of the NAS pipeline with synthetic data,
matching the exact flow from evaluation/main.py:
  1. DataProcessor -> dataloaders
  2. NAS.search() -> model
  3. Trainer.train() -> trained model
  4. Trainer.predict() -> predictions
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submission'))

import torch
from torch.utils.data import RandomSampler

from data_processor import DataProcessor
from nas import NAS
from trainer import Trainer


# Minimal Clock (same as evaluation/main.py)
class Clock:
    def __init__(self, time_limit_in_hours):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + (time_limit_in_hours * 60 * 60)

    def check(self):
        return self.time_limit - time.perf_counter()


def general_num_params(model):
    """Same as evaluation/main.py"""
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def run_pipeline(name, shape, num_classes, time_limit_hours, n_train=200, n_valid=50, n_test=50):
    """Run the full pipeline for one synthetic dataset."""
    print("\n" + "=" * 70)
    print("TEST: {} (time_limit={:.2f}h = {:.0f}min)".format(
        name, time_limit_hours, time_limit_hours * 60))
    print("  Shape: {}, Classes: {}, Train/Val/Test: {}/{}/{}".format(
        shape, num_classes, n_train, n_valid, n_test))
    print("=" * 70)

    # Generate synthetic data
    train_x = np.random.rand(n_train, *shape).astype(np.float32)
    train_y = np.random.randint(0, num_classes, n_train)
    valid_x = np.random.rand(n_valid, *shape).astype(np.float32)
    valid_y = np.random.randint(0, num_classes, n_valid)
    test_x = np.random.rand(n_test, *shape).astype(np.float32)

    full_shape = [n_train + n_valid + n_test] + list(shape)
    metadata = {
        'num_classes': num_classes,
        'input_shape': full_shape,
        'codename': 'Test_{}'.format(name.replace(' ', '_')),
        'time_limit': time_limit_hours,
    }

    clock = Clock(time_limit_hours)

    # Phase 1: DataProcessor
    print("\n--- Phase 1: DataProcessor ---")
    metadata['time_remaining'] = clock.check()
    dp = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, clock)
    train_loader, valid_loader, test_loader = dp.process()

    # Verify test loader
    assert not isinstance(test_loader.sampler, RandomSampler), "Test loader shuffling!"
    assert not test_loader.drop_last, "Test loader dropping last!"
    print("  DataProcessor OK")

    # Phase 2: NAS
    print("\n--- Phase 2: NAS ---")
    metadata['time_remaining'] = clock.check()
    model = NAS(train_loader, valid_loader, metadata, clock).search()

    # Verify model
    params = int(general_num_params(model))
    print("  Model params: {:,}".format(params))
    assert params > 0, "Model has no parameters!"

    # Quick forward pass test
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        if isinstance(test_batch, (list, tuple)):
            test_input = test_batch[0]
        else:
            test_input = test_batch
        output = model(test_input.to(device))
        assert output.shape[1] == num_classes, \
            "Output classes mismatch: expected {}, got {}".format(num_classes, output.shape[1])
    print("  Model forward pass OK (output shape: {})".format(output.shape))

    # Phase 3: Trainer
    print("\n--- Phase 3: Trainer ---")
    metadata['time_remaining'] = clock.check()
    trainer = Trainer(model, device, train_loader, valid_loader, metadata, clock)
    trained_model = trainer.train()
    print("  Training OK")

    # Phase 4: Predict
    print("\n--- Phase 4: Predict ---")
    predictions = trainer.predict(test_loader)
    assert len(predictions) == n_test, \
        "Prediction count mismatch: expected {}, got {}".format(n_test, len(predictions))

    # Verify predictions are valid class labels
    assert all(0 <= p < num_classes for p in predictions), \
        "Invalid predictions found!"
    print("  Predictions OK: {} predictions, range [{}, {}]".format(
        len(predictions), min(predictions), max(predictions)))

    # Time check
    remaining = clock.check()
    print("\n  Time remaining: {:.1f}s ({})".format(remaining, "OK" if remaining > 0 else "OVERTIME!"))
    assert remaining > 0, "OVERTIME — pipeline exceeded time limit!"

    print("\n  [PASS] Full pipeline completed successfully")
    return True


if __name__ == '__main__':
    results = []

    # Test 1: TIER 3 — very short time (2 minutes)
    results.append(run_pipeline(
        "TIER 3 fallback (2min budget)",
        shape=(3, 32, 32), num_classes=10,
        time_limit_hours=2/60,  # 2 minutes
    ))

    # Test 2: TIER 2 — medium time (10 minutes), small input
    results.append(run_pipeline(
        "TIER 2 ZCP-only (10min, 8x8)",
        shape=(1, 8, 8), num_classes=20,
        time_limit_hours=10/60,  # 10 minutes
    ))

    # Test 3: TIER 1 — long time (30 minutes)
    results.append(run_pipeline(
        "TIER 1 full search (30min)",
        shape=(3, 32, 32), num_classes=10,
        time_limit_hours=30/60,  # 30 minutes
        n_train=500, n_valid=100, n_test=100,
    ))

    # Test 4: 3D input (missing channel dim)
    results.append(run_pipeline(
        "3D input with TIER 3 (3min)",
        shape=(28, 28), num_classes=10,
        time_limit_hours=3/60,  # 3 minutes
    ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: {}/{} pipeline tests passed".format(sum(results), len(results)))
    print("=" * 70)

    if not all(results):
        sys.exit(1)
