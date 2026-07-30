"""Fast end-to-end exercise of the Tier-1 controller with a stepped clock."""

import os
import sys

import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from data_processor import DataProcessor
from nas import NAS
from trainer import Trainer


class _SteppedClock:
    """Expose a long tier while making budget guards advance in unit tests."""

    def __init__(self, remaining=1_050.0, decrement=3.0):
        self.remaining = float(remaining)
        self.decrement = float(decrement)

    def check(self):
        self.remaining -= self.decrement
        return self.remaining


def test_accelerated_tier1_pipeline_returns_all_predictions():
    rng = np.random.RandomState(23)
    train_x = rng.rand(72, 3, 16, 16).astype(np.float32)
    train_y = np.arange(72) % 4
    valid_x = rng.rand(24, 3, 16, 16).astype(np.float32)
    valid_y = np.arange(24) % 4
    test_x = rng.rand(19, 3, 16, 16).astype(np.float32)
    metadata = {
        "num_classes": 4,
        "input_shape": [115, 3, 16, 16],
        "benchmark": None,
    }
    clock = _SteppedClock()
    train_loader, valid_loader, test_loader = DataProcessor(
        train_x,
        train_y,
        valid_x,
        valid_y,
        test_x,
        metadata,
        clock,
    ).process()
    model = NAS(
        train_loader, valid_loader, metadata, clock
    ).search()
    trainer = Trainer(
        model,
        torch.device("cpu"),
        train_loader,
        valid_loader,
        metadata,
        clock,
    )
    trained = trainer.train()
    predictions = trainer.predict(test_loader)
    assert trained is trainer.model
    assert len(predictions) == len(test_x)
    assert all(0 <= prediction < 4 for prediction in predictions)
    assert metadata["nas_ensemble"] is False


if __name__ == "__main__":
    test_accelerated_tier1_pipeline_returns_all_predictions()
    print("Accelerated Tier-1 pipeline test passed")
