"""Regression tests for bounded materialization and logical microbatches."""

import os
import sys

import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from data_processor import DataProcessor, NASDataset
from nas import NAS


class _Clock:
    def check(self):
        return 10_000.0


def test_dataset_preserves_storage_dtype():
    values = np.arange(32 * 3 * 8 * 8, dtype=np.uint8).reshape(
        32, 3, 8, 8
    )
    dataset = NASDataset(values, np.arange(32) % 4)
    assert dataset.x.dtype == torch.uint8
    image, label = dataset[3]
    assert image.dtype == torch.float32
    assert int(label) == 3


def test_streaming_normalization_matches_reference():
    rng = np.random.RandomState(11)
    values = rng.randint(0, 256, size=(37, 3, 7, 5), dtype=np.uint8)
    processor = DataProcessor(
        values,
        np.arange(37) % 3,
        values[:5],
        np.arange(5) % 3,
        values[:5],
        {"num_classes": 3, "input_shape": list(values.shape)},
        _Clock(),
    )
    mean, std = processor._compute_normalization_stats(
        max_samples=37, max_chunk_megabytes=0.002
    )
    reference = values.astype(np.float32)
    expected_mean = reference.mean(axis=(0, 2, 3))
    expected_std = reference.std(axis=(0, 2, 3))
    assert np.allclose(mean, expected_mean, atol=1e-5)
    assert np.allclose(std, expected_std, atol=1e-5)
    assert processor.metadata["normalization_chunk_samples"] < len(values)


def _controller_for_loader(loader):
    controller = object.__new__(NAS)
    controller.train_loader = loader
    controller.valid_loader = loader
    controller.metadata = {}
    controller.clock = _Clock()
    controller.device = torch.device("cpu")
    controller.seed = 19
    controller.in_channels = 3
    controller.input_h = 16
    controller.input_w = 16
    controller.num_classes = 3
    controller.data_props = {}
    return controller


def test_search_cache_honors_byte_cap_on_oversized_first_batch():
    data = torch.randn(40, 3, 16, 16)
    targets = torch.arange(40) % 3
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, targets),
        batch_size=16,
        shuffle=False,
    )
    controller = _controller_for_loader(loader)
    limit_megabytes = 0.012
    batches = controller._cache_search_batches(
        max_megabytes=limit_megabytes, max_batches=10
    )
    used = sum(
        batch.numel() * batch.element_size()
        + labels.numel() * labels.element_size()
        for batch, labels in batches
    )
    assert batches
    assert batches[0][0].size(0) < 16
    assert used <= int(limit_megabytes * 1024 * 1024)


def test_validation_cache_preserves_priors_and_separates_confirmation():
    rng = np.random.RandomState(3)
    values = rng.rand(100, 3, 8, 8).astype(np.float32)
    labels = np.asarray([0] * 70 + [1] * 20 + [2] * 10)
    dataset = NASDataset(values, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=17, shuffle=False
    )
    controller = _controller_for_loader(loader)
    controller.in_channels = 3
    controller.input_h = 8
    controller.input_w = 8
    selection, confirmation = controller._cache_validation_splits(
        max_samples=60,
        max_megabytes=1,
        confirmation_fraction=0.25,
    )
    selection_labels = torch.cat([target for _, target in selection])
    confirmation_labels = torch.cat([target for _, target in confirmation])
    assert not set(selection_labels.tolist()).isdisjoint(
        confirmation_labels.tolist()
    )
    assert len(selection_labels) + len(confirmation_labels) == 60
    combined = torch.bincount(
        torch.cat([selection_labels, confirmation_labels]), minlength=3
    )
    assert combined.tolist() == [42, 12, 6]
    selection_indices = {
        tuple(row.flatten().tolist())
        for batch, _ in selection
        for row in batch
    }
    confirmation_indices = {
        tuple(row.flatten().tolist())
        for batch, _ in confirmation
        for row in batch
    }
    assert selection_indices.isdisjoint(confirmation_indices)


def test_logical_batch_is_accumulated_from_microbatches():
    data = torch.randn(7, 3, 4, 4)
    targets = torch.arange(7) % 3
    loader = [(data, targets)]
    controller = _controller_for_loader(loader)
    controller.input_h = 4
    controller.input_w = 4
    controller.search_microbatch_size = 2
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 3)
    )
    recipe = {
        "optimizer": "adamw",
        "search_lr": 1e-3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
    }
    steps, _, failed, _, stats = controller._train_low_fidelity(
        model,
        loader,
        max_steps=1,
        time_quantum=30,
        deadline_remaining=0,
        recipe=recipe,
    )
    assert not failed
    assert steps == 1
    assert stats["examples_seen"] == 7
    assert 0.0 <= stats["train_accuracy"] <= 1.0


if __name__ == "__main__":
    test_dataset_preserves_storage_dtype()
    test_streaming_normalization_matches_reference()
    test_search_cache_honors_byte_cap_on_oversized_first_batch()
    test_validation_cache_preserves_priors_and_separates_confirmation()
    test_logical_batch_is_accumulated_from_microbatches()
    print("Memory-safety regression tests passed")
