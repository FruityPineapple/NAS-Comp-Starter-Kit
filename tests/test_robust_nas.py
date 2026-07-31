"""Synthetic contract and safety tests for submission_robust_nas."""

import os
import sys
import time
import unittest

import numpy as np
import torch
from torch.utils.data import RandomSampler


SUBMISSION = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submission_robust_nas"))
if SUBMISSION not in sys.path:
    sys.path.insert(0, SUBMISSION)

from data_processor import DataProcessor  # noqa: E402
from models import build_model, fallback_config  # noqa: E402
from nas import NAS  # noqa: E402
from trainer import Trainer  # noqa: E402


class Clock(object):
    def __init__(self, seconds):
        self.deadline = time.perf_counter() + float(seconds)

    def check(self):
        return self.deadline - time.perf_counter()


def arrays(shape=(16, 1, 4, 4), dtype=np.float32, labels=None, test_count=7):
    rng = np.random.RandomState(11)
    train = rng.normal(size=shape).astype(dtype)
    valid = rng.normal(size=(8,) + shape[1:]).astype(dtype)
    test = rng.normal(size=(test_count,) + shape[1:]).astype(dtype)
    if labels is None:
        labels = np.asarray(([10, 20] * ((shape[0] + 1) // 2))[:shape[0]])
    valid_labels = np.asarray(([10, 20, 30, 10] * 2)[: len(valid)])
    return train, np.asarray(labels), valid, valid_labels, test


class DataContractTests(unittest.TestCase):
    def test_lazy_ordered_complete_and_label_union(self):
        train_x, train_y, valid_x, valid_y, test_x = arrays(dtype=np.float64, test_count=9)
        train_x[0, 0, 0, 0] = np.nan
        train_x[1, 0, 0, 1] = np.inf
        train_x[2, 0, :, :] = 1.0e300
        train_x[3, 0, :, :] = -1.0e300
        metadata = {"time_remaining": 30.0, "num_classes": 99, "input_shape": train_x.shape}
        train_loader, valid_loader, test_loader = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, Clock(30.0)
        ).process()

        self.assertIs(train_loader.dataset.x, train_x)
        self.assertFalse(isinstance(test_loader.sampler, RandomSampler))
        self.assertFalse(test_loader.drop_last)
        self.assertEqual(len(test_loader.dataset), 9)
        self.assertEqual(metadata["class_labels"], [10, 20, 30])
        self.assertEqual(metadata["num_classes"], 3)
        self.assertTrue(torch.isfinite(train_loader.dataset[0][0]).all().item())
        self.assertTrue(torch.isfinite(train_loader.dataset[2][0]).all().item())
        self.assertIsInstance(next(iter(test_loader)), torch.Tensor)

    def test_one_hot_and_defensive_nhw_bool(self):
        rng = np.random.RandomState(3)
        train_x = rng.randint(0, 2, size=(6, 4, 4)).astype(bool)
        valid_x = rng.randint(0, 2, size=(3, 4, 4)).astype(bool)
        test_x = rng.randint(0, 2, size=(5, 4, 4)).astype(bool)
        train_y = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1, 2]]
        valid_y = np.eye(3, dtype=np.float32)[[0, 1, 2]]
        metadata = {"time_remaining": 15.0}
        train_loader, _, test_loader = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, Clock(15.0)
        ).process()
        sample, target = train_loader.dataset[0]
        self.assertEqual(tuple(sample.shape), (1, 4, 4))
        self.assertEqual(target.dtype, torch.long)
        self.assertEqual(len(test_loader.dataset), 5)
        self.assertEqual(metadata["num_classes"], 3)

    def test_many_classes_uint8_and_imbalance_profile(self):
        rng = np.random.RandomState(17)
        train_x = rng.randint(0, 256, size=(61, 1, 4, 4), dtype=np.uint8)
        valid_x = rng.randint(0, 256, size=(61, 1, 4, 4), dtype=np.uint8)
        test_x = rng.randint(0, 256, size=(11, 1, 4, 4), dtype=np.uint8)
        train_y = np.arange(61)
        valid_y = np.arange(61)
        metadata = {"time_remaining": 20.0}
        train_loader, _, test_loader = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, Clock(20.0)
        ).process()
        self.assertEqual(metadata["num_classes"], 61)
        self.assertEqual(next(iter(train_loader))[0].dtype, torch.float32)
        self.assertEqual(len(test_loader.dataset), 11)

        imbalanced_y = np.asarray([0] * 60 + [1])
        metadata_imbalanced = {"time_remaining": 20.0}
        DataProcessor(
            train_x,
            imbalanced_y,
            valid_x[:2],
            np.asarray([0, 1]),
            test_x,
            metadata_imbalanced,
            Clock(20.0),
        ).process()
        self.assertTrue(metadata_imbalanced["highly_imbalanced"])


class ModelSpaceTests(unittest.TestCase):
    def test_shape_fuzz_forward_backward(self):
        cases = [
            (1, 1, 128),
            (1, 4, 4),
            (3, 32, 32),
            (3, 224, 224),
            (64, 8, 8),
        ]
        for channels, height, width in cases:
            with self.subTest(shape=(channels, height, width)):
                metadata = {
                    "num_classes": 5,
                    "input_shape": (4, channels, height, width),
                    "channels": channels,
                    "height": height,
                    "width": width,
                }
                config = fallback_config(8)
                config.update({
                    "block": "separable" if channels >= 16 or max(height, width) >= 128 else "basic",
                    "coordinates": True,
                    "pool_grid": 4,
                    "stages": 3,
                    "blocks_per_stage": [1, 1, 1],
                    "max_pool_stem": max(height, width) >= 128,
                })
                model = build_model(metadata, config)
                batch_size = 1 if max(height, width) >= 128 else 2
                inputs = torch.randn(batch_size, channels, height, width)
                logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (batch_size, 5))
                targets = torch.tensor([0]) if batch_size == 1 else torch.tensor([0, 4])
                loss = torch.nn.functional.cross_entropy(logits, targets)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()


class EndToEndTests(unittest.TestCase):
    def _pipeline(self, train_y=None, valid_y=None, seconds=20.0):
        train_x, default_y, valid_x, default_valid_y, test_x = arrays(
            shape=(12, 1, 4, 4), labels=train_y, test_count=7
        )
        train_y = default_y if train_y is None else np.asarray(train_y)
        valid_y = default_valid_y if valid_y is None else np.asarray(valid_y)
        clock = Clock(seconds)
        metadata = {"time_remaining": seconds, "input_shape": train_x.shape, "num_classes": 3}
        loaders = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, clock).process()
        model = NAS(loaders[0], loaders[1], metadata, clock).search()
        trainer = Trainer(model, torch.device("cpu"), loaders[0], loaders[1], metadata, clock)
        trainer.train()
        return trainer, loaders[2], metadata

    def test_tiny_budget_exact_original_labels(self):
        trainer, test_loader, metadata = self._pipeline(seconds=20.0)
        predictions = trainer.predict(test_loader)
        self.assertEqual(len(predictions), len(test_loader.dataset))
        self.assertTrue(set(predictions.tolist()).issubset({10, 20, 30}))
        self.assertIn("selected_config", metadata)

    def test_one_class_constant_path(self):
        trainer, test_loader, _ = self._pipeline(
            train_y=[7] * 12,
            valid_y=[7] * 8,
            seconds=10.0,
        )
        predictions = trainer.predict(test_loader)
        np.testing.assert_array_equal(predictions, np.full(7, 7))

    def test_inference_exception_returns_majority(self):
        trainer, test_loader, metadata = self._pipeline(seconds=20.0)

        def fail(_inputs):
            raise RuntimeError("injected inference failure")

        trainer.model.forward = fail
        predictions = trainer.predict(test_loader)
        self.assertEqual(len(predictions), 7)
        np.testing.assert_array_equal(
            predictions,
            np.full(7, metadata["majority_label"], dtype=predictions.dtype),
        )

    def test_short_search_contains_candidate_build_failure(self):
        train_x, train_y, valid_x, valid_y, test_x = arrays(shape=(12, 1, 4, 4), test_count=5)
        clock = Clock(330.0)
        metadata = {"time_remaining": 330.0, "input_shape": train_x.shape, "num_classes": 3}
        train_loader, valid_loader, _ = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, clock
        ).process()
        search = NAS(train_loader, valid_loader, metadata, clock)
        oversized = fallback_config(128)
        oversized["name"] = "injected_oversized"
        self.assertIsNone(search._build_candidate(99, oversized))
        original = search._build_candidate

        def injected(index, config, protected=False):
            if index == 1:
                raise RuntimeError("injected construction failure")
            return original(index, config, protected=protected)

        search._build_candidate = injected
        model = search.search()
        self.assertIsInstance(model, torch.nn.Module)
        events = [item["event"] for item in metadata["search_history"]]
        self.assertIn("candidate_build_failed", events)
        self.assertIn("candidate_scored", events)
        self.assertIn("search_selected", events)

    def test_training_recovers_from_injected_oom(self):
        train_x, train_y, valid_x, valid_y, test_x = arrays(shape=(12, 1, 4, 4), test_count=5)
        clock = Clock(20.0)
        metadata = {"time_remaining": 20.0, "input_shape": train_x.shape, "num_classes": 3}
        train_loader, valid_loader, _ = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, clock
        ).process()
        model = build_model(metadata, fallback_config())
        original_forward = model.forward
        calls = [0]

        def flaky(inputs):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("CUDA out of memory (injected)")
            return original_forward(inputs)

        model.forward = flaky
        trainer = Trainer(model, torch.device("cpu"), train_loader, valid_loader, metadata, clock)
        self.assertIs(trainer.train(), model)
        self.assertGreater(calls[0], 1)
        self.assertLess(metadata["safe_microbatch"], metadata["initial_microbatch"])

    def test_multifidelity_resume_and_confirmation(self):
        rng = np.random.RandomState(29)
        train_x = rng.normal(size=(36, 1, 4, 4)).astype(np.float32)
        valid_x = rng.normal(size=(24, 1, 4, 4)).astype(np.float32)
        test_x = rng.normal(size=(5, 1, 4, 4)).astype(np.float32)
        train_y = np.arange(36) % 3
        valid_y = np.arange(24) % 3
        clock = Clock(1000.0)
        metadata = {"time_remaining": 1000.0, "input_shape": train_x.shape, "num_classes": 3}
        train_loader, valid_loader, _ = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, clock
        ).process()
        model = NAS(train_loader, valid_loader, metadata, clock).search()
        self.assertIsInstance(model, torch.nn.Module)
        events = [item["event"] for item in metadata["search_history"]]
        self.assertGreaterEqual(events.count("rung_complete"), 2)
        self.assertIn("candidate_confirmed", events)

    def test_nonfinite_training_is_contained(self):
        train_x, train_y, valid_x, valid_y, test_x = arrays(shape=(12, 1, 4, 4), test_count=5)
        clock = Clock(20.0)
        metadata = {"time_remaining": 20.0, "input_shape": train_x.shape, "num_classes": 3}
        train_loader, valid_loader, test_loader = DataProcessor(
            train_x, train_y, valid_x, valid_y, test_x, metadata, clock
        ).process()
        model = build_model(metadata, fallback_config())

        def nonfinite(inputs):
            return torch.full(
                (len(inputs), metadata["num_classes"]),
                float("nan"),
                device=inputs.device,
            )

        model.forward = nonfinite
        trainer = Trainer(model, torch.device("cpu"), train_loader, valid_loader, metadata, clock)
        self.assertIs(trainer.train(), model)
        predictions = trainer.predict(test_loader)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(set(predictions.tolist()).issubset({10, 20, 30}))


if __name__ == "__main__":
    unittest.main()
