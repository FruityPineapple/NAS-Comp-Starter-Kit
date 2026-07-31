"""Memory-safe profiling and lazy data loading for arbitrary NCHW inputs."""

from __future__ import print_function

import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from helpers import BASE_SEED, BudgetManager, make_generator, seed_everything


def _as_label_vector(labels):
    values = np.asarray(labels)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim > 1:
        if values.shape[-1] > 1:
            values = np.argmax(values, axis=-1)
        else:
            values = values.reshape(values.shape[0], -1)[:, 0]
    return values.reshape(-1)


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _label_key(value):
    value = _python_scalar(value)
    key = (type(value).__name__, value)
    try:
        hash(key)
        return key
    except TypeError:
        return (type(value).__name__, repr(value))


def _stable_unique(values):
    result = []
    seen = set()
    for value in values:
        value = _python_scalar(value)
        key = _label_key(value)
        unseen = key not in seen
        if unseen:
            seen.add(key)
            result.append(value)
    try:
        return sorted(result)
    except (TypeError, ValueError):
        return result


def _map_labels(values, mapping):
    output = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        output[index] = mapping[_label_key(value)]
    return output


def _shape4(array):
    shape = tuple(np.shape(array))
    if len(shape) == 4:
        return shape
    if len(shape) == 3:
        return (shape[0], 1, shape[1], shape[2])
    raise ValueError("classification inputs must be NCHW 4-D arrays (or defensive NHW 3-D arrays)")


class LazyArrayDataset(Dataset):
    """A view over a NumPy array that converts and normalizes one sample at a time."""

    def __init__(self, x, y, mean, std):
        self.x = x
        self.labels = y
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self._shape = _shape4(x)

    def __len__(self):
        return int(self._shape[0])

    def __getitem__(self, index):
        array = np.asarray(self.x[index])
        # NumPy exposes unsigned integer widths that older PyTorch releases cannot
        # wrap directly. Convert only this sample, never the full backing array.
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=False)
        if not array.flags.c_contiguous or not array.flags.writeable:
            array = np.array(array, dtype=np.float32, copy=True, order="C")
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != self.mean.shape[0]:
            raise ValueError("sample channel count changed while loading")
        finite = torch.isfinite(tensor)
        tensor = torch.where(finite, tensor, self.mean)
        tensor = (tensor - self.mean) / self.std
        if not bool(torch.isfinite(tensor).all().item()):
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        if self.labels is None:
            return tensor
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


class DataProcessor(object):
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def _profile_inputs(self, shape, budget):
        n_samples, channels, height, width = shape
        scalar_cap = 4_000_000
        if budget.initial_remaining < 120.0:
            scalar_cap = 250_000
        elif budget.initial_remaining < 600.0:
            scalar_cap = 1_000_000

        profile_seconds = min(20.0, max(0.5, 0.02 * budget.remaining()))
        deadline = time.perf_counter() + profile_seconds
        spatial = max(1, height * width)
        approximate_per_sample = max(1, channels * min(spatial, 4096))
        sample_count = min(n_samples, 256, max(1, scalar_cap // approximate_per_sample))
        if n_samples == 0:
            return np.zeros(channels, dtype=np.float32), np.ones(channels, dtype=np.float32), 1.0

        sample_indices = np.linspace(0, n_samples - 1, num=max(1, sample_count), dtype=np.int64)
        sums = np.zeros(channels, dtype=np.float64)
        sum_squares = np.zeros(channels, dtype=np.float64)
        max_absolute = np.zeros(channels, dtype=np.float64)
        counts = np.zeros(channels, dtype=np.int64)
        observed = 0
        finite_observed = 0
        remaining_scalars = scalar_cap

        for sample_number, sample_index in enumerate(sample_indices):
            if (
                time.perf_counter() >= deadline
                or remaining_scalars <= 0
                or not budget.can_start(0.0, reserve="prediction", margin=0.0)
            ):
                break
            array = np.asarray(self.train_x[int(sample_index)])
            if array.ndim == 2:
                array = array[np.newaxis, :, :]
            flat = array.reshape(channels, -1)
            remaining_samples = max(1, len(sample_indices) - sample_number)
            per_channel = max(1, remaining_scalars // (remaining_samples * max(1, channels)))
            per_channel = min(flat.shape[1], per_channel)
            stride = max(1, int(math.ceil(float(flat.shape[1]) / per_channel)))
            selected = flat[:, ::stride][:, :per_channel].astype(np.float64, copy=False)
            mask = np.isfinite(selected)
            safe = np.where(mask, selected, 0.0)
            sums += safe.sum(axis=1)
            with np.errstate(over="ignore", invalid="ignore"):
                sum_squares += np.square(safe).sum(axis=1)
            max_absolute = np.maximum(max_absolute, np.max(np.abs(safe), axis=1))
            channel_counts = mask.sum(axis=1).astype(np.int64)
            counts += channel_counts
            finite_observed += int(channel_counts.sum())
            observed_here = int(selected.size)
            observed += observed_here
            remaining_scalars -= observed_here

        valid = counts > 0
        mean = np.zeros(channels, dtype=np.float64)
        mean[valid] = sums[valid] / counts[valid]
        variance = np.ones(channels, dtype=np.float64)
        variance[valid] = sum_squares[valid] / counts[valid] - np.square(mean[valid])
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        nonfinite_std = ~np.isfinite(std)
        std[nonfinite_std] = np.maximum(1.0, max_absolute[nonfinite_std])
        std[std < 1.0e-6] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        float_limit = np.finfo(np.float32).max
        mean = np.clip(mean, -float_limit, float_limit)
        std = np.clip(std, 1.0e-6, float_limit)
        finite_rate = 1.0 if observed == 0 else float(finite_observed) / float(observed)
        return mean.astype(np.float32), std.astype(np.float32), finite_rate

    @staticmethod
    def _cuda_profile():
        if not torch.cuda.is_available():
            return "none", 0, 0
        total = 0
        free = 0
        try:
            total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass
        try:
            free = int(torch.cuda.mem_get_info()[0])
        except Exception:
            free = total
        gib = free / float(1024 ** 3) if free else 0.0
        tier = "low" if gib and gib < 5.0 else ("medium" if gib and gib < 12.0 else "high")
        return tier, total, free

    @staticmethod
    def _loader_batch_size(shape, n_train):
        _, channels, height, width = shape
        elements = max(1, channels * height * width)
        target_elements = 2_000_000
        maximum = max(1, min(128, int(n_train) if n_train else 1))
        batch = 1
        while batch * 2 <= maximum and (batch * 2 * elements) <= target_elements:
            batch *= 2
        return batch

    def process(self):
        self.metadata.setdefault("_pipeline_wall_start", time.perf_counter())
        budget = BudgetManager.get(self.clock, self.metadata)
        seed_everything(BASE_SEED)

        train_shape = _shape4(self.train_x)
        valid_shape = _shape4(self.valid_x)
        test_shape = _shape4(self.test_x)
        if train_shape[1:] != valid_shape[1:] or train_shape[1:] != test_shape[1:]:
            raise ValueError("train, validation, and test input shapes must agree")

        train_labels = _as_label_vector(self.train_y)
        valid_labels = _as_label_vector(self.valid_y)
        if len(train_labels) != train_shape[0] or len(valid_labels) != valid_shape[0]:
            raise ValueError("input and label counts do not match")

        original_labels = _stable_unique(list(train_labels) + list(valid_labels))
        if not original_labels:
            original_labels = [0]
        mapping = {}
        for index, value in enumerate(original_labels):
            mapping[_label_key(value)] = index
        mapped_train = _map_labels(train_labels, mapping)
        mapped_valid = _map_labels(valid_labels, mapping)

        counts = np.bincount(mapped_train, minlength=len(original_labels))
        if counts.sum() == 0:
            valid_counts = np.bincount(mapped_valid, minlength=len(original_labels))
            majority_index = int(np.argmax(valid_counts)) if valid_counts.size else 0
        else:
            majority_index = int(np.argmax(counts))
        nonzero_counts = counts[counts > 0]
        imbalance_ratio = float(counts.max()) / float(nonzero_counts.min()) if nonzero_counts.size else 1.0

        started = time.perf_counter()
        mean, std, finite_rate = self._profile_inputs(train_shape, budget)
        budget.record("profiling", time.perf_counter() - started)

        cuda_tier, cuda_bytes, cuda_free_bytes = self._cuda_profile()
        batch_size = self._loader_batch_size(train_shape, train_shape[0])
        sample_elements = int(np.prod(train_shape[1:]))
        estimated_batch_bytes = batch_size * sample_elements * 4
        pin_memory = bool(torch.cuda.is_available() and cuda_tier != "low" and estimated_batch_bytes <= 128 * 1024 ** 2)

        train_dataset = LazyArrayDataset(self.train_x, mapped_train, mean, std)
        valid_dataset = LazyArrayDataset(self.valid_x, mapped_valid, mean, std)
        test_dataset = LazyArrayDataset(self.test_x, None, mean, std)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shape[0] > 0,
            drop_last=False,
            num_workers=0,
            pin_memory=pin_memory,
            generator=make_generator(BASE_SEED),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        dtype = np.asarray(train_labels).dtype if len(train_labels) else np.asarray(valid_labels).dtype
        channels, height, width = train_shape[1:]
        profile = {
            "n_train": train_shape[0],
            "n_valid": valid_shape[0],
            "n_test": test_shape[0],
            "channels": channels,
            "height": height,
            "width": width,
            "input_dtype": str(np.asarray(self.train_x).dtype),
            "bytes_per_sample": int(sample_elements * np.asarray(self.train_x).dtype.itemsize),
            "class_counts": counts.tolist(),
            "imbalance_ratio": imbalance_ratio,
            "finite_rate": finite_rate,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "batch_size": batch_size,
            "cuda_memory_tier": cuda_tier,
            "cuda_total_bytes": cuda_bytes,
            "cuda_free_bytes": cuda_free_bytes,
            "tiny": train_shape[0] < 512,
            "high_resolution": max(height, width) >= 128 or sample_elements >= 250_000,
            "highly_imbalanced": imbalance_ratio >= 10.0,
            "spatially_degenerate": height == 1 or width == 1,
        }
        self.metadata.update(profile)
        self.metadata["profile"] = profile
        self.metadata["num_classes"] = len(original_labels)
        self.metadata["input_shape"] = (
            train_shape[0] + valid_shape[0] + test_shape[0],
            channels,
            height,
            width,
        )
        self.metadata["class_labels"] = list(original_labels)
        self.metadata["_inverse_labels"] = list(original_labels)
        self.metadata["_label_dtype"] = str(dtype)
        self.metadata["majority_index"] = majority_index
        self.metadata["majority_label"] = original_labels[majority_index]
        self.metadata["initial_microbatch"] = batch_size
        self.metadata["pin_memory"] = pin_memory

        if isinstance(test_loader.sampler, RandomSampler) or test_loader.drop_last:
            raise AssertionError("test loader must be ordered and complete")
        if len(test_loader.dataset) != test_shape[0]:
            raise AssertionError("test loader length differs from test input length")
        return train_loader, valid_loader, test_loader
