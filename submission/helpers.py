"""Shared time, device, batch-size, and data-fingerprint helpers."""

import math

import numpy as np
import torch


def div_remainder(value, interval):
    factor = math.floor(value / interval)
    remainder = int(value - factor * interval)
    return factor, remainder


def show_time(seconds):
    seconds = float(seconds)
    if seconds < 0:
        return "-" + show_time(-seconds)
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    if seconds < 3600:
        minutes, secs = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, secs)
    hours, remainder = div_remainder(seconds, 3600)
    minutes, secs = div_remainder(remainder, 60)
    return "{}h,{}m,{}s".format(hours, minutes, secs)


TIER3_THRESHOLD = 5 * 60
TIER2_THRESHOLD = 15 * 60


def get_tier(clock):
    remaining = clock.check()
    if remaining < TIER3_THRESHOLD:
        return 3
    if remaining < TIER2_THRESHOLD:
        return 2
    return 1


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_batch_size(input_shape, base_batch_size=128, min_batch_size=8):
    """Conservative resolution- and GPU-memory-aware batch-size heuristic."""
    if len(input_shape) == 4:
        _, channels, height, width = input_shape
    elif len(input_shape) == 3:
        _, height, width = input_shape
        channels = 1
    else:
        return base_batch_size

    pixels = channels * height * width
    if pixels <= 3072:
        batch_size = base_batch_size
    elif pixels <= 12288:
        batch_size = base_batch_size // 2
    elif pixels <= 49152:
        batch_size = base_batch_size // 4
    else:
        batch_size = base_batch_size // 8

    if torch.cuda.is_available():
        try:
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if memory_gb < 8:
                batch_size //= 2
            elif memory_gb >= 20 and pixels <= 12288:
                batch_size *= 2
        except Exception:
            pass
    return max(min_batch_size, int(batch_size))


def inspect_data_properties(train_x):
    """Build scale-aware fingerprints from a deterministic sample."""
    shape = train_x.shape
    if len(shape) == 4:
        n, channels, height, width = shape
    elif len(shape) == 3:
        n, height, width = shape
        channels = 1
    else:
        return {
            "channels": 1,
            "height": 1,
            "width": 1,
            "is_small": True,
            "is_grayscale": True,
            "is_square": True,
            "spatial_variance": 0.0,
            "normalized_spatial_variance": 0.0,
            "value_mean": 0.0,
            "value_std": 0.0,
            "value_min": 0.0,
            "value_max": 0.0,
            "is_standardized": False,
            "low_variance_color": False,
            "is_structured": True,
        }

    sample_size = min(n, 256)
    indices = np.linspace(0, n - 1, sample_size, dtype=np.int64)
    sample = np.asarray(train_x[indices], dtype=np.float32)
    if len(shape) == 3:
        sample = sample.reshape(sample_size, 1, height, width)

    spatial_variance = float(np.mean(np.var(sample, axis=(2, 3))))
    total_variance = float(np.mean(np.var(sample, axis=(0, 2, 3))))
    normalized_spatial_variance = spatial_variance / max(total_variance, 1e-8)
    value_mean = float(np.mean(sample))
    value_std = float(np.std(sample))
    value_min = float(np.min(sample))
    value_max = float(np.max(sample))

    is_small = height <= 8 or width <= 8
    is_grayscale = channels == 1
    is_standardized = (
        value_min < -0.5
        and value_max > 1.5
        and abs(value_mean) < 0.5
        and 0.4 <= value_std <= 2.5
    )
    low_variance_color = (
        channels >= 3
        and value_min >= -0.1
        and value_max <= 1.5
        and spatial_variance < 0.15
    )

    return {
        "channels": int(channels),
        "height": int(height),
        "width": int(width),
        "is_small": bool(is_small),
        "is_grayscale": bool(is_grayscale),
        "is_square": bool(height == width),
        "spatial_variance": spatial_variance,
        "normalized_spatial_variance": float(normalized_spatial_variance),
        "value_mean": value_mean,
        "value_std": value_std,
        "value_min": value_min,
        "value_max": value_max,
        "is_standardized": bool(is_standardized),
        "low_variance_color": bool(low_variance_color),
        "is_structured": bool(is_small or is_grayscale or is_standardized),
    }
