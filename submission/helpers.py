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
            "binary_like_fraction": 0.0,
            "active_density": 0.0,
            "one_hot_column_ratio": 0.0,
            "one_hot_row_ratio": 0.0,
            "sequence_width_confidence": 0.0,
            "sequence_height_confidence": 0.0,
            "position_sensitive_confidence": 0.0,
            "factorized_confidence": 0.0,
            "representation_hypotheses": {"spatial": 1.0},
            "is_categorical_grid": False,
            "is_structured": True,
        }

    bytes_per_float_sample = max(1, channels * height * width * 4)
    byte_limited_samples = max(
        1, int(64 * 1024 * 1024) // bytes_per_float_sample
    )
    sample_size = min(n, 256, byte_limited_samples)
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

    # Some competition datasets encode a categorical sequence as a sparse
    # binary image: rows identify symbols and columns identify positions.  Such
    # inputs are not translation-invariant images.  Detect the representation
    # from its contents rather than from a dataset name.
    binary_distance = np.minimum(np.abs(sample), np.abs(sample - 1.0))
    binary_like_fraction = float(np.mean(binary_distance <= 1e-4))
    if binary_like_fraction >= 0.98:
        active = sample > 0.5
        active_density = float(np.mean(active))
        active_per_column = np.sum(active, axis=2)
        active_per_row = np.sum(active, axis=3)
        one_hot_column_ratio = float(np.mean(active_per_column == 1))
        one_hot_row_ratio = float(np.mean(active_per_row == 1))
    else:
        active_density = 0.0
        one_hot_column_ratio = 0.0
        one_hot_row_ratio = 0.0

    binary_confidence = min(1.0, max(0.0, (binary_like_fraction - 0.90) / 0.08))
    sparse_confidence = (
        1.0
        if 0.005 <= active_density <= 0.15
        else max(0.0, 1.0 - abs(active_density - 0.075) / 0.20)
    )
    shape_confidence = 1.0 if min(height, width) >= 12 else 0.25
    sequence_width_confidence = float(
        binary_confidence
        * sparse_confidence
        * shape_confidence
        * max(one_hot_column_ratio, 0.35 * one_hot_row_ratio)
    )
    sequence_height_confidence = float(
        binary_confidence
        * sparse_confidence
        * shape_confidence
        * max(one_hot_row_ratio, 0.35 * one_hot_column_ratio)
    )
    position_sensitive_confidence = float(
        max(
            sequence_width_confidence,
            sequence_height_confidence,
            0.55 if (channels == 1 and is_small) else 0.0,
            0.45
            if (
                channels == 1
                and max(height, width) >= 2 * max(1, min(height, width))
            )
            else 0.0,
        )
    )
    factorized_confidence = float(
        max(
            0.25,
            0.75 if channels >= 3 else 0.0,
            0.65 if max(height, width) >= 64 else 0.0,
        )
    )
    if channels > 1:
        channel_signatures = sample.mean(axis=(2, 3))
        channel_std = channel_signatures.std(axis=0)
        stable_channels = channel_std > 1e-5
        if stable_channels.sum() >= 2:
            correlations = np.corrcoef(
                channel_signatures[:, stable_channels], rowvar=False
            )
            off_diagonal = correlations[
                ~np.eye(correlations.shape[0], dtype=bool)
            ]
            mean_channel_correlation = float(
                np.nanmean(np.abs(off_diagonal))
            )
        else:
            mean_channel_correlation = 1.0
    else:
        mean_channel_correlation = 1.0
    channel_independence_confidence = float(
        max(
            0.0,
            min(
                1.0,
                (1.0 - mean_channel_correlation)
                * (1.0 if 2 <= channels <= 12 else 0.35),
            ),
        )
    )

    continuous_confidence = max(
        0.0, min(1.0, (0.95 - binary_like_fraction) / 0.50)
    )
    short_axis = min(height, width)
    long_axis = max(height, width)
    dense_sequence_confidence = float(
        continuous_confidence
        * (
            1.0
            if short_axis <= 16 and long_axis >= 64
            else (
                0.55
                if short_axis <= 24 and long_axis >= 4 * short_axis
                else 0.0
            )
        )
    )
    dense_sequence_direction = (
        "height" if height <= width else "width"
    )

    dimension_ratio = max(channels, height, width) / max(
        1, min(channels, height, width)
    )
    volumetric_confidence = float(
        (
            max(0.0, 1.0 - (dimension_ratio - 1.0) / 4.0)
            * (0.55 + 0.45 * mean_channel_correlation)
        )
        if 4 <= channels <= 32 and min(height, width) >= 8
        else 0.0
    )
    temporal_confidence = float(
        (
            0.35 + 0.65 * mean_channel_correlation
        )
        if 3 <= channels <= 32 and min(height, width) >= 8
        else 0.0
    )

    uniqueness_sample = sample[
        : min(sample.shape[0], 8)
    ]
    rounded_unique = int(
        min(
            257,
            np.unique(np.round(uniqueness_sample, decimals=4)).size,
        )
    )
    discrete_confidence = (
        1.0
        if rounded_unique <= 32
        else max(0.0, 1.0 - (rounded_unique - 32) / 224.0)
    )
    board_confidence = float(
        discrete_confidence
        * (
            1.0
            if 4 <= min(height, width) and max(height, width) <= 16
            else 0.0
        )
    )
    is_categorical_grid = (
        channels == 1
        and max(sequence_width_confidence, sequence_height_confidence) >= 0.70
    )
    natural_image_confidence = float(
        (
            0.85
            if channels == 3
            and not is_categorical_grid
            and min(height, width) >= 16
            and rounded_unique > 64
            and normalized_spatial_variance >= 0.50
            else (
                0.35
                if channels in (1, 3)
                and min(height, width) >= 24
                and not is_categorical_grid
                else 0.0
            )
        )
    )
    fixed_coordinate_confidence = float(
        max(
            position_sensitive_confidence,
            board_confidence,
            0.55 if is_standardized and channels > 1 else 0.0,
        )
    )
    representation_hypotheses = {
        # Spatial structure is never ruled out from a fingerprint alone.
        "spatial": 1.0,
        "position_sensitive": position_sensitive_confidence,
        "sequence_width": sequence_width_confidence,
        "sequence_height": sequence_height_confidence,
        "factorized": factorized_confidence,
        "dense_sequence": dense_sequence_confidence,
        "channel_independent": channel_independence_confidence,
        "volumetric": volumetric_confidence,
        "temporal": temporal_confidence,
        "board": board_confidence,
        "fixed_coordinate": fixed_coordinate_confidence,
        "natural_image": natural_image_confidence,
    }

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
        "binary_like_fraction": binary_like_fraction,
        "active_density": active_density,
        "one_hot_column_ratio": one_hot_column_ratio,
        "one_hot_row_ratio": one_hot_row_ratio,
        "sequence_width_confidence": sequence_width_confidence,
        "sequence_height_confidence": sequence_height_confidence,
        "position_sensitive_confidence": position_sensitive_confidence,
        "factorized_confidence": factorized_confidence,
        "dense_sequence_confidence": dense_sequence_confidence,
        "dense_sequence_direction": dense_sequence_direction,
        "channel_independence_confidence": (
            channel_independence_confidence
        ),
        "mean_channel_correlation": mean_channel_correlation,
        "volumetric_confidence": volumetric_confidence,
        "temporal_confidence": temporal_confidence,
        "board_confidence": board_confidence,
        "fixed_coordinate_confidence": fixed_coordinate_confidence,
        "natural_image_confidence": natural_image_confidence,
        "estimated_unique_values": rounded_unique,
        "fingerprint_samples": int(sample_size),
        "representation_hypotheses": representation_hypotheses,
        "is_categorical_grid": bool(is_categorical_grid),
        "is_structured": bool(
            is_small
            or is_grayscale
            or is_standardized
            or is_categorical_grid
            or position_sensitive_confidence >= 0.50
        ),
    }
