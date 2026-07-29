"""
helpers.py — Shared utilities for the NAS Unseen-Data 2026 submission.

Contains time-display helpers, GPU memory estimation, and logging utilities
used across DataProcessor, NAS, and Trainer.
"""

import math
import torch
import numpy as np


# =============================================================================
# Time display
# =============================================================================

def div_remainder(n, interval):
    """Find divisor and remainder given n / interval."""
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    """Format a duration in seconds as a human-readable string."""
    if seconds < 0:
        return "-" + show_time(-seconds)
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, secs = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, secs)
    else:
        hours, secs = div_remainder(seconds, 60 * 60)
        minutes, secs = div_remainder(secs, 60)
        return "{}h,{}m,{}s".format(hours, minutes, secs)


# =============================================================================
# Time budget management
# =============================================================================

# Budget fractions (of the per-dataset time_limit)
BUDGET_DATA_PROCESSING = 0.03   # ~3%
BUDGET_NAS_SEARCH      = 0.12   # ~12%
BUDGET_HPO             = 0.05   # ~5%  (conditional)
BUDGET_TRAINING        = 0.65   # ~65%
BUDGET_ENSEMBLE_TRAIN  = 0.10   # ~10% (conditional)
BUDGET_SAFETY_MARGIN   = 0.05   # ~5%

# Tier thresholds (in seconds)
TIER3_THRESHOLD = 5 * 60    # <5 min  → immediate fallback
TIER2_THRESHOLD = 15 * 60   # <15 min → ZCP-only, no learning curve


def get_tier(clock):
    """Determine the operating tier based on remaining time."""
    remaining = clock.check()
    if remaining < TIER3_THRESHOLD:
        return 3
    elif remaining < TIER2_THRESHOLD:
        return 2
    else:
        return 1


def time_budget_for(clock, fraction):
    """Return the absolute number of seconds allocated for a phase."""
    return max(0, clock.check() * fraction)


# =============================================================================
# GPU / memory helpers
# =============================================================================

def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_batch_size(input_shape, base_batch_size=128, min_batch_size=8):
    """
    Heuristic batch size based on input tensor size.

    For very large inputs (high-res, many channels) we reduce the batch size
    to avoid OOM.  For small inputs we can increase it.
    """
    # input_shape = [n, C, H, W]
    if len(input_shape) == 4:
        _, c, h, w = input_shape
    elif len(input_shape) == 3:
        # missing channel dim
        _, h, w = input_shape
        c = 1
    else:
        return base_batch_size

    pixels = c * h * w

    # Reference: 3×32×32 = 3072 pixels → batch 128 is fine
    if pixels <= 3072:
        batch_size = base_batch_size
    elif pixels <= 12288:   # e.g. 3×64×64
        batch_size = base_batch_size // 2
    elif pixels <= 49152:   # e.g. 3×128×128
        batch_size = base_batch_size // 4
    else:
        batch_size = base_batch_size // 8

    return max(min_batch_size, batch_size)


# =============================================================================
# Data inspection helpers
# =============================================================================

def inspect_data_properties(train_x):
    """
    Analyze training data to inform adaptive augmentation decisions.

    Returns a dict with:
        - 'channels': int
        - 'height': int
        - 'width': int
        - 'is_small': bool    (spatial dims <= 8)
        - 'is_grayscale': bool (1 channel)
        - 'is_square': bool
        - 'spatial_variance': float  (mean variance across spatial dims)
    """
    shape = train_x.shape
    if len(shape) == 4:
        n, c, h, w = shape
    elif len(shape) == 3:
        n, h, w = shape
        c = 1
    else:
        # Unexpected shape — return safe defaults
        return {
            'channels': 1, 'height': 1, 'width': 1,
            'is_small': True, 'is_grayscale': True, 'is_square': True,
            'spatial_variance': 0.0,
        }

    # Compute spatial variance on a small sample to save time
    sample_size = min(n, 200)
    sample = train_x[:sample_size].astype(np.float32)
    if len(shape) == 3:
        sample = sample.reshape(sample_size, 1, h, w)

    # Mean variance across spatial dimensions (per-channel, averaged)
    # High spatial variance → natural images; low → structured / synthetic
    spatial_var = np.mean(np.var(sample, axis=(2, 3)))

    return {
        'channels': c,
        'height': h,
        'width': w,
        'is_small': (h <= 8 or w <= 8),
        'is_grayscale': (c == 1),
        'is_square': (h == w),
        'spatial_variance': float(spatial_var),
    }
