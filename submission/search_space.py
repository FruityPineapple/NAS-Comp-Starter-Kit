"""
search_space.py — Macro-level NAS search space for the NAS Unseen-Data Challenge 2026.

Defines a parameterized architecture family over depth, width, kernel size,
block type, and attention, with ~2500-5000 discrete candidates. Each candidate
is a standalone nn.Module — no mixed-ops or discretization required.
"""

import math
import random
import itertools
from collections import namedtuple

import torch
import torch.nn as nn


# =============================================================================
# Architecture Specification
# =============================================================================

ArchSpec = namedtuple('ArchSpec', [
    'num_stages',       # int: 2, 3, or 4
    'init_channels',    # int: 16, 32, or 64
    'blocks_per_stage', # int: 1, 2, or 3
    'block_type',       # str: 'basic' or 'bottleneck'
    'kernel_size',      # int: 3 or 5
    'use_se',           # bool
    'stem_kernel',      # int: 3, 5, or 7
])


# =============================================================================
# Building Blocks
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two conv-BN-ReLU with residual."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=False):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.se = SEBlock(out_channels) if use_se else None

        # Shortcut projection when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out = self.relu(out + identity)
        return out


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block: 1x1 -> kxk -> 1x1 with residual."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=False):
        super().__init__()
        mid = max(8, out_channels // 4)
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.se = SEBlock(out_channels) if use_se else None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = self.relu(out + identity)
        return out


# =============================================================================
# Network Assembly
# =============================================================================

class FlexibleNetwork(nn.Module):
    """
    A complete network assembled from stem + stages + head.

    This is the model returned by SearchSpace.build_model().
    """

    def __init__(self, stem, stages, head):
        super().__init__()
        self.stem = stem
        self.stages = nn.Sequential(*stages)
        self.head = head

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x


def _init_weights(module):
    """Kaiming initialization for conv layers, constant for BN."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# =============================================================================
# Search Space
# =============================================================================

class SearchSpace:
    """
    Macro-level search space over a ResNet-like backbone.

    Searchable dimensions:
        - Number of stages (depth tiers)
        - Initial channel width (doubles per stage)
        - Blocks per stage
        - Block type (BasicBlock vs BottleneckBlock)
        - Conv kernel size
        - Squeeze-and-Excitation attention
        - Stem kernel size
    """

    NUM_STAGES = [2, 3, 4]
    INIT_CHANNELS = [16, 32, 64]
    BLOCKS_PER_STAGE = [1, 2, 3]
    BLOCK_TYPES = ['basic', 'bottleneck']
    KERNEL_SIZES = [3, 5]
    USE_SE = [True, False]
    STEM_KERNELS = [3, 5, 7]

    def __init__(self, in_channels, num_classes, input_height, input_width):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width

        # Determine maximum safe number of stages.
        # Each stage beyond stage 0 halves spatial resolution via stride=2.
        # We need the final feature map to be at least 2x2.
        min_dim = min(input_height, input_width)
        # After stage 0 (stride=1): min_dim unchanged
        # After stage 1 (stride=2): min_dim // 2
        # After stage k (k>=1):     min_dim // 2^k
        # Need min_dim // 2^(num_stages-1) >= 2
        if min_dim <= 1:
            self.max_safe_stages = 1
        else:
            # 2^(num_stages-1) <= min_dim/2  =>  num_stages-1 <= log2(min_dim/2)
            self.max_safe_stages = max(1, int(math.log2(min_dim / 2)) + 1)

        # Clamp to our search space range
        self.max_safe_stages = min(self.max_safe_stages, max(self.NUM_STAGES))

        print("  [SearchSpace] Input: {}ch x {}x{}, {} classes, max_stages={}".format(
            in_channels, input_height, input_width, num_classes, self.max_safe_stages))

    def all_specs(self):
        """Enumerate every valid architecture exactly once."""
        safe_stages = [s for s in self.NUM_STAGES if s <= self.max_safe_stages]
        if not safe_stages:
            safe_stages = [1]
        return [
            ArchSpec(*values)
            for values in itertools.product(
                safe_stages,
                self.INIT_CHANNELS,
                self.BLOCKS_PER_STAGE,
                self.BLOCK_TYPES,
                self.KERNEL_SIZES,
                self.USE_SE,
                self.STEM_KERNELS,
            )
        ]

    def sample(self, n=1, rng=None):
        """Sample unique architecture specs without replacement."""
        rng = rng or random
        population = self.all_specs()
        n = min(max(0, int(n)), len(population))
        return rng.sample(population, n)

    def build_model(self, spec):
        """Construct a full PyTorch model from an ArchSpec."""
        BlockClass = BasicBlock if spec.block_type == 'basic' else BottleneckBlock

        # --- Stem ---
        stem = nn.Sequential(
            nn.Conv2d(self.in_channels, spec.init_channels, spec.stem_kernel,
                      stride=1, padding=spec.stem_kernel // 2, bias=False),
            nn.BatchNorm2d(spec.init_channels),
            nn.ReLU(inplace=True),
        )

        # --- Stages ---
        stages = []
        current_channels = spec.init_channels

        for stage_idx in range(spec.num_stages):
            out_channels = spec.init_channels * (2 ** stage_idx)
            blocks = []

            for block_idx in range(spec.blocks_per_stage):
                # First block in stages 1+ uses stride=2 for downsampling
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1

                blocks.append(BlockClass(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=spec.kernel_size,
                    stride=stride,
                    use_se=spec.use_se,
                ))
                current_channels = out_channels

            stages.append(nn.Sequential(*blocks))

        # --- Head ---
        dropout = 0.15 if spec.init_channels >= 64 else (0.10 if spec.init_channels >= 32 else 0.05)
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(current_channels, self.num_classes),
        )

        # --- Assemble and initialize ---
        model = FlexibleNetwork(stem, stages, head)
        model.apply(_init_weights)
        return model

    def parameter_count(self, spec):
        """Return the exact trainable parameter count without building a model."""
        total = (
            self.in_channels * spec.init_channels * spec.stem_kernel ** 2
            + 2 * spec.init_channels  # stem BatchNorm
        )
        current_channels = spec.init_channels

        for stage_idx in range(spec.num_stages):
            out_channels = spec.init_channels * (2 ** stage_idx)
            for block_idx in range(spec.blocks_per_stage):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1

                if spec.block_type == 'basic':
                    total += current_channels * out_channels * spec.kernel_size ** 2
                    total += 2 * out_channels
                    total += out_channels * out_channels * spec.kernel_size ** 2
                    total += 2 * out_channels
                else:
                    mid = max(8, out_channels // 4)
                    total += current_channels * mid + 2 * mid
                    total += mid * mid * spec.kernel_size ** 2 + 2 * mid
                    total += mid * out_channels + 2 * out_channels

                if stride != 1 or current_channels != out_channels:
                    total += current_channels * out_channels + 2 * out_channels
                if spec.use_se:
                    se_mid = max(1, out_channels // 16)
                    total += 2 * out_channels * se_mid
                current_channels = out_channels

        total += current_channels * self.num_classes + self.num_classes
        return int(total)

    @property
    def size(self):
        """Total number of architectures in the search space (before safety filtering)."""
        safe_stages = len([s for s in self.NUM_STAGES if s <= self.max_safe_stages])
        return (safe_stages *
                len(self.INIT_CHANNELS) *
                len(self.BLOCKS_PER_STAGE) *
                len(self.BLOCK_TYPES) *
                len(self.KERNEL_SIZES) *
                len(self.USE_SE) *
                len(self.STEM_KERNELS))
