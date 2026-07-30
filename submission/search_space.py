"""
Compact architecture portfolio for the NAS Unseen-Data Challenge 2026.

Every candidate is an independent PyTorch module.  There is no supernet,
weight sharing, differentiable relaxation, DARTS, or einspace grammar.
"""

import itertools
import math
import random
from collections import namedtuple

import torch
import torch.nn as nn


ArchSpec = namedtuple(
    "ArchSpec",
    [
        "num_stages",
        "init_channels",
        "blocks_per_stage",
        "block_type",
        "kernel_size",
        "use_se",
        "stem_kernel",
        "model_family",
    ],
)
ArchSpec.__new__.__defaults__ = ("spatial",)


def _group_norm(channels):
    groups = min(8, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _norm_2d(channels, kind):
    return _group_norm(channels) if kind == "group" else nn.BatchNorm2d(channels)


def _norm_1d(channels, kind):
    return _group_norm(channels) if kind == "group" else nn.BatchNorm1d(channels)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, dimensions=2):
        super().__init__()
        mid = max(1, channels // reduction)
        pool = nn.AdaptiveAvgPool1d(1) if dimensions == 1 else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            pool,
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x)
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-1)
        return x * scale


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_se=False,
        norm_kind="batch",
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.norm1 = _norm_2d(out_channels, norm_kind)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=pad,
            bias=False,
        )
        self.norm2 = _norm_2d(out_channels, norm_kind)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm_2d(out_channels, norm_kind),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + identity)


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_se=False,
        norm_kind="batch",
    ):
        super().__init__()
        mid = max(8, out_channels // 4)
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.norm1 = _norm_2d(mid, norm_kind)
        self.conv2 = nn.Conv2d(
            mid,
            mid,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.norm2 = _norm_2d(mid, norm_kind)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.norm3 = _norm_2d(out_channels, norm_kind)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm_2d(out_channels, norm_kind),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + identity)


class FactorizedBlock(nn.Module):
    """Depthwise-separable residual block with GroupNorm."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_se=False,
        expansion=2,
    ):
        super().__init__()
        hidden = max(out_channels, int(out_channels * expansion))
        self.expand = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.norm1 = _group_norm(hidden)
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=hidden,
            bias=False,
        )
        self.norm2 = _group_norm(hidden)
        self.project = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.norm3 = _group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.expand(x)))
        out = self.relu(self.norm2(self.depthwise(out)))
        out = self.norm3(self.project(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + identity)


class BasicBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_se=False,
        norm_kind="group",
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.norm1 = _norm_1d(out_channels, norm_kind)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=pad,
            bias=False,
        )
        self.norm2 = _norm_1d(out_channels, norm_kind)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, dimensions=1) if use_se else None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm_1d(out_channels, norm_kind),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + identity)


class BottleneckBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_se=False,
        norm_kind="group",
    ):
        super().__init__()
        mid = max(8, out_channels // 4)
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, mid, 1, bias=False)
        self.norm1 = _norm_1d(mid, norm_kind)
        self.conv2 = nn.Conv1d(
            mid,
            mid,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.norm2 = _norm_1d(mid, norm_kind)
        self.conv3 = nn.Conv1d(mid, out_channels, 1, bias=False)
        self.norm3 = _norm_1d(out_channels, norm_kind)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, dimensions=1) if use_se else None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm_1d(out_channels, norm_kind),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + identity)


class FlexibleNetwork(nn.Module):
    def __init__(self, stem, stages, head):
        super().__init__()
        self.stem = stem
        self.stages = nn.Sequential(*stages)
        self.head = head

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class SpatialPyramidHead(nn.Module):
    """Fuse global and coarse 2x2 layout instead of discarding all position."""

    def __init__(self, channels, num_classes, dropout):
        super().__init__()
        features = 5 * channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.layout_pool = nn.AdaptiveAvgPool2d(2)
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(features, num_classes)

    def forward(self, x):
        global_features = self.global_pool(x).flatten(1)
        layout_features = self.layout_pool(x).flatten(1)
        features = torch.cat([global_features, layout_features], dim=1)
        return self.classifier(self.dropout(self.norm(features)))


class AxisEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        input_height,
        input_width,
        direction,
        stem,
        stages,
        position_bins,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.direction = direction
        self.stem = stem
        self.stages = nn.Sequential(*stages)
        self.position_bins = int(position_bins)

    @staticmethod
    def _add_position_encoding(sequence):
        """Add a parameter-free absolute encoding before sequence reduction."""
        _, channels, length = sequence.shape
        if length <= 1:
            return sequence
        position = torch.linspace(
            0.0,
            1.0,
            length,
            device=sequence.device,
            dtype=torch.float32,
        ).view(1, 1, length)
        channel = torch.arange(
            channels,
            device=sequence.device,
            dtype=torch.float32,
        ).view(1, channels, 1)
        frequency = torch.pow(
            torch.tensor(10_000.0, device=sequence.device),
            -2.0 * torch.floor(channel / 2.0) / max(1, channels),
        )
        angle = position * frequency * (2.0 * math.pi)
        encoding = torch.where(
            (channel.long() % 2) == 0,
            torch.sin(angle),
            torch.cos(angle),
        )
        return sequence + encoding.to(dtype=sequence.dtype)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if (
            channels != self.in_channels
            or height != self.input_height
            or width != self.input_width
        ):
            raise ValueError("Axis encoder received an unexpected input shape")
        if self.direction == "width":
            sequence = x.reshape(batch, channels * height, width)
            frequencies = x.mean(dim=3).flatten(1)
        else:
            sequence = x.permute(0, 1, 3, 2).reshape(
                batch, channels * width, height
            )
            frequencies = x.mean(dim=2).flatten(1)
        sequence = self._add_position_encoding(self.stem(sequence))
        sequence = self.stages(sequence)
        # Coarse ordered bins retain where evidence occurred. A single global
        # mean made row/column sequences with different orders indistinguishable.
        sequence = nn.functional.adaptive_avg_pool1d(
            sequence, self.position_bins
        ).flatten(1)
        return torch.cat([sequence, frequencies], dim=1)


class AxisAwareNetwork(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        return self.head(self.encoder(x))


class DualAxisNetwork(nn.Module):
    def __init__(self, width_encoder, height_encoder, head):
        super().__init__()
        self.width_encoder = width_encoder
        self.height_encoder = height_encoder
        self.head = head

    def forward(self, x):
        features = torch.cat(
            [self.width_encoder(x), self.height_encoder(x)], dim=1
        )
        return self.head(features)


def _init_weights(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class SearchSpace:
    NUM_STAGES = [2, 3, 4]
    INIT_CHANNELS = [16, 32, 64]
    BLOCKS_PER_STAGE = [1, 2, 3]
    BLOCK_TYPES = ["basic", "bottleneck"]
    KERNEL_SIZES = [3, 5]
    USE_SE = [True, False]
    STEM_KERNELS = [3, 5, 7]

    def __init__(
        self,
        in_channels,
        num_classes,
        input_height,
        input_width,
        data_props=None,
    ):
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.data_props = data_props or {}

        hypotheses = self.data_props.get("representation_hypotheses", {})
        width_confidence = float(
            hypotheses.get(
                "sequence_width",
                self.data_props.get("sequence_width_confidence", 0.0),
            )
        )
        height_confidence = float(
            hypotheses.get(
                "sequence_height",
                self.data_props.get("sequence_height_confidence", 0.0),
            )
        )
        position_confidence = float(
            hypotheses.get(
                "position_sensitive",
                self.data_props.get("position_sensitive_confidence", 0.0),
            )
        )

        self.model_families = ["spatial", "spatial_pyramid", "factorized"]
        if width_confidence >= 0.18:
            self.model_families.append("axis_width")
        if height_confidence >= 0.18:
            self.model_families.append("axis_height")
        if (
            max(width_confidence, height_confidence) >= 0.12
            or position_confidence >= 0.35
        ):
            self.model_families.append("dual_axis")

        min_dim = min(self.input_height, self.input_width)
        if min_dim <= 1:
            self.max_safe_stages = 1
        else:
            self.max_safe_stages = max(1, int(math.log2(min_dim / 2)) + 1)
        self.max_safe_stages = min(self.max_safe_stages, max(self.NUM_STAGES))

        print(
            "  [SearchSpace] Input: {}ch x {}x{}, {} classes, "
            "max_stages={}, portfolio={}".format(
                self.in_channels,
                self.input_height,
                self.input_width,
                self.num_classes,
                self.max_safe_stages,
                ",".join(self.model_families),
            )
        )

    def all_specs(self):
        safe_stages = [
            stages
            for stages in self.NUM_STAGES
            if stages <= self.max_safe_stages
        ] or [1]
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
                self.model_families,
            )
        ]

    def sample(self, n=1, rng=None):
        rng = rng or random
        population = self.all_specs()
        return rng.sample(population, min(max(0, int(n)), len(population)))

    @staticmethod
    def _dropout(spec):
        if spec.init_channels >= 64:
            return 0.15
        return 0.10 if spec.init_channels >= 32 else 0.05

    def _build_stages_2d(self, spec):
        if spec.model_family == "factorized":
            block_class = FactorizedBlock
        else:
            block_class = BasicBlock if spec.block_type == "basic" else BottleneckBlock
        stages = []
        current_channels = spec.init_channels
        for stage_index in range(spec.num_stages):
            out_channels = spec.init_channels * (2**stage_index)
            blocks = []
            for block_index in range(spec.blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                kwargs = {
                    "in_channels": current_channels,
                    "out_channels": out_channels,
                    "kernel_size": spec.kernel_size,
                    "stride": stride,
                    "use_se": spec.use_se,
                }
                if block_class is FactorizedBlock:
                    kwargs["expansion"] = (
                        2 if spec.block_type == "basic" else 3
                    )
                blocks.append(block_class(**kwargs))
                current_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        return stages, current_channels

    def _build_spatial_model(self, spec):
        norm_kind = "group" if spec.model_family == "factorized" else "batch"
        stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                spec.init_channels,
                spec.stem_kernel,
                padding=spec.stem_kernel // 2,
                bias=False,
            ),
            _norm_2d(spec.init_channels, norm_kind),
            nn.ReLU(inplace=True),
        )
        stages, current_channels = self._build_stages_2d(spec)
        if spec.model_family == "spatial_pyramid":
            head = SpatialPyramidHead(
                current_channels,
                self.num_classes,
                self._dropout(spec),
            )
        else:
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(self._dropout(spec)),
                nn.Linear(current_channels, self.num_classes),
            )
        return FlexibleNetwork(stem, stages, head)

    def _build_axis_encoder(self, spec, direction):
        input_features = self.in_channels * (
            self.input_height if direction == "width" else self.input_width
        )
        stem = nn.Sequential(
            nn.Conv1d(
                input_features,
                spec.init_channels,
                spec.stem_kernel,
                padding=spec.stem_kernel // 2,
                bias=False,
            ),
            _group_norm(spec.init_channels),
            nn.ReLU(inplace=True),
        )
        block_class = (
            BasicBlock1D if spec.block_type == "basic" else BottleneckBlock1D
        )
        stages = []
        current_channels = spec.init_channels
        for stage_index in range(spec.num_stages):
            out_channels = spec.init_channels * (2**stage_index)
            blocks = []
            for block_index in range(spec.blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    block_class(
                        current_channels,
                        out_channels,
                        kernel_size=spec.kernel_size,
                        stride=stride,
                        use_se=spec.use_se,
                        norm_kind="group",
                    )
                )
                current_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        input_length = (
            self.input_width if direction == "width" else self.input_height
        )
        reduced_length = max(
            1, int(math.ceil(input_length / (2 ** max(0, spec.num_stages - 1))))
        )
        position_bins = min(4, reduced_length)
        encoder = AxisEncoder(
            self.in_channels,
            self.input_height,
            self.input_width,
            direction,
            stem,
            stages,
            position_bins,
        )
        return encoder, current_channels * position_bins + input_features

    def _build_axis_model(self, spec, direction):
        encoder, features = self._build_axis_encoder(spec, direction)
        head = nn.Sequential(
            nn.LayerNorm(features),
            nn.Dropout(self._dropout(spec)),
            nn.Linear(features, self.num_classes),
        )
        return AxisAwareNetwork(encoder, head)

    def _build_dual_axis_model(self, spec):
        width_encoder, width_features = self._build_axis_encoder(spec, "width")
        height_encoder, height_features = self._build_axis_encoder(spec, "height")
        features = width_features + height_features
        head = nn.Sequential(
            nn.LayerNorm(features),
            nn.Dropout(self._dropout(spec)),
            nn.Linear(features, self.num_classes),
        )
        return DualAxisNetwork(width_encoder, height_encoder, head)

    def build_model(self, spec):
        if spec.model_family in ("spatial", "spatial_pyramid", "factorized"):
            model = self._build_spatial_model(spec)
        elif spec.model_family in ("axis", "axis_width"):
            model = self._build_axis_model(spec, "width")
        elif spec.model_family == "axis_height":
            model = self._build_axis_model(spec, "height")
        elif spec.model_family == "dual_axis":
            model = self._build_dual_axis_model(spec)
        else:
            raise ValueError("Unknown model family: {}".format(spec.model_family))
        model.apply(_init_weights)
        return model

    @staticmethod
    def _se_params(channels):
        middle = max(1, channels // 16)
        return 2 * channels * middle

    def _backbone_2d_parameter_count(self, spec):
        total = (
            self.in_channels * spec.init_channels * spec.stem_kernel**2
            + 2 * spec.init_channels
        )
        current_channels = spec.init_channels
        for stage_index in range(spec.num_stages):
            out_channels = spec.init_channels * (2**stage_index)
            for block_index in range(spec.blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                if spec.model_family == "factorized":
                    expansion = 2 if spec.block_type == "basic" else 3
                    hidden = max(out_channels, out_channels * expansion)
                    total += current_channels * hidden + 2 * hidden
                    total += hidden * spec.kernel_size**2 + 2 * hidden
                    total += hidden * out_channels + 2 * out_channels
                elif spec.block_type == "basic":
                    total += (
                        current_channels
                        * out_channels
                        * spec.kernel_size**2
                        + 2 * out_channels
                    )
                    total += (
                        out_channels
                        * out_channels
                        * spec.kernel_size**2
                        + 2 * out_channels
                    )
                else:
                    middle = max(8, out_channels // 4)
                    total += current_channels * middle + 2 * middle
                    total += (
                        middle * middle * spec.kernel_size**2 + 2 * middle
                    )
                    total += middle * out_channels + 2 * out_channels
                if stride != 1 or current_channels != out_channels:
                    total += current_channels * out_channels + 2 * out_channels
                if spec.use_se:
                    total += self._se_params(out_channels)
                current_channels = out_channels
        return total, current_channels

    def _axis_encoder_parameter_count(
        self, spec, input_features, input_length
    ):
        total = (
            input_features * spec.init_channels * spec.stem_kernel
            + 2 * spec.init_channels
        )
        current_channels = spec.init_channels
        for stage_index in range(spec.num_stages):
            out_channels = spec.init_channels * (2**stage_index)
            for block_index in range(spec.blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                if spec.block_type == "basic":
                    total += (
                        current_channels
                        * out_channels
                        * spec.kernel_size
                        + 2 * out_channels
                    )
                    total += (
                        out_channels
                        * out_channels
                        * spec.kernel_size
                        + 2 * out_channels
                    )
                else:
                    middle = max(8, out_channels // 4)
                    total += current_channels * middle + 2 * middle
                    total += middle * middle * spec.kernel_size + 2 * middle
                    total += middle * out_channels + 2 * out_channels
                if stride != 1 or current_channels != out_channels:
                    total += current_channels * out_channels + 2 * out_channels
                if spec.use_se:
                    total += self._se_params(out_channels)
                current_channels = out_channels
        reduced_length = max(
            1, int(math.ceil(input_length / (2 ** max(0, spec.num_stages - 1))))
        )
        position_bins = min(4, reduced_length)
        return total, current_channels * position_bins + input_features

    def parameter_count(self, spec):
        if spec.model_family in ("spatial", "spatial_pyramid", "factorized"):
            total, features = self._backbone_2d_parameter_count(spec)
            if spec.model_family == "spatial_pyramid":
                features *= 5
                total += 2 * features
            total += features * self.num_classes + self.num_classes
            return int(total)

        width_features = self.in_channels * self.input_height
        height_features = self.in_channels * self.input_width
        if spec.model_family in ("axis", "axis_width"):
            total, features = self._axis_encoder_parameter_count(
                spec, width_features, self.input_width
            )
        elif spec.model_family == "axis_height":
            total, features = self._axis_encoder_parameter_count(
                spec, height_features, self.input_height
            )
        elif spec.model_family == "dual_axis":
            width_total, width_output = self._axis_encoder_parameter_count(
                spec, width_features, self.input_width
            )
            height_total, height_output = self._axis_encoder_parameter_count(
                spec, height_features, self.input_height
            )
            total = width_total + height_total
            features = width_output + height_output
        else:
            raise ValueError("Unknown model family: {}".format(spec.model_family))
        total += 2 * features
        total += features * self.num_classes + self.num_classes
        return int(total)

    @property
    def size(self):
        safe_stages = max(
            1,
            len(
                [
                    stages
                    for stages in self.NUM_STAGES
                    if stages <= self.max_safe_stages
                ]
            ),
        )
        return (
            safe_stages
            * len(self.INIT_CHANNELS)
            * len(self.BLOCKS_PER_STAGE)
            * len(self.BLOCK_TYPES)
            * len(self.KERNEL_SIZES)
            * len(self.USE_SE)
            * len(self.STEM_KERNELS)
            * len(self.model_families)
        )
