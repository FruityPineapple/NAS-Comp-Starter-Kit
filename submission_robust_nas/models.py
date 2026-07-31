"""Compact, shape-safe residual CNN search space."""

from __future__ import print_function

import copy
import math

import torch
import torch.nn as nn

from helpers import count_parameters


def fallback_config(width=16):
    return {
        "name": "fallback_tiny_residual",
        "block": "basic",
        "base_width": int(width),
        "stages": 2,
        "blocks_per_stage": [1, 1],
        "kernel": 3,
        "norm": "group",
        "activation": "relu",
        "pool_grid": 1,
        "coordinates": False,
        "attention": False,
        "dropout": 0.0,
        "max_pool_stem": False,
        "adapter_kernel": 3,
        "optimizer": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "mixup": 0.0,
        "label_smoothing": 0.0,
        "class_weighting": False,
    }


def normalize_config(config):
    result = fallback_config()
    result.update(copy.deepcopy(config or {}))
    result["stages"] = max(2, min(4, int(result["stages"])))
    result["base_width"] = max(4, int(result["base_width"]))
    blocks = result.get("blocks_per_stage", 1)
    if isinstance(blocks, int):
        blocks = [blocks] * result["stages"]
    blocks = list(blocks)
    if not blocks:
        blocks = [1]
    while len(blocks) < result["stages"]:
        blocks.append(blocks[-1])
    result["blocks_per_stage"] = [max(1, min(3, int(value))) for value in blocks[:result["stages"]]]
    result["kernel"] = 5 if int(result.get("kernel", 3)) == 5 and result.get("block") == "separable" else 3
    result["pool_grid"] = max(1, min(4, int(result.get("pool_grid", 1))))
    result["dropout"] = max(0.0, min(0.75, float(result.get("dropout", 0.0))))
    return result


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


def make_norm(kind, channels):
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(_group_count(channels), channels)


def make_activation(kind):
    return nn.SiLU(inplace=True) if kind == "silu" else nn.ReLU(inplace=True)


def _kernel_for_shape(kernel, height, width):
    return (1 if height <= 1 else kernel, 1 if width <= 1 else kernel)


def _padding(kernel):
    return (kernel[0] // 2, kernel[1] // 2)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super(SqueezeExcitation, self).__init__()
        hidden = max(4, channels // 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        scale = self.pool(x)
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, norm, activation, attention):
        super(BasicResidualBlock, self).__init__()
        padding = _padding(kernel)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=False)
        self.norm1 = make_norm(norm, out_channels)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, padding=padding, bias=False)
        self.norm2 = make_norm(norm, out_channels)
        self.attention = SqueezeExcitation(out_channels) if attention else nn.Identity()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels and tuple(stride) == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                make_norm(norm, out_channels),
            )
        )
        self.out_activation = make_activation(activation)

    def forward(self, x):
        residual = self.projection(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        return self.out_activation(out + residual)


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, norm, activation, attention):
        super(SeparableResidualBlock, self).__init__()
        padding = _padding(kernel)
        self.depthwise1 = nn.Conv2d(
            in_channels, in_channels, kernel, stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = make_norm(norm, out_channels)
        self.act1 = make_activation(activation)
        self.depthwise2 = nn.Conv2d(
            out_channels, out_channels, kernel, padding=padding,
            groups=out_channels, bias=False,
        )
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm2 = make_norm(norm, out_channels)
        self.attention = SqueezeExcitation(out_channels) if attention else nn.Identity()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels and tuple(stride) == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                make_norm(norm, out_channels),
            )
        )
        self.out_activation = make_activation(activation)

    def forward(self, x):
        residual = self.projection(x)
        out = self.depthwise1(x)
        out = self.act1(self.norm1(self.pointwise1(out)))
        out = self.depthwise2(out)
        out = self.norm2(self.pointwise2(out))
        out = self.attention(out)
        return self.out_activation(out + residual)


class CoordinateChannels(nn.Module):
    def __init__(self, height, width):
        super(CoordinateChannels, self).__init__()
        if height == 1:
            yy = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        else:
            yy = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
        if width == 1:
            xx = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        else:
            xx = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        self.register_buffer("grid", torch.cat((
            yy.expand(1, 1, height, width),
            xx.expand(1, 1, height, width),
        ), dim=1))

    def forward(self, x):
        grid = self.grid
        if grid.dtype != x.dtype:
            grid = grid.to(dtype=x.dtype)
        return torch.cat((x, grid.expand(x.shape[0], -1, -1, -1)), dim=1)


class ConstantClassifier(nn.Module):
    """Trainable-shaped constant output used for genuine one-class tasks."""

    def __init__(self, num_classes=1):
        super(ConstantClassifier, self).__init__()
        self.bias = nn.Parameter(torch.zeros(max(1, int(num_classes))))
        self.config = {
            "name": "constant_classifier",
            "_constant": True,
            "optimizer": "adamw",
            "learning_rate": 0.0,
            "weight_decay": 0.0,
        }

    def forward(self, x):
        return self.bias.view(1, -1).expand(x.shape[0], -1)


class SafeResidualNet(nn.Module):
    def __init__(self, input_channels, num_classes, height, width, config):
        super(SafeResidualNet, self).__init__()
        self.config = normalize_config(config)
        cfg = self.config
        self.coordinates = CoordinateChannels(int(height), int(width)) if cfg["coordinates"] else nn.Identity()
        in_channels = int(input_channels) + (2 if cfg["coordinates"] else 0)
        current_h, current_w = int(height), int(width)
        base_width = cfg["base_width"]
        adapter_kernel_value = 1 if int(cfg.get("adapter_kernel", 3)) == 1 else 3
        adapter_kernel = _kernel_for_shape(adapter_kernel_value, current_h, current_w)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, base_width, adapter_kernel, padding=_padding(adapter_kernel), bias=False),
            make_norm(cfg["norm"], base_width),
            make_activation(cfg["activation"]),
        )

        if cfg.get("max_pool_stem", False):
            stride = (2 if current_h >= 8 else 1, 2 if current_w >= 8 else 1)
            pool_kernel = _kernel_for_shape(3, current_h, current_w)
            self.stem_pool = nn.MaxPool2d(pool_kernel, stride=stride, padding=_padding(pool_kernel))
            current_h = int(math.ceil(float(current_h) / stride[0]))
            current_w = int(math.ceil(float(current_w) / stride[1]))
        else:
            self.stem_pool = nn.Identity()

        layers = []
        stage_shapes = []
        current_channels = base_width
        block_type = BasicResidualBlock if cfg["block"] == "basic" else SeparableResidualBlock
        for stage in range(cfg["stages"]):
            out_channels = base_width * min(2 ** stage, 4)
            for block_index in range(cfg["blocks_per_stage"][stage]):
                if stage > 0 and block_index == 0:
                    stride = (2 if current_h >= 4 else 1, 2 if current_w >= 4 else 1)
                else:
                    stride = (1, 1)
                kernel = _kernel_for_shape(cfg["kernel"], current_h, current_w)
                layers.append(
                    block_type(
                        current_channels, out_channels, kernel, stride,
                        cfg["norm"], cfg["activation"], cfg["attention"],
                    )
                )
                current_channels = out_channels
                current_h = int(math.ceil(float(current_h) / stride[0]))
                current_w = int(math.ceil(float(current_w) / stride[1]))
            stage_shapes.append((current_channels, current_h, current_w))
        self.features = nn.Sequential(*layers)

        pool_h = min(cfg["pool_grid"], max(1, current_h))
        pool_w = min(cfg["pool_grid"], max(1, current_w))
        if height == 1:
            pool_h = 1
        if width == 1:
            pool_w = 1
        self.pool_shape = (pool_h, pool_w)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_shape)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.classifier = nn.Linear(current_channels * pool_h * pool_w, int(num_classes))
        self.stage_shapes = stage_shapes
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.coordinates(x)
        x = self.stem_pool(self.adapter(x))
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(self.dropout(x))


def build_model(metadata, config):
    num_classes = max(1, int(metadata.get("num_classes", 1)))
    if num_classes == 1 or (config and config.get("_constant", False)):
        return ConstantClassifier(num_classes)
    shape = metadata.get("input_shape")
    if shape is None or len(shape) < 4:
        channels = int(metadata.get("channels", 1))
        height = int(metadata.get("height", 1))
        width = int(metadata.get("width", 1))
    else:
        channels, height, width = map(int, shape[-3:])
    return SafeResidualNet(channels, num_classes, height, width, config)


def parameter_cap(metadata):
    n_train = int(metadata.get("n_train", 0))
    if n_train < 2000:
        cap = 750_000
    elif n_train < 20_000:
        cap = 2_000_000
    else:
        cap = 5_000_000
    if metadata.get("high_resolution", False):
        cap = int(cap * 0.65)
    if metadata.get("cuda_memory_tier") == "low":
        cap = int(cap * 0.55)
    return max(100_000, cap)


def estimated_activation_bytes(metadata, config, microbatch):
    cfg = normalize_config(config)
    height = max(1, int(metadata.get("height", 1)))
    width = max(1, int(metadata.get("width", 1)))
    channels = cfg["base_width"]
    total_elements = channels * height * width
    if cfg.get("max_pool_stem", False):
        height = int(math.ceil(float(height) / (2 if height >= 8 else 1)))
        width = int(math.ceil(float(width) / (2 if width >= 8 else 1)))
    for stage in range(cfg["stages"]):
        if stage > 0:
            height = int(math.ceil(float(height) / (2 if height >= 4 else 1)))
            width = int(math.ceil(float(width) / (2 if width >= 4 else 1)))
        channels = cfg["base_width"] * min(2 ** stage, 4)
        total_elements += channels * height * width * cfg["blocks_per_stage"][stage]
    # Rough forward/backward/gradient multiplier; it is a guard, not a proxy score.
    return int(max(1, microbatch) * total_elements * 4 * 7)


def configuration_is_safe(metadata, model, config, microbatch):
    params = count_parameters(model)
    cap = parameter_cap(metadata)
    if params > cap:
        return False, "parameter_cap:{}>{}".format(params, cap)
    activations = estimated_activation_bytes(metadata, config, microbatch)
    cuda_available = int(metadata.get("cuda_free_bytes", 0) or metadata.get("cuda_total_bytes", 0))
    activation_cap = int(0.35 * cuda_available) if cuda_available else 1024 ** 3
    if activations > max(128 * 1024 ** 2, activation_cap):
        return False, "activation_guard:{}>{}".format(activations, activation_cap)
    return True, "ok"
