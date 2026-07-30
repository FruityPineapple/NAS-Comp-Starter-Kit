"""Focused regression tests for structured-grid routing and axis-aware models."""

import os
import sys

import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from data_processor import build_augmentation_pipeline
from helpers import inspect_data_properties
from search_space import ArchSpec, SearchSpace


def make_categorical_grid(samples=64, height=24, width=24, axis="columns"):
    values = np.zeros((samples, 1, height, width), dtype=np.float32)
    for sample_index in range(samples):
        if axis == "columns":
            for column in range(width):
                row = (sample_index * 3 + column * 5) % height
                values[sample_index, 0, row, column] = 1.0
        else:
            for row in range(height):
                column = (sample_index * 3 + row * 5) % width
                values[sample_index, 0, row, column] = 1.0
    return values


def test_categorical_grid_disables_geometry():
    values = make_categorical_grid()
    props = inspect_data_properties(values)
    assert props["is_categorical_grid"]
    assert props["binary_like_fraction"] == 1.0
    assert props["one_hot_column_ratio"] == 1.0
    assert props["sequence_width_confidence"] > 0.9

    mean = values.mean(axis=(0, 2, 3)).tolist()
    std = np.maximum(values.std(axis=(0, 2, 3)), 1e-6).tolist()
    train_transform, eval_transform = build_augmentation_pipeline(props, mean, std)
    assert "RandomAffine" not in repr(train_transform)
    assert "RandomCrop" not in repr(train_transform)
    assert "RandomFlip" not in repr(train_transform)

    sample = torch.from_numpy(values[0])
    assert torch.equal(train_transform(sample), eval_transform(sample))
    assert torch.equal(train_transform(sample), train_transform(sample))


def test_portfolio_families_are_available_and_counted_exactly():
    props = inspect_data_properties(make_categorical_grid())
    space = SearchSpace(1, 10, 24, 24, data_props=props)
    expected = {
        "spatial",
        "spatial_pyramid",
        "factorized",
        "axis_width",
        "axis_height",
        "dual_axis",
        "categorical_sequence",
        "coord_spatial",
        "spatial_axis",
    }
    assert {spec.model_family for spec in space.all_specs()} == expected

    for family in expected:
        spec = ArchSpec(3, 32, 2, "basic", 3, True, 5, family)
        model = space.build_model(spec)
        exact = sum(parameter.numel() for parameter in model.parameters())
        assert space.parameter_count(spec) == exact, family
        output = model(torch.randn(4, 1, 24, 24))
        assert output.shape == (4, 10), family


def test_row_encoded_grid_activates_height_axis():
    props = inspect_data_properties(
        make_categorical_grid(height=20, width=28, axis="rows")
    )
    assert props["one_hot_row_ratio"] == 1.0
    assert props["sequence_height_confidence"] > 0.9
    space = SearchSpace(1, 7, 20, 28, data_props=props)
    assert "axis_height" in space.model_families


def test_axis_encoder_retains_coarse_position():
    values = make_categorical_grid(
        samples=2, height=20, width=28, axis="rows"
    )
    props = inspect_data_properties(values)
    space = SearchSpace(1, 7, 20, 28, data_props=props)
    spec = ArchSpec(2, 16, 1, "basic", 3, False, 3, "axis_height")
    model = space.build_model(spec).eval()
    assert model.encoder.position_bins > 1
    first = torch.from_numpy(values[:1])
    reversed_rows = first.flip(-2)
    with torch.no_grad():
        first_features = model.encoder(first)
        reversed_features = model.encoder(reversed_rows)
    assert not torch.allclose(first_features, reversed_features)


def test_standardized_rgb_keeps_generic_image_anchors_available():
    values = np.random.RandomState(7).randn(64, 3, 28, 28).astype(np.float32)
    props = inspect_data_properties(values)
    assert not props["is_categorical_grid"]

    space = SearchSpace(3, 20, 28, 28, data_props=props)
    assert space.size == 1953
    assert {spec.model_family for spec in space.all_specs()} == {
        "spatial",
        "spatial_pyramid",
        "factorized",
        "multiview",
        "wide_residual",
        "dense_reuse",
    }


def _assert_family_forward(values, classes, family):
    props = inspect_data_properties(values)
    space = SearchSpace(
        values.shape[1],
        classes,
        values.shape[2],
        values.shape[3],
        data_props=props,
    )
    assert family in space.model_families
    spec = next(
        spec for spec in space.all_specs() if spec.model_family == family
    )
    model = space.build_model(spec).eval()
    exact = sum(parameter.numel() for parameter in model.parameters())
    assert space.parameter_count(spec) == exact
    with torch.no_grad():
        output = model(torch.from_numpy(values[:2]).float())
    assert output.shape == (2, classes)


def test_semantic_family_activation_and_forward_paths():
    rng = np.random.RandomState(13)
    dense = rng.rand(32, 1, 8, 128).astype(np.float32)
    _assert_family_forward(dense, 4, "dense_sequence")

    multiview = rng.rand(32, 3, 32, 32).astype(np.float32)
    _assert_family_forward(multiview, 5, "multiview")
    _assert_family_forward(multiview, 5, "wide_residual")
    _assert_family_forward(multiview, 5, "dense_reuse")

    volume = rng.rand(32, 8, 16, 16).astype(np.float32)
    _assert_family_forward(volume, 3, "volumetric")

    board = rng.randint(0, 9, size=(32, 1, 9, 9)).astype(np.float32)
    _assert_family_forward(board, 6, "coord_spatial")


if __name__ == "__main__":
    test_categorical_grid_disables_geometry()
    test_portfolio_families_are_available_and_counted_exactly()
    test_row_encoded_grid_activates_height_axis()
    test_axis_encoder_retains_coarse_position()
    test_standardized_rgb_keeps_generic_image_anchors_available()
    test_semantic_family_activation_and_forward_paths()
    print("Portfolio and structured-grid regression tests passed")
