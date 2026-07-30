"""Regression tests for diverse quotas and decoupled recipe selection."""

import os
import random
import sys
from types import SimpleNamespace


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from nas import NAS
from search_space import ArchSpec


RECIPES = [
    {"name": "stable"},
    {"name": "regularized"},
    {"name": "fast_fit"},
]


def _ranked_entries(families, per_family=6):
    entries = []
    for family in reversed(families):
        for index in range(per_family):
            spec = ArchSpec(
                2,
                16,
                1 + index % 3,
                "basic",
                3 if index < 3 else 5,
                bool(index % 2),
                3,
                family,
            )
            entries.append(
                {
                    "spec": spec,
                    "recipe": RECIPES[index % len(RECIPES)],
                    "params": 20_000 * (index + 1),
                    "proxy_prior": 1.0 - len(entries) / 100.0,
                }
            )
    return entries


def test_label_aware_slots_are_family_balanced():
    controller = object.__new__(NAS)
    natural_families = ["spatial", "spatial_pyramid", "factorized"]
    selected = controller._select_label_aware_entries(
        _ranked_entries(natural_families), 12
    )
    counts = {
        family: sum(
            entry["spec"].model_family == family for entry in selected
        )
        for family in natural_families
    }
    assert set(counts.values()) == {4}
    for family in natural_families:
        family_params = sorted(
            entry["params"]
            for entry in selected
            if entry["spec"].model_family == family
        )
        assert family_params[0] < family_params[-1]

    structured_families = natural_families + [
        "axis_width",
        "axis_height",
        "dual_axis",
    ]
    selected = controller._select_label_aware_entries(
        _ranked_entries(structured_families), 12
    )
    counts = {
        family: sum(
            entry["spec"].model_family == family for entry in selected
        )
        for family in structured_families
    }
    assert set(counts.values()) == {2}


def test_architectures_are_not_confounded_with_recipes():
    controller = object.__new__(NAS)
    controller.input_h = 28
    controller.input_w = 28
    controller.rng = random.Random(7)
    families = ["spatial", "spatial_pyramid", "factorized"]
    space = SimpleNamespace(
        model_families=families,
        max_safe_stages=4,
    )

    pool = []
    for family in families:
        for index in range(12):
            spec = ArchSpec(
                3,
                32,
                1 + index % 3,
                "basic" if index < 6 else "bottleneck",
                3 if index % 2 == 0 else 5,
                bool(index % 3 == 0),
                3 if index < 9 else 5,
                family,
            )
            if spec not in {existing for existing, _ in pool}:
                pool.append((spec, 50_000 + index * 1_000))

    entries = controller._portfolio_candidates(
        space, pool, 27, RECIPES
    )
    for family in families:
        names = [
            entry["recipe"]["name"]
            for entry in entries
            if entry["spec"].model_family == family
        ]
        assert names
        assert set(names) == {"architecture_probe"}


def test_final_score_rewards_accuracy_and_rank_stability():
    controller = object.__new__(NAS)
    stable = {
        "full_val_acc": 0.82,
        "val_history": [0.60, 0.70, 0.80],
    }
    late_flip = {
        "full_val_acc": 0.825,
        "val_history": [0.40, 0.55, 0.79],
    }
    entries = controller._final_selection_scores([stable, late_flip])
    assert entries[0]["selection_score"] > entries[1]["selection_score"]


def test_search_plateau_keeps_state_and_reduces_lr():
    controller = object.__new__(NAS)
    entry = {
        "val_history": [0.70, 0.701, 0.7005],
        "optimizer_state": {
            "state": {0: {"step": 10}},
            "param_groups": [{"lr": 1e-3}],
        },
    }
    original_state = entry["optimizer_state"]["state"]
    controller._adapt_search_lr(entry)
    assert entry["optimizer_state"]["state"] is original_state
    assert abs(
        entry["optimizer_state"]["param_groups"][0]["lr"] - 3.5e-4
    ) < 1e-12


if __name__ == "__main__":
    test_label_aware_slots_are_family_balanced()
    test_architectures_are_not_confounded_with_recipes()
    test_final_score_rewards_accuracy_and_rank_stability()
    test_search_plateau_keeps_state_and_reduces_lr()
    print("Controller refinement regression tests passed")
