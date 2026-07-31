"""Regression tests for diverse quotas and decoupled recipe selection."""

import os
import random
import sys
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from nas import NAS
from search_space import ArchSpec


RECIPES = [
    {"name": "stable"},
    {"name": "regularized"},
    {"name": "fast_fit"},
]


class _ConstantClock:
    def __init__(self, remaining):
        self.remaining = float(remaining)

    def check(self):
        return self.remaining


class _TinyRecipeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, data):
        return torch.zeros((data.size(0), 2), device=data.device)


class _TinyRecipeSpace:
    def __init__(self):
        self.models = []

    def build_model(self, spec):
        model = _TinyRecipeModel()
        self.models.append(model)
        return model


def _run_recipe_race(
    schedule,
    incumbent_accuracy=0.84,
    incumbent_loss=0.70,
    samples=1000,
    remaining=1000.0,
    seconds_per_step=0.001,
    loader_steps=20,
):
    controller = object.__new__(NAS)
    controller.clock = _ConstantClock(remaining)
    controller.seed = 17
    controller.device = torch.device("cpu")
    controller.recipe_min_accuracy_gain = 0.001
    recipes = [dict(recipe) for recipe in RECIPES]
    controller._training_recipes = MethodType(
        lambda self: recipes, controller
    )
    controller._default_recipe = MethodType(
        lambda self: recipes[0], controller
    )

    stage_counts = {recipe["name"]: 0 for recipe in recipes}
    train_calls = []
    seed_calls = []
    evaluated_sources = []
    marker_metrics = {}
    for stages in schedule.values():
        for marker, accuracy, loss in stages:
            marker_metrics[float(marker)] = (float(accuracy), float(loss))

    def fake_train(
        self,
        model,
        batch_source,
        max_steps,
        time_quantum,
        deadline_remaining,
        recipe,
        optimizer_state=None,
        start_step=0,
    ):
        name = recipe["name"]
        stage = stage_counts[name]
        stage_counts[name] += 1
        marker, _, _ = schedule[name][stage]
        train_calls.append(
            {
                "name": name,
                "stage": stage + 1,
                "start_marker": float(model.marker.item()),
                "max_steps": int(max_steps),
                "start_step": int(start_step),
                "optimizer_marker": (
                    None
                    if optimizer_state is None
                    else optimizer_state["param_groups"][0]["marker"]
                ),
            }
        )
        with torch.no_grad():
            model.marker.fill_(float(marker))
        state = {
            "state": {},
            "param_groups": [{"marker": float(marker)}],
        }
        accuracy = marker_metrics[float(marker)][0]
        stats = {
            "examples_seen": int(max_steps),
            "train_accuracy": min(1.0, float(accuracy) + 0.05),
            "train_loss": 1.0 - float(accuracy),
            "final_lr": 1e-3,
            "peak_memory_mb": 0.0,
        }
        return int(max_steps), 0.1, False, state, stats

    def fake_evaluate(self, model, source=None):
        evaluated_sources.append(source)
        marker = float(model.marker.item())
        if abs(marker - 1.0) < 1e-6:
            accuracy, loss = incumbent_accuracy, incumbent_loss
        else:
            accuracy, loss = marker_metrics[marker]
        return {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(accuracy),
            "loss": float(loss),
            "samples": int(samples),
        }

    controller._train_low_fidelity = MethodType(fake_train, controller)
    controller._evaluate = MethodType(fake_evaluate, controller)
    controller._seed_everything = seed_calls.append

    architecture_model = _TinyRecipeModel()
    with torch.no_grad():
        architecture_model.marker.fill_(1.0)
    spec = ArchSpec(2, 16, 1, "basic", 3, False, 3, "spatial")
    winner = {
        "model": architecture_model,
        "spec": spec,
        "params": 1,
        "val_acc": incumbent_accuracy,
        "seconds_per_step": seconds_per_step,
        "trained_steps": 10,
        "train_seconds": 1.0,
        "optimizer_state": {
            "state": {},
            "param_groups": [{"marker": 1.0}],
        },
    }
    space = _TinyRecipeSpace()
    validation_source = [None]
    result = controller._recipe_tournament(
        space,
        winner,
        [None] * loader_steps,
        validation_source,
        0.0,
    )
    return result, train_calls, space, {
        "seed_calls": seed_calls,
        "evaluated_sources": evaluated_sources,
        "validation_source": validation_source,
    }


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


def test_recipe_race_preserves_a_stronger_incumbent():
    result, calls, _, observations = _run_recipe_race(
        {
            "stable": [(2, 0.80, 0.75), (5, 0.805, 0.72)],
            "regularized": [(3, 0.79, 0.78), (6, 0.795, 0.74)],
            "fast_fit": [(4, 0.78, 0.80)],
        }
    )
    assert abs(result["val_acc"] - 0.84) < 1e-12
    assert abs(result["model"].marker.item() - 1.0) < 1e-12
    assert (
        result["model"].search_optimizer_state["param_groups"][0]["marker"]
        == 1.0
    )
    stage_one = [call for call in calls if call["stage"] == 1]
    stage_two = [call for call in calls if call["stage"] == 2]
    assert [call["start_marker"] for call in stage_one] == [1.0, 1.0, 1.0]
    assert len(stage_two) == 2
    assert len({call["max_steps"] for call in stage_one}) == 1
    assert len({call["max_steps"] for call in stage_two}) == 1
    assert observations["seed_calls"] == [
        12017,
        12017,
        12017,
        13017,
        13017,
    ]
    assert all(
        source is observations["validation_source"]
        for source in observations["evaluated_sources"]
    )


def test_recipe_race_restores_best_stage_and_matching_optimizer():
    result, calls, space, _ = _run_recipe_race(
        {
            "stable": [(2, 0.860, 0.60), (5, 0.850, 0.55)],
            "regularized": [(3, 0.858, 0.52), (6, 0.870, 0.45)],
            "fast_fit": [(4, 0.800, 0.68)],
        }
    )
    assert result["recipe"]["name"] == "regularized"
    assert abs(result["val_acc"] - 0.870) < 1e-12
    assert abs(result["model"].marker.item() - 6.0) < 1e-12
    assert (
        result["model"].search_optimizer_state["param_groups"][0]["marker"]
        == 6.0
    )
    assert abs(
        result["model"].independent_retry_state["marker"].item() - 1.0
    ) < 1e-12

    stable_model = next(
        model
        for model in space.models
        if model.training_recipe["name"] == "stable"
    )
    assert abs(stable_model.marker.item() - 2.0) < 1e-12
    stage_two = [call for call in calls if call["stage"] == 2]
    assert {call["start_marker"] for call in stage_two} == {2.0, 3.0}
    assert {call["optimizer_marker"] for call in stage_two} == {2.0, 3.0}


def test_recipe_trial_score_uses_validation_loss_slope():
    flat = {
        "best_val_acc": 0.800,
        "metrics_history": [
            {"loss": 1.00},
            {"loss": 0.99},
        ],
    }
    improving = {
        "best_val_acc": 0.799,
        "metrics_history": [
            {"loss": 1.00},
            {"loss": 0.70},
        ],
    }
    NAS._recipe_trial_score(flat)
    NAS._recipe_trial_score(improving)
    assert improving["selection_score"] > flat["selection_score"]


def test_controller_evaluation_reports_cross_entropy_loss():
    controller = object.__new__(NAS)
    controller.device = torch.device("cpu")
    controller.num_classes = 2
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    metrics = controller._evaluate(torch.nn.Identity(), [(logits, targets)])
    expected = torch.nn.functional.cross_entropy(logits, targets).item()
    assert abs(metrics["loss"] - expected) < 1e-12
    assert metrics["accuracy"] == 1.0
    assert metrics["samples"] == 2


def test_controller_evaluation_places_fresh_model_on_controller_device():
    controller = object.__new__(NAS)
    controller.device = torch.device("cpu")
    controller.num_classes = 2
    model = torch.nn.Linear(3, 2)
    observed_devices = []
    placement_calls = []

    class _DeviceRecorder(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def to(self, *args, **kwargs):
            placement_calls.append(str(args[0]))
            return super().to(*args, **kwargs)

        def forward(self, data):
            observed_devices.append(
                (
                    data.device.type,
                    next(self.parameters()).device.type,
                )
            )
            return self.wrapped(data)

    recorded_model = _DeviceRecorder(model)
    controller._evaluate(
        recorded_model,
        [(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))],
    )
    assert placement_calls == ["cpu"]
    assert observed_devices == [("cpu", "cpu")]


def test_controller_evaluation_splits_oom_batches_in_order():
    controller = object.__new__(NAS)
    controller.device = torch.device("cpu")
    controller.num_classes = 2
    controller.metadata = {}

    class _LimitedBatchModel(torch.nn.Module):
        def forward(self, data):
            if data.size(0) > 2:
                raise RuntimeError("out of memory")
            return data

    logits = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1, 0])
    metrics = controller._evaluate(
        _LimitedBatchModel(), [(logits, targets)]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["samples"] == 5
    assert controller.metadata["nas_evaluation_microbatch_size"] <= 2


def test_recipe_race_short_budget_falls_back_without_training():
    result, calls, _, _ = _run_recipe_race(
        {
            "stable": [(2, 0.90, 0.50)],
            "regularized": [(3, 0.91, 0.45)],
            "fast_fit": [(4, 0.89, 0.55)],
        },
        remaining=20.0,
        seconds_per_step=1.0,
    )
    assert calls == []
    assert abs(result["val_acc"] - 0.84) < 1e-12
    assert abs(result["model"].marker.item() - 1.0) < 1e-12


if __name__ == "__main__":
    test_label_aware_slots_are_family_balanced()
    test_architectures_are_not_confounded_with_recipes()
    test_final_score_rewards_accuracy_and_rank_stability()
    test_search_plateau_keeps_state_and_reduces_lr()
    test_recipe_race_preserves_a_stronger_incumbent()
    test_recipe_race_restores_best_stage_and_matching_optimizer()
    test_recipe_trial_score_uses_validation_loss_slope()
    test_controller_evaluation_reports_cross_entropy_loss()
    test_controller_evaluation_places_fresh_model_on_controller_device()
    test_controller_evaluation_splits_oom_batches_in_order()
    test_recipe_race_short_budget_falls_back_without_training()
    print("Controller refinement regression tests passed")
