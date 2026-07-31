"""Regression tests for uncertainty-aware objectives and optimizer diversity."""

import os
import sys
from types import MethodType, SimpleNamespace

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from nas import NAS
from search_space import ArchSpec


def _bare_controller():
    controller = object.__new__(NAS)
    controller.data_props = {
        "class_imbalance_ratio": 1.0,
        "train_samples": 1_000,
    }
    controller.metadata = {"input_shape": [1_000, 1, 8, 8]}
    controller.num_classes = 5
    return controller


def test_recipe_inventory_has_control_and_optimizer_diversity():
    controller = _bare_controller()
    recipes = controller._training_recipes()
    assert any(
        recipe["optimizer"] == "sgd"
        and recipe["scheduler"] == "cosine"
        and recipe["nesterov"]
        for recipe in recipes
    )
    assert any(recipe["label_smoothing"] == 0.0 for recipe in recipes)
    linear = torch.nn.Linear(4, 2)
    sgd_recipe = next(
        recipe for recipe in recipes if recipe["optimizer"] == "sgd"
    )
    optimizer = controller._make_search_optimizer(linear, sgd_recipe)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["nesterov"]


def test_confirmation_accuracy_dominates_adaptive_selection_history():
    controller = _bare_controller()
    overfit = {
        "selection_val_acc": 0.92,
        "confirmation_val_acc": 0.70,
        "val_history": [0.80, 0.90],
    }
    general = {
        "selection_val_acc": 0.84,
        "confirmation_val_acc": 0.82,
        "val_history": [0.78, 0.83],
    }
    scored = controller._final_selection_scores([overfit, general])
    assert scored[1]["selection_score"] > scored[0]["selection_score"]


def test_train_validation_gap_can_resolve_a_near_tie():
    overfit = {
        "best_val_acc": 0.801,
        "metrics_history": [
            {"loss": 1.0, "accuracy": 0.78, "train_accuracy": 0.99},
            {"loss": 0.8, "accuracy": 0.801, "train_accuracy": 0.99},
        ],
    }
    general = {
        "best_val_acc": 0.800,
        "metrics_history": [
            {"loss": 1.0, "accuracy": 0.78, "train_accuracy": 0.83},
            {"loss": 0.8, "accuracy": 0.800, "train_accuracy": 0.83},
        ],
    }
    NAS._recipe_trial_score(overfit)
    NAS._recipe_trial_score(general)
    assert general["selection_score"] > overfit["selection_score"]


def test_family_probe_rewards_learning_and_penalizes_gap():
    improving = {
        "metrics_history": [
            {"loss": 2.0, "accuracy": 0.20, "samples": 200},
            {"loss": 1.0, "accuracy": 0.50, "samples": 200},
        ],
        "train_stats": {"train_accuracy": 0.55},
    }
    memorizing = {
        "metrics_history": [
            {"loss": 2.0, "accuracy": 0.20, "samples": 200},
            {"loss": 1.8, "accuracy": 0.50, "samples": 200},
        ],
        "train_stats": {"train_accuracy": 0.99},
    }
    assert NAS._family_probe_score(improving) > NAS._family_probe_score(
        memorizing
    )


def test_representation_probe_promotes_with_progressive_offsets():
    controller = _bare_controller()
    controller.clock = SimpleNamespace(check=lambda: 1_000.0)
    controller.device = torch.device("cpu")
    controller.seed = 31
    controller.input_h = 8
    controller.input_w = 8
    controller.in_channels = 1
    controller.metadata = {}
    gains = {"spatial": 0.03, "factorized": 0.06, "axis_width": 0.09}
    starts = []
    seed_calls = []
    controller._seed_everything = seed_calls.append

    class _ProbeModel(torch.nn.Module):
        def __init__(self, family):
            super().__init__()
            self.family = family
            self.marker = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, data):
            return data

    class _Space:
        max_safe_stages = 2

        @staticmethod
        def build_model(spec):
            return _ProbeModel(spec.model_family)

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
        starts.append((model.family, int(start_step)))
        with torch.no_grad():
            model.marker.add_(gains[model.family])
        stats = {
            "examples_seen": int(max_steps),
            "train_accuracy": 0.55,
            "train_loss": 1.0,
            "final_lr": 1e-3,
            "peak_memory_mb": 0.0,
        }
        return int(max_steps), 0.1, False, {"param_groups": []}, stats

    def fake_evaluate(self, model, source=None):
        accuracy = 0.30 + float(model.marker.item())
        return {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "loss": 2.0 - 4.0 * float(model.marker.item()),
            "margin": 0.1,
            "samples": 400,
        }

    controller._train_low_fidelity = MethodType(fake_train, controller)
    controller._evaluate = MethodType(fake_evaluate, controller)
    ranked = []
    for family in gains:
        ranked.append(
            {
                "spec": ArchSpec(
                    2, 16, 1, "basic", 3, False, 3, family
                ),
                "params": 10_000,
            }
        )
    selected = controller._representation_probe_race(
        _Space(),
        ranked,
        [None] * 20,
        [None],
        deadline_remaining=0,
    )
    assert selected == {"axis_width", "factorized"}
    assert any(
        family == "axis_width" and start > 0
        for family, start in starts
    )
    assert controller.metadata["nas_promoted_families"] == [
        "axis_width",
        "factorized",
    ]
    assert seed_calls[:6] == [1231, 1831] * 3
    assert seed_calls[6:] == [2231, 2231]


if __name__ == "__main__":
    test_recipe_inventory_has_control_and_optimizer_diversity()
    test_confirmation_accuracy_dominates_adaptive_selection_history()
    test_train_validation_gap_can_resolve_a_near_tie()
    test_family_probe_rewards_learning_and_penalizes_gap()
    test_representation_probe_promotes_with_progressive_offsets()
    print("Search-objective regression tests passed")
