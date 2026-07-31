"""Focused checks for final optimizer, EMA, TTA, and clock-driven training."""

import os
import sys

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submission"))

from trainer import Trainer


class _ConstantClock:
    def check(self):
        return 1_000.0


class _AcceleratedClock:
    def __init__(self, remaining=120.0, decrement=2.0):
        self.remaining = float(remaining)
        self.decrement = float(decrement)

    def check(self):
        self.remaining -= self.decrement
        return self.remaining


class _CountingModel(torch.nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.linear = torch.nn.Linear(4, classes)
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        return self.linear(data.flatten(1))


def _loaders(samples=24):
    data = torch.randn(samples, 1, 2, 2)
    labels = (data.flatten(1).sum(dim=1) > 0).long()
    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False
    )
    return loader, data


def _metadata(policy="identity"):
    return {
        "num_classes": 2,
        "input_shape": [48, 1, 2, 2],
        "batch_size": 8,
        "test_num_batches": 3,
        "test_num_samples": 24,
        "augmentation_policy": policy,
        "data_props": {
            "class_weights": [1.0, 1.0],
            "horizontal_flip_safe": policy == "safe_flips",
            "vertical_flip_safe": policy == "safe_flips",
        },
    }


def test_sgd_nesterov_uses_cosine_and_rejects_adam_moments():
    loader, _ = _loaders()
    model = _CountingModel()
    recipe = {
        "name": "sgd_nesterov",
        "optimizer": "sgd",
        "scheduler": "cosine",
        "final_lr": 0.05,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 5e-4,
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
    }
    model.training_recipe = recipe
    trainer = Trainer(
        model,
        torch.device("cpu"),
        loader,
        loader,
        _metadata(),
        _ConstantClock(),
    )
    trainer.safe_epochs = 10
    adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer, scheduler = trainer._make_optimizer_and_scheduler(
        model, adam.state_dict()
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["nesterov"]
    assert isinstance(
        scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
    )


def test_cosine_scheduler_cannot_rebound_after_initial_horizon():
    loader, _ = _loaders()
    model = _CountingModel()
    model.training_recipe = {
        "name": "sgd_nesterov",
        "optimizer": "sgd",
        "scheduler": "cosine",
        "final_lr": 0.02,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 5e-4,
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
    }
    trainer = Trainer(
        model,
        torch.device("cpu"),
        loader,
        loader,
        _metadata(),
        _ConstantClock(),
    )
    trainer.safe_epochs = 2
    optimizer, scheduler = trainer._make_optimizer_and_scheduler(model)
    optimizer.param_groups[0]["lr"] = trainer.base_lr
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        previous = optimizer.param_groups[0]["lr"]
        optimizer.step()
        trainer._step_scheduler(scheduler, 0.5)
        trainer._apply_time_cooldown(
            optimizer, elapsed=0.0, previous_lr=previous
        )
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert all(
        current <= previous + 1e-12
        for previous, current in zip(
            learning_rates, learning_rates[1:]
        )
    )
    assert learning_rates[-1] == learning_rates[2]


def test_attempt_rotation_distinguishes_failure_from_slow_recovery():
    failure = Trainer._attempt_rotation_reason(
        attempt_epochs=12,
        attempt_best_val=0.3967,
        attempt_start_best_val=0.4123,
        epochs_without_improvement=6,
        train_accuracy=0.6631,
        validation_accuracy=0.3949,
        current_lr=0.019,
        base_lr=0.020,
        validation_samples=5_000,
        plateau_patience=4,
    )
    assert failure is not None

    slow_recovery = Trainer._attempt_rotation_reason(
        attempt_epochs=32,
        attempt_best_val=0.8465,
        attempt_start_best_val=0.8410,
        epochs_without_improvement=24,
        train_accuracy=1.0,
        validation_accuracy=0.8271,
        current_lr=0.016,
        base_lr=0.020,
        validation_samples=5_000,
        plateau_patience=5,
    )
    assert slow_recovery is None

    exhausted_success = Trainer._attempt_rotation_reason(
        attempt_epochs=80,
        attempt_best_val=0.8643,
        attempt_start_best_val=0.8410,
        epochs_without_improvement=30,
        train_accuracy=1.0,
        validation_accuracy=0.8521,
        current_lr=0.0075,
        base_lr=0.020,
        validation_samples=5_000,
        plateau_patience=5,
    )
    assert exhausted_success is not None


def test_warm_checkpoint_is_a_target_not_trained_recovery_evidence():
    """Model the post-start accounting used by Trainer.train()."""
    preserved_baseline = 0.3121
    trained_best = -float("inf")
    epochs_without_improvement = 0
    validation_trajectory = [
        0.2876,
        0.2864,
        0.2715,
        0.2999,
        0.2798,
        0.2805,
        0.2986,
        0.2374,
        0.2853,
        0.2899,
        0.2869,
        0.2454,
    ]
    for score in validation_trajectory:
        if score > trained_best + 1e-6:
            trained_best = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    reason = Trainer._attempt_rotation_reason(
        attempt_epochs=len(validation_trajectory),
        attempt_best_val=trained_best,
        attempt_start_best_val=preserved_baseline,
        epochs_without_improvement=epochs_without_improvement,
        train_accuracy=0.6149,
        validation_accuracy=validation_trajectory[-1],
        current_lr=0.020,
        base_lr=0.020,
        validation_samples=4_096,
        plateau_patience=5,
    )
    assert reason == "failed to recover the preserved baseline"


def test_large_generalization_gap_prioritizes_regularized_retry():
    loader, _ = _loaders()
    model = _CountingModel()
    trainer = Trainer(
        model,
        torch.device("cpu"),
        loader,
        loader,
        _metadata(),
        _ConstantClock(),
    )
    trainer.recipe = {"name": "sgd_nesterov"}
    trainer.alternative_recipes = [
        {"name": "stable"},
        {"name": "regularized"},
        {"name": "fast_fit"},
    ]
    selected = trainer._select_alternative_recipe(
        {"sgd_nesterov"}, prefer_regularized=True
    )
    assert selected["name"] == "regularized"


def test_safe_flip_tta_averages_three_views():
    loader, data = _loaders(samples=8)
    model = _CountingModel()
    trainer = Trainer(
        model,
        torch.device("cpu"),
        loader,
        loader,
        _metadata("safe_flips"),
        _ConstantClock(),
    )
    predictions = trainer.predict([data])
    assert len(predictions) == len(data)
    assert model.calls == 3


def test_dormant_challenger_is_not_a_registered_branch():
    loader, _ = _loaders()
    primary = _CountingModel()
    challenger = _CountingModel()
    primary_count = sum(
        parameter.numel() for parameter in primary.parameters()
    )
    primary.architecture_challenger_bundle = {
        "model": challenger,
        "spec": "candidate",
        "val_acc": 0.5,
        "params": 123,
        "reason": "efficient_overfit_insurance",
        "risk": 0.25,
    }
    assert (
        sum(parameter.numel() for parameter in primary.parameters())
        == primary_count
    )
    trainer = Trainer(
        primary,
        torch.device("cpu"),
        loader,
        loader,
        _metadata(),
        _ConstantClock(),
    )
    assert trainer.challenger_model is challenger
    assert trainer.challenger_params == 123
    assert trainer.challenger_reason == "efficient_overfit_insurance"
    assert trainer.challenger_risk == 0.25
    assert not hasattr(primary, "architecture_challenger_bundle")


def test_clock_driven_training_and_prediction_complete():
    loader, data = _loaders()
    model = _CountingModel()
    model.training_recipe = {
        "name": "stable",
        "optimizer": "adamw",
        "scheduler": "plateau",
        "lr_scale": 1.0,
        "weight_decay": 1e-3,
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "dropout_scale": 1.0,
        "use_class_weights": False,
    }
    clock = _AcceleratedClock()
    trainer = Trainer(
        model,
        torch.device("cpu"),
        loader,
        loader,
        _metadata(),
        clock,
    )
    trained = trainer.train()
    assert trained is trainer.model
    predictions = trainer.predict([data])
    assert len(predictions) == len(data)
    assert trainer.max_epochs > 400
    assert trainer.metadata["trainer_prediction_reserve"] >= 20.0


if __name__ == "__main__":
    test_sgd_nesterov_uses_cosine_and_rejects_adam_moments()
    test_cosine_scheduler_cannot_rebound_after_initial_horizon()
    test_attempt_rotation_distinguishes_failure_from_slow_recovery()
    test_warm_checkpoint_is_a_target_not_trained_recovery_evidence()
    test_large_generalization_gap_prioritizes_regularized_retry()
    test_safe_flip_tta_averages_three_views()
    test_dormant_challenger_is_not_a_registered_branch()
    test_clock_driven_training_and_prediction_complete()
    print("Trainer anytime regression tests passed")
