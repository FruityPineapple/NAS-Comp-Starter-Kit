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
    test_safe_flip_tta_averages_three_views()
    test_dormant_challenger_is_not_a_registered_branch()
    test_clock_driven_training_and_prediction_complete()
    print("Trainer anytime regression tests passed")
