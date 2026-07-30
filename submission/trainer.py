"""Measured, recipe-aware final training with strict wall-clock protection."""

import math
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from helpers import show_time


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_dataloader,
        valid_dataloader,
        metadata,
        clock,
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.use_amp = device.type == "cuda"
        self.max_epochs = 400
        self.grad_clip_norm = 5.0

        self.recipe = dict(
            getattr(
                model,
                "training_recipe",
                {
                    "name": "stable",
                    "lr_scale": 1.0,
                    "weight_decay": 1e-2,
                    "label_smoothing": 0.05,
                    "mixup_alpha": 0.0,
                    "use_class_weights": False,
                },
            )
        )
        self.alternative_recipes = [
            dict(recipe)
            for recipe in getattr(
                model, "alternative_training_recipes", []
            )
        ]
        self.search_optimizer_state = getattr(
            model, "search_optimizer_state", None
        )
        if hasattr(model, "search_optimizer_state"):
            delattr(model, "search_optimizer_state")
        retry_state = getattr(model, "independent_retry_state", None)
        self.independent_retry_state = (
            retry_state if isinstance(retry_state, dict) else None
        )
        if hasattr(model, "independent_retry_state"):
            delattr(model, "independent_retry_state")
        if hasattr(model, "alternative_training_recipes"):
            delattr(model, "alternative_training_recipes")
        self.benchmark = (
            float(metadata["benchmark"])
            if metadata.get("benchmark") is not None
            else None
        )
        batch_size = getattr(train_dataloader, "batch_size", 128) or 128
        self.base_lr = self._base_lr_for_batch(batch_size)
        self.weight_decay = float(
            self.recipe.get("weight_decay", 1e-2)
        )
        self.label_smoothing = float(
            self.recipe.get("label_smoothing", 0.05)
        )
        self.mixup_alpha = float(self.recipe.get("mixup_alpha", 0.0))

        self.num_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.epoch_time_estimate = None
        self.validation_time_estimate = None
        self.prediction_reserve = 60.0
        self.safe_epochs = 1
        self.warmup_steps = 1
        self.training_time_budget = 1.0
        self.attempt_time_budget = 1.0
        self.plateau_patience = 3

    def _apply_recipe(self, recipe):
        self.recipe = dict(recipe)
        batch_size = (
            getattr(self.train_dataloader, "batch_size", 128) or 128
        )
        self.base_lr = self._base_lr_for_batch(batch_size)
        self.weight_decay = float(
            self.recipe.get("weight_decay", 1e-2)
        )
        self.label_smoothing = float(
            self.recipe.get("label_smoothing", 0.05)
        )
        self.mixup_alpha = float(self.recipe.get("mixup_alpha", 0.0))

    def _criterion(self):
        return nn.CrossEntropyLoss(
            weight=self._class_weights(),
            label_smoothing=self.label_smoothing,
        )

    def _select_alternative_recipe(self, excluded=None):
        if not self.alternative_recipes:
            return None
        excluded = set(excluded or ())
        current = self.recipe.get("name")
        if current == "regularized":
            for name in ("fast_fit", "stable", "balanced"):
                for recipe in self.alternative_recipes:
                    if (
                        recipe.get("name") == name
                        and name not in excluded
                    ):
                        return dict(recipe)
        return next(
            (
                dict(recipe)
                for recipe in self.alternative_recipes
                if recipe.get("name") != current
                and recipe.get("name") not in excluded
            ),
            None,
        )

    @staticmethod
    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _is_oom(error):
        message = str(error).lower()
        return "out of memory" in message or "cuda error: memory" in message

    def _base_lr_for_batch(self, batch_size):
        batch_scaled_lr = 1e-3 * math.sqrt(float(batch_size) / 128.0)
        return float(
            max(3e-4, min(2.5e-3, batch_scaled_lr))
            * float(self.recipe.get("lr_scale", 1.0))
        )

    @staticmethod
    def _save_batchnorm_state(model):
        state = []
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                state.append(
                    (
                        module,
                        module.running_mean.detach().clone()
                        if module.running_mean is not None
                        else None,
                        module.running_var.detach().clone()
                        if module.running_var is not None
                        else None,
                        module.num_batches_tracked.detach().clone()
                        if module.num_batches_tracked is not None
                        else None,
                    )
                )
        return state

    @staticmethod
    def _restore_batchnorm_state(state):
        with torch.no_grad():
            for module, running_mean, running_var, batches in state:
                if running_mean is not None:
                    module.running_mean.copy_(running_mean)
                if running_var is not None:
                    module.running_var.copy_(running_var)
                if batches is not None:
                    module.num_batches_tracked.copy_(batches)

    def _benchmark_steps(self, model, dataloader, training, max_batches=5):
        if len(dataloader) == 0:
            return 1.0
        criterion = nn.CrossEntropyLoss()
        timings = []
        bn_state = self._save_batchnorm_state(model)
        iterator = iter(dataloader)
        try:
            for _ in range(min(max_batches, len(dataloader))):
                self._sync(self.device)
                started = time.perf_counter()
                try:
                    data, target = next(iterator)
                except StopIteration:
                    break
                data = data.to(
                    self.device, non_blocking=self.use_amp
                )
                target = target.to(
                    self.device, non_blocking=self.use_amp
                )
                if training:
                    model.train()
                    model.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        output = model(data)
                        loss = criterion(output, target)
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                else:
                    model.eval()
                    with torch.no_grad(), torch.cuda.amp.autocast(
                        enabled=self.use_amp
                    ):
                        model(data)
                self._sync(self.device)
                timings.append(time.perf_counter() - started)
        finally:
            model.zero_grad(set_to_none=True)
            self._restore_batchnorm_state(bn_state)
        if not timings:
            return 1.0
        stable = timings[1:] if len(timings) > 1 else timings
        return max(1e-3, statistics.median(stable))

    def _rebuild_loader(self, loader, batch_size, training):
        generator = torch.Generator()
        generator.manual_seed(1729)
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=training,
            drop_last=training and len(loader.dataset) > batch_size,
            num_workers=0,
            pin_memory=self.use_amp,
            generator=generator if training else None,
        )

    def _halve_training_batch(self):
        current = int(
            getattr(self.train_dataloader, "batch_size", 1) or 1
        )
        if current <= 4:
            return False
        reduced = max(4, current // 2)
        print(
            "  [Trainer] CUDA OOM during calibration; reducing batch {} -> {}".format(
                current, reduced
            )
        )
        self.train_dataloader = self._rebuild_loader(
            self.train_dataloader, reduced, True
        )
        valid_batch = min(
            reduced,
            int(getattr(self.valid_dataloader, "batch_size", reduced) or reduced),
        )
        self.valid_dataloader = self._rebuild_loader(
            self.valid_dataloader, valid_batch, False
        )
        self.metadata["batch_size"] = reduced
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def _calibrate_budget(self, model):
        print("  [Trainer] Calibrating real model throughput...")
        while True:
            try:
                train_step = self._benchmark_steps(
                    model, self.train_dataloader, training=True
                )
                valid_step = self._benchmark_steps(
                    model, self.valid_dataloader, training=False
                )
                break
            except RuntimeError as error:
                if not self._is_oom(error) or not self._halve_training_batch():
                    raise

        self.base_lr = self._base_lr_for_batch(
            getattr(self.train_dataloader, "batch_size", 128) or 128
        )

        train_time = train_step * max(1, len(self.train_dataloader))
        valid_time = valid_step * max(1, len(self.valid_dataloader))
        self.validation_time_estimate = valid_time
        self.epoch_time_estimate = max(1.0, train_time + valid_time)

        recorded_test_batches = int(
            self.metadata.get("test_num_batches", len(self.valid_dataloader))
        )
        effective_test_batches = int(
            math.ceil(
                float(self.metadata.get("test_num_samples", 0))
                / max(
                    1,
                    int(
                        getattr(
                            self.valid_dataloader,
                            "batch_size",
                            1,
                        )
                        or 1
                    ),
                )
            )
        )
        test_batches = max(recorded_test_batches, effective_test_batches)
        predicted_test_time = valid_step * max(1, test_batches)
        self.prediction_reserve = max(
            30.0,
            min(180.0, predicted_test_time * 1.75 + 15.0),
        )
        self.training_time_budget = max(
            0.0, self.clock.check() - self.prediction_reserve
        )
        self.attempt_time_budget = self.training_time_budget
        self.safe_epochs = max(
            0,
            min(
                self.max_epochs,
                int(
                    self.training_time_budget
                    / max(1.0, self.epoch_time_estimate * 1.06)
                ),
            ),
        )
        steps_per_epoch = max(1, len(self.train_dataloader))
        warm_started = int(
            self.metadata.get("nas_pretrained_steps", 0)
        ) > 0
        if warm_started and self.search_optimizer_state is not None:
            self.warmup_steps = 1
        elif warm_started:
            # Search weights already saw several optimizer steps. Replaying a
            # complete epoch at warm-up LR can erase that useful checkpoint.
            self.warmup_steps = max(1, min(32, steps_per_epoch // 4))
        else:
            warmup_epochs = min(3, max(1, self.safe_epochs // 20))
            self.warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        self.plateau_patience = max(
            2, min(5, max(1, self.safe_epochs // 25))
        )

        self.metadata["trainer_epoch_time_estimate"] = (
            self.epoch_time_estimate
        )
        self.metadata["trainer_safe_epochs"] = self.safe_epochs
        self.metadata["trainer_prediction_reserve"] = (
            self.prediction_reserve
        )
        print(
            "  [Trainer] Estimated epoch: {}, prediction reserve: {}, "
            "safe horizon: {} epochs".format(
                show_time(self.epoch_time_estimate),
                show_time(self.prediction_reserve),
                self.safe_epochs,
            )
        )

    def _class_weights(self):
        if not self.recipe.get("use_class_weights", False):
            return None
        values = self.metadata.get("data_props", {}).get("class_weights")
        if not values or len(values) != int(self.metadata["num_classes"]):
            return None
        return torch.tensor(
            values, dtype=torch.float32, device=self.device
        )

    def _make_plateau_scheduler(
        self,
        optimizer,
        min_lr_factor=0.01,
        patience=None,
    ):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.35,
            patience=(
                self.plateau_patience
                if patience is None
                else int(patience)
            ),
            threshold=1e-3,
            threshold_mode="abs",
            min_lr=self.base_lr * float(min_lr_factor),
        )

    def _make_optimizer_and_scheduler(
        self, model, optimizer_state=None
    ):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.base_lr * 0.10,
            weight_decay=self.weight_decay,
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = min(
                    self.base_lr, float(group.get("lr", self.base_lr))
                )
                group["weight_decay"] = self.weight_decay
        scheduler = self._make_plateau_scheduler(optimizer)
        return optimizer, scheduler

    def _set_warmup_lr(self, optimizer, global_step):
        progress = min(1.0, global_step / max(1, self.warmup_steps))
        learning_rate = self.base_lr * (0.10 + 0.90 * progress)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def _time_cooldown_ceiling(self, elapsed):
        progress = min(
            1.0, max(0.0, elapsed / max(1.0, self.attempt_time_budget))
        )
        minimum = self.base_lr * 0.01
        return minimum + 0.5 * (self.base_lr - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply_time_cooldown(self, optimizer, elapsed):
        """Monotonic LR ceiling within the current optimization attempt."""
        ceiling = self._time_cooldown_ceiling(elapsed)
        for group in optimizer.param_groups:
            group["lr"] = min(group["lr"], ceiling)

    def _mixup(self, data, target):
        if self.mixup_alpha <= 0 or data.size(0) < 2:
            return data, target, target, 1.0
        coefficient = float(
            np.random.beta(self.mixup_alpha, self.mixup_alpha)
        )
        permutation = torch.randperm(data.size(0), device=data.device)
        mixed = coefficient * data + (1.0 - coefficient) * data[permutation]
        return mixed, target, target[permutation], coefficient

    def train(self):
        seed = int(self.metadata.get("nas_seed", 1729)) + 9001
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self.model.to(self.device)
        self._calibrate_budget(model)
        if self.safe_epochs <= 0:
            print("  [Trainer] No safe full epoch; preserving search weights")
            self.search_optimizer_state = None
            self.independent_retry_state = None
            return model

        criterion = self._criterion()
        optimizer, plateau_scheduler = self._make_optimizer_and_scheduler(
            model, self.search_optimizer_state
        )
        self.search_optimizer_state = None
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        initial_val_acc = self._evaluate(model)
        best_val_acc = initial_val_acc
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_recipe = dict(self.recipe)
        attempt_best_val = initial_val_acc
        epochs_without_improvement = 0
        lr_reductions = 0
        independent_retry_count = 0
        attempted_recipes = {self.recipe.get("name")}
        completed_epochs = 0
        attempt_step = 0
        measured_epoch_time = self.epoch_time_estimate
        training_started = time.perf_counter()
        attempt_started = training_started

        print(
            "  [Trainer] Recipe: {} | LR {:.2e} | WD {:.2e} | "
            "smoothing {:.2f} | mixup {:.2f}".format(
                self.recipe.get("name", "stable"),
                self.base_lr,
                self.weight_decay,
                self.label_smoothing,
                self.mixup_alpha,
            )
        )
        print(
            "  [Trainer] Warmup {} steps; plateau patience {}; AMP {}".format(
                self.warmup_steps,
                self.plateau_patience,
                self.use_amp,
            )
        )
        print(
            "  [Trainer] Warm-start baseline valid: {:.2f}%".format(
                initial_val_acc * 100
            )
        )
        if self.benchmark is not None:
            print(
                "  [Trainer] Benchmark-aware retry target: {:.2f}% "
                "(baseline gap {:+.2f}pp)".format(
                    self.benchmark,
                    initial_val_acc * 100 - self.benchmark,
                )
            )

        while completed_epochs < self.max_epochs:
            remaining = self.clock.check()
            required = self.prediction_reserve + measured_epoch_time * 1.05
            if completed_epochs > 0 and remaining < required:
                print(
                    "  [Trainer] Stopping before next epoch: need ~{}, have {}".format(
                        show_time(required), show_time(remaining)
                    )
                )
                break

            epoch_started = time.perf_counter()
            model.train()
            correct = 0.0
            total = 0
            loss_sum = 0.0
            batch_count = 0
            partial_epoch = False
            oom_epoch = False

            for batch_index, (data, target) in enumerate(
                self.train_dataloader
            ):
                if batch_index > 0 and batch_index % 10 == 0:
                    guard = self.prediction_reserve + max(
                        10.0, self.validation_time_estimate * 1.25
                    )
                    if self.clock.check() <= guard:
                        partial_epoch = True
                        break

                data = data.to(
                    self.device, non_blocking=self.use_amp
                )
                target = target.to(
                    self.device, non_blocking=self.use_amp
                )
                mixed, first, second, coefficient = self._mixup(data, target)
                optimizer.zero_grad(set_to_none=True)
                if attempt_step < self.warmup_steps:
                    self._set_warmup_lr(optimizer, attempt_step + 1)

                try:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        output = model(mixed)
                        loss = (
                            coefficient * criterion(output, first)
                            + (1.0 - coefficient) * criterion(output, second)
                        )
                    if not torch.isfinite(loss):
                        print(
                            "  [Trainer] Non-finite loss; skipping batch"
                        )
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.grad_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError as error:
                    if not self._is_oom(error):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(
                        "  [Trainer] CUDA OOM inside epoch; preserving checkpoint"
                    )
                    partial_epoch = True
                    oom_epoch = True
                    break

                attempt_step += 1
                loss_sum += float(loss.detach().item())
                batch_count += 1
                total += target.size(0)
                predictions = output.detach().argmax(dim=1)
                correct += float(
                    coefficient
                    * (predictions == first).sum().item()
                    + (1.0 - coefficient)
                    * (predictions == second).sum().item()
                )

            if batch_count == 0:
                if oom_epoch and self._halve_training_batch():
                    print(
                        "  [Trainer] Retrying the epoch with a smaller loader"
                    )
                    continue
                break

            val_acc = self._evaluate(model)
            completed_epochs += 1
            train_acc = correct / max(1, total)
            actual_epoch_time = time.perf_counter() - epoch_started
            measured_epoch_time = (
                actual_epoch_time
                if completed_epochs == 1
                else 0.70 * measured_epoch_time + 0.30 * actual_epoch_time
            )

            if val_acc > attempt_best_val + 1e-6:
                attempt_best_val = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                best_recipe = dict(self.recipe)
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            previous_lr = optimizer.param_groups[0]["lr"]
            if attempt_step >= self.warmup_steps:
                plateau_scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < previous_lr * 0.99:
                lr_reductions += 1
                print(
                    "  [Trainer] Plateau: LR reduced {:.2e} -> {:.2e}".format(
                        previous_lr, current_lr
                    )
                )
            self._apply_time_cooldown(
                optimizer, time.perf_counter() - attempt_started
            )
            current_lr = optimizer.param_groups[0]["lr"]

            available = max(
                0.0, self.clock.check() - self.prediction_reserve
            )
            additional = int(
                available / max(1.0, measured_epoch_time * 1.06)
            )
            self.safe_epochs = min(
                self.max_epochs, completed_epochs + additional
            )
            print(
                "  Epoch {:>3}/{:<3} | Loss: {:.3f} | {}: {:>6.2f}% | "
                "Valid: {:>6.2f}% | LR: {:.1e} | Time: {:<7} | Remaining: ~{}".format(
                    completed_epochs,
                    self.safe_epochs,
                    loss_sum / batch_count,
                    "MixFit" if self.mixup_alpha > 0 else "Train",
                    train_acc * 100,
                    val_acc * 100,
                    current_lr,
                    show_time(actual_epoch_time),
                    show_time(self.clock.check()),
                )
            )

            if partial_epoch:
                if oom_epoch and self._halve_training_batch():
                    print(
                        "  [Trainer] Retrying with the smaller loader on the next epoch"
                    )
                    continue
                print(
                    "  [Trainer] Partial epoch ended to protect prediction time"
                )
                break

            # Once an attempt is exhausted, change both the optimization path
            # and recipe. Reloading the same best basin at a smaller LR did not
            # improve any of the measured datasets.
            below_benchmark = (
                self.benchmark is not None
                and best_val_acc * 100.0 < self.benchmark
            )
            low_lr = current_lr <= self.base_lr * 0.015
            benchmark_plateau = (
                below_benchmark
                and current_lr <= self.base_lr * 0.08
                and epochs_without_improvement
                >= max(8, 2 * self.plateau_patience)
            )
            attempt_exhausted = (
                low_lr and epochs_without_improvement >= 10
            ) or benchmark_plateau
            if attempt_exhausted:
                alternative = self._select_alternative_recipe(
                    attempted_recipes
                )
                retry_epochs = 4 if below_benchmark else 6
                enough_for_retry = (
                    available >= retry_epochs * measured_epoch_time
                )
                max_retries = 2 if below_benchmark else 1
                if (
                    independent_retry_count < max_retries
                    and self.independent_retry_state is not None
                    and alternative is not None
                    and enough_for_retry
                ):
                    model.load_state_dict(self.independent_retry_state)
                    model.to(self.device)
                    self._apply_recipe(alternative)
                    criterion = self._criterion()
                    optimizer, plateau_scheduler = (
                        self._make_optimizer_and_scheduler(model)
                    )
                    scaler = torch.cuda.amp.GradScaler(
                        enabled=self.use_amp
                    )
                    retry_seed = seed + 7919 * (
                        independent_retry_count + 1
                    )
                    np.random.seed(retry_seed % (2**32 - 1))
                    torch.manual_seed(retry_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(retry_seed)
                    attempt_best_val = self._evaluate(model)
                    if attempt_best_val > best_val_acc:
                        best_val_acc = attempt_best_val
                        best_recipe = dict(self.recipe)
                        best_model_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    epochs_without_improvement = 0
                    lr_reductions = 0
                    attempt_step = 0
                    self.warmup_steps = max(
                        1,
                        min(
                            32,
                            max(1, len(self.train_dataloader)) // 4,
                        ),
                    )
                    self.attempt_time_budget = max(
                        1.0, available
                    )
                    attempt_started = time.perf_counter()
                    independent_retry_count += 1
                    attempted_recipes.add(
                        self.recipe.get("name")
                    )
                    print(
                        "  [Trainer] Independent attempt {} from common NAS "
                        "checkpoint: recipe={} | baseline={:.2f}%{}".format(
                            independent_retry_count + 1,
                            self.recipe.get("name", "stable"),
                            attempt_best_val * 100,
                            " | benchmark still unmet"
                            if below_benchmark
                            else "",
                        )
                    )
                    continue

                reason = (
                    "all safe recipe attempts are exhausted"
                    if independent_retry_count
                    else "no safe independent attempt remains"
                )
                print(
                    "  [Trainer] Stopping exhausted attempt: {}".format(
                        reason
                    )
                )
                break

        elapsed = time.perf_counter() - training_started
        print(
            "  [Trainer] Training complete in {} ({} epochs). "
            "Best valid acc: {:.2f}%".format(
                show_time(elapsed),
                completed_epochs,
                max(0.0, best_val_acc) * 100,
            )
        )
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)
            model.training_recipe = dict(best_recipe)
            self.metadata["trainer_best_recipe"] = best_recipe.get(
                "name", "stable"
            )
            print("  [Trainer] Restored best checkpoint")
        self.independent_retry_state = None
        return model

    def _evaluate(self, model=None):
        model = model or self.model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                data = data.to(
                    self.device, non_blocking=self.use_amp
                )
                target = target.to(
                    self.device, non_blocking=self.use_amp
                )
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = model(data)
                correct += int(
                    (output.argmax(dim=1) == target).sum().item()
                )
                total += target.size(0)
        return correct / max(1, total)

    def _predict_tensor(self, data):
        cpu_data = data.detach().cpu()
        try:
            device_data = cpu_data.to(
                self.device, non_blocking=self.use_amp
            )
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=self.use_amp
            ):
                return self.model(device_data).argmax(dim=1).cpu()
        except RuntimeError as error:
            if not self._is_oom(error) or cpu_data.size(0) <= 1:
                raise
            if "device_data" in locals():
                del device_data
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            midpoint = cpu_data.size(0) // 2
            return torch.cat(
                [
                    self._predict_tensor(cpu_data[:midpoint]),
                    self._predict_tensor(cpu_data[midpoint:]),
                ],
                dim=0,
            )

    def predict(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        for data in test_loader:
            predictions.extend(self._predict_tensor(data).tolist())
        return predictions
