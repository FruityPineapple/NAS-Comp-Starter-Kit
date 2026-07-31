"""Measured, recipe-aware final training with strict wall-clock protection."""

import copy
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
        # The external clock is the real termination condition.  This only
        # protects against a broken/constant clock in a synthetic harness.
        self.max_epochs = 100_000
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
        challenger_bundle = getattr(
            model, "architecture_challenger_bundle", {}
        )
        if hasattr(model, "architecture_challenger_bundle"):
            delattr(model, "architecture_challenger_bundle")
        self.challenger_model = challenger_bundle.get("model")
        self.challenger_spec = challenger_bundle.get("spec")
        self.challenger_val_acc = float(
            challenger_bundle.get("val_acc", 0.0)
        )
        self.challenger_params = int(challenger_bundle.get("params", 0))
        self.challenger_reason = challenger_bundle.get(
            "reason", "uncertainty_tie"
        )
        self.challenger_risk = float(challenger_bundle.get("risk", 0.0))
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
        props = metadata.get("data_props", {})
        self.tta_dimensions = []
        if metadata.get("augmentation_policy") == "safe_flips":
            if props.get("horizontal_flip_safe", False):
                self.tta_dimensions.append(-1)
            if props.get("vertical_flip_safe", False):
                self.tta_dimensions.append(-2)

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

    def _apply_model_regularization(self, model):
        scale = float(self.recipe.get("dropout_scale", 1.0))
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                if not hasattr(module, "_nas_base_dropout"):
                    module._nas_base_dropout = float(module.p)
                module.p = min(
                    0.60,
                    max(0.0, module._nas_base_dropout * scale),
                )

    def _criterion(self):
        return nn.CrossEntropyLoss(
            weight=self._class_weights(),
            label_smoothing=self.label_smoothing,
        )

    def _select_alternative_recipe(
        self,
        excluded=None,
        prefer_regularized=False,
    ):
        if not self.alternative_recipes:
            return None
        excluded = set(excluded or ())
        current = self.recipe.get("name")
        if prefer_regularized:
            for name in (
                "regularized",
                "balanced",
                "stable",
                "fast_fit",
                "sgd_nesterov",
            ):
                for recipe in self.alternative_recipes:
                    if (
                        recipe.get("name") == name
                        and name != current
                        and name not in excluded
                    ):
                        return dict(recipe)
        if current == "regularized":
            for name in (
                "sgd_nesterov",
                "fast_fit",
                "stable",
                "balanced",
            ):
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
    def _attempt_recovery_margin(reference_accuracy, samples):
        """Bound noise when comparing an attempt with its starting best."""
        samples = max(1, int(samples))
        accuracy = min(1.0, max(0.0, float(reference_accuracy)))
        standard_error = math.sqrt(
            2.0 * accuracy * (1.0 - accuracy) / samples
        )
        return max(
            0.005,
            1.0 / samples,
            min(0.015, 0.75 * standard_error),
        )

    @classmethod
    def _attempt_rotation_reason(
        cls,
        *,
        attempt_epochs,
        attempt_best_val,
        attempt_start_best_val,
        epochs_without_improvement,
        train_accuracy,
        validation_accuracy,
        current_lr,
        base_lr,
        validation_samples,
        plateau_patience,
    ):
        """Detect unproductive attempts without cutting slow recoveries."""
        margin = cls._attempt_recovery_margin(
            attempt_start_best_val, validation_samples
        )
        generalization_gap = max(
            0.0, float(train_accuracy) - float(validation_accuracy)
        )
        minimum_recovery_epochs = max(
            10, 2 * int(plateau_patience) + 2
        )
        failed_to_recover = (
            int(attempt_epochs) >= minimum_recovery_epochs
            and float(attempt_best_val) + margin
            < float(attempt_start_best_val)
            and int(epochs_without_improvement)
            >= max(5, int(plateau_patience) + 2)
            and generalization_gap >= 0.12
        )
        if failed_to_recover:
            return "failed to recover the preserved baseline"

        # An attempt that already proved useful gets a much longer runway.
        # Requiring both a decayed LR and a long plateau avoids cutting the
        # delayed recovery observed on structured tasks.
        productive_plateau = (
            float(attempt_best_val)
            >= float(attempt_start_best_val) + margin
            and int(attempt_epochs)
            >= max(30, 6 * int(plateau_patience))
            and int(epochs_without_improvement)
            >= max(24, 6 * int(plateau_patience))
            and float(current_lr) <= 0.40 * max(1e-12, float(base_lr))
            and generalization_gap >= 0.10
        )
        if productive_plateau:
            return "productive attempt plateaued with a large generalization gap"
        return None

    @staticmethod
    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _is_oom(error):
        message = str(error).lower()
        return "out of memory" in message or "cuda error: memory" in message

    def _base_lr_for_batch(self, batch_size):
        if str(self.recipe.get("optimizer", "adamw")).lower() == "sgd":
            reference = float(self.recipe.get("final_lr", 2e-2))
            batch_scaled_lr = reference * math.sqrt(
                float(batch_size) / 128.0
            )
            return float(
                max(5e-3, min(1.5e-1, batch_scaled_lr))
                * float(self.recipe.get("lr_scale", 1.0))
            )
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
        prediction_views = 1 + len(self.tta_dimensions)
        predicted_test_time = (
            valid_step * max(1, test_batches) * prediction_views
        )
        self.prediction_reserve = max(
            20.0,
            predicted_test_time * 1.75 + 15.0,
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
        self.metadata["trainer_prediction_views"] = prediction_views
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
        optimizer_name = str(
            self.recipe.get("optimizer", "adamw")
        ).lower()
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.base_lr * 0.10,
                momentum=float(self.recipe.get("momentum", 0.9)),
                nesterov=bool(self.recipe.get("nesterov", True)),
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.base_lr * 0.10,
                weight_decay=self.weight_decay,
            )
        if optimizer_state is not None:
            groups = optimizer_state.get("param_groups", [])
            group = groups[0] if groups else {}
            compatible = (
                ("momentum" in group and "betas" not in group)
                if optimizer_name == "sgd"
                else ("betas" in group)
            )
            try:
                if not compatible:
                    raise ValueError("incompatible optimizer state")
                optimizer.load_state_dict(optimizer_state)
            except (KeyError, ValueError):
                # An incumbent architecture checkpoint can carry AdamW search
                # moments while a different final optimizer won the recipe
                # policy. Weights remain valid; incompatible moments do not.
                optimizer_state = None
            if optimizer_state is not None:
                for group in optimizer.param_groups:
                    group["lr"] = self.base_lr * 0.10
                    group["weight_decay"] = self.weight_decay
        if str(self.recipe.get("scheduler", "plateau")).lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.safe_epochs)),
                eta_min=self.base_lr * 0.01,
            )
        else:
            scheduler = self._make_plateau_scheduler(optimizer)
        return optimizer, scheduler

    @staticmethod
    def _step_scheduler(scheduler, validation_accuracy):
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validation_accuracy)
        else:
            # CosineAnnealingLR is periodic after T_max. The controller wants
            # a one-way cooldown, not an implicit warm restart when measured
            # epochs exceed the initial estimate.
            if (
                isinstance(
                    scheduler, optim.lr_scheduler.CosineAnnealingLR
                )
                and scheduler.last_epoch >= scheduler.T_max
            ):
                return
            scheduler.step()

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

    def _apply_time_cooldown(
        self,
        optimizer,
        elapsed,
        previous_lr=None,
    ):
        """Monotonic LR ceiling within the current optimization attempt."""
        ceiling = self._time_cooldown_ceiling(elapsed)
        for group in optimizer.param_groups:
            candidates = [float(group["lr"]), float(ceiling)]
            if previous_lr is not None:
                candidates.append(float(previous_lr))
            group["lr"] = min(candidates)

    @staticmethod
    def _cpu_state_dict(model):
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _update_ema_state(ema_state, model, decay=0.995):
        current = model.state_dict()
        with torch.no_grad():
            for key, value in current.items():
                cpu_value = value.detach().cpu()
                if key not in ema_state or not torch.is_floating_point(
                    cpu_value
                ):
                    ema_state[key] = cpu_value.clone()
                else:
                    ema_state[key].mul_(decay).add_(
                        cpu_value, alpha=1.0 - decay
                    )

    @staticmethod
    def _average_states(states, reference):
        if not states:
            return reference
        averaged = {}
        for key, reference_value in reference.items():
            if torch.is_floating_point(reference_value):
                averaged[key] = torch.stack(
                    [state[key].float() for state in states], dim=0
                ).mean(dim=0).to(dtype=reference_value.dtype)
            else:
                averaged[key] = reference_value.clone()
        return averaged

    def _evaluate_state(self, model, candidate_state):
        original = self._cpu_state_dict(model)
        model.load_state_dict(candidate_state)
        model.to(self.device)
        score = self._evaluate(model)
        model.load_state_dict(original)
        model.to(self.device)
        return score

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
        self._apply_model_regularization(model)
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
        best_model_prototype = copy.deepcopy(model).cpu()
        current_architecture_id = 0
        best_architecture_id = 0
        ema_state = self._cpu_state_dict(model)
        checkpoint_pool = [(initial_val_acc, self._cpu_state_dict(model))]
        best_recipe = dict(self.recipe)
        # The immutable starting checkpoint is the recovery target, not
        # evidence that the optimizer path has recovered it. Keeping it as
        # the attempt best made the first attempt impossible to classify as
        # a failed recovery even when every trained epoch was worse.
        attempt_trained_best_val = -float("inf")
        attempt_start_best_val = best_val_acc
        attempt_epochs = 0
        epochs_without_improvement = 0
        lr_reductions = 0
        independent_retry_count = 0
        continuation_count = 0
        attempted_recipes = {self.recipe.get("name")}
        completed_epochs = 0
        attempt_step = 0
        measured_epoch_time = self.epoch_time_estimate
        training_started = time.perf_counter()
        attempt_started = training_started
        try:
            validation_samples = max(
                1, int(len(self.valid_dataloader.dataset))
            )
        except (AttributeError, TypeError):
            validation_samples = max(
                1, int(self.metadata.get("valid_num_samples", 1))
            )

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
        if str(self.recipe.get("scheduler", "plateau")).lower() == "cosine":
            scheduler_description = (
                "cosine, no restart, initial horizon {} epochs".format(
                    self.safe_epochs
                )
            )
        else:
            scheduler_description = "plateau, patience {}".format(
                self.plateau_patience
            )
        print(
            "  [Trainer] Warmup {} steps; scheduler {}; AMP {}".format(
                self.warmup_steps,
                scheduler_description,
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
            attempt_epochs += 1
            train_acc = correct / max(1, total)
            actual_epoch_time = time.perf_counter() - epoch_started
            measured_epoch_time = (
                actual_epoch_time
                if completed_epochs == 1
                else 0.70 * measured_epoch_time + 0.30 * actual_epoch_time
            )

            if val_acc > attempt_trained_best_val + 1e-6:
                attempt_trained_best_val = val_acc
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
                if current_architecture_id != best_architecture_id:
                    best_model_prototype = copy.deepcopy(model).cpu()
                    best_architecture_id = current_architecture_id
            self._update_ema_state(
                ema_state,
                model,
                decay=min(
                    0.999,
                    max(0.90, 1.0 - 1.0 / (completed_epochs + 10.0)),
                ),
            )
            checkpoint_pool.append((val_acc, self._cpu_state_dict(model)))
            checkpoint_pool = sorted(
                checkpoint_pool,
                key=lambda item: item[0],
                reverse=True,
            )[:3]

            previous_lr = optimizer.param_groups[0]["lr"]
            if attempt_step >= self.warmup_steps:
                self._step_scheduler(plateau_scheduler, val_acc)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < previous_lr * 0.99:
                lr_reductions += 1
                scheduler_name = (
                    "Plateau"
                    if isinstance(
                        plateau_scheduler,
                        optim.lr_scheduler.ReduceLROnPlateau,
                    )
                    else "Cosine"
                )
                print(
                    "  [Trainer] {}: LR reduced {:.2e} -> {:.2e}".format(
                        scheduler_name, previous_lr, current_lr
                    )
                )
            self._apply_time_cooldown(
                optimizer,
                time.perf_counter() - attempt_started,
                previous_lr=previous_lr,
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
            rotation_reason = self._attempt_rotation_reason(
                attempt_epochs=attempt_epochs,
                attempt_best_val=attempt_trained_best_val,
                attempt_start_best_val=attempt_start_best_val,
                epochs_without_improvement=epochs_without_improvement,
                train_accuracy=train_acc,
                validation_accuracy=val_acc,
                current_lr=current_lr,
                base_lr=self.base_lr,
                validation_samples=validation_samples,
                plateau_patience=self.plateau_patience,
            )
            attempt_exhausted = (
                low_lr and epochs_without_improvement >= 10
            ) or benchmark_plateau or rotation_reason is not None
            if attempt_exhausted:
                alternative = self._select_alternative_recipe(
                    attempted_recipes,
                    prefer_regularized=(
                        rotation_reason is not None
                        and train_acc - val_acc >= 0.12
                    ),
                )
                retry_epochs = 4 if below_benchmark else 6
                enough_for_retry = (
                    available >= retry_epochs * measured_epoch_time
                )
                if (
                    self.independent_retry_state is not None
                    and alternative is not None
                    and enough_for_retry
                ):
                    model.load_state_dict(self.independent_retry_state)
                    model.to(self.device)
                    self._apply_recipe(alternative)
                    self._apply_model_regularization(model)
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
                    retry_baseline_val = self._evaluate(model)
                    if retry_baseline_val > best_val_acc:
                        best_val_acc = retry_baseline_val
                        best_recipe = dict(self.recipe)
                        best_model_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    attempt_trained_best_val = -float("inf")
                    attempt_start_best_val = best_val_acc
                    attempt_epochs = 0
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
                    ema_state = self._cpu_state_dict(model)
                    independent_retry_count += 1
                    attempted_recipes.add(
                        self.recipe.get("name")
                    )
                    print(
                        "  [Trainer] Independent attempt {} from common NAS "
                        "checkpoint: recipe={} | baseline={:.2f}%{}{}".format(
                            independent_retry_count + 1,
                            self.recipe.get("name", "stable"),
                            retry_baseline_val * 100,
                            " | benchmark still unmet"
                            if below_benchmark
                            else "",
                            (
                                " | trigger={}".format(rotation_reason)
                                if rotation_reason is not None
                                else ""
                            ),
                        )
                    )
                    continue

                if (
                    self.challenger_model is not None
                    and enough_for_retry
                ):
                    model = self.challenger_model.to(self.device)
                    self.challenger_model = None
                    self.model = model
                    current_architecture_id = 1
                    challenger_recipe = dict(
                        getattr(model, "training_recipe", best_recipe)
                    )
                    challenger_recipe["name"] = "challenger_{}".format(
                        challenger_recipe.get("name", "stable")
                    )
                    self._apply_recipe(challenger_recipe)
                    self._apply_model_regularization(model)
                    challenger_optimizer_state = getattr(
                        model, "search_optimizer_state", None
                    )
                    if hasattr(model, "search_optimizer_state"):
                        delattr(model, "search_optimizer_state")
                    criterion = self._criterion()
                    optimizer, plateau_scheduler = (
                        self._make_optimizer_and_scheduler(
                            model, challenger_optimizer_state
                        )
                    )
                    scaler = torch.cuda.amp.GradScaler(
                        enabled=self.use_amp
                    )
                    challenger_seed = seed + 65_537
                    np.random.seed(challenger_seed % (2**32 - 1))
                    torch.manual_seed(challenger_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(challenger_seed)
                    challenger_baseline_val = self._evaluate(model)
                    if challenger_baseline_val > best_val_acc + 1e-6:
                        best_val_acc = challenger_baseline_val
                        best_recipe = dict(self.recipe)
                        best_model_state = self._cpu_state_dict(model)
                        best_model_prototype = copy.deepcopy(model).cpu()
                        best_architecture_id = current_architecture_id
                    attempt_trained_best_val = -float("inf")
                    attempt_start_best_val = best_val_acc
                    attempt_epochs = 0
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
                    self.attempt_time_budget = max(1.0, available)
                    attempt_started = time.perf_counter()
                    ema_state = self._cpu_state_dict(model)
                    checkpoint_pool = [
                        (
                            challenger_baseline_val,
                            self._cpu_state_dict(model),
                        )
                    ]
                    print(
                        "  [Trainer] Anytime second-architecture attempt: "
                        "{} | baseline={:.2f}% | reason={}".format(
                            self.challenger_spec,
                            challenger_baseline_val * 100,
                            self.challenger_reason,
                        )
                    )
                    continue

                # Once all distinct recipes have been tried, keep exploiting
                # the immutable global best with a fresh seed and a smaller
                # learning-rate basin. This action can repeat while the clock
                # says another full epoch is safe.
                if (
                    enough_for_retry
                    and best_model_state is not None
                ):
                    model = copy.deepcopy(best_model_prototype)
                    model.load_state_dict(best_model_state)
                    model.to(self.device)
                    self.model = model
                    current_architecture_id = best_architecture_id
                    continuation_count += 1
                    continuation_recipe = dict(best_recipe)
                    continuation_recipe["name"] = "{}_continue{}".format(
                        best_recipe.get("name", "stable"),
                        continuation_count,
                    )
                    continuation_recipe["lr_scale"] = float(
                        best_recipe.get("lr_scale", 1.0)
                    ) * max(
                        0.08, 0.55 ** min(continuation_count, 5)
                    )
                    self._apply_recipe(continuation_recipe)
                    self._apply_model_regularization(model)
                    criterion = self._criterion()
                    optimizer, plateau_scheduler = (
                        self._make_optimizer_and_scheduler(model)
                    )
                    scaler = torch.cuda.amp.GradScaler(
                        enabled=self.use_amp
                    )
                    continuation_seed = seed + 104729 * continuation_count
                    np.random.seed(
                        continuation_seed % (2**32 - 1)
                    )
                    torch.manual_seed(continuation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(continuation_seed)
                    attempt_trained_best_val = -float("inf")
                    attempt_start_best_val = best_val_acc
                    attempt_epochs = 0
                    epochs_without_improvement = 0
                    lr_reductions = 0
                    attempt_step = 0
                    self.warmup_steps = max(
                        1,
                        min(
                            16,
                            max(1, len(self.train_dataloader)) // 8,
                        ),
                    )
                    self.attempt_time_budget = max(1.0, available)
                    attempt_started = time.perf_counter()
                    ema_state = self._cpu_state_dict(model)
                    checkpoint_pool = [
                        (best_val_acc, self._cpu_state_dict(model))
                    ]
                    print(
                        "  [Trainer] Anytime continuation {} from global "
                        "best: LR {:.2e}".format(
                            continuation_count, self.base_lr
                        )
                    )
                    continue

                # The guard above can be conservative for a final partial
                # horizon. Keep the current optimizer alive; the outer clock
                # check decides whether another epoch actually fits.
                epochs_without_improvement = 0
                print(
                    "  [Trainer] No restart fits; continuing the current "
                    "checkpoint until the prediction guard"
                )

        elapsed = time.perf_counter() - training_started
        # Evaluate rule-neutral same-architecture temporal ensembles only
        # when a complete validation pass still fits before prediction.
        evaluation_guard = (
            self.prediction_reserve
            + max(2.0, self.validation_time_estimate * 1.25)
        )
        if self.clock.check() > evaluation_guard:
            ema_score = self._evaluate_state(model, ema_state)
            if ema_score > best_val_acc + 1e-6:
                best_val_acc = ema_score
                best_model_state = {
                    key: value.clone() for key, value in ema_state.items()
                }
                best_recipe = dict(self.recipe)
                best_recipe["name"] = "{}_ema".format(
                    best_recipe.get("name", "stable")
                )
                if current_architecture_id != best_architecture_id:
                    best_model_prototype = copy.deepcopy(model).cpu()
                    best_architecture_id = current_architecture_id
                print(
                    "  [Trainer] EMA checkpoint improved valid to {:.2f}%".format(
                        ema_score * 100
                    )
                )
        if (
            len(checkpoint_pool) >= 2
            and self.clock.check() > evaluation_guard
        ):
            averaged_state = self._average_states(
                [state for _, state in checkpoint_pool],
                checkpoint_pool[0][1],
            )
            averaged_score = self._evaluate_state(
                model, averaged_state
            )
            if averaged_score > best_val_acc + 1e-6:
                best_val_acc = averaged_score
                best_model_state = averaged_state
                best_recipe = dict(self.recipe)
                best_recipe["name"] = "{}_avg".format(
                    best_recipe.get("name", "stable")
                )
                if current_architecture_id != best_architecture_id:
                    best_model_prototype = copy.deepcopy(model).cpu()
                    best_architecture_id = current_architecture_id
                print(
                    "  [Trainer] Checkpoint average improved valid to "
                    "{:.2f}%".format(averaged_score * 100)
                )
        print(
            "  [Trainer] Training complete in {} ({} epochs). "
            "Best valid acc: {:.2f}%".format(
                show_time(elapsed),
                completed_epochs,
                max(0.0, best_val_acc) * 100,
            )
        )
        if best_model_state is not None:
            model = best_model_prototype.to(self.device)
            model.load_state_dict(best_model_state)
            model.to(self.device)
            self._apply_recipe(best_recipe)
            self._apply_model_regularization(model)
            model.training_recipe = dict(best_recipe)
            self.metadata["trainer_best_recipe"] = best_recipe.get(
                "name", "stable"
            )
            print("  [Trainer] Restored best checkpoint")
        self.model = model
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

    def _predict_logits(self, data):
        cpu_data = data.detach().cpu()
        try:
            device_data = cpu_data.to(
                self.device, non_blocking=self.use_amp
            )
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=self.use_amp
            ):
                return self.model(device_data).float().cpu()
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
                    self._predict_logits(cpu_data[:midpoint]),
                    self._predict_logits(cpu_data[midpoint:]),
                ],
                dim=0,
            )

    def _predict_tensor(self, data):
        logits = self._predict_logits(data)
        for dimension in self.tta_dimensions:
            logits.add_(self._predict_logits(data.flip(dimension)))
        logits.div_(1 + len(self.tta_dimensions))
        return logits.argmax(dim=1)

    def predict(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        for data in test_loader:
            predictions.extend(self._predict_tensor(data).tolist())
        return predictions
