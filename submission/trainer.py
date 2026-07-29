"""
Time-aware training for the NAS Unseen-Data Challenge.

The trainer measures the actual model/data-loader throughput before choosing a
schedule.  This is important for unseen datasets: parameter count alone is a
poor predictor of how many useful epochs fit into the remaining wall-clock
budget.
"""

import math
import statistics
import time

import torch
import torch.nn as nn
from torch import optim

from helpers import show_time


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        self.is_ensemble = hasattr(model, "models") and hasattr(model, "ensemble_weights")
        self.max_epochs = 200
        self.grad_clip_norm = 5.0
        self.use_amp = device.type == "cuda"

        batch_size = getattr(train_dataloader, "batch_size", 128) or 128
        # AdamW is fairly insensitive to linear batch-size scaling.  Square-root
        # scaling is a safer compromise for unknown domains and small batches.
        self.base_lr = float(max(5e-4, min(2e-3, 1e-3 * math.sqrt(batch_size / 128.0))))

        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.weight_decay = 2e-2 if self.num_params >= 3_000_000 else 1e-2

        self.epoch_time_estimate = None
        self.validation_time_estimate = None
        self.prediction_reserve = 60.0
        self.safe_epochs = 1
        self.warmup_steps = 1
        self.total_steps = max(1, len(train_dataloader))
        self.patience = 8

    @staticmethod
    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _save_batchnorm_state(model):
        state = []
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                state.append((
                    module,
                    module.running_mean.detach().clone() if module.running_mean is not None else None,
                    module.running_var.detach().clone() if module.running_var is not None else None,
                    module.num_batches_tracked.detach().clone()
                    if module.num_batches_tracked is not None else None,
                ))
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

    def _benchmark_steps(self, model, dataloader, training, max_batches=3):
        """Measure data loading plus model compute on a few real batches."""
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
                    batch = next(iterator)
                except StopIteration:
                    break

                data, target = batch
                data = data.to(self.device, non_blocking=self.use_amp)
                target = target.to(self.device, non_blocking=self.use_amp)

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
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
                        model(data)

                self._sync(self.device)
                timings.append(time.perf_counter() - started)
        finally:
            model.zero_grad(set_to_none=True)
            self._restore_batchnorm_state(bn_state)

        if not timings:
            return 1.0
        # The first CUDA batch includes kernel/cudnn warm-up and is deliberately
        # excluded when more measurements are available.
        stable = timings[1:] if len(timings) > 1 else timings
        return max(1e-3, statistics.median(stable))

    def _calibrate_budget(self, model, max_time=None):
        remaining = max(0.0, self.clock.check())
        if max_time is not None:
            remaining = min(remaining, max_time)

        print("  [Trainer] Calibrating real model throughput...")
        train_step = self._benchmark_steps(model, self.train_dataloader, training=True)
        valid_step = self._benchmark_steps(model, self.valid_dataloader, training=False)

        train_time = train_step * max(1, len(self.train_dataloader))
        valid_time = valid_step * max(1, len(self.valid_dataloader))
        self.validation_time_estimate = valid_time
        self.epoch_time_estimate = max(1.0, train_time + valid_time)

        # Test and validation splits normally have comparable sizes.  Keep at
        # least 45 seconds, and more for high-resolution/slow models.
        self.prediction_reserve = max(45.0, min(180.0, valid_time * 1.75 + 20.0))
        usable = max(0.0, remaining - self.prediction_reserve)

        # A 12% calibration/variance allowance prevents a fast micro-benchmark
        # from causing an overrun while still using almost all of the budget.
        self.safe_epochs = max(
            0,
            min(self.max_epochs, int(usable / max(1.0, self.epoch_time_estimate * 1.12))),
        )
        self.total_steps = max(1, self.safe_epochs * max(1, len(self.train_dataloader)))
        warm_started = int(self.metadata.get("nas_pretrained_steps", 0)) > 0
        warmup_fraction = 0.02 if warm_started else 0.05
        max_warmup_epochs = 2 if warm_started else 5
        self.warmup_steps = min(
            max(1, max_warmup_epochs * max(1, len(self.train_dataloader))),
            max(1, int(warmup_fraction * self.total_steps)),
        )
        self.patience = max(8, min(20, self.safe_epochs // 3))

        self.metadata["trainer_epoch_time_estimate"] = self.epoch_time_estimate
        self.metadata["trainer_safe_epochs"] = self.safe_epochs
        self.metadata["trainer_prediction_reserve"] = self.prediction_reserve

        print(
            "  [Trainer] Estimated epoch: {}, prediction reserve: {}, safe epochs: {}".format(
                show_time(self.epoch_time_estimate),
                show_time(self.prediction_reserve),
                self.safe_epochs,
            )
        )

    def _make_optimizer_and_scheduler(self, model):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )

        warmup_steps = self.warmup_steps
        total_steps = self.total_steps

        def lr_multiplier(step):
            # step starts at zero.  The final warmup step reaches exactly 1.0.
            if step < warmup_steps:
                return 0.1 + 0.9 * ((step + 1) / warmup_steps)
            if total_steps <= warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
        return optimizer, scheduler

    def train(self):
        if self.is_ensemble:
            return self._train_ensemble()
        return self._train_single(self.model)

    def _train_ensemble(self):
        """Compatibility path; the compute-aware NAS normally returns one model."""
        n_models = len(self.model.models)
        for index, sub_model in enumerate(self.model.models):
            remaining = self.clock.check()
            models_left = n_models - index
            budget = max(30.0, (remaining - 60.0) / max(1, models_left))
            print(
                "\n  [Trainer] Training ensemble member {}/{} (budget ~{})".format(
                    index + 1, n_models, show_time(budget)
                )
            )
            self._train_single(sub_model, max_time=budget)
            if self.clock.check() <= 60:
                break
        return self.model

    def _train_single(self, model, max_time=None):
        model.to(self.device)
        self._calibrate_budget(model, max_time=max_time)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer, scheduler = self._make_optimizer_and_scheduler(model)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        best_val_acc = -1.0
        best_model_state = None
        plateau_count = 0
        completed_epochs = 0
        measured_epoch_time = self.epoch_time_estimate
        training_started = time.perf_counter()

        print("  [Trainer] Starting measured time-aware training loop")
        print(
            "  [Trainer] LR: {:.2e}, weight decay: {:.2e}, warmup steps: {}, "
            "AMP: {}, patience: {}".format(
                self.base_lr,
                self.weight_decay,
                self.warmup_steps,
                self.use_amp,
                self.patience,
            )
        )

        for epoch in range(self.safe_epochs):
            remaining = self.clock.check()
            required = self.prediction_reserve + measured_epoch_time * 1.05
            if epoch > 0 and remaining < required:
                print(
                    "  [Trainer] Stopping before epoch {} — need ~{}, have {}".format(
                        epoch + 1, show_time(required), show_time(remaining)
                    )
                )
                break
            if max_time is not None and time.perf_counter() - training_started >= max_time:
                break

            epoch_started = time.perf_counter()
            model.train()
            correct = 0
            total = 0
            loss_sum = 0.0
            batch_count = 0
            partial_epoch = False

            for batch_index, (data, target) in enumerate(self.train_dataloader):
                # Batch-level guard protects against unexpectedly slow/OOM-retried
                # epochs.  Leave enough time for validation and prediction.
                if batch_index > 0 and batch_index % 10 == 0:
                    guard = self.prediction_reserve + max(
                        10.0, self.validation_time_estimate * 1.25
                    )
                    if self.clock.check() <= guard:
                        partial_epoch = True
                        break

                data = data.to(self.device, non_blocking=self.use_amp)
                target = target.to(self.device, non_blocking=self.use_amp)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = model(data)
                    loss = criterion(output, target)

                if not torch.isfinite(loss):
                    print("  [Trainer] Non-finite loss encountered; skipping batch")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                loss_sum += float(loss.detach().item())
                batch_count += 1
                total += target.size(0)
                correct += int((output.detach().argmax(dim=1) == target).sum().item())

            if batch_count == 0:
                break

            val_acc = self._evaluate(model)
            train_acc = correct / max(1, total)
            completed_epochs += 1

            actual_epoch_time = time.perf_counter() - epoch_started
            measured_epoch_time = (
                actual_epoch_time
                if completed_epochs == 1
                else 0.7 * measured_epoch_time + 0.3 * actual_epoch_time
            )
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                "  Epoch {:>3}/{:<3} | Loss: {:.3f} | Train: {:>6.2f}% | "
                "Valid: {:>6.2f}% | LR: {:.1e} | Time: {:<7} | Remaining: ~{}".format(
                    completed_epochs,
                    self.safe_epochs,
                    loss_sum / batch_count,
                    train_acc * 100,
                    val_acc * 100,
                    current_lr,
                    show_time(actual_epoch_time),
                    show_time(self.clock.check()),
                )
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                plateau_count = 0
            else:
                plateau_count += 1

            if partial_epoch:
                print("  [Trainer] Stopping after a partial epoch to preserve prediction time")
                break
            if plateau_count >= self.patience:
                print(
                    "  [Trainer] Early stopping — no validation improvement for {} epochs".format(
                        self.patience
                    )
                )
                break

        total_time = time.perf_counter() - training_started
        print(
            "  [Trainer] Training complete in {} ({} epochs). Best valid acc: {:.2f}%".format(
                show_time(total_time), completed_epochs, max(0.0, best_val_acc) * 100
            )
        )

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)
            print("  [Trainer] Restored best checkpoint")

        return model

    def _evaluate(self, model=None):
        if model is None:
            model = self.model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                data = data.to(self.device, non_blocking=self.use_amp)
                target = target.to(self.device, non_blocking=self.use_amp)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = model(data)
                correct += int((output.argmax(dim=1) == target).sum().item())
                total += target.size(0)
        return correct / max(1, total)

    def predict(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device, non_blocking=self.use_amp)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(data)
                predictions.extend(output.argmax(dim=1).cpu().tolist())
        return predictions
