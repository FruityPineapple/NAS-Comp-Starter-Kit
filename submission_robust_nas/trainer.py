"""Clock-aware final optimization and guaranteed-length prediction."""

from __future__ import print_function

import math
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader

from helpers import (
    BASE_SEED,
    BudgetManager,
    append_history,
    autocast,
    clear_memory,
    finite_logits,
    is_oom_error,
    make_generator,
    make_grad_scaler,
    move_optimizer_state,
    practical_batch_time,
    safe_device,
    seed_everything,
    state_dict_to_cpu,
)
from models import fallback_config, normalize_config
from nas import _logical_loss, _optimizer_for


class Trainer(object):
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = safe_device(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.budget = BudgetManager.get(clock, metadata)
        self.config = normalize_config(
            getattr(model, "_nas_config", metadata.get("selected_config", fallback_config()))
        )
        self.microbatch = max(
            1,
            int(getattr(model, "_safe_microbatch", metadata.get("safe_microbatch", metadata.get("initial_microbatch", 1)))),
        )
        self.best_state = state_dict_to_cpu(model)
        self.best_accuracy = float(getattr(model, "_nas_validation_accuracy", float("-inf")))
        self.best_loss = float("inf")
        self.total_updates = int(getattr(model, "_nas_steps", 0))
        self.optimizer_state = getattr(model, "_nas_optimizer_state", None)
        self._class_weights_cpu = self._make_class_weights()

    def _make_class_weights(self):
        if not self.config.get("class_weighting", False):
            return None
        counts = np.asarray(self.metadata.get("class_counts", []), dtype=np.float64)
        if counts.size == 0 or np.any(counts <= 0):
            return None
        weights = counts.sum() / (counts.size * counts)
        weights /= max(1.0e-12, weights.mean())
        return torch.tensor(weights, dtype=torch.float32)

    def _set_learning_rate(self, optimizer, update, max_updates, base_lr):
        warmup = max(1, min(20, int(0.05 * max_updates)))
        if update < warmup:
            factor = float(update + 1) / float(warmup)
        else:
            progress = float(update - warmup) / float(max(1, max_updates - warmup))
            factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for group in optimizer.param_groups:
            group["lr"] = max(1.0e-7, base_lr * factor)

    def _train_batch(self, optimizer, scaler, inputs, targets, rng):
        logical_size = int(inputs.shape[0])
        microbatch = max(1, min(self.microbatch, logical_size))
        attempts = 0
        class_weights = None if self._class_weights_cpu is None else self._class_weights_cpu.to(self.device)
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                for start in range(0, logical_size, microbatch):
                    stop = min(logical_size, start + microbatch)
                    x = inputs[start:stop].to(self.device, non_blocking=True)
                    y = targets[start:stop].to(self.device, non_blocking=True)
                    mixed_y = None
                    mix_lambda = 1.0
                    mixup = float(self.config.get("mixup", 0.0))
                    if mixup > 0.0 and len(x) > 1:
                        mix_lambda = float(rng.beta(mixup, mixup))
                        permutation = torch.randperm(len(x), device=x.device)
                        x = mix_lambda * x + (1.0 - mix_lambda) * x[permutation]
                        mixed_y = y[permutation]
                    with autocast(self.device.type == "cuda"):
                        logits = self.model(x)
                        if not bool(torch.isfinite(logits).all().item()):
                            raise FloatingPointError("non-finite final-training logits")
                        loss = _logical_loss(
                            logits,
                            y,
                            self.config,
                            class_weights,
                            mixed_targets=mixed_y,
                            mix_lambda=mix_lambda,
                        )
                        loss = loss * (float(stop - start) / float(logical_size))
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("non-finite final-training loss")
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = clip_grad_norm_(self.model.parameters(), 5.0)
                if not bool(torch.isfinite(gradient_norm).item()):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("non-finite final-training gradient")
                scaler.step(optimizer)
                scaler.update()
                self.microbatch = microbatch
                self.metadata["safe_microbatch"] = microbatch
                return True
            except Exception as exc:
                optimizer.zero_grad(set_to_none=True)
                if is_oom_error(exc) and attempts == 0 and microbatch > 1:
                    microbatch = max(1, microbatch // 2)
                    self.microbatch = microbatch
                    self.metadata["safe_microbatch"] = microbatch
                    attempts += 1
                    clear_memory()
                    continue
                raise

    def _evaluate(self):
        if len(self.valid_dataloader.dataset) == 0:
            return float("-inf"), float("inf"), False
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        started = time.perf_counter()
        complete = True
        try:
            with torch.no_grad():
                for inputs, targets in self.valid_dataloader:
                    if not self.budget.can_start(
                        self.budget.estimate("inference_batch", 0.05), reserve="prediction"
                    ):
                        complete = False
                        break
                    batch_micro = max(1, min(self.microbatch, len(inputs)))
                    attempts = 0
                    while True:
                        try:
                            batch_started = time.perf_counter()
                            for offset in range(0, len(inputs), batch_micro):
                                stop = min(len(inputs), offset + batch_micro)
                                x = inputs[offset:stop].to(self.device, non_blocking=True)
                                y = targets[offset:stop].to(self.device, non_blocking=True)
                                with autocast(self.device.type == "cuda"):
                                    raw_logits = self.model(x)
                                    if not bool(torch.isfinite(raw_logits).all().item()):
                                        raise FloatingPointError("non-finite validation logits")
                                    logits = finite_logits(raw_logits)
                                    loss = _logical_loss(logits, y, self.config, None)
                                count = stop - offset
                                correct += int((logits.argmax(dim=1) == y).sum().item())
                                total += count
                                loss_sum += float(loss.item()) * count
                            self.budget.record("inference_batch", time.perf_counter() - batch_started)
                            self.microbatch = batch_micro
                            break
                        except Exception as exc:
                            if is_oom_error(exc) and attempts == 0 and batch_micro > 1:
                                batch_micro = max(1, batch_micro // 2)
                                attempts += 1
                                self.microbatch = batch_micro
                                clear_memory()
                                continue
                            raise
        finally:
            elapsed = time.perf_counter() - started
            observed = max(1, total)
            self.budget.record("validation_example", elapsed, units=observed)
            full_equivalent = elapsed * max(1, int(self.metadata.get("n_valid", observed))) / observed
            self.budget.record("validation_pass", full_equivalent)
        if total == 0:
            return float("-inf"), float("inf"), False
        return float(correct) / total, loss_sum / total, complete

    def _max_updates(self):
        remaining = self.budget.remaining()
        batches = max(1, len(self.train_dataloader))
        if remaining < 30.0:
            hard_cap = 1
        elif remaining < 120.0:
            hard_cap = 16
        elif remaining < 300.0:
            hard_cap = 64
        elif remaining < 900.0:
            hard_cap = 1000
        elif remaining < 2700.0:
            hard_cap = 20_000
        else:
            hard_cap = 40_000
        n_train = int(self.metadata.get("n_train", 0))
        epoch_cap = 20 if n_train < 2000 else 12
        available = max(0.0, remaining - self.budget.prediction_reserve())
        estimated_batch = max(0.005, practical_batch_time(self.budget))
        time_cap = max(1, int(0.80 * available / estimated_batch))
        return max(1, min(hard_cap, epoch_cap * batches, time_cap))

    def _is_better(self, accuracy, loss):
        if not math.isfinite(loss):
            return False
        if accuracy > self.best_accuracy + 0.001:
            return True
        return accuracy >= self.best_accuracy - 0.0001 and loss < self.best_loss

    def _restore_best(self):
        try:
            self.model.load_state_dict(self.best_state)
            return True
        except Exception:
            return False

    def _refit(self):
        if self.budget.initial_remaining < 900.0:
            return False
        estimated_batch = practical_batch_time(self.budget)
        refit_steps = max(2, min(32, len(self.train_dataloader)))
        if not self.budget.can_start(
            estimated_batch * refit_steps * 1.5, reserve="prediction"
        ):
            return False
        combined = ConcatDataset((self.train_dataloader.dataset, self.valid_dataloader.dataset))
        loader = DataLoader(
            combined,
            batch_size=self.train_dataloader.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=bool(getattr(self.train_dataloader, "pin_memory", False)),
            generator=make_generator(BASE_SEED + 911),
        )
        optimizer = _optimizer_for(self.model, self.config)
        for group in optimizer.param_groups:
            group["lr"] = max(1.0e-6, 0.15 * float(self.config.get("learning_rate", 1.0e-3)))
        scaler = make_grad_scaler(self.device.type == "cuda")
        rng = np.random.RandomState(BASE_SEED + 911)
        updates = 0
        self.model.train()
        for inputs, targets in loader:
            if updates >= refit_steps or not self.budget.can_start(
                practical_batch_time(self.budget), reserve="prediction"
            ):
                break
            started = time.perf_counter()
            self._train_batch(optimizer, scaler, inputs, targets, rng)
            self.budget.record("train_batch", time.perf_counter() - started)
            updates += 1
        append_history(self.metadata, "final_refit", updates=updates, microbatch=self.microbatch)
        return updates > 0

    def train(self):
        seed_everything(BASE_SEED + 503)
        if int(self.metadata.get("num_classes", 1)) == 1 or len(self.train_dataloader) == 0:
            append_history(self.metadata, "final_training_skipped", reason="one_class_or_empty")
            return self.model

        try:
            self.model.to(self.device)
        except Exception as exc:
            append_history(self.metadata, "final_training_device_failed", reason=str(exc)[:200])
            # A late CUDA failure is not a safe reason to begin an unbounded CPU
            # training run. Preserve the preflighted checkpoint and let prediction's
            # exact-length majority fallback protect the submission.
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self.metadata["_force_majority_prediction"] = True
            return self.model

        optimizer = _optimizer_for(self.model, self.config)
        if self.optimizer_state is not None:
            try:
                optimizer.load_state_dict(self.optimizer_state)
                move_optimizer_state(optimizer, self.device)
            except Exception:
                optimizer = _optimizer_for(self.model, self.config)
            finally:
                self.optimizer_state = None
                try:
                    self.model._nas_optimizer_state = None
                except Exception:
                    pass
        scaler = make_grad_scaler(self.device.type == "cuda")
        rng = np.random.RandomState(BASE_SEED + 503)
        max_updates = self._max_updates()
        validation_interval = max(1, min(len(self.train_dataloader), 32))
        base_lr = max(1.0e-7, float(self.config.get("learning_rate", 1.0e-3)))
        updates = 0
        consecutive_nonfinite = 0
        stopped_reason = "update_cap"
        contained_failure = False

        try:
            while updates < max_updates:
                made_progress = False
                for inputs, targets in self.train_dataloader:
                    estimate = practical_batch_time(self.budget)
                    if not self.budget.can_start(estimate, reserve="prediction"):
                        stopped_reason = "prediction_reserve"
                        break
                    self._set_learning_rate(optimizer, updates, max_updates, base_lr)
                    self.model.train()
                    started = time.perf_counter()
                    try:
                        self._train_batch(optimizer, scaler, inputs, targets, rng)
                        updates += 1
                        self.total_updates += 1
                        made_progress = True
                        consecutive_nonfinite = 0
                    except FloatingPointError:
                        consecutive_nonfinite += 1
                        if consecutive_nonfinite >= 3:
                            stopped_reason = "non_finite_batches"
                            break
                        continue
                    finally:
                        self.budget.record("train_batch", time.perf_counter() - started)

                    should_validate = updates % validation_interval == 0 or updates >= max_updates
                    if should_validate and self.budget.can_start(
                        self.budget.estimate("validation_pass", 0.25), reserve="prediction"
                    ):
                        accuracy, loss, complete = self._evaluate()
                        if complete and self._is_better(accuracy, loss):
                            self.best_accuracy = accuracy
                            self.best_loss = loss
                            self.best_state = state_dict_to_cpu(self.model)
                            append_history(
                                self.metadata,
                                "final_checkpoint",
                                updates=updates,
                                accuracy=accuracy,
                                loss=loss,
                                microbatch=self.microbatch,
                            )
                    if updates >= max_updates:
                        break
                if (
                    not made_progress
                    and 0 < consecutive_nonfinite < 3
                    and self.budget.can_start(practical_batch_time(self.budget), reserve="prediction")
                ):
                    continue
                if not made_progress or stopped_reason != "update_cap":
                    break
        except Exception as exc:
            contained_failure = True
            stopped_reason = "{}: {}".format(type(exc).__name__, str(exc)[:200])
            append_history(self.metadata, "final_training_contained_failure", reason=stopped_reason)

        del optimizer, scaler
        clear_memory()
        self._restore_best()
        if not contained_failure and stopped_reason != "non_finite_batches":
            try:
                self._refit()
            except Exception as exc:
                self._restore_best()
                append_history(self.metadata, "final_refit_contained_failure", reason=str(exc)[:200])
        self.model.eval()
        self.metadata["final_training_updates"] = updates
        self.metadata["safe_microbatch"] = self.microbatch
        append_history(
            self.metadata,
            "final_training_complete",
            updates=updates,
            total_updates=self.total_updates,
            best_accuracy=self.best_accuracy,
            best_loss=self.best_loss,
            stopped_reason=stopped_reason,
            remaining=self.budget.remaining(),
        )
        return self.model

    def _prediction_dtype(self):
        try:
            return np.dtype(self.metadata.get("_label_dtype", "int64"))
        except (TypeError, ValueError):
            return np.dtype("O")

    def _majority_predictions(self, count):
        value = self.metadata.get("majority_label", 0)
        try:
            return np.full(int(count), value, dtype=self._prediction_dtype())
        except (TypeError, ValueError):
            return np.asarray([value] * int(count), dtype=object)

    @staticmethod
    def _unwrap_test_batch(batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) != 1:
                raise ValueError("test loader yielded labels or multiple tensors")
            return batch[0]
        return batch

    def _map_predictions(self, indices):
        inverse = self.metadata.get("_inverse_labels", [0])
        mapped = []
        for index in indices:
            index = int(index)
            if index < 0 or index >= len(inverse):
                raise ValueError("predicted class index is outside the label map")
            mapped.append(inverse[index])
        try:
            return np.asarray(mapped, dtype=self._prediction_dtype())
        except (TypeError, ValueError):
            return np.asarray(mapped, dtype=object)

    def predict(self, test_loader):
        n_test = int(len(test_loader.dataset))
        if n_test == 0:
            return self._majority_predictions(0)
        if self.metadata.get("_force_majority_prediction", False):
            append_history(self.metadata, "prediction_majority_fallback", reason="late_device_failure", count=n_test)
            return self._majority_predictions(n_test)
        estimated_batch = self.budget.estimate("inference_batch", 0.05)
        estimated_total = max(0.1, estimated_batch * max(1, len(test_loader)) * 1.5)
        if self.budget.remaining() <= estimated_total:
            append_history(self.metadata, "prediction_majority_fallback", reason="insufficient_live_time", count=n_test)
            return self._majority_predictions(n_test)

        try:
            self.model.to(self.device)
            self.model.eval()
            indices = []
            prediction_started = time.perf_counter()
            with torch.no_grad():
                for batch_number, raw_batch in enumerate(test_loader):
                    remaining_batches = max(1, len(test_loader) - batch_number)
                    if self.budget.remaining() <= max(0.1, estimated_batch * remaining_batches * 1.15):
                        # Preserve exact output length even if the external clock is
                        # unexpectedly tighter than the measured reserve.
                        majority_index = int(self.metadata.get("majority_index", 0))
                        indices.extend([majority_index] * (n_test - len(indices)))
                        append_history(
                            self.metadata,
                            "prediction_partial_majority_fill",
                            completed=len(indices),
                            count=n_test,
                        )
                        break
                    inputs = self._unwrap_test_batch(raw_batch)
                    batch_micro = max(1, min(self.microbatch, len(inputs)))
                    attempts = 0
                    while True:
                        try:
                            batch_started = time.perf_counter()
                            batch_indices = []
                            for start in range(0, len(inputs), batch_micro):
                                stop = min(len(inputs), start + batch_micro)
                                x = inputs[start:stop].to(self.device, non_blocking=True)
                                with autocast(self.device.type == "cuda"):
                                    logits = finite_logits(self.model(x))
                                batch_indices.extend(logits.argmax(dim=1).detach().cpu().tolist())
                            indices.extend(batch_indices)
                            elapsed = time.perf_counter() - batch_started
                            estimated_batch = self.budget.record("inference_batch", elapsed)
                            self.microbatch = batch_micro
                            break
                        except Exception as exc:
                            if is_oom_error(exc) and attempts == 0 and batch_micro > 1:
                                batch_micro = max(1, batch_micro // 2)
                                self.microbatch = batch_micro
                                attempts += 1
                                clear_memory()
                                continue
                            raise
            self.budget.record("prediction_pass", time.perf_counter() - prediction_started)
            if len(indices) != n_test:
                raise ValueError("prediction count {} does not equal {}".format(len(indices), n_test))
            predictions = self._map_predictions(indices)
            if len(predictions) != n_test:
                raise ValueError("inverse label mapping changed prediction length")
            allowed = self.metadata.get("_inverse_labels", [])
            if allowed and any(value not in allowed for value in predictions.tolist()):
                raise ValueError("prediction contains a label outside the training/validation domain")
            append_history(
                self.metadata,
                "prediction_complete",
                count=n_test,
                microbatch=self.microbatch,
                remaining=self.budget.remaining(),
            )
            return predictions
        except Exception as exc:
            clear_memory()
            append_history(
                self.metadata,
                "prediction_majority_fallback",
                reason="{}: {}".format(type(exc).__name__, str(exc)[:200]),
                count=n_test,
            )
            return self._majority_predictions(n_test)
