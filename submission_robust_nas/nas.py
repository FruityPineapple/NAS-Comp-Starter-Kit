"""Anytime, failure-contained neural architecture search."""

from __future__ import print_function

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from helpers import (
    BASE_SEED,
    BudgetManager,
    append_history,
    autocast,
    candidate_seed,
    clear_memory,
    count_parameters,
    finite_logits,
    is_oom_error,
    make_generator,
    make_grad_scaler,
    move_optimizer_state,
    optimizer_state_to_cpu,
    practical_batch_time,
    safe_device,
    seed_everything,
    state_dict_to_cpu,
)
from models import (
    ConstantClassifier,
    build_model,
    configuration_is_safe,
    fallback_config,
    normalize_config,
)


class Candidate(object):
    def __init__(self, index, config, seed, model, microbatch):
        self.index = int(index)
        self.config = normalize_config(config)
        self.seed = int(seed)
        self.state = state_dict_to_cpu(model)
        self.optimizer_state = None
        self.steps = 0
        self.microbatch = max(1, int(microbatch))
        self.accuracy = float("-inf")
        self.loss = float("inf")
        self.confirm_accuracy = None
        self.confirm_loss = None
        self.params = count_parameters(model)
        self.failed = False
        self.failure_reason = None
        self.completed_fidelity = 0

    def checkpoint_bytes(self):
        def tensor_bytes(value):
            if torch.is_tensor(value):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(tensor_bytes(item) for item in value)
            return 0
        return int(tensor_bytes(self.state) + tensor_bytes(self.optimizer_state))


def _optimizer_for(model, config):
    lr = max(0.0, float(config.get("learning_rate", 1.0e-3)))
    weight_decay = max(0.0, float(config.get("weight_decay", 1.0e-4)))
    if config.get("optimizer") == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _per_example_loss(logits, targets, smoothing, class_weights):
    log_probabilities = F.log_softmax(logits, dim=1)
    nll = -log_probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
    if smoothing > 0.0:
        nll = (1.0 - smoothing) * nll + smoothing * (-log_probabilities.mean(dim=1))
    if class_weights is not None:
        nll = nll * class_weights[targets]
    return nll


def _logical_loss(logits, targets, config, class_weights, mixed_targets=None, mix_lambda=1.0):
    smoothing = max(0.0, min(0.2, float(config.get("label_smoothing", 0.0))))
    primary = _per_example_loss(logits, targets, smoothing, class_weights)
    if mixed_targets is not None:
        secondary = _per_example_loss(logits, mixed_targets, smoothing, class_weights)
        primary = mix_lambda * primary + (1.0 - mix_lambda) * secondary
    return primary.mean()


class NAS(object):
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.budget = BudgetManager.get(clock, metadata)
        self.device = safe_device()
        self.base_seed = BASE_SEED
        self.class_weights = self._class_weights()
        self.search_indices, self.confirm_indices = self._validation_split()
        self._train_order = self._deterministic_train_order()

    def _class_weights(self):
        counts = np.asarray(self.metadata.get("class_counts", []), dtype=np.float64)
        if counts.size == 0 or np.any(counts <= 0):
            return None
        weights = counts.sum() / (counts.size * counts)
        weights = weights / max(1.0e-12, weights.mean())
        return torch.tensor(weights, dtype=torch.float32)

    def _validation_split(self):
        dataset = self.valid_loader.dataset
        labels = getattr(dataset, "labels", None)
        n_valid = len(dataset)
        if labels is None or n_valid < max(20, 2 * int(self.metadata.get("num_classes", 1))):
            return list(range(n_valid)), None
        labels = np.asarray(labels)
        class_indices = [np.flatnonzero(labels == class_index) for class_index in np.unique(labels)]
        if not class_indices or any(len(indices) < 2 for indices in class_indices):
            return list(range(n_valid)), None
        search, confirm = [], []
        for indices in class_indices:
            search.extend(indices[::2].tolist())
            confirm.extend(indices[1::2].tolist())
        if not search or not confirm:
            return list(range(n_valid)), None
        return sorted(search), sorted(confirm)

    def _deterministic_train_order(self):
        n_train = len(self.train_loader.dataset)
        rng = np.random.RandomState(self.base_seed)
        labels = getattr(self.train_loader.dataset, "labels", None)
        if labels is None or len(labels) != n_train:
            return rng.permutation(n_train).tolist()
        buckets = []
        labels = np.asarray(labels)
        for class_index in np.unique(labels):
            indices = np.flatnonzero(labels == class_index)
            rng.shuffle(indices)
            buckets.append(indices.tolist())
        order = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    order.append(bucket.pop())
        return order

    def _loader_for_indices(self, source_loader, indices, shuffle, seed):
        subset = Subset(source_loader.dataset, list(indices))
        shuffle = bool(shuffle) and len(subset) > 0
        kwargs = {
            "batch_size": source_loader.batch_size,
            "shuffle": shuffle,
            "drop_last": False,
            "num_workers": 0,
            "pin_memory": bool(getattr(source_loader, "pin_memory", False)),
        }
        if shuffle:
            kwargs["generator"] = make_generator(seed)
        return DataLoader(subset, **kwargs)

    def _train_loader_for_rung(self, rung):
        fractions = (0.35, 0.70, 1.0)
        fraction = fractions[min(int(rung), len(fractions) - 1)]
        count = max(1, int(math.ceil(len(self._train_order) * fraction))) if self._train_order else 0
        return self._loader_for_indices(
            self.train_loader,
            self._train_order[:count],
            shuffle=True,
            seed=self.base_seed + 37 * int(rung),
        )

    def _valid_loader_for_rung(self, rung):
        fractions = (0.40, 0.75, 1.0)
        fraction = fractions[min(int(rung), len(fractions) - 1)]
        indices = self.search_indices
        count = max(1, int(math.ceil(len(indices) * fraction))) if indices else 0
        return self._loader_for_indices(
            self.valid_loader,
            indices[:count],
            shuffle=False,
            seed=self.base_seed,
        )

    def _class_weight_for(self, config, device):
        if not config.get("class_weighting", False) or self.class_weights is None:
            return None
        return self.class_weights.to(device)

    def _train_logical_batch(self, model, optimizer, scaler, inputs, targets, candidate, rng):
        microbatch = max(1, min(candidate.microbatch, int(inputs.shape[0])))
        attempts = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                logical_size = int(inputs.shape[0])
                for start in range(0, logical_size, microbatch):
                    stop = min(logical_size, start + microbatch)
                    x = inputs[start:stop].to(self.device, non_blocking=True)
                    y = targets[start:stop].to(self.device, non_blocking=True)
                    mixed_y = None
                    mix_lambda = 1.0
                    mixup = float(candidate.config.get("mixup", 0.0))
                    if mixup > 0.0 and len(x) > 1:
                        mix_lambda = float(rng.beta(mixup, mixup))
                        permutation = torch.randperm(len(x), device=x.device)
                        x = mix_lambda * x + (1.0 - mix_lambda) * x[permutation]
                        mixed_y = y[permutation]
                    with autocast(self.device.type == "cuda"):
                        logits = model(x)
                        if not bool(torch.isfinite(logits).all().item()):
                            raise FloatingPointError("non-finite training logits")
                        loss = _logical_loss(
                            logits,
                            y,
                            candidate.config,
                            self._class_weight_for(candidate.config, self.device),
                            mixed_targets=mixed_y,
                            mix_lambda=mix_lambda,
                        )
                        loss = loss * (float(stop - start) / float(logical_size))
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("non-finite training loss")
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), 5.0)
                if not bool(torch.isfinite(grad_norm).item()):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("non-finite gradient")
                scaler.step(optimizer)
                scaler.update()
                candidate.microbatch = microbatch
                return
            except Exception as exc:
                optimizer.zero_grad(set_to_none=True)
                if is_oom_error(exc) and attempts == 0 and microbatch > 1:
                    microbatch = max(1, microbatch // 2)
                    candidate.microbatch = microbatch
                    attempts += 1
                    clear_memory()
                    continue
                raise

    def _train_candidate(self, candidate, target_steps, train_loader):
        if candidate.failed or candidate.steps >= target_steps:
            return candidate.steps >= target_steps
        model = build_model(self.metadata, candidate.config)
        model.load_state_dict(candidate.state)
        model.to(self.device)
        optimizer = _optimizer_for(model, candidate.config)
        if candidate.optimizer_state is not None:
            try:
                optimizer.load_state_dict(candidate.optimizer_state)
                move_optimizer_state(optimizer, self.device)
            except Exception:
                optimizer = _optimizer_for(model, candidate.config)
        scaler = make_grad_scaler(self.device.type == "cuda")
        rng = np.random.RandomState(candidate.seed + candidate.steps)
        nonfinite_failures = 0
        completed = True

        while candidate.steps < target_steps:
            made_progress = False
            for inputs, targets in train_loader:
                estimate = practical_batch_time(self.budget)
                if not self.budget.can_start(estimate, reserve="search_stop"):
                    completed = False
                    break
                started = time.perf_counter()
                try:
                    self._train_logical_batch(model, optimizer, scaler, inputs, targets, candidate, rng)
                    candidate.steps += 1
                    made_progress = True
                    nonfinite_failures = 0
                except FloatingPointError:
                    nonfinite_failures += 1
                    if nonfinite_failures >= 3:
                        raise
                finally:
                    self.budget.record("train_batch", time.perf_counter() - started)
                if candidate.steps >= target_steps:
                    break
            if (
                not made_progress
                and 0 < nonfinite_failures < 3
                and len(train_loader) > 0
                and self.budget.can_start(practical_batch_time(self.budget), reserve="search_stop")
            ):
                continue
            if not made_progress or not completed or len(train_loader) == 0:
                completed = False
                break

        candidate.state = state_dict_to_cpu(model)
        candidate.optimizer_state = optimizer_state_to_cpu(optimizer)
        model.to("cpu")
        del model, optimizer, scaler
        clear_memory()
        return completed and candidate.steps >= target_steps

    def _evaluate(self, candidate, loader, reserve="search_stop"):
        if len(loader.dataset) == 0:
            return float("-inf"), float("inf"), False
        model = build_model(self.metadata, candidate.config)
        model.load_state_dict(candidate.state)
        model.to(self.device)
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        microbatch = max(1, candidate.microbatch)
        pass_started = time.perf_counter()
        complete = True
        try:
            with torch.no_grad():
                for inputs, targets in loader:
                    if not self.budget.can_start(
                        self.budget.estimate("inference_batch", 0.05), reserve=reserve
                    ):
                        complete = False
                        break
                    attempts = 0
                    batch_micro = min(microbatch, len(inputs))
                    while True:
                        try:
                            batch_started = time.perf_counter()
                            for start in range(0, len(inputs), batch_micro):
                                stop = min(len(inputs), start + batch_micro)
                                x = inputs[start:stop].to(self.device, non_blocking=True)
                                y = targets[start:stop].to(self.device, non_blocking=True)
                                with autocast(self.device.type == "cuda"):
                                    raw_logits = model(x)
                                    was_finite = bool(torch.isfinite(raw_logits).all().item())
                                    logits = finite_logits(raw_logits)
                                    loss = _logical_loss(logits, y, candidate.config, None)
                                if not was_finite or not bool(torch.isfinite(loss).item()):
                                    raise FloatingPointError("non-finite validation output")
                                correct += int((logits.argmax(dim=1) == y).sum().item())
                                count = stop - start
                                total += count
                                loss_sum += float(loss.item()) * count
                            self.budget.record("inference_batch", time.perf_counter() - batch_started)
                            microbatch = batch_micro
                            break
                        except Exception as exc:
                            if is_oom_error(exc) and attempts == 0 and batch_micro > 1:
                                batch_micro = max(1, batch_micro // 2)
                                attempts += 1
                                candidate.microbatch = batch_micro
                                clear_memory()
                                continue
                            raise
        finally:
            elapsed = time.perf_counter() - pass_started
            observed = max(1, total)
            self.budget.record("validation_example", elapsed, units=observed)
            full_equivalent = elapsed * max(1, int(self.metadata.get("n_valid", observed))) / observed
            self.budget.record("validation_pass", full_equivalent)
            model.to("cpu")
            del model
            clear_memory()
        if total == 0:
            return float("-inf"), float("inf"), False
        return float(correct) / total, loss_sum / total, complete

    def _anchor_configs(self):
        height = int(self.metadata.get("height", 1))
        width = int(self.metadata.get("width", 1))
        n_train = int(self.metadata.get("n_train", 0))
        high_resolution = bool(self.metadata.get("high_resolution", False))
        effective_batch = int(self.metadata.get("safe_microbatch", self.metadata.get("initial_microbatch", 1)))
        imbalance = bool(self.metadata.get("highly_imbalanced", False))

        balanced_width = 24 if n_train < 2000 or high_resolution else 32
        balanced = fallback_config(balanced_width)
        balanced.update({
            "name": "balanced_residual",
            "stages": 3,
            "blocks_per_stage": [1, 2, 1],
            "norm": "batch" if effective_batch >= 16 else "group",
            "activation": "silu",
            "pool_grid": 2 if min(height, width) >= 4 else 1,
            "dropout": 0.25 if n_train < 1000 else 0.1,
            "label_smoothing": 0.05,
            "class_weighting": imbalance,
        })

        positional = fallback_config(24)
        positional.update({
            "name": "position_aware_residual",
            "stages": 3,
            "blocks_per_stage": [1, 1, 1],
            "coordinates": True,
            "pool_grid": 4 if max(height, width) >= 4 else 1,
            "activation": "silu",
            "dropout": 0.1,
            "learning_rate": 7.0e-4,
        })

        efficient = fallback_config(24)
        efficient.update({
            "name": "efficient_large_input",
            "block": "separable",
            "stages": 4 if max(height, width) >= 32 else 3,
            "blocks_per_stage": [1, 2, 1, 1],
            "kernel": 5,
            "max_pool_stem": high_resolution,
            "adapter_kernel": 1,
            "pool_grid": 2 if min(height, width) >= 4 else 1,
            "attention": not high_resolution,
            "activation": "silu",
            "learning_rate": 1.5e-3,
        })
        if high_resolution:
            ordered = [fallback_config(), efficient, balanced, positional]
        elif min(height, width) == 1:
            ordered = [fallback_config(), positional, balanced, efficient]
        else:
            ordered = [fallback_config(), balanced, positional, efficient]
        if n_train >= 10_000:
            sgd_anchor = copy.deepcopy(balanced)
            sgd_anchor.update({
                "name": "sgd_residual_anchor",
                "optimizer": "sgd",
                "learning_rate": 3.0e-2,
                "weight_decay": 1.0e-4,
                "activation": "relu",
                "mixup": 0.0,
            })
            ordered.append(sgd_anchor)
        return ordered

    def _random_config(self, rng, index):
        height = int(self.metadata.get("height", 1))
        width = int(self.metadata.get("width", 1))
        n_train = int(self.metadata.get("n_train", 0))
        high_resolution = bool(self.metadata.get("high_resolution", False))
        effective_batch = int(self.metadata.get("safe_microbatch", self.metadata.get("initial_microbatch", 1)))
        block = "separable" if high_resolution or rng.rand() < 0.35 else "basic"
        widths = [16, 24] if high_resolution else [16, 24, 32, 48]
        stages = int(rng.choice([2, 3, 4] if max(height, width) >= 16 else [2, 3]))
        max_blocks = 3 if n_train > 5000 and height * width <= 4096 else 2
        optimizer_name = "sgd" if n_train >= 10_000 and rng.rand() < 0.15 else "adamw"
        if optimizer_name == "sgd":
            learning_rate = float(10 ** rng.uniform(math.log10(1.0e-2), math.log10(8.0e-2)))
        else:
            learning_rate = float(10 ** rng.uniform(math.log10(3.0e-4), math.log10(3.0e-3)))
        config = fallback_config(int(rng.choice(widths)))
        config.update({
            "name": "random_{:02d}".format(index),
            "block": block,
            "stages": stages,
            "blocks_per_stage": [int(rng.choice(list(range(1, max_blocks + 1)))) for _ in range(stages)],
            "kernel": int(rng.choice([3, 5])) if block == "separable" else 3,
            "norm": "batch" if effective_batch >= 16 and rng.rand() < 0.25 else "group",
            "activation": "silu" if rng.rand() < 0.6 else "relu",
            "pool_grid": int(rng.choice([1, 2, 4] if max(height, width) >= 8 else [1, 2])),
            "coordinates": bool(rng.rand() < 0.25),
            "attention": bool(not high_resolution and rng.rand() < 0.3),
            "dropout": float(rng.choice([0.0, 0.1, 0.25])),
            "max_pool_stem": bool(high_resolution and rng.rand() < 0.7),
            "adapter_kernel": int(rng.choice([1, 3])),
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": float(rng.choice([1.0e-5, 1.0e-4, 1.0e-3])),
            "mixup": float(rng.choice([0.0, 0.15])) if n_train >= 256 else 0.0,
            "label_smoothing": float(rng.choice([0.0, 0.05])),
            "class_weighting": bool(self.metadata.get("highly_imbalanced", False) and rng.rand() < 0.5),
        })
        return config

    def _mutate_config(self, parent, rng, index):
        config = copy.deepcopy(parent)
        field = str(rng.choice(["learning_rate", "dropout", "pool_grid", "coordinates", "attention"]))
        if field == "learning_rate":
            config[field] = float(np.clip(float(config.get(field, 1.0e-3)) * rng.choice([0.6, 1.6]), 2.0e-4, 4.0e-3))
        elif field == "dropout":
            choices = [value for value in (0.0, 0.1, 0.25) if value != float(config.get(field, 0.0))]
            config[field] = float(rng.choice(choices))
        elif field == "pool_grid":
            choices = [value for value in (1, 2, 4) if value != int(config.get(field, 1))]
            config[field] = int(rng.choice(choices))
        else:
            config[field] = not bool(config.get(field, False))
        config["name"] = "mutation_{:02d}_{}".format(index, field)
        return config

    def _build_candidate(self, index, config, protected=False):
        seed = candidate_seed(index, self.base_seed)
        seed_everything(seed)
        model = build_model(self.metadata, config)
        guarded_microbatch = int(self.metadata.get("safe_microbatch", self.metadata.get("initial_microbatch", 1)))
        safe, reason = configuration_is_safe(
            self.metadata, model, config, guarded_microbatch
        )
        if not safe and not protected:
            append_history(self.metadata, "candidate_rejected", index=index, name=config.get("name"), reason=reason)
            del model
            return None
        candidate = Candidate(index, config, seed, model, guarded_microbatch)
        append_history(
            self.metadata,
            "candidate_built",
            index=index,
            name=candidate.config.get("name"),
            seed=seed,
            params=candidate.params,
            config=candidate.config,
            guard=reason,
        )
        del model
        return candidate

    def _preflight(self):
        self.budget.set_reserve("search_stop", self.budget.prediction_reserve())
        last_exception = None
        widths = [16, 8, 4]
        for attempt, width in enumerate(widths):
            config = fallback_config(width)
            config["name"] = "fallback_tiny_residual" if width == 16 else "fallback_width_{}".format(width)
            candidate = None
            try:
                candidate = self._build_candidate(0, config, protected=True)
                if candidate is None:
                    continue
                if len(self.train_loader) and self.budget.can_start(
                    practical_batch_time(self.budget), reserve="search_stop"
                ):
                    # Use the exact first-rung order so the protected preflight
                    # update remains comparable with later candidates.
                    loader = self._train_loader_for_rung(0)
                    self._train_candidate(candidate, 1, loader)
                if len(self.valid_loader):
                    valid_indices = self.search_indices[:max(1, self.valid_loader.batch_size)]
                    loader = self._loader_for_indices(
                        self.valid_loader, valid_indices, shuffle=False, seed=self.base_seed
                    )
                    candidate.accuracy, candidate.loss, _ = self._evaluate(candidate, loader)
                append_history(
                    self.metadata,
                    "preflight_ok",
                    width=width,
                    microbatch=candidate.microbatch,
                    accuracy=candidate.accuracy,
                    loss=candidate.loss,
                )
                self.metadata["safe_microbatch"] = candidate.microbatch
                return candidate
            except Exception as exc:
                last_exception = exc
                if is_oom_error(exc):
                    previous_microbatch = int(self.metadata.get("initial_microbatch", 1))
                    attempted_microbatch = int(getattr(candidate, "microbatch", previous_microbatch))
                    self.metadata["initial_microbatch"] = max(1, min(previous_microbatch // 2, attempted_microbatch))
                clear_memory()
                append_history(
                    self.metadata,
                    "preflight_retry",
                    attempt=attempt,
                    width=width,
                    reason="{}: {}".format(type(exc).__name__, str(exc)[:200]),
                )
                if not is_oom_error(exc) and width == widths[-1]:
                    break

        # This model has no activation-sized internal state and is the final
        # construction fallback even for a multi-class dataset.
        model = ConstantClassifier(max(1, int(self.metadata.get("num_classes", 1))))
        candidate = Candidate(0, model.config, candidate_seed(0), model, 1)
        candidate.failure_reason = None if last_exception is None else str(last_exception)[:200]
        append_history(self.metadata, "preflight_constant_fallback", reason=candidate.failure_reason)
        return candidate

    @staticmethod
    def _winner(candidates, confirmation=False):
        usable = [candidate for candidate in candidates if not candidate.failed and math.isfinite(candidate.loss)]
        if not usable:
            usable = [candidate for candidate in candidates if not candidate.failed]
        if not usable:
            return None
        def metrics(candidate):
            if confirmation and candidate.confirm_accuracy is not None:
                return candidate.confirm_accuracy, candidate.confirm_loss
            return candidate.accuracy, candidate.loss
        maximum = max(metrics(candidate)[0] for candidate in usable)
        close = [candidate for candidate in usable if metrics(candidate)[0] >= maximum - 0.005]
        close.sort(key=lambda candidate: (metrics(candidate)[1], candidate.params, candidate.index))
        return close[0]

    def _budget_design(self):
        initial = self.budget.initial_remaining
        if initial < 300.0:
            return 1, [], 0.0
        if initial < 900.0:
            return 3, [1], 0.18
        if initial < 2700.0:
            return 6, [1, 2, 4], 0.25
        if initial < 7200.0:
            return 8, [1, 2, 4], 0.32
        return 10, [1, 2, 4], 0.35

    def _base_fidelity(self):
        # The initial 16-64 update target is capped at one pass through the
        # deterministic rung-zero subset. Later targets are exact multiples of it.
        loader_length = len(self._train_loader_for_rung(0))
        return max(1, min(64, loader_length))

    def _mark_failure(self, candidate, exc, stage):
        candidate.failed = True
        candidate.failure_reason = "{}: {}".format(type(exc).__name__, str(exc)[:240])
        candidate.optimizer_state = None
        append_history(
            self.metadata,
            "candidate_failed",
            index=candidate.index,
            name=candidate.config.get("name"),
            stage=stage,
            reason=candidate.failure_reason,
        )
        clear_memory()

    def _attach_and_return(self, candidate):
        try:
            model = build_model(self.metadata, candidate.config)
            model.load_state_dict(candidate.state)
        except Exception as exc:
            append_history(self.metadata, "winner_rebuild_failed", reason=str(exc)[:200])
            model = ConstantClassifier(max(1, int(self.metadata.get("num_classes", 1))))
            candidate = Candidate(-1, model.config, self.base_seed, model, 1)
        model._nas_config = copy.deepcopy(candidate.config)
        model._nas_optimizer_state = candidate.optimizer_state
        model._nas_steps = candidate.steps
        model._nas_validation_accuracy = candidate.confirm_accuracy if candidate.confirm_accuracy is not None else candidate.accuracy
        model._safe_microbatch = candidate.microbatch
        model._search_history = self.metadata.get("search_history", [])
        self.metadata["selected_config"] = copy.deepcopy(candidate.config)
        self.metadata["selected_candidate"] = candidate.index
        self.metadata["selected_validation_accuracy"] = model._nas_validation_accuracy
        self.metadata["safe_microbatch"] = candidate.microbatch
        append_history(
            self.metadata,
            "search_selected",
            index=candidate.index,
            name=candidate.config.get("name"),
            accuracy=model._nas_validation_accuracy,
            loss=candidate.confirm_loss if candidate.confirm_loss is not None else candidate.loss,
            steps=candidate.steps,
            params=candidate.params,
            microbatch=candidate.microbatch,
        )
        return model

    def search(self):
        seed_everything(self.base_seed)
        fallback = self._preflight()
        candidate_limit, rung_multipliers, search_fraction = self._budget_design()
        if int(self.metadata.get("num_classes", 1)) == 1 or candidate_limit == 1:
            return self._attach_and_return(fallback)

        search_start_remaining = self.budget.remaining()
        search_allowance = search_fraction * search_start_remaining
        search_floor = max(
            self.budget.prediction_reserve() + 2.0,
            search_start_remaining - search_allowance,
        )
        self.budget.set_reserve("search_stop", search_floor)
        append_history(
            self.metadata,
            "search_budget",
            initial=self.budget.initial_remaining,
            remaining=search_start_remaining,
            allowance=search_allowance,
            stop_floor=search_floor,
            candidate_limit=candidate_limit,
            rungs=rung_multipliers,
        )

        configs = self._anchor_configs()
        rng = np.random.RandomState(self.base_seed + 19)
        while len(configs) < candidate_limit:
            configs.append(self._random_config(rng, len(configs)))
        if self.confirm_indices is None and len(configs) > 3:
            configs = configs[:-1]

        candidates = [fallback]
        checkpoint_cap = 384 * 1024 ** 2
        checkpoint_bytes = fallback.checkpoint_bytes()
        for index, config in enumerate(configs[1:candidate_limit], start=1):
            if not self.budget.can_start(0.05, reserve="search_stop"):
                break
            try:
                candidate = self._build_candidate(index, config)
                if candidate is None:
                    continue
                if checkpoint_bytes + candidate.checkpoint_bytes() > checkpoint_cap:
                    append_history(
                        self.metadata,
                        "candidate_rejected",
                        index=index,
                        name=config.get("name"),
                        reason="checkpoint_memory_cap",
                    )
                    continue
                candidates.append(candidate)
                checkpoint_bytes += candidate.checkpoint_bytes()
            except Exception as exc:
                append_history(
                    self.metadata,
                    "candidate_build_failed",
                    index=index,
                    name=config.get("name"),
                    reason="{}: {}".format(type(exc).__name__, str(exc)[:200]),
                )
                clear_memory()

        active = candidates
        base_fidelity = self._base_fidelity()
        for rung, multiplier in enumerate(rung_multipliers):
            if not active or not self.budget.can_start(
                practical_batch_time(self.budget), reserve="search_stop"
            ):
                break
            train_loader = self._train_loader_for_rung(rung)
            valid_loader = self._valid_loader_for_rung(rung)
            target_steps = max(1, base_fidelity * int(multiplier))
            completed = []
            for candidate in active:
                if not self.budget.can_start(
                    practical_batch_time(self.budget), reserve="search_stop"
                ):
                    break
                candidate_started = time.perf_counter()
                try:
                    # Recreate the shuffled loader so every candidate in a rung sees
                    # the same examples in the same deterministic order.
                    candidate_train_loader = self._train_loader_for_rung(rung)
                    trained = self._train_candidate(candidate, target_steps, candidate_train_loader)
                    if not trained:
                        append_history(
                            self.metadata,
                            "candidate_interrupted",
                            index=candidate.index,
                            rung=rung,
                            steps=candidate.steps,
                        )
                        continue
                    if sum(item.checkpoint_bytes() for item in candidates if not item.failed) > checkpoint_cap:
                        candidate.optimizer_state = None
                        append_history(
                            self.metadata,
                            "optimizer_checkpoint_dropped",
                            index=candidate.index,
                            reason="checkpoint_memory_cap",
                        )
                    accuracy, loss, evaluated = self._evaluate(candidate, valid_loader)
                    if not evaluated:
                        append_history(
                            self.metadata,
                            "validation_interrupted",
                            index=candidate.index,
                            rung=rung,
                        )
                        continue
                    candidate.accuracy = accuracy
                    candidate.loss = loss
                    candidate.completed_fidelity = rung + 1
                    completed.append(candidate)
                    append_history(
                        self.metadata,
                        "candidate_scored",
                        index=candidate.index,
                        name=candidate.config.get("name"),
                        rung=rung,
                        steps=candidate.steps,
                        fidelity=target_steps,
                        accuracy=accuracy,
                        loss=loss,
                        runtime=time.perf_counter() - candidate_started,
                        runtime_per_batch=self.budget.estimate("train_batch", 0.0),
                        microbatch=candidate.microbatch,
                    )
                except Exception as exc:
                    self._mark_failure(candidate, exc, "rung_{}".format(rung))

            if not completed:
                break
            completed.sort(key=lambda item: (-item.accuracy, item.loss, item.params, item.index))
            keep = max(1, int(math.ceil(len(completed) / 2.0)))
            # The fallback remains protected until a challenger reaches this rung.
            fallback_completed = fallback in completed
            challenger_completed = any(item is not fallback for item in completed)
            promoted = completed[:keep]
            if fallback_completed and not challenger_completed and fallback not in promoted:
                promoted[-1] = fallback
            active = promoted
            for eliminated in candidates:
                if eliminated.completed_fidelity == rung + 1 and eliminated not in active:
                    eliminated.optimizer_state = None
            append_history(
                self.metadata,
                "rung_complete",
                rung=rung,
                completed=[item.index for item in completed],
                promoted=[item.index for item in active],
            )

        # Longer budgets earn a small local-search phase. Each child changes one
        # conditional field of a completed incumbent and is measured at comparable
        # real-training fidelity; no proxy-only promotion is used.
        mutation_count = 2 if self.budget.initial_remaining >= 7200.0 else (
            1 if self.budget.initial_remaining >= 2700.0 else 0
        )
        mutation_rng = np.random.RandomState(self.base_seed + 701)
        for mutation_number in range(mutation_count):
            parent = self._winner(active)
            if parent is None or parent.completed_fidelity <= 0 or not self.budget.can_start(
                practical_batch_time(self.budget), reserve="search_stop"
            ):
                break
            index = candidate_limit + mutation_number
            config = self._mutate_config(parent.config, mutation_rng, index)
            child = None
            try:
                child = self._build_candidate(index, config)
                if child is None:
                    continue
                rung = max(0, parent.completed_fidelity - 1)
                child_loader = self._train_loader_for_rung(rung)
                target_steps = max(1, parent.steps)
                if not self._train_candidate(child, target_steps, child_loader):
                    continue
                accuracy, loss, complete = self._evaluate(child, self._valid_loader_for_rung(rung))
                if not complete:
                    continue
                child.accuracy = accuracy
                child.loss = loss
                child.completed_fidelity = parent.completed_fidelity
                candidates.append(child)
                if active is not candidates:
                    active.append(child)
                append_history(
                    self.metadata,
                    "mutation_scored",
                    index=child.index,
                    parent=parent.index,
                    fidelity=child.completed_fidelity,
                    steps=child.steps,
                    accuracy=accuracy,
                    loss=loss,
                )
            except Exception as exc:
                if child is not None:
                    self._mark_failure(child, exc, "mutation")
                else:
                    append_history(
                        self.metadata,
                        "mutation_failed",
                        index=index,
                        reason="{}: {}".format(type(exc).__name__, str(exc)[:200]),
                    )

        completed_candidates = [candidate for candidate in candidates if candidate.completed_fidelity > 0]
        if not completed_candidates:
            completed_candidates = [fallback]
        provisional = sorted(
            completed_candidates,
            key=lambda item: (-item.completed_fidelity, -item.accuracy, item.loss, item.params),
        )
        highest_fidelity = provisional[0].completed_fidelity
        comparable = [item for item in provisional if item.completed_fidelity == highest_fidelity]

        if self.confirm_indices is not None and len(comparable) > 0:
            top_two = sorted(comparable, key=lambda item: (-item.accuracy, item.loss, item.params))[:2]
            confirmation_loader = self._loader_for_indices(
                self.valid_loader, self.confirm_indices, shuffle=False, seed=self.base_seed
            )
            confirmed = []
            for candidate in top_two:
                if not self.budget.can_start(
                    self.budget.estimate("validation_pass", 0.1), reserve="search_stop"
                ):
                    break
                try:
                    accuracy, loss, complete = self._evaluate(candidate, confirmation_loader)
                    if complete:
                        candidate.confirm_accuracy = accuracy
                        candidate.confirm_loss = loss
                        confirmed.append(candidate)
                        append_history(
                            self.metadata,
                            "candidate_confirmed",
                            index=candidate.index,
                            accuracy=accuracy,
                            loss=loss,
                        )
                except Exception as exc:
                    self._mark_failure(candidate, exc, "confirmation")
            winner = self._winner(confirmed, confirmation=True) if confirmed else self._winner(comparable)
        else:
            winner = self._winner(comparable)

        if winner is None:
            winner = fallback
        return self._attach_and_return(winner)
