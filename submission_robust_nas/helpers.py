"""Shared safety, reproducibility, and clock utilities for the submission.

The evaluator copies submission files into one flat directory, so this module is
deliberately self contained and depends only on NumPy and PyTorch.
"""

from __future__ import print_function

import copy
import math
import random
import time

import numpy as np
import torch


BASE_SEED = 1729


def _finite_seconds(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return max(0.0, value)


class BudgetManager(object):
    """Read-only wrapper around the evaluator clock.

    The wrapper never changes the supplied clock and never treats a locally
    decremented counter as authoritative.  Its estimates are used only to decide
    whether another interruptible unit of work is safe to begin.
    """

    METADATA_KEY = "_robust_budget_manager"

    def __init__(self, clock, metadata):
        self.clock = clock
        self.metadata = metadata
        snapshot = metadata.get("time_remaining", 1800.0)
        self.initial_remaining = self._clock_value(snapshot)
        self.started_at = time.perf_counter()
        self.ema = {}
        self.ema_weight = 0.25

        base_prediction = max(90.0, 0.12 * self.initial_remaining)
        # A ninety-second minimum would consume all of a tiny local smoke run.
        if self.initial_remaining < 600.0:
            base_prediction = min(base_prediction, 0.25 * self.initial_remaining)
        self._base_prediction_reserve = max(0.5, base_prediction)
        self.reserves = {
            "none": 0.0,
            "prediction": self._base_prediction_reserve,
            "final_training": max(
                self._base_prediction_reserve + 2.0,
                0.68 * self.initial_remaining,
            ),
        }
        metadata["initial_time_remaining"] = self.initial_remaining
        metadata["prediction_reserve"] = self._base_prediction_reserve

    @classmethod
    def get(cls, clock, metadata):
        existing = metadata.get(cls.METADATA_KEY)
        if isinstance(existing, cls) and existing.clock is clock:
            return existing
        manager = cls(clock, metadata)
        metadata[cls.METADATA_KEY] = manager
        return manager

    def _clock_value(self, default):
        try:
            return _finite_seconds(self.clock.check(), default)
        except Exception:
            return _finite_seconds(default, 0.0)

    def remaining(self):
        remaining = self._clock_value(self.metadata.get("time_remaining", 0.0))
        self.metadata["_last_live_time_remaining"] = remaining
        return remaining

    def record(self, name, seconds, units=1.0):
        units = max(float(units), 1.0e-12)
        sample = max(0.0, float(seconds)) / units
        old = self.ema.get(name)
        self.ema[name] = sample if old is None else (
            (1.0 - self.ema_weight) * old + self.ema_weight * sample
        )
        return self.ema[name]

    def estimate(self, name, default=0.0, units=1.0):
        return max(0.0, float(self.ema.get(name, default))) * max(0.0, float(units))

    def prediction_reserve(self):
        reserve = self._base_prediction_reserve
        n_valid = max(1, int(self.metadata.get("n_valid", 1)))
        n_test = max(0, int(self.metadata.get("n_test", 0)))
        if "validation_example" in self.ema:
            measured = 2.5 * self.ema["validation_example"] * n_test
            reserve = max(reserve, measured)
        elif "validation_pass" in self.ema:
            measured = 2.5 * self.ema["validation_pass"] * float(n_test) / n_valid
            reserve = max(reserve, measured)
        # Retain the short-budget cap even after a noisy timing observation.
        if self.initial_remaining < 600.0:
            reserve = min(reserve, max(0.5, 0.25 * self.initial_remaining))
        self.reserves["prediction"] = reserve
        self.metadata["prediction_reserve"] = reserve
        return reserve

    def set_reserve(self, name, seconds):
        self.reserves[str(name)] = max(0.0, float(seconds))

    def reserve(self, name_or_seconds):
        if isinstance(name_or_seconds, str):
            if name_or_seconds == "prediction":
                return self.prediction_reserve()
            return float(self.reserves.get(name_or_seconds, 0.0))
        return max(0.0, float(name_or_seconds))

    def can_start(self, estimated_cost=0.0, reserve="prediction", margin=0.25):
        needed = self.reserve(reserve) + max(0.0, float(estimated_cost)) + max(0.0, float(margin))
        return self.remaining() > needed

    def should_stop(self, reserve="prediction"):
        return not self.can_start(0.0, reserve=reserve, margin=0.0)


def seed_everything(seed=BASE_SEED):
    seed = int(seed) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    return seed


def candidate_seed(index, base_seed=BASE_SEED):
    return int((int(base_seed) + 1009 * int(index)) % (2 ** 31 - 1))


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def state_dict_to_cpu(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def object_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: object_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [object_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(object_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def optimizer_state_to_cpu(optimizer):
    return object_to_cpu(optimizer.state_dict())


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def is_oom_error(exc):
    text = str(exc).lower()
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    typed_oom = oom_type is not None and isinstance(exc, oom_type)
    return typed_oom or "out of memory" in text or "cuda error: memory" in text


def clear_memory():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def safe_device(requested=None):
    if requested is not None:
        requested = torch.device(requested)
        if requested.type != "cuda" or torch.cuda.is_available():
            return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grad_scaler(enabled):
    # torch.cuda.amp.GradScaler is available in the competition's PyTorch 1.10.
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def autocast(enabled):
    return torch.cuda.amp.autocast(enabled=bool(enabled))


def finite_logits(logits):
    if bool(torch.isfinite(logits).all().item()):
        return logits
    return torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)


def append_history(metadata, event, **fields):
    history = metadata.setdefault("search_history", [])
    record = {"event": str(event), "elapsed": float(time.perf_counter() - metadata.get("_pipeline_wall_start", time.perf_counter()))}
    for key, value in fields.items():
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, torch.device):
            value = str(value)
        record[key] = value
    history.append(record)
    # Compact, deterministic logging is useful to diagnose hidden-dataset failures.
    printable = []
    for key in sorted(record):
        value = record[key]
        if isinstance(value, (dict, list, tuple)):
            value = str(value)
            if len(value) > 180:
                value = value[:177] + "..."
        printable.append("{}={}".format(key, value))
    print("[robust-nas] " + " ".join(printable))
    return record


def practical_batch_time(budget, default=0.25):
    return max(0.005, budget.estimate("train_batch", default=default))
