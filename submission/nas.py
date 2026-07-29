"""Deterministic, compute-aware NAS for unseen image datasets."""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from helpers import (
    get_device,
    get_tier,
    show_time,
)
from search_space import ArchSpec, SearchSpace
from zero_cost_proxies import compute_all_proxies, rank_aggregate


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.device = get_device()

        input_shape = metadata["input_shape"]
        if len(input_shape) == 4:
            self.in_channels = int(input_shape[1])
            self.input_h = int(input_shape[2])
            self.input_w = int(input_shape[3])
        elif len(input_shape) == 3:
            self.in_channels = 1
            self.input_h = int(input_shape[1])
            self.input_w = int(input_shape[2])
        else:
            self.in_channels, self.input_h, self.input_w = 1, 32, 32

        self.num_classes = int(metadata["num_classes"])
        # Shape-derived seed stays stable even when hidden dataset codenames change.
        self.seed = int(
            1729
            + 17 * self.in_channels
            + 31 * self.input_h
            + 37 * self.input_w
            + 41 * self.num_classes
        )
        self.rng = random.Random(self.seed)
        self._seed_everything(self.seed)

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def search(self):
        tier = get_tier(self.clock)
        remaining = self.clock.check()
        print("  [NAS] Time remaining: ~{}".format(show_time(remaining)))
        print("  [NAS] Operating in TIER {} mode (seed={})".format(tier, self.seed))

        if tier == 3:
            return self._tier3_fallback()
        if tier == 2:
            return self._search_pipeline(
                n_candidates=32,
                n_top=6,
                search_fraction=0.08,
                max_search_seconds=75,
                round_plan=((20, 3), (40, 1)),
            )
        return self._search_pipeline(
            n_candidates=72,
            n_top=12,
            search_fraction=0.12,
            max_search_seconds=240,
            round_plan=((32, 6), (64, 2), (64, 1)),
        )

    def _make_space(self):
        return SearchSpace(
            self.in_channels,
            self.num_classes,
            self.input_h,
            self.input_w,
        )

    def _fallback_spec(self):
        props = self.metadata.get("data_props", {})
        min_dim = min(self.input_h, self.input_w)
        structured = bool(props.get("is_structured", self.in_channels == 1))

        if min_dim <= 8:
            return ArchSpec(2, 32, 2, "basic", 3, False, 3)
        if structured and min_dim <= 32:
            return ArchSpec(3, 32, 2, "basic", 3, False, 5)
        if min_dim <= 32:
            return ArchSpec(3, 32, 2, "basic", 3, True, 3)
        if min_dim <= 64:
            return ArchSpec(4, 32, 2, "basic", 3, True, 3)
        return ArchSpec(3, 32, 2, "basic", 3, True, 5)

    def _tier3_fallback(self):
        """Return a compact input-adaptive residual model without search."""
        space = self._make_space()
        spec = self._fallback_spec()
        if spec.num_stages > space.max_safe_stages:
            spec = spec._replace(num_stages=space.max_safe_stages)
        model = space.build_model(spec)
        self.metadata["nas_tier"] = 3
        self.metadata["nas_spec"] = spec
        self.metadata["nas_ensemble"] = False
        self.metadata["nas_pretrained_steps"] = 0
        print(
            "  [NAS] TIER 3 fallback: {} ({:,} parameters)".format(
                spec, self._num_params(model)
            )
        )
        return model

    @staticmethod
    def _num_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _parameter_bounds(self):
        min_dim = min(self.input_h, self.input_w)
        if min_dim <= 8:
            return 20_000, 4_000_000
        if min_dim <= 32:
            return 80_000, 6_000_000
        if min_dim <= 64:
            return 200_000, 4_000_000
        return 150_000, 2_500_000

    def _build_candidate_pool(self, space):
        """Filter architectures that cannot be trained sufficiently in budget."""
        min_params, max_params = self._parameter_bounds()
        pool = []
        for spec in space.all_specs():
            params = space.parameter_count(spec)
            if min_params <= params <= max_params:
                pool.append((spec, params))

        if not pool:
            spec = self._fallback_spec()
            pool = [(spec, space.parameter_count(spec))]
        return pool

    def _stratified_candidates(self, pool, n_candidates):
        """Sample evenly across parameter-count quartiles, without duplicates."""
        ordered = sorted(pool, key=lambda item: item[1])
        n_candidates = min(n_candidates, len(ordered))
        buckets = []
        for bucket_index in range(4):
            start = bucket_index * len(ordered) // 4
            end = (bucket_index + 1) * len(ordered) // 4
            buckets.append(ordered[start:end])

        chosen = []
        per_bucket = max(1, n_candidates // 4)
        for bucket in buckets:
            if bucket:
                chosen.extend(self.rng.sample(bucket, min(per_bucket, len(bucket))))

        selected_specs = {spec for spec, _ in chosen}
        remaining = [item for item in ordered if item[0] not in selected_specs]
        if len(chosen) < n_candidates:
            chosen.extend(self.rng.sample(remaining, n_candidates - len(chosen)))

        # Always include the robust fallback as an anchor when it survives the
        # parameter bounds; replace the last random candidate if necessary.
        fallback = self._fallback_spec()
        fallback_item = next((item for item in ordered if item[0] == fallback), None)
        if fallback_item is not None and fallback not in {item[0] for item in chosen}:
            chosen[-1] = fallback_item
        return chosen[:n_candidates]

    def _fixed_calibration_inputs(self, num_samples=24):
        self._seed_everything(self.seed)
        batch = next(iter(self.train_loader))
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        return data[:num_samples].detach().cpu().clone()

    def _cache_search_batches(self, max_megabytes=48, max_batches=24):
        """Materialize one fair, deterministic low-fidelity training stream."""
        self._seed_everything(self.seed + 1)
        cached = []
        byte_budget = max_megabytes * 1024 * 1024
        used = 0
        for data, target in self.train_loader:
            batch_bytes = data.numel() * data.element_size() + target.numel() * target.element_size()
            if cached and used + batch_bytes > byte_budget:
                break
            cached.append((data.detach().cpu().clone(), target.detach().cpu().clone()))
            used += batch_bytes
            if len(cached) >= max_batches:
                break
        return cached

    def _proxy_screen(self, space, candidates, calibration_inputs, deadline_remaining):
        evaluated = []
        input_shape = (self.in_channels, self.input_h, self.input_w)
        started = time.perf_counter()

        for index, (spec, params) in enumerate(candidates):
            if self.clock.check() <= deadline_remaining + 5:
                break

            self._seed_everything(self.seed + 100)
            model = space.build_model(spec)
            self._sync(self.device)
            proxy_started = time.perf_counter()
            scores = compute_all_proxies(
                model,
                calibration_inputs,
                input_shape,
                self.num_classes,
                self.device,
            )
            self._sync(self.device)
            latency = max(1e-4, time.perf_counter() - proxy_started)

            # A mild efficiency rank counters the strong size bias of SynFlow,
            # while validation accuracy remains the primary later-stage metric.
            scores["efficiency"] = -math.log10(max(1, params)) - 0.25 * math.log1p(latency)
            evaluated.append({
                "spec": spec,
                "params": params,
                "scores": scores,
                "proxy_time": latency,
            })

            del model
            if self.device.type == "cuda" and (index + 1) % 12 == 0:
                torch.cuda.empty_cache()
            if (index + 1) % 12 == 0:
                print(
                    "  [NAS] Proxies: {}/{} in {} (~{} remaining)".format(
                        index + 1,
                        len(candidates),
                        show_time(time.perf_counter() - started),
                        show_time(self.clock.check()),
                    )
                )

        if not evaluated:
            return []

        ranking = rank_aggregate([entry["scores"] for entry in evaluated])
        return [evaluated[index] for index in ranking]

    def _train_low_fidelity(self, model, cached_batches, steps, deadline_remaining):
        model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        completed = 0
        started = time.perf_counter()
        for step in range(steps):
            if self.clock.check() <= deadline_remaining + 3:
                break
            data, target = cached_batches[step % len(cached_batches)]
            data = data.to(self.device, non_blocking=self.device.type == "cuda")
            target = target.to(self.device, non_blocking=self.device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                output = model(data)
                loss = criterion(output, target)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            completed += 1

        self._sync(self.device)
        elapsed = max(1e-4, time.perf_counter() - started)
        del optimizer
        return completed, elapsed

    def _evaluate_low_fidelity(self, model, max_batches=12):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.valid_loader):
                if batch_index >= max_batches:
                    break
                data = data.to(self.device, non_blocking=self.device.type == "cuda")
                target = target.to(self.device, non_blocking=self.device.type == "cuda")
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    output = model(data)
                correct += int((output.argmax(dim=1) == target).sum().item())
                total += target.size(0)
        return correct / max(1, total)

    @staticmethod
    def _candidate_utility(entry):
        accuracy = entry.get("val_acc", 0.0)
        params = entry["params"]
        seconds_per_step = entry.get("seconds_per_step", 1.0)
        projected_epochs = entry.get("projected_epochs", 0.0)
        param_penalty = 0.0025 * max(0.0, math.log10(params / 100_000.0))
        latency_penalty = 0.001 * math.log1p(max(0.0, seconds_per_step * 1000.0))
        epoch_shortfall_penalty = 0.002 * max(0.0, 15.0 - projected_epochs)
        return accuracy - param_penalty - latency_penalty - epoch_shortfall_penalty

    def _successive_halving(
        self,
        space,
        ranked_entries,
        n_top,
        cached_batches,
        round_plan,
        deadline_remaining,
    ):
        active = []
        for entry in ranked_entries[:n_top]:
            candidate = dict(entry)
            self._seed_everything(self.seed + 500)
            candidate["model"] = space.build_model(entry["spec"])
            candidate["trained_steps"] = 0
            candidate["train_seconds"] = 0.0
            candidate["val_acc"] = 0.0
            active.append(candidate)

        for round_index, (additional_steps, survivors) in enumerate(round_plan):
            if not active or self.clock.check() <= deadline_remaining + 5:
                break

            results = []
            print(
                "  [NAS] Fidelity round {}: {} candidates × up to {} steps".format(
                    round_index + 1, len(active), additional_steps
                )
            )
            for candidate_index, candidate in enumerate(active):
                if self.clock.check() <= deadline_remaining + 5:
                    break

                self._seed_everything(self.seed + 1000 * round_index)
                steps, elapsed = self._train_low_fidelity(
                    candidate["model"],
                    cached_batches,
                    additional_steps,
                    deadline_remaining,
                )
                candidate["trained_steps"] += steps
                candidate["train_seconds"] += elapsed
                candidate["seconds_per_step"] = (
                    candidate["train_seconds"] / max(1, candidate["trained_steps"])
                )
                projected_epoch_time = (
                    candidate["seconds_per_step"]
                    * max(1, len(self.train_loader))
                    * 1.20
                )
                candidate["projected_epochs"] = max(
                    0.0,
                    (deadline_remaining - 60.0) / max(1e-3, projected_epoch_time),
                )
                candidate["val_acc"] = self._evaluate_low_fidelity(candidate["model"])
                candidate["utility"] = self._candidate_utility(candidate)
                candidate["model"].cpu()
                results.append(candidate)

                print(
                    "    {}/{} val={:.2f}% utility={:.4f} projected_epochs={:.1f} "
                    "steps={} params={:,}".format(
                        candidate_index + 1,
                        len(active),
                        candidate["val_acc"] * 100,
                        candidate["utility"],
                        candidate["projected_epochs"],
                        candidate["trained_steps"],
                        candidate["params"],
                    )
                )

            if not results:
                break

            trainable = [
                item for item in results if item.get("projected_epochs", 0.0) >= 12.0
            ]
            if trainable:
                for item in results:
                    if item.get("projected_epochs", 0.0) < 12.0:
                        del item["model"]
                results = trainable
            results.sort(key=lambda item: item["utility"], reverse=True)
            keep = min(max(1, survivors), len(results))
            discarded = results[keep:]
            for candidate in discarded:
                del candidate["model"]
            active = results[:keep]

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if not active:
            return None
        active.sort(key=lambda item: item.get("utility", -float("inf")), reverse=True)
        winner = active[0]
        for candidate in active[1:]:
            del candidate["model"]
        return winner

    def _search_pipeline(
        self,
        n_candidates,
        n_top,
        search_fraction,
        max_search_seconds,
        round_plan,
    ):
        start_remaining = self.clock.check()
        search_budget = min(max_search_seconds, max(30.0, start_remaining * search_fraction))
        deadline_remaining = max(0.0, start_remaining - search_budget)
        print(
            "  [NAS] Compute-aware search budget: {} (training protected after ~{} remaining)".format(
                show_time(search_budget), show_time(deadline_remaining)
            )
        )

        space = self._make_space()
        pool = self._build_candidate_pool(space)
        candidates = self._stratified_candidates(pool, n_candidates)
        min_params, max_params = self._parameter_bounds()
        print(
            "  [NAS] {} unique candidates from {} feasible specs "
            "({:,}–{:,} parameter guard)".format(
                len(candidates), len(pool), min_params, max_params
            )
        )

        calibration_inputs = self._fixed_calibration_inputs()
        ranked = self._proxy_screen(
            space, candidates, calibration_inputs, deadline_remaining
        )
        if not ranked:
            print("  [NAS] Proxy phase exhausted; using robust fallback")
            return self._tier3_fallback()
        print("  [NAS] Proxy top-3:")
        for rank, entry in enumerate(ranked[:3], start=1):
            finite_scores = {
                name: round(value, 3)
                for name, value in entry["scores"].items()
                if math.isfinite(value)
            }
            print(
                "    {}. {} | {:,} params | {}".format(
                    rank, entry["spec"], entry["params"], finite_scores
                )
            )

        cached_batches = self._cache_search_batches()
        if not cached_batches or self.clock.check() <= deadline_remaining + 5:
            winner = ranked[0]
            model = space.build_model(winner["spec"])
            trained_steps = 0
            val_acc = None
        else:
            winner = self._successive_halving(
                space,
                ranked,
                n_top,
                cached_batches,
                round_plan,
                deadline_remaining,
            )
            if winner is None:
                winner = ranked[0]
                model = space.build_model(winner["spec"])
                trained_steps = 0
                val_acc = None
            else:
                model = winner["model"]
                trained_steps = winner["trained_steps"]
                val_acc = winner["val_acc"]

        self.metadata["nas_tier"] = 1 if start_remaining >= 15 * 60 else 2
        self.metadata["nas_spec"] = winner["spec"]
        self.metadata["nas_ensemble"] = False
        self.metadata["nas_pretrained_steps"] = trained_steps
        self.metadata["nas_search_seconds"] = start_remaining - self.clock.check()

        summary = "  [NAS] Winner: {} | {:,} params | {} warm-start steps".format(
            winner["spec"], winner["params"], trained_steps
        )
        if val_acc is not None:
            summary += " | low-fidelity val={:.2f}%".format(val_acc * 100)
        print(summary)
        print("  [NAS] Search used {}".format(show_time(start_remaining - self.clock.check())))
        return model
