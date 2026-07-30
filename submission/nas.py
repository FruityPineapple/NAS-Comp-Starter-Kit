"""Hierarchical, portfolio-based NAS for unseen classification datasets."""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from helpers import get_device, get_tier, show_time
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
        self.data_props = metadata.get("data_props", {})

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

    @staticmethod
    def _is_oom(error):
        message = str(error).lower()
        return "out of memory" in message or "cuda error: memory" in message

    @staticmethod
    def _num_params(model):
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def search(self):
        tier = get_tier(self.clock)
        remaining = self.clock.check()
        print("  [NAS] Time remaining: ~{}".format(show_time(remaining)))
        print(
            "  [NAS] Hierarchical portfolio in TIER {} mode (seed={})".format(
                tier, self.seed
            )
        )

        if tier == 3:
            return self._tier3_fallback()
        if tier == 2:
            return self._search_pipeline(
                n_candidates=30,
                n_top=7,
                search_fraction=0.12,
                max_search_seconds=120,
                round_plan=((0.35, 3), (0.85, 2)),
            )
        return self._search_pipeline(
            n_candidates=54,
            n_top=12,
            search_fraction=0.18,
            max_search_seconds=360,
            round_plan=((0.35, 7), (0.70, 4), (1.00, 2)),
        )

    def _make_space(self):
        return SearchSpace(
            self.in_channels,
            self.num_classes,
            self.input_h,
            self.input_w,
            data_props=self.data_props,
        )

    def _training_recipes(self):
        imbalance = float(self.data_props.get("class_imbalance_ratio", 1.0))
        examples_per_class = (
            float(
                self.data_props.get(
                    "train_samples",
                    self.metadata.get("input_shape", [0])[0],
                )
            )
            / max(1, self.num_classes)
        )
        recipes = [
            {
                "name": "stable",
                "lr_scale": 1.0,
                "weight_decay": 1e-2,
                "label_smoothing": 0.05,
                "mixup_alpha": 0.0,
                "use_class_weights": False,
            },
            {
                "name": "regularized",
                "lr_scale": 0.85,
                "weight_decay": 2e-2,
                "label_smoothing": 0.10,
                "mixup_alpha": 0.10 if self.num_classes > 2 else 0.0,
                "use_class_weights": False,
            },
        ]
        if imbalance >= 1.35:
            recipes.append(
                {
                    "name": "balanced",
                    "lr_scale": 0.90,
                    "weight_decay": 1e-2,
                    "label_smoothing": 0.03,
                    "mixup_alpha": 0.0,
                    "use_class_weights": True,
                }
            )
        elif examples_per_class >= 500:
            recipes.append(
                {
                    "name": "fast_fit",
                    "lr_scale": 1.20,
                    "weight_decay": 5e-3,
                    "label_smoothing": 0.03,
                    "mixup_alpha": 0.0,
                    "use_class_weights": False,
                }
            )
        return recipes

    def _architecture_search_recipe(self):
        """Use one neutral recipe so architecture and recipe are not confounded."""
        return {
            "name": "architecture_probe",
            "lr_scale": 1.0,
            "weight_decay": 1e-2,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "use_class_weights": (
                float(self.data_props.get("class_imbalance_ratio", 1.0))
                >= 1.75
            ),
        }

    def _fallback_family(self, space):
        width = float(self.data_props.get("sequence_width_confidence", 0.0))
        height = float(self.data_props.get("sequence_height_confidence", 0.0))
        position = float(
            self.data_props.get("position_sensitive_confidence", 0.0)
        )
        if "dual_axis" in space.model_families and max(width, height) >= 0.45:
            return "dual_axis"
        if (
            "axis_width" in space.model_families
            and width >= max(0.55, height + 0.15)
        ):
            return "axis_width"
        if (
            "axis_height" in space.model_families
            and height >= max(0.55, width + 0.15)
        ):
            return "axis_height"
        if position >= 0.55:
            return "spatial_pyramid"
        return "spatial"

    def _anchor_spec(self, family, space):
        min_dim = min(self.input_h, self.input_w)
        stages = 2 if min_dim <= 8 else (3 if min_dim <= 48 else 4)
        stages = min(stages, space.max_safe_stages)
        stem_kernel = 3 if min_dim <= 32 else 5
        use_se = family in ("spatial", "spatial_pyramid") and min_dim >= 16
        return ArchSpec(
            stages,
            32,
            2,
            "basic",
            3,
            use_se,
            stem_kernel,
            family,
        )

    def _anchor_specs(self, family, space):
        """Generic compact, central and high-capacity portfolio anchors."""
        central = self._anchor_spec(family, space)
        compact = ArchSpec(
            min(2, space.max_safe_stages),
            16,
            1,
            "bottleneck",
            3,
            False,
            3,
            family,
        )
        capacity = ArchSpec(
            min(3, space.max_safe_stages),
            32,
            2,
            "basic",
            5,
            family in ("spatial", "spatial_pyramid", "axis_width", "axis_height"),
            3,
            family,
        )
        return list(dict.fromkeys((compact, central, capacity)))

    @staticmethod
    def _spec_label(spec):
        return (
            "{}[S{} C{} B{} {} K{} SE{} stem{}]".format(
                spec.model_family,
                spec.num_stages,
                spec.init_channels,
                spec.blocks_per_stage,
                spec.block_type,
                spec.kernel_size,
                int(spec.use_se),
                spec.stem_kernel,
            )
        )

    def _fallback_spec(self, space=None):
        space = space or self._make_space()
        return self._anchor_spec(self._fallback_family(space), space)

    def _default_recipe(self):
        recipes = self._training_recipes()
        if float(self.data_props.get("class_imbalance_ratio", 1.0)) >= 1.75:
            return next(
                (recipe for recipe in recipes if recipe["name"] == "balanced"),
                recipes[0],
            )
        return recipes[0]

    def _attach_recipe(self, model, recipe):
        model.training_recipe = dict(recipe)
        return model

    def _tier3_fallback(self):
        space = self._make_space()
        spec = self._fallback_spec(space)
        recipe = self._default_recipe()
        model = self._attach_recipe(space.build_model(spec), recipe)
        model.independent_retry_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.alternative_training_recipes = [
            dict(candidate)
            for candidate in self._training_recipes()
            if candidate["name"] != recipe["name"]
        ]
        self.metadata["nas_tier"] = 3
        self.metadata["nas_seed"] = self.seed
        self.metadata["nas_spec"] = spec
        self.metadata["nas_recipe"] = recipe["name"]
        self.metadata["nas_ensemble"] = False
        self.metadata["nas_pretrained_steps"] = 0
        print(
            "  [NAS] TIER 3 anchor: {} + {} ({:,} parameters)".format(
                spec, recipe["name"], self._num_params(model)
            )
        )
        return model

    def _parameter_bounds(self):
        min_dim = min(self.input_h, self.input_w)
        if max(
            float(self.data_props.get("sequence_width_confidence", 0.0)),
            float(self.data_props.get("sequence_height_confidence", 0.0)),
        ) >= 0.45:
            return 15_000, 8_000_000
        if min_dim <= 8:
            return 10_000, 4_000_000
        if min_dim <= 32:
            return 40_000, 8_000_000
        if min_dim <= 64:
            return 80_000, 6_000_000
        return 100_000, 3_500_000

    def _build_candidate_pool(self, space):
        min_params, max_params = self._parameter_bounds()
        pool = []
        for spec in space.all_specs():
            params = space.parameter_count(spec)
            if min_params <= params <= max_params:
                pool.append((spec, params))
        if not pool:
            spec = self._fallback_spec(space)
            pool = [(spec, space.parameter_count(spec))]
        return pool

    def _portfolio_candidates(self, space, pool, n_candidates, recipes):
        """Sample diverse architectures; recipes are evaluated only later."""
        ordered = sorted(pool, key=lambda item: item[1])
        by_family = {}
        for item in ordered:
            by_family.setdefault(item[0].model_family, []).append(item)

        selected = []
        selected_specs = set()

        # Strong, deterministic anchors are evaluated first so an interrupted
        # proxy phase still covers the complete active portfolio.
        anchors_by_family = {
            family: self._anchor_specs(family, space)
            for family in space.model_families
        }
        max_anchor_count = max(
            (len(anchors) for anchors in anchors_by_family.values()),
            default=0,
        )
        for anchor_index in range(max_anchor_count):
            for family in space.model_families:
                anchors = anchors_by_family[family]
                if anchor_index >= len(anchors):
                    continue
                anchor = anchors[anchor_index]
                item = next(
                    (
                        candidate
                        for candidate in by_family.get(family, [])
                        if candidate[0] == anchor
                    ),
                    None,
                )
                if item is not None and item[0] not in selected_specs:
                    selected.append(item)
                    selected_specs.add(item[0])
        for family in space.model_families:
            if not any(
                item[0].model_family == family for item in selected
            ) and by_family.get(family):
                group = by_family[family]
                item = group[len(group) // 2]
                selected.append(item)
                selected_specs.add(item[0])

        family_index = 0
        while len(selected) < min(n_candidates, len(ordered)):
            family = space.model_families[family_index % len(space.model_families)]
            family_index += 1
            group = [
                item
                for item in by_family.get(family, [])
                if item[0] not in selected_specs
            ]
            if not group:
                if all(
                    not [
                        item
                        for item in by_family.get(name, [])
                        if item[0] not in selected_specs
                    ]
                    for name in space.model_families
                ):
                    break
                continue
            quartile = (len(selected) // max(1, len(space.model_families))) % 4
            start = quartile * len(group) // 4
            end = max(start + 1, (quartile + 1) * len(group) // 4)
            bucket = group[start:end]
            item = self.rng.choice(bucket)
            selected.append(item)
            selected_specs.add(item[0])

        entries = []
        for spec, params in selected:
            entries.append(
                {
                    "spec": spec,
                    "params": params,
                    "recipe": {
                        "name": "architecture_probe",
                        "lr_scale": 1.0,
                        "weight_decay": 1e-2,
                        "label_smoothing": 0.0,
                        "mixup_alpha": 0.0,
                        "use_class_weights": False,
                    },
                }
            )
        return entries

    def _fixed_calibration_inputs(self, num_samples=24):
        self._seed_everything(self.seed)
        batch = next(iter(self.train_loader))
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        return data[:num_samples].detach().cpu().clone()

    def _cache_search_batches(self, max_megabytes=128, max_batches=128):
        """Materialize a fair low-fidelity stream, up to one full epoch."""
        self._seed_everything(self.seed + 1)
        cached = []
        byte_budget = max_megabytes * 1024 * 1024
        used = 0
        for data, target in self.train_loader:
            batch_bytes = (
                data.numel() * data.element_size()
                + target.numel() * target.element_size()
            )
            if cached and used + batch_bytes > byte_budget:
                break
            cached.append(
                (data.detach().cpu().clone(), target.detach().cpu().clone())
            )
            used += batch_bytes
            if len(cached) >= min(max_batches, len(self.train_loader)):
                break
        print(
            "  [NAS] Cached {} fair training batches ({:.1f}% epoch coverage)".format(
                len(cached),
                100.0 * len(cached) / max(1, len(self.train_loader)),
            )
        )
        return cached

    def _cache_validation_batches(self, max_samples=4096):
        dataset = getattr(self.valid_loader, "dataset", None)
        labels = getattr(dataset, "y", None)
        if dataset is None or labels is None or len(dataset) == 0:
            return []
        labels = labels.detach().cpu().numpy()
        classes = np.unique(labels)
        selected = []
        base = max_samples // max(1, len(classes))
        remainder = max_samples % max(1, len(classes))
        for class_index, class_id in enumerate(classes):
            positions = np.flatnonzero(labels == class_id)
            quota = min(
                len(positions),
                base + (1 if class_index < remainder else 0),
            )
            if quota:
                offsets = np.linspace(
                    0, len(positions) - 1, quota, dtype=np.int64
                )
                selected.extend(positions[offsets].tolist())
        if not selected:
            return []
        subset = torch.utils.data.Subset(dataset, sorted(selected))
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=getattr(self.valid_loader, "batch_size", 128) or 128,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return [
            (data.detach().cpu().clone(), target.detach().cpu().clone())
            for data, target in loader
        ]

    def _make_refinement_loader(self):
        """Deterministic full-data stream for the last two candidates."""
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or len(dataset) == 0:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed + 77)
        indices = torch.randperm(
            len(dataset), generator=generator
        ).tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        batch_size = int(
            getattr(self.train_loader, "batch_size", 128) or 128
        )
        return torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=len(subset) > batch_size,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _proxy_screen(self, space, candidates, inputs, deadline_remaining):
        evaluated = []
        input_shape = (self.in_channels, self.input_h, self.input_w)
        started = time.perf_counter()
        for index, entry in enumerate(candidates):
            if self.clock.check() <= deadline_remaining + 8:
                break
            self._seed_everything(self.seed + 100)
            model = space.build_model(entry["spec"])
            try:
                self._sync(self.device)
                proxy_started = time.perf_counter()
                scores = compute_all_proxies(
                    model,
                    inputs,
                    input_shape,
                    self.num_classes,
                    self.device,
                )
                self._sync(self.device)
                latency = max(1e-4, time.perf_counter() - proxy_started)
                scores["efficiency"] = -math.log1p(latency)
                candidate = dict(entry)
                candidate.update(
                    {
                        "scores": scores,
                        "proxy_time": latency,
                    }
                )
                evaluated.append(candidate)
            except RuntimeError as error:
                if not self._is_oom(error):
                    raise
                print(
                    "  [NAS] Proxy OOM; rejecting {}".format(
                        entry["spec"].model_family
                    )
                )
            finally:
                model.cpu()
                del model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            if (index + 1) % 9 == 0:
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
        ranked = [evaluated[index] for index in ranking]
        for proxy_rank, entry in enumerate(ranked):
            entry["proxy_prior"] = 1.0 - proxy_rank / max(1, len(ranked) - 1)
        return ranked

    def _select_label_aware_entries(self, ranked, n_top):
        """Allocate family quotas and parameter strata before proxy tie-breaks."""
        if not ranked or n_top <= 0:
            return []

        family_order = []
        for entry in ranked:
            family = entry["spec"].model_family
            if family not in family_order:
                family_order.append(family)

        n_top = min(int(n_top), len(ranked))
        base_quota = n_top // len(family_order)
        remainder = n_top % len(family_order)
        quotas = {
            family: base_quota + (1 if index < remainder else 0)
            for index, family in enumerate(family_order)
        }
        selected = []
        keys = set()

        def add(entry):
            key = entry["spec"]
            if key not in keys:
                selected.append(entry)
                keys.add(key)

        for family in family_order:
            family_entries = [
                entry
                for entry in ranked
                if entry["spec"].model_family == family
            ]
            family_selected = 0
            quota = quotas[family]
            # Proxy ranks only break ties inside size strata. This prevents an
            # unreliable proxy from selecting several near-identical models.
            sized = sorted(
                family_entries,
                key=lambda item: (
                    int(item.get("params", 0)),
                    item["spec"].init_channels,
                    item["spec"].num_stages,
                ),
            )
            for stratum in range(quota):
                start = stratum * len(sized) // quota
                end = max(start + 1, (stratum + 1) * len(sized) // quota)
                bucket_specs = {
                    item["spec"] for item in sized[start:end]
                }
                entry = next(
                    (
                        item
                        for item in family_entries
                        if item["spec"] in bucket_specs
                        and item["spec"] not in keys
                    ),
                    None,
                )
                if entry is not None:
                    add(entry)
                    family_selected += 1

            for entry in family_entries:
                if family_selected >= quota:
                    break
                before = len(selected)
                add(entry)
                family_selected += int(len(selected) > before)

        # A heavily filtered family may not fill its quota. Only then may
        # proxy order fill the remaining global slots.
        for entry in ranked:
            if len(selected) >= n_top:
                break
            add(entry)

        selected = selected[:n_top]
        counts = {
            family: sum(
                entry["spec"].model_family == family for entry in selected
            )
            for family in family_order
        }
        print("  [NAS] Label-aware family quotas selected: {}".format(counts))
        return selected

    def _criterion(self, recipe):
        weight = None
        if recipe.get("use_class_weights", False):
            values = self.data_props.get("class_weights")
            if values and len(values) == self.num_classes:
                weight = torch.tensor(
                    values, dtype=torch.float32, device=self.device
                )
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(recipe.get("label_smoothing", 0.05)),
        )

    def _mixup(self, data, target, alpha):
        if alpha <= 0 or data.size(0) < 2:
            return data, target, target, 1.0
        coefficient = float(np.random.beta(alpha, alpha))
        permutation = torch.randperm(data.size(0), device=data.device)
        mixed = coefficient * data + (1.0 - coefficient) * data[permutation]
        return mixed, target, target[permutation], coefficient

    def _train_low_fidelity(
        self,
        model,
        batch_source,
        max_steps,
        time_quantum,
        deadline_remaining,
        recipe,
        optimizer_state=None,
    ):
        model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3 * float(recipe.get("lr_scale", 1.0)),
            weight_decay=float(recipe.get("weight_decay", 1e-2)),
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        criterion = self._criterion(recipe)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        completed = 0
        started = time.perf_counter()
        failed = False
        saved_optimizer_state = None
        is_cached = isinstance(batch_source, (list, tuple))
        batch_iterator = None if is_cached else iter(batch_source)
        try:
            for step in range(max_steps):
                if (
                    self.clock.check() <= deadline_remaining + 4
                    or time.perf_counter() - started >= time_quantum
                ):
                    break
                if is_cached:
                    data, target = batch_source[step % len(batch_source)]
                else:
                    try:
                        data, target = next(batch_iterator)
                    except StopIteration:
                        batch_iterator = iter(batch_source)
                        data, target = next(batch_iterator)
                data = data.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                target = target.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                data, first, second, coefficient = self._mixup(
                    data,
                    target,
                    float(recipe.get("mixup_alpha", 0.0)),
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(
                    enabled=self.device.type == "cuda"
                ):
                    output = model(data)
                    loss = (
                        coefficient * criterion(output, first)
                        + (1.0 - coefficient) * criterion(output, second)
                    )
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                completed += 1
        except RuntimeError as error:
            if not self._is_oom(error):
                raise
            failed = True
            print("    Candidate rejected after CUDA OOM")
        finally:
            optimizer.zero_grad(set_to_none=True)
            saved_optimizer_state = optimizer.state_dict()
            for state in saved_optimizer_state["state"].values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.detach().cpu().clone()
            del optimizer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self._sync(self.device)
        return (
            completed,
            max(1e-4, time.perf_counter() - started),
            failed,
            saved_optimizer_state,
        )

    def _evaluate(self, model, source=None):
        source = source if source is not None else self.valid_loader
        model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        class_total = torch.zeros(self.num_classes, dtype=torch.long)
        with torch.no_grad():
            for data, target in source:
                data = data.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                target = target.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                with torch.cuda.amp.autocast(
                    enabled=self.device.type == "cuda"
                ):
                    output = model(data)
                predictions = output.argmax(dim=1)
                matches = predictions == target
                correct += int(matches.sum().item())
                total += target.size(0)
                cpu_target = target.detach().cpu()
                cpu_matches = matches.detach().cpu().long()
                class_total += torch.bincount(
                    cpu_target, minlength=self.num_classes
                )
                class_correct += torch.bincount(
                    cpu_target,
                    weights=cpu_matches,
                    minlength=self.num_classes,
                ).long()
        present = class_total > 0
        balanced = (
            float(
                (
                    class_correct[present].float()
                    / class_total[present].float()
                ).mean()
            )
            if present.any()
            else 0.0
        )
        return {
            "accuracy": correct / max(1, total),
            "balanced_accuracy": balanced,
            "samples": total,
        }

    def _candidate_utility(self, entry):
        accuracy = float(entry.get("val_acc", 0.0))
        balanced = float(entry.get("balanced_acc", accuracy))
        imbalance = float(self.data_props.get("class_imbalance_ratio", 1.0))
        balanced_weight = 0.15 if imbalance >= 1.35 else 0.04
        validation_score = (
            (1.0 - balanced_weight) * accuracy
            + balanced_weight * balanced
        )
        seconds_per_step = float(entry.get("seconds_per_step", 1.0))
        projected_epochs = float(entry.get("projected_epochs", 0.0))
        history = entry.get("val_history", [])
        gain = max(0.0, history[-1] - history[-2]) if len(history) >= 2 else 0.0
        consistency = (
            sum(history[-3:]) / min(3, len(history))
            if history
            else validation_score
        )
        latency_penalty = 0.001 * math.log1p(
            max(0.0, seconds_per_step * 1000.0)
        )
        epoch_penalty = 0.0015 * max(0.0, 12.0 - projected_epochs)
        return (
            0.90 * validation_score
            + 0.10 * consistency
            + min(0.003, 0.10 * gain)
            - latency_penalty
            - epoch_penalty
        )

    @staticmethod
    def _adapt_search_lr(entry):
        """Small plateau schedule while retaining AdamW moments across rounds."""
        history = entry.get("val_history", [])
        state = entry.get("optimizer_state")
        if len(history) < 3 or not state:
            return
        if history[-1] > max(history[-3:-1]) + 0.001:
            return
        for group in state.get("param_groups", []):
            current = float(group.get("lr", 1e-3))
            group["lr"] = max(1e-5, 0.35 * current)

    @staticmethod
    def _history_rank_scores(entries):
        """Mean percentile rank over comparable fidelity observations."""
        if len(entries) <= 1:
            return {id(entry): 1.0 for entry in entries}
        rounds = max(
            (len(entry.get("val_history", [])) for entry in entries),
            default=0,
        )
        totals = {id(entry): 0.0 for entry in entries}
        counts = {id(entry): 0 for entry in entries}
        for round_index in range(rounds):
            observed = [
                entry
                for entry in entries
                if len(entry.get("val_history", [])) > round_index
            ]
            ordered = sorted(
                observed,
                key=lambda item: item["val_history"][round_index],
            )
            denominator = max(1, len(ordered) - 1)
            for rank, entry in enumerate(ordered):
                totals[id(entry)] += rank / denominator
                counts[id(entry)] += 1
        return {
            id(entry): totals[id(entry)] / max(1, counts[id(entry)])
            for entry in entries
        }

    def _final_selection_scores(self, entries):
        """Blend final accuracy, preceding fidelity and rank consistency."""
        rank_scores = self._history_rank_scores(entries)
        for entry in entries:
            full_score = float(
                entry.get("full_val_acc", entry.get("val_acc", 0.0))
            )
            history = entry.get("val_history", [])
            preceding = float(history[-1]) if history else full_score
            entry["selection_score"] = (
                0.65 * full_score
                + 0.25 * preceding
                + 0.10 * rank_scores.get(id(entry), 0.5)
            )
        return entries

    def _recipe_tournament(
        self,
        space,
        architecture_winner,
        refinement_loader,
        validation_batches,
        deadline_remaining,
    ):
        """Compare recipes on clones of one architecture checkpoint."""
        recipes = self._training_recipes()
        common_state = {
            key: value.detach().cpu().clone()
            for key, value in architecture_winner["model"].state_dict().items()
        }
        available = max(
            0.0, self.clock.check() - deadline_remaining - 6.0
        )
        seconds_per_step = max(
            1e-4, architecture_winner.get("seconds_per_step", 0.05)
        )
        if refinement_loader is None:
            common_steps = 0
        else:
            common_steps = min(
                len(refinement_loader),
                int(0.70 * available / max(1e-4, len(recipes) * seconds_per_step)),
            )

        if common_steps < 12:
            selected_recipe = self._default_recipe()
            architecture_winner["model"].load_state_dict(common_state)
            self._attach_recipe(
                architecture_winner["model"], selected_recipe
            )
            architecture_winner["recipe"] = dict(selected_recipe)
            architecture_winner["model"].independent_retry_state = common_state
            architecture_winner["model"].search_optimizer_state = (
                architecture_winner.pop("optimizer_state", None)
            )
            architecture_winner[
                "model"
            ].alternative_training_recipes = [
                dict(recipe)
                for recipe in recipes
                if recipe["name"] != selected_recipe["name"]
            ]
            print(
                "  [NAS] Recipe tournament skipped safely; using {} "
                "(only {} equal steps affordable)".format(
                    selected_recipe["name"], common_steps
                )
            )
            return architecture_winner

        print(
            "  [NAS] Decoupled recipe tournament on {}: {} recipes, "
            "{} equal steps".format(
                self._spec_label(architecture_winner["spec"]),
                len(recipes),
                common_steps,
            )
        )
        trials = []
        for recipe_index, recipe in enumerate(recipes):
            if self.clock.check() <= deadline_remaining + 4:
                break
            self._seed_everything(self.seed + 12000)
            model = self._attach_recipe(
                space.build_model(architecture_winner["spec"]), recipe
            )
            model.load_state_dict(common_state)
            (
                steps,
                elapsed,
                failed,
                optimizer_state,
            ) = self._train_low_fidelity(
                model,
                refinement_loader,
                common_steps,
                max(5.0, 1.80 * seconds_per_step * common_steps),
                deadline_remaining,
                recipe,
                None,
            )
            if failed or steps < common_steps:
                model.cpu()
                del model
                continue
            metrics = self._evaluate(model, validation_batches)
            model.cpu()
            trials.append(
                {
                    "model": model,
                    "recipe": dict(recipe),
                    "val_acc": metrics["accuracy"],
                    "steps": steps,
                    "elapsed": elapsed,
                    "optimizer_state": optimizer_state,
                }
            )
            print(
                "    recipe {} | val={:.2f}% | steps={}".format(
                    recipe["name"], metrics["accuracy"] * 100, steps
                )
            )

        if len(trials) < 2:
            selected_recipe = self._default_recipe()
            selected_trial = next(
                (
                    trial
                    for trial in trials
                    if trial["recipe"]["name"]
                    == selected_recipe["name"]
                ),
                None,
            )
            for trial in trials:
                if trial is not selected_trial:
                    del trial["model"]
            if selected_trial is None:
                architecture_winner["model"].load_state_dict(common_state)
                self._attach_recipe(
                    architecture_winner["model"], selected_recipe
                )
                search_state = architecture_winner.pop(
                    "optimizer_state", None
                )
            else:
                del architecture_winner["model"]
                architecture_winner["model"] = selected_trial["model"]
                architecture_winner["recipe"] = selected_trial["recipe"]
                architecture_winner["val_acc"] = selected_trial["val_acc"]
                architecture_winner["trained_steps"] += selected_trial[
                    "steps"
                ]
                architecture_winner["train_seconds"] += selected_trial[
                    "elapsed"
                ]
                search_state = selected_trial["optimizer_state"]
            architecture_winner["recipe"] = dict(selected_recipe)
            architecture_winner["model"].independent_retry_state = common_state
            architecture_winner[
                "model"
            ].search_optimizer_state = search_state
            architecture_winner[
                "model"
            ].alternative_training_recipes = [
                dict(recipe)
                for recipe in recipes
                if recipe["name"] != selected_recipe["name"]
                ]
            architecture_winner.pop("optimizer_state", None)
            return architecture_winner
        trials.sort(key=lambda item: item["val_acc"], reverse=True)
        selected = trials[0]
        for trial in trials[1:]:
            del trial["model"]
        del architecture_winner["model"]
        architecture_winner["model"] = selected["model"]
        architecture_winner["model"].search_optimizer_state = selected[
            "optimizer_state"
        ]
        architecture_winner["recipe"] = selected["recipe"]
        architecture_winner["val_acc"] = selected["val_acc"]
        architecture_winner["trained_steps"] += selected["steps"]
        architecture_winner["train_seconds"] += selected["elapsed"]
        architecture_winner["model"].independent_retry_state = common_state
        architecture_winner["model"].alternative_training_recipes = [
            trial["recipe"]
            for trial in trials[1:]
        ]
        architecture_winner.pop("optimizer_state", None)
        return architecture_winner

    def _successive_halving(
        self,
        space,
        ranked,
        n_top,
        cached_batches,
        validation_batches,
        round_plan,
        deadline_remaining,
    ):
        finalists = self._select_label_aware_entries(ranked, n_top)
        architecture_recipe = self._architecture_search_recipe()
        active = []
        for index, entry in enumerate(finalists):
            candidate = dict(entry)
            self._seed_everything(self.seed + 500 + index)
            candidate["recipe"] = dict(architecture_recipe)
            candidate["model"] = self._attach_recipe(
                space.build_model(entry["spec"]), architecture_recipe
            )
            candidate["trained_steps"] = 0
            candidate["train_seconds"] = 0.0
            candidate["val_history"] = []
            candidate["optimizer_state"] = None
            active.append(candidate)

        coverage_steps = max(1, len(cached_batches))
        for round_index, (coverage, survivors) in enumerate(round_plan):
            if not active or self.clock.check() <= deadline_remaining + 8:
                break
            max_steps = max(12, int(math.ceil(coverage_steps * coverage)))
            round_available = max(
                0.0, self.clock.check() - deadline_remaining - 8.0
            )
            # Preserve budget for later, higher-fidelity rounds instead of
            # letting the first broad round consume the whole search window.
            if round_index == len(round_plan) - 1:
                round_share = 0.85
            elif round_index == 0:
                round_share = 0.35
            else:
                round_share = 0.50
            time_quantum = max(
                2.0,
                round_share * round_available / max(1, len(active)),
            )
            print(
                "  [NAS] Fidelity round {}: {} candidates, up to {} steps "
                "or {} each".format(
                    round_index + 1,
                    len(active),
                    max_steps,
                    show_time(time_quantum),
                )
            )
            results = []
            for candidate_index, candidate in enumerate(active):
                if self.clock.check() <= deadline_remaining + 6:
                    break
                self._seed_everything(
                    self.seed + 1000 * (round_index + 1) + candidate_index
                )
                (
                    steps,
                    elapsed,
                    failed,
                    optimizer_state,
                ) = self._train_low_fidelity(
                    candidate["model"],
                    cached_batches,
                    max_steps,
                    time_quantum,
                    deadline_remaining,
                    candidate["recipe"],
                    candidate.get("optimizer_state"),
                )
                if failed or (steps == 0 and candidate["trained_steps"] == 0):
                    candidate["model"].cpu()
                    del candidate["model"]
                    continue
                candidate["trained_steps"] += steps
                candidate["optimizer_state"] = optimizer_state
                candidate["train_seconds"] += elapsed
                candidate["seconds_per_step"] = (
                    candidate["train_seconds"]
                    / max(1, candidate["trained_steps"])
                )
                projected_epoch_time = (
                    candidate["seconds_per_step"]
                    * max(1, len(self.train_loader))
                    * 1.20
                )
                candidate["projected_epochs"] = max(
                    0.0,
                    (deadline_remaining - 45.0)
                    / max(1e-3, projected_epoch_time),
                )
                metrics = self._evaluate(
                    candidate["model"], validation_batches
                )
                candidate["val_acc"] = metrics["accuracy"]
                candidate["balanced_acc"] = metrics["balanced_accuracy"]
                candidate["val_history"].append(candidate["val_acc"])
                self._adapt_search_lr(candidate)
                candidate["utility"] = self._candidate_utility(candidate)
                candidate["model"].cpu()
                results.append(candidate)
                print(
                    "    {}/{} {} + {} | val={:.2f}% bal={:.2f}% "
                    "steps={} projected_epochs={:.1f}".format(
                        candidate_index + 1,
                        len(active),
                        self._spec_label(candidate["spec"]),
                        candidate["recipe"]["name"],
                        candidate["val_acc"] * 100,
                        candidate["balanced_acc"] * 100,
                        candidate["trained_steps"],
                        candidate["projected_epochs"],
                    )
                )
            if not results:
                break
            results.sort(
                key=lambda item: item["utility"], reverse=True
            )
            keep = min(max(1, survivors), len(results))
            for candidate in results[keep:]:
                del candidate["model"]
                candidate.pop("optimizer_state", None)
            active = results[:keep]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if not active:
            return None

        refinement_remaining = max(
            0.0, self.clock.check() - deadline_remaining - 12.0
        )
        refinement_loader = self._make_refinement_loader()
        if (
            len(active) >= 2
            and refinement_loader is not None
            and refinement_remaining >= 20.0
        ):
            recipe_reserve = max(12.0, 0.22 * refinement_remaining)
            refinement_floor = deadline_remaining + recipe_reserve
            print(
                "  [NAS] Adaptive full-data refinement: {} candidates, "
                "{} batches per pass; {} reserved for recipe comparison".format(
                    len(active),
                    len(refinement_loader),
                    show_time(recipe_reserve),
                )
            )
            for candidate in active:
                candidate["refinement_history"] = []

            pass_index = 0
            while pass_index < 12 and len(active) >= 2:
                available = self.clock.check() - refinement_floor - 6.0
                if available <= 4.0:
                    break
                estimated_pass = sum(
                    max(1e-4, candidate.get("seconds_per_step", 0.05))
                    * len(refinement_loader)
                    for candidate in active
                )
                if estimated_pass * 1.25 <= available:
                    common_steps = len(refinement_loader)
                else:
                    common_steps = int(
                        available
                        / max(
                            1e-4,
                            1.25
                            * sum(
                                max(
                                    1e-4,
                                    candidate.get(
                                        "seconds_per_step", 0.05
                                    ),
                                )
                                for candidate in active
                            ),
                        )
                    )
                    if common_steps < max(12, len(refinement_loader) // 4):
                        break
                print(
                    "    refinement pass {}: {} equal steps per candidate".format(
                        pass_index + 1, common_steps
                    )
                )
                refined = []
                for candidate_index, candidate in enumerate(active):
                    if self.clock.check() <= refinement_floor + 4:
                        break
                    seconds_per_step = max(
                        1e-4, candidate.get("seconds_per_step", 0.05)
                    )
                    time_quantum = max(
                        5.0, 1.80 * seconds_per_step * common_steps
                    )
                    self._seed_everything(
                        self.seed + 8000 + 101 * pass_index + candidate_index
                    )
                    (
                        steps,
                        elapsed,
                        failed,
                        optimizer_state,
                    ) = self._train_low_fidelity(
                        candidate["model"],
                        refinement_loader,
                        common_steps,
                        time_quantum,
                        refinement_floor,
                        candidate["recipe"],
                        candidate.get("optimizer_state"),
                    )
                    if failed or steps == 0:
                        candidate["model"].cpu()
                        del candidate["model"]
                        candidate.pop("optimizer_state", None)
                        continue
                    candidate["optimizer_state"] = optimizer_state
                    candidate["trained_steps"] += steps
                    candidate["train_seconds"] += elapsed
                    candidate["seconds_per_step"] = (
                        candidate["train_seconds"]
                        / max(1, candidate["trained_steps"])
                    )
                    metrics = self._evaluate(
                        candidate["model"], validation_batches
                    )
                    candidate["val_acc"] = metrics["accuracy"]
                    candidate["balanced_acc"] = metrics[
                        "balanced_accuracy"
                    ]
                    candidate["val_history"].append(candidate["val_acc"])
                    candidate["refinement_history"].append(
                        candidate["val_acc"]
                    )
                    self._adapt_search_lr(candidate)
                    candidate["utility"] = self._candidate_utility(candidate)
                    candidate["model"].cpu()
                    refined.append(candidate)
                    print(
                        "      {} | val={:.2f}% | total steps={}".format(
                            self._spec_label(candidate["spec"]),
                            candidate["val_acc"] * 100,
                            candidate["trained_steps"],
                        )
                    )
                if len(refined) < 2:
                    if refined:
                        active = refined
                    break
                active = refined
                pass_index += 1
                recent_gains = [
                    history[-1] - history[-2]
                    for history in (
                        candidate["refinement_history"]
                        for candidate in active
                    )
                    if len(history) >= 2
                ]
                if (
                    pass_index >= 2
                    and recent_gains
                    and max(recent_gains) <= 0.003
                ):
                    print(
                        "    refinement converged; all finalist gains <= 0.30pp"
                    )
                    break
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # The same complete validation split decides between the last
        # challengers.  Earlier subset scores never compete directly with it.
        print(
            "  [NAS] Full-validation champion/challenger comparison ({})".format(
                len(active)
            )
        )
        final_results = []
        for candidate in active:
            if self.clock.check() <= deadline_remaining + 4 and final_results:
                break
            candidate["model"].to(self.device)
            metrics = self._evaluate(candidate["model"], None)
            candidate["full_val_acc"] = metrics["accuracy"]
            candidate["val_acc"] = candidate["full_val_acc"]
            candidate["balanced_acc"] = metrics["balanced_accuracy"]
            candidate["model"].cpu()
            final_results.append(candidate)
            print(
                "    {} + {} | full val={:.2f}% bal={:.2f}%".format(
                    self._spec_label(candidate["spec"]),
                    candidate["recipe"]["name"],
                    candidate["val_acc"] * 100,
                    candidate["balanced_acc"] * 100,
                )
            )
        if not final_results:
            final_results = active
        self._final_selection_scores(final_results)
        final_results.sort(
            key=lambda item: item.get("selection_score", -float("inf")),
            reverse=True,
        )
        winner = final_results[0]
        print(
            "  [NAS] Robust architecture score selected {} ({:.4f})".format(
                self._spec_label(winner["spec"]),
                winner.get("selection_score", 0.0),
            )
        )
        for candidate in active:
            if candidate is not winner and "model" in candidate:
                del candidate["model"]
                candidate.pop("optimizer_state", None)
        winner = self._recipe_tournament(
            space,
            winner,
            refinement_loader,
            validation_batches,
            deadline_remaining,
        )
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
        search_budget = min(
            max_search_seconds,
            max(40.0, start_remaining * search_fraction),
        )
        deadline_remaining = max(0.0, start_remaining - search_budget)
        print(
            "  [NAS] Search budget: {} (final training protected after ~{})".format(
                show_time(search_budget), show_time(deadline_remaining)
            )
        )

        space = self._make_space()
        recipes = self._training_recipes()
        pool = self._build_candidate_pool(space)
        candidates = self._portfolio_candidates(
            space, pool, n_candidates, recipes
        )
        min_params, max_params = self._parameter_bounds()
        print(
            "  [NAS] {} diverse architecture candidates from {} feasible specs; "
            "recipes tested only after architecture selection={} "
            "({:,}-{:,} parameter guard)".format(
                len(candidates),
                len(pool),
                ",".join(recipe["name"] for recipe in recipes),
                min_params,
                max_params,
            )
        )

        inputs = self._fixed_calibration_inputs()
        ranked = self._proxy_screen(
            space, candidates, inputs, deadline_remaining
        )
        if not ranked:
            print("  [NAS] Proxy phase exhausted; using robust anchor")
            return self._tier3_fallback()
        print("  [NAS] Proxy top-3 (pre-screen only):")
        for rank, entry in enumerate(ranked[:3], start=1):
            finite_scores = {
                name: round(value, 3)
                for name, value in entry["scores"].items()
                if math.isfinite(value)
            }
            print(
                "    {}. {} | {:,} params | {}".format(
                    rank,
                    self._spec_label(entry["spec"]),
                    entry["params"],
                    finite_scores,
                )
            )

        cached_batches = self._cache_search_batches()
        validation_batches = self._cache_validation_batches() or None
        if (
            not cached_batches
            or self.clock.check() <= deadline_remaining + 8
        ):
            winner = ranked[0]
            winner["recipe"] = dict(self._default_recipe())
            model = self._attach_recipe(
                space.build_model(winner["spec"]), winner["recipe"]
            )
            trained_steps = 0
            val_acc = None
        else:
            winner = self._successive_halving(
                space,
                ranked,
                n_top,
                cached_batches,
                validation_batches,
                round_plan,
                deadline_remaining,
            )
            if winner is None:
                winner = ranked[0]
                winner["recipe"] = dict(self._default_recipe())
                model = self._attach_recipe(
                    space.build_model(winner["spec"]), winner["recipe"]
                )
                trained_steps = 0
                val_acc = None
            else:
                model = winner["model"]
                trained_steps = winner["trained_steps"]
                val_acc = winner["val_acc"]

        self.metadata["nas_tier"] = (
            1 if start_remaining >= 15 * 60 else 2
        )
        self.metadata["nas_seed"] = self.seed
        self.metadata["nas_spec"] = winner["spec"]
        self.metadata["nas_recipe"] = winner["recipe"]["name"]
        self.metadata["nas_ensemble"] = False
        self.metadata["nas_pretrained_steps"] = trained_steps
        if not hasattr(model, "independent_retry_state"):
            model.independent_retry_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if not hasattr(model, "alternative_training_recipes"):
            model.alternative_training_recipes = [
                dict(recipe)
                for recipe in recipes
                if recipe["name"] != winner["recipe"]["name"]
            ]
        self.metadata["nas_recipe_candidates"] = [
            recipe["name"] for recipe in recipes
        ]
        self.metadata["nas_search_seconds"] = (
            start_remaining - self.clock.check()
        )

        summary = (
            "  [NAS] Winner: {} + {} | {:,} params | {} warm-start steps".format(
                winner["spec"],
                winner["recipe"]["name"],
                winner["params"],
                trained_steps,
            )
        )
        if val_acc is not None:
            summary += " | selection val={:.2f}%".format(val_acc * 100)
        print(summary)
        print(
            "  [NAS] Search used {}".format(
                show_time(start_remaining - self.clock.check())
            )
        )
        return model
