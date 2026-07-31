"""Hierarchical, portfolio-based NAS for unseen classification datasets."""

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _architecture_initialization_seed(self):
        """Common initialization seed for every macro architecture.

        Candidate position is deliberately absent. Reordering or inserting a
        candidate must not silently change the stochastic trial assigned to an
        existing architecture.
        """
        return self.seed + 500

    def _architecture_round_seed(self, round_index):
        """Common-random-number seed for a macro fidelity round."""
        return self.seed + 1000 * (int(round_index) + 1)

    def _architecture_refinement_seed(self, pass_index):
        """Common-random-number seed for an equal-work refinement pass."""
        return self.seed + 8000 + 101 * int(pass_index)

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

    @staticmethod
    def _cpu_model_state(model):
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

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
        # Scale the architecture/recipe investigation on long final-phase
        # budgets instead of stopping after six minutes regardless of the
        # supplied clock. Warm-started search updates are retained.
        scalable_search_cap = min(
            30 * 60.0,
            max(360.0, 0.24 * remaining),
        )
        return self._search_pipeline(
            n_candidates=54,
            n_top=12,
            search_fraction=0.22,
            max_search_seconds=scalable_search_cap,
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
                "optimizer": "adamw",
                "scheduler": "plateau",
                "lr_scale": 1.0,
                "weight_decay": 1e-2,
                # The unregularized control is important on symbolic and
                # already-noisy tasks where smoothing can suppress the best
                # attainable confidence.
                "label_smoothing": 0.0,
                "mixup_alpha": 0.0,
                "dropout_scale": 0.75,
                "use_class_weights": False,
            },
            {
                "name": "regularized",
                "optimizer": "adamw",
                "scheduler": "plateau",
                "lr_scale": 0.85,
                "weight_decay": 2e-2,
                "label_smoothing": 0.10,
                "mixup_alpha": 0.10 if self.num_classes > 2 else 0.0,
                "dropout_scale": 1.50,
                "use_class_weights": False,
            },
        ]
        if imbalance >= 1.35:
            recipes.append(
                {
                    "name": "balanced",
                    "optimizer": "adamw",
                    "scheduler": "plateau",
                    "lr_scale": 0.90,
                    "weight_decay": 1e-2,
                    "label_smoothing": 0.03,
                    "mixup_alpha": 0.0,
                    "dropout_scale": 1.0,
                    "use_class_weights": True,
                }
            )
        elif examples_per_class >= 500:
            recipes.append(
                {
                    "name": "fast_fit",
                    "optimizer": "adamw",
                    "scheduler": "plateau",
                    "lr_scale": 1.20,
                    "weight_decay": 5e-3,
                    "label_smoothing": 0.03,
                    "mixup_alpha": 0.0,
                    "dropout_scale": 0.50,
                    "use_class_weights": False,
                }
            )
        recipes.append(
            {
                "name": "sgd_nesterov",
                "optimizer": "sgd",
                "scheduler": "cosine",
                "search_lr": 2e-2,
                "final_lr": 2e-2,
                "lr_scale": 1.0,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 5e-4,
                "label_smoothing": 0.0,
                "mixup_alpha": 0.0,
                "dropout_scale": 1.0,
                "use_class_weights": imbalance >= 1.75,
            }
        )
        return recipes

    def _architecture_search_recipe(self):
        """Use one neutral recipe so architecture and recipe are not confounded."""
        return {
            "name": "architecture_probe",
            "optimizer": "adamw",
            "scheduler": "plateau",
            "lr_scale": 1.0,
            "weight_decay": 1e-2,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "dropout_scale": 1.0,
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
        full_grid_family = family in {
            "spatial",
            "spatial_pyramid",
            "factorized",
            "axis_width",
            "axis_height",
            "dual_axis",
        }
        if full_grid_family:
            # Use a genuinely distinct wide anchor when the parameter guard
            # permits it. The former nominal capacity anchor was another C32
            # model and could leave an entire promoted family without a C64
            # label-trained candidate.
            capacity = ArchSpec(
                min(2, space.max_safe_stages),
                64,
                3,
                "basic",
                3,
                False,
                3,
                family,
            )
        else:
            capacity = ArchSpec(
                min(3, space.max_safe_stages),
                32,
                2,
                "bottleneck",
                5,
                True,
                5,
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
        scale = float(recipe.get("dropout_scale", 1.0))
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                if not hasattr(module, "_nas_base_dropout"):
                    module._nas_base_dropout = float(module.p)
                module.p = min(
                    0.60,
                    max(0.0, module._nas_base_dropout * scale),
                )
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
                        "optimizer": "adamw",
                        "scheduler": "plateau",
                        "lr_scale": 1.0,
                        "weight_decay": 1e-2,
                        "label_smoothing": 0.0,
                        "mixup_alpha": 0.0,
                        "dropout_scale": 1.0,
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
            remaining_bytes = byte_budget - used
            if remaining_bytes <= 0:
                break
            if batch_bytes > remaining_bytes:
                bytes_per_example = max(
                    1, batch_bytes // max(1, data.size(0))
                )
                fitting = int(remaining_bytes // bytes_per_example)
                if fitting <= 0:
                    break
                data = data[:fitting]
                target = target[:fitting]
                batch_bytes = (
                    data.numel() * data.element_size()
                    + target.numel() * target.element_size()
                )
            cached.append(
                (data.detach().cpu().clone(), target.detach().cpu().clone())
            )
            used += batch_bytes
            if data.size(0) < getattr(
                self.train_loader, "batch_size", data.size(0)
            ):
                break
            if len(cached) >= min(max_batches, len(self.train_loader)):
                break
        print(
            "  [NAS] Cached {} fair training batches ({:.1f}% epoch coverage)".format(
                len(cached),
                100.0 * len(cached) / max(1, len(self.train_loader)),
            )
        )
        return cached

    @staticmethod
    def _materialize_validation_subset(dataset, indices, batch_size):
        if not indices:
            return []
        subset = torch.utils.data.Subset(dataset, sorted(indices))
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return [
            (data.detach().cpu().clone(), target.detach().cpu().clone())
            for data, target in loader
        ]

    def _cache_validation_splits(
        self,
        max_samples=4096,
        max_megabytes=64,
        confirmation_fraction=0.25,
    ):
        """Cache prior-preserving selection and untouched confirmation sets."""
        dataset = getattr(self.valid_loader, "dataset", None)
        labels = getattr(dataset, "y", None)
        if dataset is None or labels is None or len(dataset) == 0:
            return [], []
        labels = labels.detach().cpu().numpy()
        classes = np.unique(labels)
        sample_tensor = getattr(dataset, "x", None)
        if sample_tensor is not None and sample_tensor.ndim >= 2:
            output_elements = int(np.prod(sample_tensor.shape[1:]))
        else:
            output_elements = (
                self.in_channels * self.input_h * self.input_w
            )
        bytes_per_sample = max(1, 4 * output_elements + 8)
        byte_limited = max(
            1, int(max_megabytes * 1024 * 1024) // bytes_per_sample
        )
        target_samples = min(
            len(dataset), int(max_samples), byte_limited
        )
        class_counts = np.asarray(
            [np.sum(labels == class_id) for class_id in classes],
            dtype=np.int64,
        )
        exact_quotas = class_counts * (
            float(target_samples) / max(1, len(dataset))
        )
        quotas = np.floor(exact_quotas).astype(np.int64)
        remainder = target_samples - int(quotas.sum())
        fractional_order = np.argsort(
            -(exact_quotas - quotas), kind="stable"
        )
        for class_index in fractional_order[:remainder]:
            quotas[class_index] += 1

        selection_indices = []
        confirmation_indices = []
        for class_index, class_id in enumerate(classes):
            positions = np.flatnonzero(labels == class_id)
            quota = min(len(positions), int(quotas[class_index]))
            if quota:
                offsets = np.linspace(
                    0, len(positions) - 1, quota, dtype=np.int64
                )
                chosen = positions[offsets].tolist()
                confirmation_count = (
                    max(1, int(round(quota * confirmation_fraction)))
                    if quota >= 2
                    else 0
                )
                confirmation_offsets = set(
                    np.linspace(
                        0,
                        quota - 1,
                        confirmation_count,
                        dtype=np.int64,
                    ).tolist()
                )
                confirmation = [
                    value
                    for offset, value in enumerate(chosen)
                    if offset in confirmation_offsets
                ]
                selection = [
                    value
                    for offset, value in enumerate(chosen)
                    if offset not in confirmation_offsets
                ]
                confirmation_indices.extend(confirmation)
                selection_indices.extend(selection)

        # A tiny validation split may not support a separate confirmation
        # sample. In that case retain the prior-preserving selection set.
        if not selection_indices:
            selection_indices = confirmation_indices
            confirmation_indices = []
        batch_size = (
            getattr(self.valid_loader, "batch_size", 128) or 128
        )
        selection_batches = self._materialize_validation_subset(
            dataset, selection_indices, batch_size
        )
        confirmation_batches = self._materialize_validation_subset(
            dataset, confirmation_indices, batch_size
        )
        self.metadata["nas_selection_samples"] = len(selection_indices)
        self.metadata["nas_confirmation_samples"] = len(
            confirmation_indices
        )
        self.metadata["nas_validation_cache_megabytes"] = (
            target_samples * bytes_per_sample / (1024.0 * 1024.0)
        )
        print(
            "  [NAS] Validation split: {} selection + {} confirmation "
            "samples ({:.1f} MiB cap use)".format(
                len(selection_indices),
                len(confirmation_indices),
                self.metadata["nas_validation_cache_megabytes"],
            )
        )
        return selection_batches, confirmation_batches

    def _cache_validation_batches(self, max_samples=4096):
        """Backward-compatible selection-cache accessor used by tests."""
        selection, _ = self._cache_validation_splits(
            max_samples=max_samples
        )
        return selection

    def _calibrate_search_microbatch(
        self, space, cached_batches, deadline_remaining
    ):
        """Exercise one compact update so OOM fallback is fixed before racing."""
        if not cached_batches:
            return
        logical_batch = int(cached_batches[0][1].size(0))
        self.search_microbatch_size = logical_batch
        if self.device.type != "cuda":
            self.metadata["nas_search_microbatch_size"] = logical_batch
            return
        if self.clock.check() <= deadline_remaining + 12:
            return
        spec = self._anchor_spec(self._fallback_family(space), space)
        model = space.build_model(spec)
        try:
            self._seed_everything(self.seed + 71)
            self._train_low_fidelity(
                model,
                cached_batches[:1],
                max_steps=1,
                time_quantum=min(
                    12.0,
                    max(
                        2.0,
                        self.clock.check() - deadline_remaining - 10.0,
                    ),
                ),
                deadline_remaining=deadline_remaining,
                recipe=self._architecture_search_recipe(),
            )
        finally:
            model.cpu()
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.metadata["nas_search_microbatch_size"] = int(
            self.search_microbatch_size
        )
        print(
            "  [NAS] Calibrated logical/microbatch: {}/{}".format(
                logical_batch, self.search_microbatch_size
            )
        )

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

    def _policy_probe_batches(self, max_batches=24):
        """Materialize the same deterministic examples under the active policy."""
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or len(dataset) == 0:
            return []
        generator = torch.Generator()
        generator.manual_seed(self.seed + 59)
        indices = torch.randperm(
            len(dataset), generator=generator
        ).tolist()
        batch_size = int(
            getattr(self.train_loader, "batch_size", 128) or 128
        )
        sample_limit = min(
            len(indices), max(1, int(max_batches)) * batch_size
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, indices[:sample_limit]),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return [
            (data.detach().cpu().clone(), target.detach().cpu().clone())
            for data, target in loader
        ]

    def _select_augmentation_policy(
        self,
        space,
        validation_batches,
        deadline_remaining,
    ):
        """Run an incumbent-safe compact functional probe over safe policies."""
        dataset = getattr(self.train_loader, "dataset", None)
        policies = getattr(dataset, "augmentation_policies", {})
        current = getattr(dataset, "augmentation_policy", "identity")
        if (
            dataset is None
            or not hasattr(dataset, "set_augmentation_policy")
            or len(policies) <= 1
            or not validation_batches
        ):
            return current
        available = self.clock.check() - deadline_remaining - 12.0
        if available < max(12.0, 3.0 * len(policies)):
            return current

        spec = self._anchor_spec(self._fallback_family(space), space)
        common_model = space.build_model(spec)
        common_state = {
            key: value.detach().cpu().clone()
            for key, value in common_model.state_dict().items()
        }
        del common_model
        recipe = self._architecture_search_recipe()
        quantum = max(
            2.0,
            min(12.0, 0.16 * available / max(1, len(policies))),
        )
        results = []
        print(
            "  [NAS] Functional augmentation probe: {}".format(
                ",".join(policies)
            )
        )
        try:
            for index, name in enumerate(policies):
                if self.clock.check() <= deadline_remaining + 10:
                    break
                dataset.set_augmentation_policy(name)
                self._seed_everything(self.seed + 600)
                batches = self._policy_probe_batches(max_batches=24)
                if not batches:
                    continue
                model = space.build_model(spec)
                model.load_state_dict(common_state)
                (
                    steps,
                    _,
                    failed,
                    _,
                    train_stats,
                ) = self._train_low_fidelity(
                    model,
                    batches,
                    max_steps=max(8, min(24, len(batches))),
                    time_quantum=quantum,
                    deadline_remaining=deadline_remaining,
                    recipe=recipe,
                )
                if failed or steps == 0:
                    model.cpu()
                    del model
                    continue
                metrics = self._evaluate(model, validation_batches)
                model.cpu()
                del model
                results.append(
                    {
                        "name": name,
                        "accuracy": float(metrics["accuracy"]),
                        "loss": float(metrics["loss"]),
                        "samples": int(metrics["samples"]),
                        "train_accuracy": float(
                            train_stats.get("train_accuracy", 0.0)
                        ),
                    }
                )
                print(
                    "    {} | val={:.2f}% loss={:.4f} steps={}".format(
                        name,
                        metrics["accuracy"] * 100,
                        metrics["loss"],
                        steps,
                    )
                )
        finally:
            dataset.set_augmentation_policy(current)

        incumbent = next(
            (result for result in results if result["name"] == current),
            None,
        )
        if incumbent is None:
            return current
        best = max(
            results,
            key=lambda result: (
                result["accuracy"],
                -result["loss"],
            ),
        )
        margin = self._recipe_acceptance_margin(
            incumbent["samples"],
            incumbent["accuracy"],
            best["accuracy"],
        )
        accuracy_gain = best["accuracy"] - incumbent["accuracy"]
        relative_loss_gain = (
            incumbent["loss"] - best["loss"]
        ) / max(1e-8, abs(incumbent["loss"]))
        accepted = (
            accuracy_gain + 1e-12 >= margin
            or (
                accuracy_gain >= -margin
                and relative_loss_gain >= 0.01
            )
        )
        selected = best["name"] if accepted else current
        dataset.set_augmentation_policy(selected)
        self.metadata["augmentation_policy"] = selected
        self.metadata["augmentation_probe"] = results
        print(
            "  [NAS] Augmentation policy: {}{}".format(
                selected,
                " (incumbent retained)" if selected == current else "",
            )
        )
        return selected

    @staticmethod
    def _family_probe_score(probe):
        history = probe.get("metrics_history", [])
        latest = history[-1]
        first = history[0]
        relative_loss_gain = (
            float(first["loss"]) - float(latest["loss"])
        ) / max(1e-8, abs(float(first["loss"])))
        train_accuracy = float(
            probe.get("train_stats", {}).get("train_accuracy", 0.0)
        )
        gap = max(0.0, train_accuracy - float(latest["accuracy"]))
        samples = max(1, int(latest.get("samples", 1)))
        accuracy = float(latest["accuracy"])
        uncertainty = math.sqrt(
            max(1e-8, accuracy * (1.0 - accuracy)) / samples
        )
        score = (
            accuracy
            + min(0.01, 0.025 * max(-0.20, relative_loss_gain))
            + min(0.006, 0.50 * uncertainty)
            - 0.006 * min(0.50, gap)
        )
        probe["loss_slope"] = relative_loss_gain
        probe["train_validation_gap"] = gap
        probe["probe_score"] = score
        return score

    def _representation_probe_race(
        self,
        space,
        ranked,
        cached_batches,
        validation_batches,
        deadline_remaining,
    ):
        """Promote representation families before racing macro dimensions."""
        family_entries = {}
        for entry in ranked:
            family_entries.setdefault(
                entry["spec"].model_family, []
            ).append(entry)
        families = list(family_entries)
        if len(families) <= 2 or not cached_batches or not validation_batches:
            return set(families)
        available = self.clock.check() - deadline_remaining - 12.0
        if available < max(18.0, 2.5 * len(families)):
            return set(families)

        recipe = self._architecture_search_recipe()
        stage_steps = max(
            8, min(len(cached_batches), int(math.ceil(len(cached_batches) / 2)))
        )
        probe_budget = min(
            0.32 * available,
            max(18.0, 7.0 * len(families)),
        )
        stage_one_quantum = max(
            2.0, 0.55 * probe_budget / max(1, len(families))
        )
        probes = []
        print(
            "  [NAS] Representation probe race: {} families, "
            "{} progressive steps/stage".format(
                len(families), stage_steps
            )
        )
        for family in families:
            if self.clock.check() <= deadline_remaining + 10:
                break
            entries = family_entries[family]
            anchor = self._anchor_spec(family, space)
            entry = min(
                entries,
                key=lambda item: (
                    item["spec"] != anchor,
                    item["params"],
                ),
            )
            self._seed_everything(self.seed + 1200)
            model = self._attach_recipe(
                space.build_model(entry["spec"]), recipe
            )
            try:
                initial = self._evaluate(model, validation_batches)
            except (RuntimeError, ValueError) as error:
                print(
                    "    {} probe skipped before training: {}".format(
                        family, str(error).splitlines()[0]
                    )
                )
                model.cpu()
                del model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            # Model constructors consume different numbers of random values.
            # Reset once more so label-order, augmentation and dropout draws
            # begin from a family-order-independent common seed.
            self._seed_everything(self.seed + 1800)
            (
                steps,
                elapsed,
                failed,
                optimizer_state,
                train_stats,
            ) = self._train_low_fidelity(
                model,
                cached_batches,
                stage_steps,
                stage_one_quantum,
                deadline_remaining,
                recipe,
                start_step=0,
            )
            if failed or steps == 0:
                model.cpu()
                del model
                continue
            metrics = self._evaluate(model, validation_batches)
            probe = {
                "family": family,
                "entry": entry,
                "model": model,
                "optimizer_state": optimizer_state,
                "steps": steps,
                "elapsed": elapsed,
                "data_cursor": steps,
                "metrics_history": [initial, metrics],
                "train_stats": train_stats,
            }
            self._family_probe_score(probe)
            model.cpu()
            probes.append(probe)
            print(
                "    {} | val={:.2f}% loss={:.4f} slope={:+.3f}".format(
                    family,
                    metrics["accuracy"] * 100,
                    metrics["loss"],
                    probe["loss_slope"],
                )
            )

        if len(probes) < 2:
            successful_families = {
                probe["family"] for probe in probes
            }
            for probe in probes:
                del probe["model"]
            if successful_families:
                return successful_families
            safe_families = {
                family
                for family in families
                if family in {"spatial", "spatial_pyramid", "factorized"}
            }
            return safe_families or set(families)
        probes.sort(key=lambda item: item["probe_score"], reverse=True)
        promoted = probes[:2]
        if len(probes) >= 3:
            second_metrics = promoted[-1]["metrics_history"][-1]
            third_metrics = probes[2]["metrics_history"][-1]
            samples = max(
                1,
                min(
                    int(second_metrics.get("samples", 1)),
                    int(third_metrics.get("samples", 1)),
                ),
            )
            threshold = max(
                1.0 / samples,
                min(
                    0.01,
                    math.sqrt(
                        max(
                            1e-8,
                            float(second_metrics["accuracy"])
                            * (1.0 - float(second_metrics["accuracy"]))
                            / samples,
                        )
                    ),
                ),
            )
            if (
                promoted[-1]["probe_score"] - probes[2]["probe_score"]
                <= threshold
            ):
                promoted.append(probes[2])

        stage_two_quantum = max(
            2.0,
            0.40 * probe_budget / max(1, len(promoted)),
        )
        completed = []
        for probe in promoted:
            if self.clock.check() <= deadline_remaining + 9:
                break
            self._seed_everything(self.seed + 2200)
            (
                steps,
                elapsed,
                failed,
                optimizer_state,
                train_stats,
            ) = self._train_low_fidelity(
                probe["model"],
                cached_batches,
                stage_steps,
                stage_two_quantum,
                deadline_remaining,
                recipe,
                probe["optimizer_state"],
                start_step=probe["data_cursor"],
            )
            if failed or steps == 0:
                continue
            try:
                metrics = self._evaluate(
                    probe["model"], validation_batches
                )
            except (RuntimeError, ValueError) as error:
                print(
                    "    promoted {} evaluation skipped: {}".format(
                        probe["family"], str(error).splitlines()[0]
                    )
                )
                probe["model"].cpu()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            probe["optimizer_state"] = optimizer_state
            probe["steps"] += steps
            probe["elapsed"] += elapsed
            probe["data_cursor"] += steps
            probe["train_stats"] = train_stats
            probe["metrics_history"].append(metrics)
            self._family_probe_score(probe)
            completed.append(probe)
            print(
                "    promoted {} | val={:.2f}% loss={:.4f} "
                "total_steps={}".format(
                    probe["family"],
                    metrics["accuracy"] * 100,
                    metrics["loss"],
                    probe["steps"],
                )
            )

        selected_probes = completed if len(completed) >= 2 else promoted
        selected_probes.sort(
            key=lambda item: item["probe_score"], reverse=True
        )
        selected_families = {
            probe["family"] for probe in selected_probes[:2]
        }
        if len(selected_probes) >= 3:
            first = selected_probes[1]["metrics_history"][-1]
            second = selected_probes[2]["metrics_history"][-1]
            samples = max(
                1,
                min(
                    int(first.get("samples", 1)),
                    int(second.get("samples", 1)),
                ),
            )
            tie = max(1.0 / samples, 0.005)
            if (
                selected_probes[1]["probe_score"]
                - selected_probes[2]["probe_score"]
                <= tie
            ):
                selected_families.add(selected_probes[2]["family"])
        for probe in probes:
            probe["model"].cpu()
            del probe["model"]
        self.metadata["nas_promoted_families"] = sorted(
            selected_families
        )
        self.metadata["nas_family_probe_results"] = [
            {
                "family": probe["family"],
                "score": float(probe["probe_score"]),
                "steps": int(probe["steps"]),
                "loss_slope": float(probe["loss_slope"]),
                "train_validation_gap": float(
                    probe["train_validation_gap"]
                ),
            }
            for probe in probes
        ]
        print(
            "  [NAS] Promoted representation families: {}".format(
                ",".join(sorted(selected_families))
            )
        )
        return selected_families

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

    def _expand_promoted_macro_pool(self, ranked, pool, promoted_families):
        """Restore full size coverage after the representation race."""
        promoted_families = set(promoted_families)
        expanded = [
            entry
            for entry in ranked
            if entry["spec"].model_family in promoted_families
        ]
        seen = {entry["spec"] for entry in expanded}
        for spec, params in sorted(
            pool,
            key=lambda item: (
                item[0].model_family,
                int(item[1]),
                tuple(item[0]),
            ),
        ):
            if (
                spec.model_family not in promoted_families
                or spec in seen
            ):
                continue
            expanded.append(
                {
                    "spec": spec,
                    "params": int(params),
                    "scores": {},
                    "proxy_prior": 0.0,
                    "proxy_evaluated": False,
                }
            )
            seen.add(spec)

        counts = {
            family: sum(
                entry["spec"].model_family == family
                for entry in expanded
            )
            for family in sorted(promoted_families)
        }
        self.metadata["nas_promoted_macro_pool_size"] = int(len(expanded))
        print(
            "  [NAS] Promoted macro pool restored from full feasible "
            "size strata: {}".format(counts)
        )
        return expanded

    def _select_label_aware_entries(
        self,
        ranked,
        n_top,
        anchor_specs_by_family=None,
    ):
        """Allocate family quotas and parameter strata before proxy tie-breaks."""
        if not ranked or n_top <= 0:
            return []

        anchor_specs_by_family = anchor_specs_by_family or {}

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
        proxy_order = {
            entry["spec"]: index
            for index, entry in enumerate(ranked)
            if entry.get("proxy_evaluated", True)
        }

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
            # Reserve evenly spaced deterministic reference points before
            # filling the remaining quota from size strata. With two slots,
            # this keeps the compact and capacity endpoints; with three, it
            # keeps compact, central, and capacity anchors. Missing or
            # parameter-infeasible anchors consume no slot.
            available_anchors = [
                spec
                for spec in anchor_specs_by_family.get(family, ())
                if any(entry["spec"] == spec for entry in family_entries)
            ]
            if available_anchors:
                anchor_slots = min(quota, len(available_anchors))
                if anchor_slots == 1:
                    anchor_indices = [len(available_anchors) // 2]
                else:
                    anchor_indices = [
                        int(
                            round(
                                index
                                * (len(available_anchors) - 1)
                                / (anchor_slots - 1)
                            )
                        )
                        for index in range(anchor_slots)
                    ]
                for anchor_index in dict.fromkeys(anchor_indices):
                    anchor_spec = available_anchors[anchor_index]
                    anchor_entry = next(
                        entry
                        for entry in family_entries
                        if entry["spec"] == anchor_spec
                    )
                    before = len(selected)
                    add(anchor_entry)
                    family_selected += int(len(selected) > before)

            # Place deterministic targets uniformly in log-parameter space.
            # Proxy ranks only break equal-distance ties, so a cross-family
            # proxy bias cannot silently remove compact or wide capacity.
            sized = sorted(
                family_entries,
                key=lambda item: (
                    int(item.get("params", 0)),
                    item["spec"].init_channels,
                    item["spec"].num_stages,
                ),
            )
            log_min = math.log1p(max(0, int(sized[0].get("params", 0))))
            log_max = math.log1p(max(0, int(sized[-1].get("params", 0))))
            targets = (
                [(log_min + log_max) / 2.0]
                if quota == 1
                else [
                    log_min
                    + (log_max - log_min) * stratum / max(1, quota - 1)
                    for stratum in range(quota)
                ]
            )
            for target_size in targets:
                if family_selected >= quota:
                    break
                bucket = [
                    item
                    for item in sized
                    if item["spec"] not in keys
                ]
                if not bucket:
                    continue
                entry = min(
                    bucket,
                    key=lambda item: (
                        abs(
                            math.log1p(
                                max(0, int(item.get("params", 0)))
                            )
                            - target_size
                        ),
                        proxy_order.get(item["spec"], len(ranked)),
                        tuple(item["spec"]),
                    ),
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

    def _make_search_optimizer(self, model, recipe):
        optimizer_name = str(recipe.get("optimizer", "adamw")).lower()
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=float(recipe.get("search_lr", 2e-2))
                * float(recipe.get("lr_scale", 1.0)),
                momentum=float(recipe.get("momentum", 0.9)),
                nesterov=bool(recipe.get("nesterov", True)),
                weight_decay=float(recipe.get("weight_decay", 5e-4)),
            )
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(recipe.get("search_lr", 1e-3))
            * float(recipe.get("lr_scale", 1.0)),
            weight_decay=float(recipe.get("weight_decay", 1e-2)),
        )

    def _train_low_fidelity(
        self,
        model,
        batch_source,
        max_steps,
        time_quantum,
        deadline_remaining,
        recipe,
        optimizer_state=None,
        start_step=0,
    ):
        model.to(self.device)
        model.train()
        optimizer = self._make_search_optimizer(model, recipe)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        cosine_search = (
            str(recipe.get("scheduler", "plateau")).lower() == "cosine"
        )
        if cosine_search:
            segment_start_lr = float(
                recipe.get("search_lr", 2e-2)
            ) * float(recipe.get("lr_scale", 1.0))
            for group in optimizer.param_groups:
                group["lr"] = segment_start_lr
        else:
            segment_start_lr = 0.0
        criterion = self._criterion(recipe)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        completed = 0
        examples_seen = 0
        train_correct = 0.0
        train_loss_sum = 0.0
        started = time.perf_counter()
        failed = False
        saved_optimizer_state = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        is_cached = isinstance(batch_source, (list, tuple))
        batch_iterator = None if is_cached else iter(batch_source)
        if not is_cached and start_step and len(batch_source):
            skip = int(start_step) % len(batch_source)
            for _ in range(skip):
                try:
                    next(batch_iterator)
                except StopIteration:
                    batch_iterator = iter(batch_source)
                    next(batch_iterator)
        try:
            for step in range(max_steps):
                if (
                    self.clock.check() <= deadline_remaining + 4
                    or time.perf_counter() - started >= time_quantum
                ):
                    break
                if is_cached:
                    data, target = batch_source[
                        (int(start_step) + step) % len(batch_source)
                    ]
                else:
                    try:
                        data, target = next(batch_iterator)
                    except StopIteration:
                        batch_iterator = iter(batch_source)
                        data, target = next(batch_iterator)
                logical_batch_size = int(target.size(0))
                configured_microbatch = int(
                    getattr(
                        self,
                        "search_microbatch_size",
                        logical_batch_size,
                    )
                )
                microbatch_size = max(
                    1, min(logical_batch_size, configured_microbatch)
                )
                batch_correct = 0.0
                batch_loss_sum = 0.0
                batch_completed = False
                while not batch_completed:
                    optimizer.zero_grad(set_to_none=True)
                    batch_correct = 0.0
                    batch_loss_sum = 0.0
                    finite_batch = True
                    try:
                        for micro_start in range(
                            0, logical_batch_size, microbatch_size
                        ):
                            micro_end = min(
                                logical_batch_size,
                                micro_start + microbatch_size,
                            )
                            micro_data = data[
                                micro_start:micro_end
                            ].to(
                                self.device,
                                non_blocking=self.device.type == "cuda",
                            )
                            micro_target = target[
                                micro_start:micro_end
                            ].to(
                                self.device,
                                non_blocking=self.device.type == "cuda",
                            )
                            (
                                mixed,
                                first,
                                second,
                                coefficient,
                            ) = self._mixup(
                                micro_data,
                                micro_target,
                                float(recipe.get("mixup_alpha", 0.0)),
                            )
                            with torch.cuda.amp.autocast(
                                enabled=self.device.type == "cuda"
                            ):
                                output = model(mixed)
                                loss = (
                                    coefficient
                                    * criterion(output, first)
                                    + (1.0 - coefficient)
                                    * criterion(output, second)
                                )
                            if not torch.isfinite(loss):
                                finite_batch = False
                                break
                            weight = (
                                float(micro_end - micro_start)
                                / logical_batch_size
                            )
                            scaler.scale(loss * weight).backward()
                            predictions = output.detach().argmax(dim=1)
                            batch_correct += float(
                                coefficient
                                * (predictions == first).sum().item()
                                + (1.0 - coefficient)
                                * (predictions == second).sum().item()
                            )
                            batch_loss_sum += (
                                float(loss.detach().item())
                                * (micro_end - micro_start)
                            )
                            del micro_data, micro_target, mixed, output, loss
                        if not finite_batch:
                            optimizer.zero_grad(set_to_none=True)
                            break
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 5.0
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        batch_completed = True
                    except RuntimeError as error:
                        if not self._is_oom(error):
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        if microbatch_size <= 1:
                            raise
                        previous = microbatch_size
                        microbatch_size = max(1, microbatch_size // 2)
                        self.search_microbatch_size = microbatch_size
                        self.metadata["nas_search_microbatch_size"] = (
                            microbatch_size
                        )
                        print(
                            "    [NAS] Search microbatch {} -> {} after OOM".format(
                                previous, microbatch_size
                            )
                        )
                if not batch_completed:
                    continue
                completed += 1
                if cosine_search:
                    progress = min(
                        1.0, completed / max(1, int(max_steps))
                    )
                    minimum_lr = 0.02 * segment_start_lr
                    learning_rate = minimum_lr + 0.5 * (
                        segment_start_lr - minimum_lr
                    ) * (1.0 + math.cos(math.pi * progress))
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                examples_seen += logical_batch_size
                train_correct += batch_correct
                train_loss_sum += batch_loss_sum
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
        final_lr = 0.0
        if saved_optimizer_state:
            groups = saved_optimizer_state.get("param_groups", [])
            if groups:
                final_lr = float(groups[0].get("lr", 0.0))
        train_stats = {
            "examples_seen": examples_seen,
            "train_accuracy": train_correct / max(1, examples_seen),
            "train_loss": train_loss_sum / max(1, examples_seen),
            "final_lr": final_lr,
            "peak_memory_mb": (
                torch.cuda.max_memory_allocated(self.device)
                / (1024.0 * 1024.0)
                if self.device.type == "cuda"
                else 0.0
            ),
        }
        return (
            completed,
            max(1e-4, time.perf_counter() - started),
            failed,
            saved_optimizer_state,
            train_stats,
        )

    def _evaluation_logits(self, model, data):
        """Forward an evaluation batch with ordered recursive OOM splitting."""
        cpu_data = data.detach().cpu()
        try:
            device_data = cpu_data.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=self.device.type == "cuda"
            ):
                return model(device_data).float().cpu()
        except RuntimeError as error:
            if not self._is_oom(error) or cpu_data.size(0) <= 1:
                raise
            if "device_data" in locals():
                del device_data
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            midpoint = cpu_data.size(0) // 2
            self.metadata["nas_evaluation_microbatch_size"] = min(
                int(
                    self.metadata.get(
                        "nas_evaluation_microbatch_size",
                        cpu_data.size(0),
                    )
                ),
                max(midpoint, cpu_data.size(0) - midpoint),
            )
            return torch.cat(
                [
                    self._evaluation_logits(model, cpu_data[:midpoint]),
                    self._evaluation_logits(model, cpu_data[midpoint:]),
                ],
                dim=0,
            )

    def _evaluate(self, model, source=None):
        source = source if source is not None else self.valid_loader
        # Evaluation is a public controller boundary and must not depend on
        # every caller remembering to place a freshly built probe first.
        # Representation probes are intentionally stored on CPU between
        # stages; on CUDA, omitting this move produced HalfTensor inputs
        # against CPU FloatTensor weights and failed every dataset.
        model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        total_margin = 0.0
        class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        class_total = torch.zeros(self.num_classes, dtype=torch.long)
        with torch.no_grad():
            for data, target in source:
                output = self._evaluation_logits(model, data)
                target = target.detach().cpu().long()
                total_loss += float(
                    F.cross_entropy(
                        output.float(), target, reduction="sum"
                    ).item()
                )
                if output.size(1) >= 2:
                    top_two = torch.topk(
                        torch.softmax(output.float(), dim=1),
                        k=2,
                        dim=1,
                    ).values
                    total_margin += float(
                        (top_two[:, 0] - top_two[:, 1]).sum().item()
                    )
                predictions = output.argmax(dim=1)
                matches = predictions == target
                correct += int(matches.sum().item())
                total += target.size(0)
                cpu_target = target
                cpu_matches = matches.long()
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
            "loss": total_loss / max(1, total),
            "margin": total_margin / max(1, total),
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
        loss_history = entry.get("val_loss_history", [])
        relative_loss_gain = (
            max(0.0, loss_history[-2] - loss_history[-1])
            / max(1e-8, abs(loss_history[-2]))
            if len(loss_history) >= 2
            else 0.0
        )
        consistency = (
            sum(history[-3:]) / min(3, len(history))
            if history
            else validation_score
        )
        latency_penalty = 0.001 * math.log1p(
            max(0.0, seconds_per_step * 1000.0)
        )
        epoch_penalty = 0.0015 * max(0.0, 12.0 - projected_epochs)
        samples = max(1, int(entry.get("val_samples", 1)))
        uncertainty = math.sqrt(
            max(1e-8, accuracy * (1.0 - accuracy)) / samples
        )
        # Absolute loss and train/validation separation are too noisy during
        # the broad low-fidelity race. They become a small, bounded risk term
        # only after a candidate has received at least two full-data
        # refinement observations.
        late_risk_penalty = 0.0
        if len(entry.get("refinement_history", [])) >= 2:
            late_risk_penalty = min(
                0.006,
                0.006 * self._architecture_risk(entry),
            )
        return (
            0.88 * validation_score
            + 0.09 * consistency
            + min(0.003, 0.10 * gain)
            + min(0.003, 0.02 * relative_loss_gain)
            + min(0.002, 0.25 * uncertainty)
            - latency_penalty
            - epoch_penalty
            - late_risk_penalty
        )

    def _architecture_risk(self, entry):
        """Return bounded late evidence of brittle generalization.

        Accuracy remains the competition objective. This signal is used only
        after meaningful fidelity, for uncertainty ties and for deciding
        whether a materially smaller dormant challenger is worth retaining.
        """
        accuracy = float(
            entry.get(
                "confirmation_val_acc",
                entry.get(
                    "selection_val_acc",
                    entry.get("val_acc", 0.0),
                ),
            )
        )
        train_accuracy = float(
            entry.get("train_stats", {}).get("train_accuracy", accuracy)
        )
        gap = max(0.0, train_accuracy - accuracy)
        loss = float(
            entry.get(
                "confirmation_val_loss",
                entry.get(
                    "selection_val_loss",
                    entry.get("val_loss", float("nan")),
                ),
            )
        )
        chance_loss = math.log(max(2, int(getattr(self, "num_classes", 2))))
        loss_excess = (
            max(0.0, loss / max(1e-8, chance_loss) - 1.0)
            if math.isfinite(loss)
            else 0.0
        )
        ready = len(entry.get("refinement_history", [])) >= 2
        risk = 0.0
        if ready:
            risk = (
                0.65 * min(1.0, gap / 0.35)
                + 0.35 * min(1.0, loss_excess / 0.50)
            )
        entry["train_validation_gap"] = gap
        entry["absolute_loss_excess"] = loss_excess
        entry["architecture_risk_ready"] = ready
        entry["architecture_risk"] = risk
        return risk

    @staticmethod
    def _candidate_accuracy(entry):
        return float(
            entry.get(
                "confirmation_val_acc",
                entry.get(
                    "selection_val_acc",
                    entry.get("val_acc", 0.0),
                ),
            )
        )

    @staticmethod
    def _recent_accuracy_gain(entry):
        history = entry.get("val_history", [])
        if len(history) < 2:
            return 0.0
        return max(0.0, float(history[-1]) - float(history[-2]))

    def _anchor_insurance_allowance(self, entry):
        """Bound how far a still-learning reference may trail the cutoff."""
        accuracy = self._candidate_accuracy(entry)
        samples = max(1, int(entry.get("val_samples", 1)))
        uncertainty = math.sqrt(
            max(1e-8, accuracy * (1.0 - accuracy)) / samples
        )
        return min(
            0.08,
            max(
                0.025,
                2.0 * self._recent_accuracy_gain(entry)
                + 2.0 * uncertainty,
            ),
        )

    def _select_halving_survivors(
        self,
        results,
        survivor_count,
        final_round=False,
    ):
        """Apply one bounded capacity-anchor repechage slot.

        Earlier rounds replace the weakest normal survivor, so the scheduled
        candidate count is unchanged. The last round may carry one eligible
        reference into exactly two refinement passes, after which only the
        best two continue.
        """
        if not results:
            return [], None
        keep = min(max(1, int(survivor_count)), len(results))
        selected = list(results[:keep])
        if keep < 2 or len(results) <= keep:
            return selected, None

        capacity_anchors = [
            entry
            for entry in results
            if entry.get("anchor_kind") == "capacity"
        ]
        if not capacity_anchors:
            return selected, None
        reference = max(
            capacity_anchors,
            key=lambda item: (
                float(item.get("utility", -float("inf"))),
                self._candidate_accuracy(item),
                -int(item.get("params", 0)),
            ),
        )
        if any(reference is entry for entry in selected):
            return selected, None
        cutoff = selected[-1]
        deficit = (
            self._candidate_accuracy(cutoff)
            - self._candidate_accuracy(reference)
        )
        if deficit > self._anchor_insurance_allowance(reference):
            return selected, None

        reference["anchor_insurance"] = True
        reference["anchor_insurance_deficit"] = max(0.0, deficit)
        if final_round:
            selected.append(reference)
        else:
            selected[-1] = reference
            selected.sort(
                key=lambda item: float(
                    item.get("utility", -float("inf"))
                ),
                reverse=True,
            )
        return selected, reference

    def _snapshot_refinement_checkpoint(self, candidate):
        """Capture a finalist state and the optimizer that produced it."""
        return {
            "model_state": self._cpu_model_state(candidate["model"]),
            "optimizer_state": copy.deepcopy(
                candidate.get("optimizer_state")
            ),
            "val_acc": float(candidate.get("val_acc", 0.0)),
            "balanced_acc": float(
                candidate.get(
                    "balanced_acc",
                    candidate.get("val_acc", 0.0),
                )
            ),
            "val_loss": float(candidate.get("val_loss", float("inf"))),
            "val_margin": float(candidate.get("val_margin", 0.0)),
            "val_samples": int(candidate.get("val_samples", 0)),
            "trained_steps": int(candidate.get("trained_steps", 0)),
            "data_cursor": int(candidate.get("data_cursor", 0)),
            "train_stats": copy.deepcopy(candidate.get("train_stats", {})),
        }

    @staticmethod
    def _is_better_refinement_checkpoint(candidate, checkpoint):
        accuracy = float(candidate.get("val_acc", 0.0))
        loss = float(candidate.get("val_loss", float("inf")))
        best_accuracy = float(checkpoint.get("val_acc", 0.0))
        best_loss = float(checkpoint.get("val_loss", float("inf")))
        return accuracy > best_accuracy + 1e-12 or (
            abs(accuracy - best_accuracy) <= 1e-12
            and loss < best_loss
        )

    @staticmethod
    def _restore_refinement_checkpoint(candidate, checkpoint):
        candidate["model"].load_state_dict(checkpoint["model_state"])
        candidate["optimizer_state"] = copy.deepcopy(
            checkpoint.get("optimizer_state")
        )
        for name in (
            "val_acc",
            "balanced_acc",
            "val_loss",
            "val_margin",
            "val_samples",
            "trained_steps",
            "data_cursor",
        ):
            candidate[name] = checkpoint[name]
        candidate["train_stats"] = copy.deepcopy(
            checkpoint.get("train_stats", {})
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
        """Score final holdouts while retaining history only as a tie-break."""
        rank_scores = self._history_rank_scores(entries)
        for entry in entries:
            self._architecture_risk(entry)
            entry["history_rank_score"] = rank_scores.get(id(entry), 0.5)
            if "confirmation_val_acc" in entry:
                selection_accuracy = float(
                    entry.get(
                        "selection_val_acc",
                        entry.get("val_acc", 0.0),
                    )
                )
                confirmation_accuracy = float(
                    entry["confirmation_val_acc"]
                )
                entry["selection_score"] = (
                    0.80 * confirmation_accuracy
                    + 0.20 * selection_accuracy
                )
                continue
            full_score = float(
                entry.get("full_val_acc", entry.get("val_acc", 0.0))
            )
            entry["selection_score"] = full_score
        return entries

    @staticmethod
    def _confirmation_tie_threshold(first, second):
        """Return a bounded uncertainty band for two holdout accuracies."""
        samples = max(
            1,
            min(
                int(first.get("confirmation_samples", 1)),
                int(second.get("confirmation_samples", 1)),
            ),
        )
        first_accuracy = float(first["confirmation_val_acc"])
        second_accuracy = float(second["confirmation_val_acc"])
        pooled_se = math.sqrt(
            (
                first_accuracy * (1.0 - first_accuracy)
                + second_accuracy * (1.0 - second_accuracy)
            )
            / samples
        )
        return max(1.0 / samples, min(0.01, 0.75 * pooled_se))

    def _order_final_results(self, entries):
        """Order finalists with confirmation dominance and tie-only history."""
        self._final_selection_scores(entries)
        if not entries:
            return entries, None
        if not all("confirmation_val_acc" in entry for entry in entries):
            entries.sort(
                key=lambda item: (
                    item.get("selection_score", -float("inf")),
                    -float(item.get("selection_val_loss", float("inf"))),
                    item.get("history_rank_score", 0.5),
                ),
                reverse=True,
            )
            return entries, None

        # Confirmation accuracy is the independent decision source. Only
        # candidates inside its bounded uncertainty band may use loss,
        # selection accuracy, or historical rank as tie-breaks.
        entries.sort(
            key=lambda item: (
                float(item["confirmation_val_acc"]),
                float(item.get("selection_val_acc", 0.0)),
            ),
            reverse=True,
        )
        leader = entries[0]
        overlapping = [leader]
        clear_losers = []
        thresholds = []
        for entry in entries[1:]:
            threshold = self._confirmation_tie_threshold(leader, entry)
            if (
                float(leader["confirmation_val_acc"])
                - float(entry["confirmation_val_acc"])
                <= threshold
            ):
                overlapping.append(entry)
                thresholds.append(threshold)
            else:
                clear_losers.append(entry)

        if len(overlapping) >= 2:
            overlapping.sort(
                key=lambda item: (
                    -float(item.get("architecture_risk", 0.0)),
                    -float(
                        item.get(
                            "confirmation_val_loss",
                            float("inf"),
                        )
                    ),
                    item.get("selection_score", -float("inf")),
                    item.get("history_rank_score", 0.5),
                ),
                reverse=True,
            )
        clear_losers.sort(
            key=lambda item: (
                float(item["confirmation_val_acc"]),
                item.get("selection_score", -float("inf")),
            ),
            reverse=True,
        )
        entries[:] = overlapping + clear_losers
        return entries, (max(thresholds) if thresholds else None)

    def _select_retained_architecture_challenger(self, winner, entries):
        """Choose a tied or efficient overfit-insurance checkpoint."""
        alternatives = [entry for entry in entries if entry is not winner]
        tied = [
            entry
            for entry in alternatives
            if entry.get("uncertainty_tie", False)
        ]
        if winner.get("uncertainty_tie", False) and tied:
            return tied[0], "uncertainty_tie"

        self._architecture_risk(winner)
        if not winner.get("architecture_risk_ready", False):
            return None, None
        # Either a large measured fit gap or cross-entropy materially worse
        # than the uniform baseline is enough to justify one recovery option.
        if (
            float(winner.get("train_validation_gap", 0.0)) < 0.12
            and float(winner.get("absolute_loss_excess", 0.0)) < 0.08
        ):
            return None, None

        winner_params = max(1, int(winner.get("params", 0)))
        winner_accuracy = self._candidate_accuracy(winner)
        efficient = []
        for entry in alternatives:
            params = int(entry.get("params", winner_params))
            if params <= 0 or params > 0.75 * winner_params:
                continue
            if self._candidate_accuracy(entry) < winner_accuracy - 0.08:
                continue
            self._architecture_risk(entry)
            efficient.append(entry)
        if not efficient:
            return None, None
        efficient.sort(
            key=lambda item: (
                self._candidate_accuracy(item),
                -float(item.get("architecture_risk", 0.0)),
                -int(item.get("params", 0)),
            ),
            reverse=True,
        )
        return efficient[0], "efficient_overfit_insurance"

    @staticmethod
    def _recipe_trial_score(trial):
        """Favor validation accuracy while using loss slope near a tie."""
        history = trial.get("metrics_history", [])
        if len(history) >= 2:
            first_loss = float(history[0].get("loss", 0.0))
            latest_loss = float(history[-1].get("loss", first_loss))
            scale = max(1e-8, abs(first_loss))
            loss_slope = (
                (first_loss - latest_loss)
                / scale
                / max(1, len(history) - 1)
            )
        else:
            loss_slope = 0.0
        # Loss is deliberately a bounded near-tie signal. Competition score
        # remains accuracy-based, but a still-improving recipe should survive
        # a tiny early accuracy deficit.
        bounded_slope = max(-0.25, min(0.25, loss_slope))
        latest = history[-1] if history else {}
        generalization_gap = max(
            0.0,
            float(latest.get("train_accuracy", 0.0))
            - float(latest.get("accuracy", 0.0)),
        )
        gap_penalty = 0.01 * min(0.50, generalization_gap)
        score = (
            float(trial.get("best_val_acc", 0.0))
            + 0.01 * bounded_slope
            - gap_penalty
        )
        trial["loss_slope"] = loss_slope
        trial["train_validation_gap"] = generalization_gap
        trial["selection_score"] = score
        return score

    def _recipe_acceptance_margin(
        self,
        samples,
        incumbent_accuracy=None,
        challenger_accuracy=None,
    ):
        """Return a bounded uncertainty-aware replacement margin."""
        configured_floor = float(
            getattr(self, "recipe_min_accuracy_gain", 0.001)
        )
        samples = max(1, int(samples))
        uncertainty_margin = 0.0
        if (
            incumbent_accuracy is not None
            and challenger_accuracy is not None
        ):
            incumbent_accuracy = min(
                1.0, max(0.0, float(incumbent_accuracy))
            )
            challenger_accuracy = min(
                1.0, max(0.0, float(challenger_accuracy))
            )
            pooled_se = math.sqrt(
                (
                    incumbent_accuracy * (1.0 - incumbent_accuracy)
                    + challenger_accuracy
                    * (1.0 - challenger_accuracy)
                )
                / samples
            )
            uncertainty_margin = min(0.01, 0.75 * pooled_se)
        return max(
            0.0,
            configured_floor,
            1.0 / samples,
            uncertainty_margin,
        )

    def _recipe_tournament(
        self,
        space,
        architecture_winner,
        refinement_loader,
        validation_batches,
        deadline_remaining,
        confirmation_batches=None,
    ):
        """Run an incumbent-safe, two-stage race from one checkpoint."""
        recipes = self._training_recipes()
        architecture_model = architecture_winner["model"]
        common_state = {
            key: value.detach().cpu().clone()
            for key, value in architecture_model.state_dict().items()
        }
        common_optimizer_state = copy.deepcopy(
            architecture_winner.pop("optimizer_state", None)
        )

        # The incumbent and every recipe endpoint are measured on this exact
        # source. A cached validation source is never compared with a full-
        # validation number left over from architecture selection.
        architecture_model.to(self.device)
        incumbent_metrics = self._evaluate(
            architecture_model, validation_batches
        )
        architecture_model.cpu()
        incumbent_accuracy = float(incumbent_metrics["accuracy"])
        incumbent_loss = float(
            incumbent_metrics.get("loss", 1.0 - incumbent_accuracy)
        )
        incumbent_samples = int(incumbent_metrics.get("samples", 0))
        acceptance_margin = self._recipe_acceptance_margin(
            incumbent_samples,
            incumbent_accuracy,
            incumbent_accuracy,
        )
        default_recipe = self._default_recipe()

        def restore_incumbent(reason):
            architecture_model.load_state_dict(common_state)
            self._attach_recipe(architecture_model, default_recipe)
            architecture_model.independent_retry_state = {
                key: value.detach().cpu().clone()
                for key, value in common_state.items()
            }
            architecture_model.search_optimizer_state = copy.deepcopy(
                common_optimizer_state
            )
            architecture_model.alternative_training_recipes = [
                dict(recipe)
                for recipe in recipes
                if recipe["name"] != default_recipe["name"]
            ]
            architecture_winner["model"] = architecture_model
            architecture_winner["recipe"] = dict(default_recipe)
            architecture_winner["val_acc"] = incumbent_accuracy
            architecture_winner["balanced_acc"] = float(
                incumbent_metrics.get(
                    "balanced_accuracy", incumbent_accuracy
                )
            )
            architecture_winner.pop("optimizer_state", None)
            print(
                "  [NAS] Recipe race preserved incumbent ({:.2f}%): {}".format(
                    incumbent_accuracy * 100, reason
                )
            )
            return architecture_winner

        available = max(
            0.0, self.clock.check() - deadline_remaining - 6.0
        )
        seconds_per_step = max(
            1e-4, architecture_winner.get("seconds_per_step", 0.05)
        )
        if refinement_loader is None:
            stage_one_steps = 0
        else:
            promoted_count = min(2, len(recipes))
            safe_seconds_per_step = 1.80 * seconds_per_step
            total_step_capacity = int(
                0.70 * available / max(1e-4, safe_seconds_per_step)
            )
            stage_one_steps = min(
                len(refinement_loader),
                total_step_capacity
                // max(1, len(recipes) + promoted_count),
            )

        if stage_one_steps < 12 or len(recipes) < 2:
            return restore_incumbent(
                "only {} equal first-stage steps affordable".format(
                    stage_one_steps
                )
            )

        print(
            "  [NAS] Incumbent-safe recipe race on {}: incumbent={:.2f}% "
            "(loss {:.4f}), {} recipes, {} equal stage-1 steps".format(
                self._spec_label(architecture_winner["spec"]),
                incumbent_accuracy * 100,
                incumbent_loss,
                len(recipes),
                stage_one_steps,
            )
        )
        trials = []
        for recipe in recipes:
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
                train_stats,
            ) = self._train_low_fidelity(
                model,
                refinement_loader,
                stage_one_steps,
                max(5.0, 1.80 * seconds_per_step * stage_one_steps),
                deadline_remaining,
                recipe,
                None,
            )
            if failed or steps < stage_one_steps:
                model.cpu()
                del model
                continue
            metrics = self._evaluate(model, validation_batches)
            model.cpu()
            trial_accuracy = float(metrics["accuracy"])
            trial_loss = float(
                metrics.get("loss", 1.0 - trial_accuracy)
            )
            trial = {
                "model": model,
                "recipe": dict(recipe),
                "metrics_history": [
                    {
                        "accuracy": incumbent_accuracy,
                        "loss": incumbent_loss,
                    },
                    {
                        "accuracy": trial_accuracy,
                        "loss": trial_loss,
                        "train_accuracy": float(
                            train_stats.get("train_accuracy", 0.0)
                        ),
                    },
                ],
                "best_val_acc": trial_accuracy,
                "best_val_loss": trial_loss,
                "best_balanced_acc": float(
                    metrics.get("balanced_accuracy", trial_accuracy)
                ),
                "best_steps": steps,
                "best_elapsed": elapsed,
                "best_optimizer_state": copy.deepcopy(optimizer_state),
                "train_stats": dict(train_stats),
            }
            self._recipe_trial_score(trial)
            trials.append(trial)
            print(
                "    stage 1 {} | val={:.2f}% loss={:.4f} "
                "slope={:+.4f} steps={}".format(
                    recipe["name"],
                    trial_accuracy * 100,
                    trial_loss,
                    trial["loss_slope"],
                    steps,
                )
            )

        if len(trials) < len(recipes):
            for trial in trials:
                del trial["model"]
            return restore_incumbent(
                "not every recipe completed the fair first stage"
            )

        trials.sort(
            key=lambda item: (
                item["selection_score"],
                item["best_val_acc"],
                -item["best_val_loss"],
            ),
            reverse=True,
        )
        promoted = trials[:2]
        for trial in trials[2:]:
            del trial["model"]

        stage_two_available = max(
            0.0, self.clock.check() - deadline_remaining - 6.0
        )
        stage_two_steps = min(
            len(refinement_loader),
            int(
                0.70
                * stage_two_available
                / max(
                    1e-4,
                    len(promoted) * 1.80 * seconds_per_step,
                )
            ),
        )
        if stage_two_steps < 12:
            for trial in promoted:
                del trial["model"]
            return restore_incumbent(
                "only {} equal second-stage steps affordable".format(
                    stage_two_steps
                )
            )

        print(
            "  [NAS] Recipe race stage 2: promoting {} for {} equal steps".format(
                ",".join(
                    trial["recipe"]["name"] for trial in promoted
                ),
                stage_two_steps,
            )
        )
        completed_stage_two = 0
        for trial in promoted:
            if self.clock.check() <= deadline_remaining + 4:
                break
            model = trial["model"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            previous_optimizer_state = copy.deepcopy(
                trial["best_optimizer_state"]
            )
            previous_best_steps = int(trial["best_steps"])
            previous_best_elapsed = float(trial["best_elapsed"])
            self._seed_everything(self.seed + 13000)
            (
                steps,
                elapsed,
                failed,
                optimizer_state,
                train_stats,
            ) = self._train_low_fidelity(
                model,
                refinement_loader,
                stage_two_steps,
                max(5.0, 1.80 * seconds_per_step * stage_two_steps),
                deadline_remaining,
                trial["recipe"],
                previous_optimizer_state,
                start_step=previous_best_steps,
            )
            if failed or steps < stage_two_steps:
                model.load_state_dict(best_state)
                model.cpu()
                continue

            metrics = self._evaluate(model, validation_batches)
            current_accuracy = float(metrics["accuracy"])
            current_loss = float(
                metrics.get("loss", 1.0 - current_accuracy)
            )
            trial["metrics_history"].append(
                {
                    "accuracy": current_accuracy,
                    "loss": current_loss,
                    "train_accuracy": float(
                        train_stats.get("train_accuracy", 0.0)
                    ),
                }
            )
            is_better_checkpoint = (
                current_accuracy > trial["best_val_acc"] + 1e-12
                or (
                    abs(current_accuracy - trial["best_val_acc"]) <= 1e-12
                    and current_loss < trial["best_val_loss"]
                )
            )
            if is_better_checkpoint:
                trial["best_val_acc"] = current_accuracy
                trial["best_val_loss"] = current_loss
                trial["best_balanced_acc"] = float(
                    metrics.get("balanced_accuracy", current_accuracy)
                )
                trial["best_steps"] = previous_best_steps + steps
                trial["best_elapsed"] = previous_best_elapsed + elapsed
                trial["best_optimizer_state"] = copy.deepcopy(
                    optimizer_state
                )
                trial["train_stats"] = dict(train_stats)
            else:
                model.load_state_dict(best_state)
                trial["best_steps"] = previous_best_steps
                trial["best_elapsed"] = previous_best_elapsed
                trial["best_optimizer_state"] = previous_optimizer_state
            model.cpu()
            self._recipe_trial_score(trial)
            completed_stage_two += 1
            print(
                "    stage 2 {} | val={:.2f}% loss={:.4f} "
                "best={:.2f}% slope={:+.4f}".format(
                    trial["recipe"]["name"],
                    current_accuracy * 100,
                    current_loss,
                    trial["best_val_acc"] * 100,
                    trial["loss_slope"],
                )
            )

        if completed_stage_two < 2:
            for trial in promoted:
                del trial["model"]
            return restore_incumbent(
                "fewer than two recipes completed the fair second stage"
            )

        promoted.sort(
            key=lambda item: (
                item["selection_score"],
                item["best_val_acc"],
                -item["best_val_loss"],
            ),
            reverse=True,
        )
        eligible = []
        selection_margins = []
        for trial in promoted:
            trial_margin = self._recipe_acceptance_margin(
                incumbent_samples,
                incumbent_accuracy,
                trial["best_val_acc"],
            )
            trial["acceptance_margin"] = trial_margin
            selection_margins.append(trial_margin)
            if (
                trial["best_val_acc"]
                - incumbent_accuracy
                + 1e-12
                >= trial_margin
            ):
                eligible.append(trial)
        if not eligible:
            for trial in promoted:
                del trial["model"]
            return restore_incumbent(
                "no recipe beat it by the {:.2f}pp margin".format(
                    min(selection_margins or [acceptance_margin]) * 100
                )
            )

        if confirmation_batches:
            architecture_model.to(self.device)
            confirmation_incumbent = self._evaluate(
                architecture_model, confirmation_batches
            )
            architecture_model.cpu()
            confirmation_accuracy = float(
                confirmation_incumbent["accuracy"]
            )
            confirmed = []
            for trial in eligible:
                trial["model"].to(self.device)
                metrics = self._evaluate(
                    trial["model"], confirmation_batches
                )
                trial["model"].cpu()
                trial["confirmation_val_acc"] = float(
                    metrics["accuracy"]
                )
                trial["confirmation_val_loss"] = float(metrics["loss"])
                confirmation_margin = self._recipe_acceptance_margin(
                    confirmation_incumbent.get("samples", 0),
                    confirmation_accuracy,
                    trial["confirmation_val_acc"],
                )
                trial["confirmation_acceptance_margin"] = (
                    confirmation_margin
                )
                if (
                    trial["confirmation_val_acc"]
                    - confirmation_accuracy
                    + 1e-12
                    >= confirmation_margin
                ):
                    confirmed.append(trial)
            if not confirmed:
                for trial in promoted:
                    del trial["model"]
                return restore_incumbent(
                    "selection gain did not confirm on the held-out split"
                )
            confirmed.sort(
                key=lambda item: (
                    item["confirmation_val_acc"],
                    -item["confirmation_val_loss"],
                    item["selection_score"],
                ),
                reverse=True,
            )
            eligible = confirmed

        selected = eligible[0]
        for trial in promoted:
            if trial is not selected:
                del trial["model"]
        selected_model = selected["model"]
        self._attach_recipe(selected_model, selected["recipe"])
        selected_model.search_optimizer_state = copy.deepcopy(
            selected["best_optimizer_state"]
        )
        selected_model.independent_retry_state = {
            key: value.detach().cpu().clone()
            for key, value in common_state.items()
        }
        selected_model.alternative_training_recipes = [
            dict(recipe)
            for recipe in recipes
            if recipe["name"] != selected["recipe"]["name"]
        ]
        architecture_winner["model"] = selected_model
        architecture_winner["recipe"] = dict(selected["recipe"])
        architecture_winner["val_acc"] = selected.get(
            "confirmation_val_acc", selected["best_val_acc"]
        )
        architecture_winner["balanced_acc"] = selected[
            "best_balanced_acc"
        ]
        architecture_winner["trained_steps"] = int(
            architecture_winner.get("trained_steps", 0)
        ) + int(selected["best_steps"])
        architecture_winner["train_seconds"] = float(
            architecture_winner.get("train_seconds", 0.0)
        ) + float(selected["best_elapsed"])
        architecture_winner.pop("optimizer_state", None)
        print(
            "  [NAS] Recipe race selected {} at {:.2f}% "
            "(incumbent {:.2f}%, margin {:.2f}pp)".format(
                selected["recipe"]["name"],
                selected["best_val_acc"] * 100,
                incumbent_accuracy * 100,
                selected.get(
                    "confirmation_acceptance_margin",
                    selected.get(
                        "acceptance_margin", acceptance_margin
                    ),
                )
                * 100,
            )
        )
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
        confirmation_batches=None,
    ):
        represented_families = list(
            dict.fromkeys(
                entry["spec"].model_family for entry in ranked
            )
        )
        anchor_specs_by_family = {
            family: self._anchor_specs(family, space)
            for family in represented_families
        }
        finalists = self._select_label_aware_entries(
            ranked,
            n_top,
            anchor_specs_by_family=anchor_specs_by_family,
        )
        anchor_kind_by_spec = {}
        for family, specs in anchor_specs_by_family.items():
            for kind, spec in zip(
                ("compact", "central", "capacity"), specs
            ):
                anchor_kind_by_spec[spec] = kind
        architecture_recipe = self._architecture_search_recipe()
        active = []
        initialization_seed = self._architecture_initialization_seed()
        for entry in finalists:
            candidate = dict(entry)
            self._seed_everything(initialization_seed)
            candidate["recipe"] = dict(architecture_recipe)
            candidate["model"] = self._attach_recipe(
                space.build_model(entry["spec"]), architecture_recipe
            )
            candidate["anchor_kind"] = anchor_kind_by_spec.get(
                entry["spec"]
            )
            candidate["trained_steps"] = 0
            candidate["train_seconds"] = 0.0
            candidate["val_history"] = []
            candidate["val_loss_history"] = []
            candidate["optimizer_state"] = None
            candidate["data_cursor"] = 0
            candidate["seed_schedule"] = []
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
                candidate_seed = self._architecture_round_seed(round_index)
                self._seed_everything(candidate_seed)
                candidate["seed_schedule"].append(candidate_seed)
                (
                    steps,
                    elapsed,
                    failed,
                    optimizer_state,
                    train_stats,
                ) = self._train_low_fidelity(
                    candidate["model"],
                    cached_batches,
                    max_steps,
                    time_quantum,
                    deadline_remaining,
                    candidate["recipe"],
                    candidate.get("optimizer_state"),
                    start_step=candidate.get("data_cursor", 0),
                )
                if failed or (steps == 0 and candidate["trained_steps"] == 0):
                    candidate["model"].cpu()
                    del candidate["model"]
                    continue
                candidate["trained_steps"] += steps
                candidate["data_cursor"] = int(
                    candidate.get("data_cursor", 0)
                ) + steps
                candidate["optimizer_state"] = optimizer_state
                candidate["train_stats"] = dict(train_stats)
                candidate["train_seconds"] += elapsed
                candidate["examples_per_second"] = (
                    float(train_stats.get("examples_seen", 0))
                    / max(1e-4, elapsed)
                )
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
                candidate["val_loss"] = metrics["loss"]
                candidate["val_margin"] = metrics["margin"]
                candidate["val_samples"] = metrics["samples"]
                candidate["examples_seen"] = int(
                    candidate.get("examples_seen", 0)
                ) + int(train_stats.get("examples_seen", 0))
                candidate["peak_memory_mb"] = max(
                    float(candidate.get("peak_memory_mb", 0.0)),
                    float(train_stats.get("peak_memory_mb", 0.0)),
                )
                candidate["val_history"].append(candidate["val_acc"])
                candidate["val_loss_history"].append(
                    candidate["val_loss"]
                )
                self._adapt_search_lr(candidate)
                candidate["utility"] = self._candidate_utility(candidate)
                candidate["model"].cpu()
                results.append(candidate)
                print(
                    "    {}/{} {} + {} | val={:.2f}% bal={:.2f}% "
                    "loss={:.4f} steps={} seed={} lr={:.1e} "
                    "ex/s={:.1f} peak={:.0f}MiB "
                    "projected_epochs={:.1f}".format(
                        candidate_index + 1,
                        len(active),
                        self._spec_label(candidate["spec"]),
                        candidate["recipe"]["name"],
                        candidate["val_acc"] * 100,
                        candidate["balanced_acc"] * 100,
                        candidate["val_loss"],
                        candidate["trained_steps"],
                        candidate_seed,
                        train_stats.get("final_lr", 0.0),
                        candidate["examples_per_second"],
                        candidate["peak_memory_mb"],
                        candidate["projected_epochs"],
                    )
                )
            if not results:
                break
            results.sort(
                key=lambda item: item["utility"], reverse=True
            )
            active, insured = self._select_halving_survivors(
                results,
                survivors,
                final_round=(round_index == len(round_plan) - 1),
            )
            if insured is not None:
                self.metadata["nas_anchor_insurance_uses"] = int(
                    self.metadata.get("nas_anchor_insurance_uses", 0)
                ) + 1
                print(
                    "  [NAS] Capacity-anchor insurance retained {} "
                    "({:.2f}pp behind cutoff)".format(
                        self._spec_label(insured["spec"]),
                        float(
                            insured.get("anchor_insurance_deficit", 0.0)
                        )
                        * 100,
                    )
                )
            survivor_ids = {id(candidate) for candidate in active}
            for candidate in results:
                if id(candidate) in survivor_ids:
                    continue
                del candidate["model"]
                candidate.pop("optimizer_state", None)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if not active:
            return None

        refinement_remaining = max(
            0.0, self.clock.check() - deadline_remaining - 12.0
        )
        refinement_loader = self._make_refinement_loader()
        insurance_candidates = []
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
                candidate["_refinement_best_checkpoint"] = (
                    self._snapshot_refinement_checkpoint(candidate)
                )

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
                for candidate in active:
                    if self.clock.check() <= refinement_floor + 4:
                        break
                    seconds_per_step = max(
                        1e-4, candidate.get("seconds_per_step", 0.05)
                    )
                    time_quantum = max(
                        5.0, 1.80 * seconds_per_step * common_steps
                    )
                    candidate_seed = self._architecture_refinement_seed(
                        pass_index
                    )
                    self._seed_everything(candidate_seed)
                    candidate.setdefault("seed_schedule", []).append(
                        candidate_seed
                    )
                    (
                        steps,
                        elapsed,
                        failed,
                        optimizer_state,
                        train_stats,
                    ) = self._train_low_fidelity(
                        candidate["model"],
                        refinement_loader,
                        common_steps,
                        time_quantum,
                        refinement_floor,
                        candidate["recipe"],
                        candidate.get("optimizer_state"),
                        start_step=candidate.get("data_cursor", 0),
                    )
                    if failed or steps == 0:
                        candidate["model"].cpu()
                        del candidate["model"]
                        candidate.pop("optimizer_state", None)
                        continue
                    candidate["optimizer_state"] = optimizer_state
                    candidate["train_stats"] = dict(train_stats)
                    candidate["trained_steps"] += steps
                    candidate["data_cursor"] = int(
                        candidate.get("data_cursor", 0)
                    ) + steps
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
                    candidate["val_loss"] = metrics["loss"]
                    candidate["val_margin"] = metrics["margin"]
                    candidate["val_samples"] = metrics["samples"]
                    candidate["examples_seen"] = int(
                        candidate.get("examples_seen", 0)
                    ) + int(train_stats.get("examples_seen", 0))
                    candidate["peak_memory_mb"] = max(
                        float(candidate.get("peak_memory_mb", 0.0)),
                        float(train_stats.get("peak_memory_mb", 0.0)),
                    )
                    candidate["val_history"].append(candidate["val_acc"])
                    candidate["val_loss_history"].append(
                        candidate["val_loss"]
                    )
                    candidate["refinement_history"].append(
                        candidate["val_acc"]
                    )
                    self._adapt_search_lr(candidate)
                    best_checkpoint = candidate[
                        "_refinement_best_checkpoint"
                    ]
                    if self._is_better_refinement_checkpoint(
                        candidate, best_checkpoint
                    ):
                        candidate["_refinement_best_checkpoint"] = (
                            self._snapshot_refinement_checkpoint(candidate)
                        )
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
                # The final-round repechage buys exactly two equal full-data
                # passes. Afterwards, the best two continue normally and the
                # parked checkpoint remains available only for the final
                # holdout comparison / dormant recovery path.
                if pass_index >= 2 and len(active) > 2:
                    active.sort(
                        key=lambda item: float(
                            item.get("utility", -float("inf"))
                        ),
                        reverse=True,
                    )
                    parked = active[2:]
                    active = active[:2]
                    for candidate in parked:
                        best_checkpoint = candidate.pop(
                            "_refinement_best_checkpoint", None
                        )
                        if best_checkpoint is not None:
                            self._restore_refinement_checkpoint(
                                candidate, best_checkpoint
                            )
                        candidate["anchor_insurance_parked"] = True
                        insurance_candidates.append(candidate)
                    self.metadata["nas_anchor_insurance_parked"] = int(
                        len(insurance_candidates)
                    )
                    print(
                        "    anchor insurance completed two passes; "
                        "continuing the best two finalists"
                    )
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
            for candidate in active:
                best_checkpoint = candidate.pop(
                    "_refinement_best_checkpoint", None
                )
                if best_checkpoint is None:
                    continue
                endpoint_accuracy = float(candidate.get("val_acc", 0.0))
                self._restore_refinement_checkpoint(
                    candidate, best_checkpoint
                )
                if (
                    candidate["val_acc"]
                    > endpoint_accuracy + 1e-12
                ):
                    print(
                        "    restored {} best refinement checkpoint: "
                        "{:.2f}% (endpoint {:.2f}%)".format(
                            self._spec_label(candidate["spec"]),
                            candidate["val_acc"] * 100,
                            endpoint_accuracy * 100,
                        )
                    )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # The last challengers are measured on the same prior-preserving
        # selection source and, when available, an untouched confirmation
        # source. Earlier adaptive scores never masquerade as confirmation.
        comparison_candidates = list(active)
        comparison_ids = {id(candidate) for candidate in comparison_candidates}
        for candidate in insurance_candidates:
            if id(candidate) not in comparison_ids:
                comparison_candidates.append(candidate)
                comparison_ids.add(id(candidate))
        print(
            "  [NAS] Selection/confirmation champion comparison ({})".format(
                len(comparison_candidates)
            )
        )
        final_results = []
        for candidate in comparison_candidates:
            if self.clock.check() <= deadline_remaining + 4 and final_results:
                break
            candidate["model"].to(self.device)
            selection_metrics = self._evaluate(
                candidate["model"], validation_batches
            )
            candidate["selection_val_acc"] = selection_metrics["accuracy"]
            candidate["selection_val_loss"] = selection_metrics["loss"]
            candidate["val_acc"] = candidate["selection_val_acc"]
            candidate["balanced_acc"] = selection_metrics[
                "balanced_accuracy"
            ]
            if confirmation_batches:
                confirmation_metrics = self._evaluate(
                    candidate["model"], confirmation_batches
                )
                candidate["confirmation_val_acc"] = (
                    confirmation_metrics["accuracy"]
                )
                candidate["confirmation_val_loss"] = (
                    confirmation_metrics["loss"]
                )
                candidate["confirmation_samples"] = (
                    confirmation_metrics["samples"]
                )
                total_samples = (
                    selection_metrics["samples"]
                    + confirmation_metrics["samples"]
                )
                candidate["full_val_acc"] = (
                    selection_metrics["accuracy"]
                    * selection_metrics["samples"]
                    + confirmation_metrics["accuracy"]
                    * confirmation_metrics["samples"]
                ) / max(1, total_samples)
            else:
                candidate["full_val_acc"] = candidate[
                    "selection_val_acc"
                ]
            candidate["model"].cpu()
            final_results.append(candidate)
            print(
                "    {} + {} | select={:.2f}% confirm={} bal={:.2f}%".format(
                    self._spec_label(candidate["spec"]),
                    candidate["recipe"]["name"],
                    candidate["selection_val_acc"] * 100,
                    (
                        "{:.2f}%".format(
                            candidate["confirmation_val_acc"] * 100
                        )
                        if "confirmation_val_acc" in candidate
                        else "n/a"
                    ),
                    candidate["balanced_acc"] * 100,
                )
            )
        if not final_results:
            final_results = comparison_candidates
        final_results, tie_threshold = self._order_final_results(
            final_results
        )
        if tie_threshold is not None and len(final_results) >= 2:
            final_results[0]["uncertainty_tie"] = True
            final_results[1]["uncertainty_tie"] = True
            print(
                "  [NAS] Finalists overlap within {:.2f}pp; "
                "confirmation loss resolves the tie".format(
                    tie_threshold * 100
                )
            )
        winner = final_results[0]
        runner_up = final_results[1] if len(final_results) >= 2 else None
        if runner_up is not None:
            self.metadata["nas_runner_up_spec"] = runner_up["spec"]
            self.metadata["nas_runner_up_score"] = float(
                runner_up.get("selection_score", 0.0)
            )
        (
            retained_challenger,
            challenger_reason,
        ) = self._select_retained_architecture_challenger(
            winner, final_results
        )
        if retained_challenger is not None:
            self._attach_recipe(
                retained_challenger["model"], self._default_recipe()
            )
            retained_challenger["model"].search_optimizer_state = (
                copy.deepcopy(retained_challenger.get("optimizer_state"))
            )
            self.metadata["nas_challenger_spec"] = retained_challenger[
                "spec"
            ]
            self.metadata["nas_challenger_score"] = float(
                retained_challenger.get("selection_score", 0.0)
            )
            self.metadata["nas_challenger_retained"] = True
            self.metadata["nas_challenger_reason"] = challenger_reason
            self.metadata["nas_challenger_params"] = int(
                retained_challenger.get("params", 0)
            )
            print(
                "  [NAS] Retained dormant challenger {} ({})".format(
                    self._spec_label(retained_challenger["spec"]),
                    challenger_reason,
                )
            )
        else:
            self.metadata["nas_challenger_retained"] = False
        print(
            "  [NAS] Robust architecture score selected {} ({:.4f})".format(
                self._spec_label(winner["spec"]),
                winner.get("selection_score", 0.0),
            )
        )
        for candidate in comparison_candidates:
            if (
                candidate is not winner
                and candidate is not retained_challenger
                and "model" in candidate
            ):
                del candidate["model"]
                candidate.pop("optimizer_state", None)
        winner = self._recipe_tournament(
            space,
            winner,
            refinement_loader,
            validation_batches,
            deadline_remaining,
            confirmation_batches=confirmation_batches,
        )
        if retained_challenger is not None:
            # Keep the dormant module inside a plain bundle so nn.Module does
            # not register/count/move it as a branch of the returned model.
            # Trainer removes the bundle before training the primary model.
            winner["model"].architecture_challenger_bundle = {
                "model": retained_challenger["model"],
                "spec": retained_challenger["spec"],
                "val_acc": float(
                    retained_challenger.get(
                        "confirmation_val_acc",
                        retained_challenger.get("val_acc", 0.0),
                    )
                ),
                "params": int(retained_challenger.get("params", 0)),
                "reason": challenger_reason,
                "risk": float(
                    retained_challenger.get("architecture_risk", 0.0)
                ),
            }
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

        (
            validation_batches,
            confirmation_batches,
        ) = self._cache_validation_splits()
        validation_batches = validation_batches or None
        confirmation_batches = confirmation_batches or None
        self._select_augmentation_policy(
            space, validation_batches, deadline_remaining
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
        self._calibrate_search_microbatch(
            space, cached_batches, deadline_remaining
        )
        promoted_families = self._representation_probe_race(
            space,
            ranked,
            cached_batches,
            validation_batches,
            deadline_remaining,
        )
        expanded_ranked = self._expand_promoted_macro_pool(
            ranked, pool, promoted_families
        )
        if expanded_ranked:
            ranked = expanded_ranked
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
                confirmation_batches=confirmation_batches,
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
        for metric in (
            "val_loss",
            "val_margin",
            "examples_per_second",
            "peak_memory_mb",
            "train_validation_gap",
        ):
            if metric in winner:
                self.metadata["nas_winner_{}".format(metric)] = winner[
                    metric
                ]

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
