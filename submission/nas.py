"""
nas.py — Tiered NAS pipeline for the NAS Unseen-Data Challenge 2026.

Implements a 3-tier search strategy that adapts to the available time budget:
  TIER 1 (>15 min): Full pipeline — sample 500 archs, ZCP ranking, learning curve
                     validation on top-20, return best (or ensemble of top-3)
  TIER 2 (5-15 min): ZCP-only — sample 100 archs, rank with proxies, return top-1
  TIER 3 (<5 min):   Immediate fallback — return a pre-configured ResNet-18
"""

import time
import copy
import torch
import torch.nn as nn
import torchvision

from helpers import show_time, get_tier, get_device, TIER2_THRESHOLD, TIER3_THRESHOLD
from search_space import SearchSpace
from zero_cost_proxies import compute_all_proxies, rank_aggregate
from ensemble import EnsembleModule


class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """

    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

        # Extract dataset properties
        input_shape = metadata['input_shape']
        if len(input_shape) == 4:
            self.in_channels = input_shape[1]
            self.input_h = input_shape[2]
            self.input_w = input_shape[3]
        else:
            self.in_channels = 1
            self.input_h = input_shape[1] if len(input_shape) == 3 else 32
            self.input_w = input_shape[2] if len(input_shape) == 3 else 32

        self.num_classes = metadata['num_classes']
        self.device = get_device()

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output.
    """

    def search(self):
        tier = get_tier(self.clock)
        remaining = self.clock.check()

        print("  [NAS] Time remaining: ~{}".format(show_time(remaining)))
        print("  [NAS] Operating in TIER {} mode".format(tier))

        if tier == 3:
            return self._tier3_fallback()
        elif tier == 2:
            return self._tier2_zcp_only()
        else:
            return self._tier1_full_search()

    # =========================================================================
    # TIER 3: Immediate fallback — no search
    # =========================================================================

    def _tier3_fallback(self):
        """Return a pre-configured ResNet-18 immediately."""
        print("  [NAS] TIER 3: Returning ResNet-18 fallback (no search)")

        model = torchvision.models.resnet18(weights=None)

        # Adjust stem for actual input channels — use smaller kernel for small inputs
        stem_kernel = 3 if min(self.input_h, self.input_w) <= 32 else 7
        stem_stride = 1 if min(self.input_h, self.input_w) <= 32 else 2
        model.conv1 = nn.Conv2d(
            self.in_channels, 64,
            kernel_size=stem_kernel, stride=stem_stride,
            padding=stem_kernel // 2, bias=False
        )

        # Remove maxpool for small inputs
        if min(self.input_h, self.input_w) <= 32:
            model.maxpool = nn.Identity()

        # Adjust classifier
        model.fc = nn.Linear(model.fc.in_features, self.num_classes, bias=True)

        return model

    # =========================================================================
    # TIER 2: ZCP-only search — fast, no training
    # =========================================================================

    def _tier2_zcp_only(self):
        """Sample architectures, rank with ZCPs, return the top-1."""
        print("  [NAS] TIER 2: ZCP-only search (no learning curve)")

        n_candidates = 100
        space = SearchSpace(
            self.in_channels, self.num_classes,
            self.input_h, self.input_w
        )
        print("  [NAS] Search space size: {} architectures".format(space.size))

        # Sample candidates
        specs = space.sample(n_candidates)
        print("  [NAS] Sampled {} candidates".format(len(specs)))

        # Evaluate with ZCPs
        input_shape = (self.in_channels, self.input_h, self.input_w)
        proxy_scores = []
        t_start = time.time()

        for i, spec in enumerate(specs):
            # Time check — stop early if running low
            if self.clock.check() < TIER3_THRESHOLD:
                print("  [NAS] Time running low at candidate {}/{}, stopping ZCP eval".format(
                    i, len(specs)))
                break

            model = space.build_model(spec)
            scores = compute_all_proxies(
                model, self.train_loader, input_shape,
                self.num_classes, self.device
            )
            proxy_scores.append(scores)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t_start
                print("  [NAS] Evaluated {}/{} candidates ({:.1f}s, ~{:.2f}s/arch)".format(
                    i + 1, len(specs), elapsed, elapsed / (i + 1)))

        if not proxy_scores:
            print("  [NAS] No candidates evaluated — falling back to TIER 3")
            return self._tier3_fallback()

        # Rank-aggregate and pick top-1
        ranked = rank_aggregate(proxy_scores)
        best_idx = ranked[0]
        best_spec = specs[best_idx]
        best_model = space.build_model(best_spec)

        print("  [NAS] Best architecture: {}".format(best_spec))
        print("  [NAS] Proxy scores: {}".format(
            {k: round(v, 3) for k, v in proxy_scores[best_idx].items()}))

        # Store search results in metadata for trainer
        self.metadata['nas_tier'] = 2
        self.metadata['nas_spec'] = best_spec
        self.metadata['nas_ensemble'] = False

        return best_model

    # =========================================================================
    # TIER 1: Full search — ZCPs + learning curve + optional ensemble
    # =========================================================================

    def _tier1_full_search(self):
        """Full pipeline: sample, ZCP rank, learning curve eval, return best."""
        print("  [NAS] TIER 1: Full search pipeline")

        # --- Phase 1: Sample and ZCP screening ---
        n_candidates = 200
        n_top_zcp = 10
        n_top_lc = 3

        space = SearchSpace(
            self.in_channels, self.num_classes,
            self.input_h, self.input_w
        )
        print("  [NAS] Search space size: {} architectures".format(space.size))

        specs = space.sample(n_candidates)
        print("  [NAS] Sampled {} candidates for ZCP screening".format(len(specs)))

        input_shape = (self.in_channels, self.input_h, self.input_w)
        proxy_scores = []
        t_start = time.time()

        for i, spec in enumerate(specs):
            if self.clock.check() < TIER2_THRESHOLD:
                print("  [NAS] Time running low, stopping ZCP eval at {}/{}".format(
                    i, len(specs)))
                break

            model = space.build_model(spec)
            scores = compute_all_proxies(
                model, self.train_loader, input_shape,
                self.num_classes, self.device
            )
            proxy_scores.append(scores)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                print("  [NAS] ZCP: {}/{} candidates ({:.1f}s, ~{:.2f}s/arch, ~{} remaining)".format(
                    i + 1, len(specs), elapsed, elapsed / (i + 1),
                    show_time(self.clock.check())))

        if len(proxy_scores) < 3:
            print("  [NAS] Too few candidates — falling back to TIER 2")
            return self._tier2_zcp_only()

        # Rank and select top candidates
        ranked = rank_aggregate(proxy_scores)
        top_indices = ranked[:n_top_zcp]

        print("  [NAS] Top {} architectures selected by ZCP ranking".format(len(top_indices)))

        # --- Phase 2: Learning curve evaluation (1-2 epochs) ---
        if self.clock.check() < TIER2_THRESHOLD:
            # Not enough time for learning curve — just return ZCP top-1
            best_spec = specs[top_indices[0]]
            print("  [NAS] Skipping learning curve (time constraint)")
            print("  [NAS] Best by ZCP: {}".format(best_spec))
            self.metadata['nas_tier'] = 1
            self.metadata['nas_spec'] = best_spec
            return space.build_model(best_spec)

        print("\n  [NAS] Phase 2: Learning curve evaluation on top-{} architectures".format(
            len(top_indices)))

        lc_results = []
        lc_epochs = 2

        for rank, idx in enumerate(top_indices):
            if self.clock.check() < TIER2_THRESHOLD:
                print("  [NAS] Time limit — stopping learning curve at {}/{}".format(
                    rank, len(top_indices)))
                break

            spec = specs[idx]
            model = space.build_model(spec)
            val_acc = self._quick_train_eval(model, lc_epochs)
            lc_results.append((idx, spec, val_acc))

            print("  [NAS] LC {}/{}: val_acc={:.2f}% | {} | ~{} remaining".format(
                rank + 1, len(top_indices),
                val_acc * 100, spec, show_time(self.clock.check())))

        if not lc_results:
            best_spec = specs[top_indices[0]]
            self.metadata['nas_tier'] = 1
            self.metadata['nas_spec'] = best_spec
            return space.build_model(best_spec)

        # Sort by validation accuracy
        lc_results.sort(key=lambda x: x[2], reverse=True)
        best_idx, best_spec, best_acc = lc_results[0]

        print("\n  [NAS] Best architecture: val_acc={:.2f}%, spec={}".format(
            best_acc * 100, best_spec))

        self.metadata['nas_tier'] = 1
        self.metadata['nas_spec'] = best_spec

        # --- Phase 3: Optional ensemble ---
        # Only if we have multiple good candidates and enough time
        if len(lc_results) >= 2 and self.clock.check() > TIER2_THRESHOLD * 2:
            top_n = min(n_top_lc, len(lc_results))
            top_specs = [lc_results[i][1] for i in range(top_n)]

            print("  [NAS] Preparing ensemble of top-{} architectures".format(top_n))
            models = [space.build_model(s) for s in top_specs]
            ensemble = EnsembleModule(models)

            self.metadata['nas_ensemble'] = True
            self.metadata['nas_ensemble_specs'] = top_specs
            return ensemble

        # Single model
        self.metadata['nas_ensemble'] = False
        return space.build_model(best_spec)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _quick_train_eval(self, model, epochs=2):
        """
        Train a model for a few epochs and return validation accuracy.

        Used for learning curve evaluation during the NAS search.
        Uses a subset of training data for speed.
        """
        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        # Train for a few epochs
        for epoch in range(epochs):
            batch_count = 0
            for data, target in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_count += 1
                # Limit to ~50 batches per epoch for speed
                if batch_count >= 50:
                    break

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data = data.to(self.device)
                output = model(data)
                preds = torch.argmax(output, dim=1)
                correct += (preds == target.to(self.device)).sum().item()
                total += target.size(0)

        return correct / max(total, 1)
