"""
ensemble.py — API-compatible ensemble wrapper for the NAS Unseen-Data Challenge 2026.

Provides:
  - EnsembleModule: nn.Module that wraps multiple models with weighted averaging
  - greedy_ensemble_selection: picks the best subset of models for the ensemble
  - optimize_ensemble_weights: grid-search over weight combinations
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score


class EnsembleModule(nn.Module):
    """
    Wraps multiple sub-models into a single nn.Module.

    Compatible with the competition evaluation pipeline:
      - general_num_params() works (nn.ModuleList registers all parameters)
      - forward() produces a single output tensor
      - .to(device) moves all sub-models
    """

    def __init__(self, models, weights=None):
        """
        Args:
            models: list of nn.Module — the sub-models to ensemble
            weights: optional list of float — per-model weights (default: equal)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer(
            'ensemble_weights',
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, x):
        """Weighted average of all sub-model logits."""
        outputs = [model(x) for model in self.models]
        weighted = sum(
            w.item() * o
            for w, o in zip(self.ensemble_weights, outputs)
        )
        return weighted

    def __repr__(self):
        return "EnsembleModule({} models, weights={})".format(
            len(self.models),
            [round(w.item(), 3) for w in self.ensemble_weights]
        )


# =============================================================================
# Greedy Ensemble Selection
# =============================================================================

def _collect_logits_and_labels(models, valid_loader, device):
    """
    Run all models on the validation set, collecting logits and true labels.

    Returns:
        all_logits: list of tensors, each [N, num_classes]
        labels: tensor [N]
    """
    all_logits = [[] for _ in models]
    labels_list = []

    for model in models:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch in valid_loader:
            data, target = batch
            data = data.to(device)
            labels_list.append(target)

            for i, model in enumerate(models):
                logits = model(data)
                all_logits[i].append(logits.cpu())

    # Concatenate
    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    labels = torch.cat(labels_list, dim=0)

    return all_logits, labels


def greedy_ensemble_selection(models, valid_loader, device, max_ensemble_size=3):
    """
    Greedily select the best subset of models for ensembling.

    At each step, add the model that most improves validation accuracy
    when its logits are averaged with the current ensemble.

    Args:
        models: list of nn.Module candidates
        valid_loader: DataLoader with (data, label) tuples
        device: torch device
        max_ensemble_size: maximum number of models in the ensemble

    Returns:
        (selected_models, weights): tuple of (list[nn.Module], list[float])
    """
    try:
        all_logits, labels = _collect_logits_and_labels(models, valid_loader, device)
        labels_np = labels.numpy()

        selected_indices = []
        best_acc = 0.0
        remaining = list(range(len(models)))

        for round_num in range(min(max_ensemble_size, len(models))):
            best_candidate = None
            best_candidate_acc = best_acc

            for idx in remaining:
                # Try adding this model to the ensemble
                trial = selected_indices + [idx]

                # Average logits of the trial ensemble
                avg_logits = torch.stack([all_logits[i] for i in trial]).mean(dim=0)
                preds = torch.argmax(avg_logits, dim=1).numpy()
                acc = accuracy_score(labels_np, preds)

                if acc > best_candidate_acc:
                    best_candidate = idx
                    best_candidate_acc = acc

            if best_candidate is None:
                # No improvement — stop
                break

            selected_indices.append(best_candidate)
            remaining.remove(best_candidate)
            best_acc = best_candidate_acc

            print("  [Ensemble] Round {}: added model {}, acc={:.2f}%".format(
                round_num + 1, best_candidate, best_acc * 100))

        if not selected_indices:
            # Fallback: just use the first model
            selected_indices = [0]

        selected_models = [models[i] for i in selected_indices]
        weights = [1.0 / len(selected_models)] * len(selected_models)

        return selected_models, weights

    except Exception as e:
        print("  [Ensemble] Selection failed: {}. Using first model.".format(e))
        return [models[0]], [1.0]


# =============================================================================
# Weight Optimization
# =============================================================================

def optimize_ensemble_weights(models, valid_loader, device, grid_resolution=10):
    """
    Grid-search for optimal ensemble weights (2-3 models only).

    Args:
        models: list of nn.Module (2 or 3 models)
        valid_loader: DataLoader with (data, label) tuples
        device: torch device
        grid_resolution: number of grid steps per weight dimension

    Returns:
        list[float]: optimized weights summing to 1.0
    """
    n = len(models)

    if n == 1:
        return [1.0]
    if n > 3:
        # Grid search too expensive for >3 models
        return [1.0 / n] * n

    try:
        all_logits, labels = _collect_logits_and_labels(models, valid_loader, device)
        labels_np = labels.numpy()

        best_weights = [1.0 / n] * n
        best_acc = 0.0

        steps = np.linspace(0, 1, grid_resolution + 1)

        if n == 2:
            for w1 in steps:
                w2 = 1.0 - w1
                avg = w1 * all_logits[0] + w2 * all_logits[1]
                preds = torch.argmax(avg, dim=1).numpy()
                acc = accuracy_score(labels_np, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = [w1, w2]

        elif n == 3:
            for w1 in steps:
                for w2 in steps:
                    w3 = 1.0 - w1 - w2
                    if w3 < -1e-6:
                        continue
                    w3 = max(0.0, w3)
                    avg = w1 * all_logits[0] + w2 * all_logits[1] + w3 * all_logits[2]
                    preds = torch.argmax(avg, dim=1).numpy()
                    acc = accuracy_score(labels_np, preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = [w1, w2, w3]

        print("  [Ensemble] Optimized weights: {} (acc={:.2f}%)".format(
            [round(w, 3) for w in best_weights], best_acc * 100))
        return best_weights

    except Exception as e:
        print("  [Ensemble] Weight optimization failed: {}".format(e))
        return [1.0 / n] * n
