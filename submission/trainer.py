"""
trainer.py — Time-aware Trainer for the NAS Unseen-Data Challenge 2026.

Features:
  - Time-aware training loop with per-epoch clock checks
  - Early stopping with patience
  - Learning rate warmup (linear, first 5% of epochs)
  - Gradient clipping to prevent NaN on exotic datasets
  - Ensemble-aware: trains each sub-model sequentially if EnsembleModule is passed
  - Best-checkpoint restoration
"""

import time
import math
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score

from helpers import show_time


class Trainer:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * device: The torch device to use
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor or NAS classes
    """

    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        # Check if this is an ensemble model
        self.is_ensemble = hasattr(model, 'models') and hasattr(model, 'ensemble_weights')

        # Training config — sensible defaults for unknown domains
        self.base_lr = 1e-3
        self.max_epochs = 200  # upper bound; time-aware loop will stop earlier
        self.grad_clip_norm = 5.0  # max gradient norm for clipping

        # Safety margin: stop training this many seconds before deadline
        self.safety_margin_seconds = 30

        # Early stopping
        self.patience = 15  # epochs without improvement before stopping

        # Estimate actual training epochs from time budget
        remaining = max(0, clock.check() - self.safety_margin_seconds)
        n_batches = len(train_dataloader)
        estimated_epoch_time = max(5, n_batches * 0.08)
        self.estimated_epochs = max(10, min(self.max_epochs, int(remaining / estimated_epoch_time)))

        # Warmup config: 5% of estimated actual epochs (min 1, max 5)
        self.warmup_epochs = max(1, min(5, int(0.05 * self.estimated_epochs)))

    def _make_optimizer_and_scheduler(self, model, max_epochs):
        """Create fresh optimizer and scheduler for a model."""
        optimizer = optim.AdamW(
            model.parameters(), lr=self.base_lr, weight_decay=1e-2
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - self.warmup_epochs)
        )
        return optimizer, scheduler

    def _get_warmup_lr(self, epoch):
        """Linear warmup: scale LR from 0.1x to 1.0x over warmup_epochs."""
        if epoch >= self.warmup_epochs:
            return 1.0
        return 0.1 + 0.9 * (epoch / self.warmup_epochs)

    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    """

    def train(self):
        if self.is_ensemble:
            return self._train_ensemble()
        else:
            return self._train_single(self.model)

    def _train_ensemble(self):
        """Train each sub-model in the ensemble sequentially."""
        n_models = len(self.model.models)
        remaining = self.clock.check()

        # Split remaining time roughly equally among sub-models, with safety buffer
        usable_time = remaining - self.safety_margin_seconds * 2
        per_model_time = max(30, usable_time / n_models)

        print("  [Trainer] Ensemble mode: training {} sub-models (~{} each)".format(
            n_models, show_time(per_model_time)))

        for i, sub_model in enumerate(self.model.models):
            model_remaining = self.clock.check()
            models_left = n_models - i
            time_for_this = max(30, (model_remaining - self.safety_margin_seconds * 2) / models_left)

            print("\n  [Trainer] Training sub-model {}/{} (~{} budget)".format(
                i + 1, n_models, show_time(time_for_this)))

            self._train_single(sub_model, max_time=time_for_this)

            if self.clock.check() < self.safety_margin_seconds * 2:
                print("  [Trainer] Time limit — skipping remaining sub-models")
                break

        return self.model

    def _train_single(self, model, max_time=None):
        """Train a single model with time-awareness, warmup, and gradient clipping."""
        model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer, scheduler = self._make_optimizer_and_scheduler(model, self.estimated_epochs)

        best_val_acc = 0.0
        best_model_state = None
        plateau_count = 0

        print("  [Trainer] Starting time-aware training loop")
        print("  [Trainer] Estimated ~{} epochs, T_max={}, warmup: {} epochs".format(
            self.estimated_epochs, max(1, self.estimated_epochs - self.warmup_epochs), self.warmup_epochs))
        print("  [Trainer] Safety margin: {}s, patience: {}, grad_clip: {}, label_smoothing: 0.1".format(
            self.safety_margin_seconds, self.patience, self.grad_clip_norm))

        t_start = time.time()

        for epoch in range(self.max_epochs):
            # ---- Check time before each epoch ----
            remaining = self.clock.check()
            if remaining < self.safety_margin_seconds:
                print("  [Trainer] Stopping — only {:.1f}s remaining (safety margin: {}s)".format(
                    remaining, self.safety_margin_seconds))
                break

            # Check per-model time budget if set
            elapsed = time.time() - t_start
            if max_time is not None and elapsed > max_time:
                print("  [Trainer] Sub-model time budget ({}) exhausted".format(show_time(max_time)))
                break

            # Estimate if we have time for another full epoch
            if epoch > 0:
                time_per_epoch = elapsed / epoch
                if remaining < (time_per_epoch * 2 + self.safety_margin_seconds):
                    print("  [Trainer] Stopping — not enough time for another epoch "
                          "(need ~{}, have {})".format(
                              show_time(time_per_epoch + self.safety_margin_seconds),
                              show_time(remaining)))
                    break

            # ---- Apply learning rate warmup ----
            if epoch < self.warmup_epochs:
                warmup_scale = self._get_warmup_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.base_lr * warmup_scale

            # ---- Train one epoch ----
            model.train()
            labels, predictions = [], []
            epoch_loss = 0.0
            batch_count = 0

            for data, target in self.train_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping to prevent NaN/explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

            # Step the cosine scheduler (only after warmup completes)
            if epoch >= self.warmup_epochs:
                scheduler.step()

            # ---- Evaluate ----
            train_acc = accuracy_score(labels, predictions)
            val_acc = self._evaluate(model)

            current_lr = optimizer.param_groups[0]['lr']
            print("  Epoch {:>3}/{:<3} | Train: {:>6.2f}% | Valid: {:>6.2f}% | "
                  "LR: {:.1e} | T/Epoch: {:<7} | Remaining: ~{}".format(
                      epoch + 1, self.max_epochs,
                      train_acc * 100, val_acc * 100,
                      current_lr,
                      show_time((time.time() - t_start) / (epoch + 1)),
                      show_time(self.clock.check())))

            # ---- Track best model ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= self.patience:
                print("  [Trainer] Early stopping — validation accuracy plateaued for {} epochs".format(
                    self.patience))
                break

        # Restore best model
        total_time = time.time() - t_start
        print("  [Trainer] Training complete in {} ({} epochs). Best valid acc: {:.2f}%".format(
            show_time(total_time), epoch + 1, best_val_acc * 100))

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)
            print("  [Trainer] Restored best checkpoint")

        return model

    def _evaluate(self, model=None):
        """Compute accuracy over the validation set."""
        if model is None:
            model = self.model
        model.eval()
        labels, predictions = [], []
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                data = data.to(self.device)
                output = model(data)
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)

    """
    ====================================================================================================================
    PREDICT ============================================================================================================
    ====================================================================================================================
    """

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions
