"""
data_processor.py — Adaptive DataProcessor for the NAS Unseen-Data Challenge 2026.

Handles arbitrary input domains by:
  - Auto-detecting and fixing missing channel dimensions (3D → 4D)
  - Computing per-channel normalization statistics from training data
  - Applying data-adaptive augmentation (skip spatial transforms for tiny inputs,
    skip color transforms for grayscale, etc.)
  - Dynamically selecting batch size based on input tensor size
  - Enriching metadata with data properties for downstream NAS and Trainer use
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from helpers import estimate_batch_size, inspect_data_properties, show_time


# =============================================================================
# Dataset wrapper
# =============================================================================

class NASDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that wraps numpy arrays with optional per-sample transforms.

    Handles:
      - 3D arrays (missing channel dim) → reshaped to 4D
      - Float32 conversion for stable training
      - Optional label tensor (None for test set)
      - Lazy transform application (per-sample, not precomputed)
    """

    def __init__(self, x, y=None, transform=None):
        # Convert to float32 tensor
        self.x = torch.tensor(x, dtype=torch.float32)

        # Fix missing channel dimension: [N, H, W] → [N, 1, H, W]
        if self.x.ndim == 3:
            self.x = self.x.unsqueeze(1)

        # Labels (None for test split)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.y is None:
            return img
        return img, self.y[idx]


# =============================================================================
# Augmentation builder
# =============================================================================

class Cutout:
    """
    Randomly masks out a square patch of the image.

    Simple but effective regularization for image classification.
    Works on tensor inputs [C, H, W].
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        c, h, w = img.shape
        mask = torch.ones(c, h, w, dtype=img.dtype)

        # Random center for the cutout patch
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()

        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)

        mask[:, y1:y2, x1:x2] = 0.0
        return img * mask


def build_augmentation_pipeline(data_props, mean, std):
    """
    Build a data-adaptive augmentation pipeline based on inspected data properties.

    Strategy:
      - ALWAYS: normalize with training mean/std
      - Small inputs (<=8×8): normalize only — spatial transforms destroy structure
      - Medium inputs (9-32): mild augmentation (flip, small pad+crop)
      - Large inputs (>32): stronger augmentation (flip, crop, cutout)
      - Grayscale: skip color-based transforms
    """
    h = data_props['height']
    w = data_props['width']
    is_small = data_props['is_small']
    is_grayscale = data_props['is_grayscale']

    normalize = transforms.Normalize(mean, std)

    # ----- Small spatial resolution (<=8): normalize only -----
    if is_small:
        train_transform = transforms.Compose([normalize])
        eval_transform = transforms.Compose([normalize])
        return train_transform, eval_transform

    # ----- Medium spatial resolution (9-32) -----
    train_augmentations = []
    min_dim = min(h, w)

    # Skip horizontal flip for low-variance grayscale data (likely digits/symbols/structured)
    is_structured = data_props.get('is_grayscale', False) and data_props.get('spatial_variance', 1.0) < 0.15
    if min_dim >= 12 and not is_structured:
        # Horizontal flip — safe for most natural-ish data
        train_augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
    elif is_structured:
        print("  [DataProcessor] Skipping horizontal flip (low-variance grayscale — likely structured data)")

    if min_dim >= 16:
        # Random crop with padding — mild spatial jitter
        pad = max(2, min_dim // 8)
        train_augmentations.append(transforms.RandomCrop((h, w), padding=pad))

    # ----- Large spatial resolution (>32): add cutout -----
    if min_dim > 32:
        cutout_size = max(4, min_dim // 4)
        train_augmentations.append(Cutout(cutout_size))

    # Always normalize at the end
    train_augmentations.append(normalize)

    train_transform = transforms.Compose(train_augmentations)
    eval_transform = transforms.Compose([normalize])

    return train_transform, eval_transform


# =============================================================================
# DataProcessor
# =============================================================================

class DataProcessor:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes.
    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    """
    ====================================================================================================================
    PROCESS ============================================================================================================
    ====================================================================================================================
    This function will be called, and it expects you to return three outputs:
        * train_loader: A Pytorch dataloader of (input, label) tuples
        * valid_loader: A Pytorch dataloader of (input, label) tuples
        * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!

    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.

    Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
    """

    def process(self):
        print("  [DataProcessor] Analyzing data properties...")

        # ---- Step 1: Inspect raw data ----
        data_props = inspect_data_properties(self.train_x)
        self._log_data_props(data_props)

        # Enrich metadata so NAS and Trainer can use these properties
        self.metadata['data_props'] = data_props

        # Fix the input_shape in metadata if channels were missing
        if len(self.train_x.shape) == 3:
            n, h, w = self.train_x.shape
            self.metadata['input_shape'] = [n, 1, h, w]
            print("  [DataProcessor] Fixed input_shape (added channel dim): {}".format(
                self.metadata['input_shape']))

        # ---- Step 2: Compute normalization statistics from training data ----
        mean, std = self._compute_normalization_stats()
        print("  [DataProcessor] Normalization — mean: {}, std: {}".format(
            [round(m, 4) for m in mean],
            [round(s, 4) for s in std]))

        # ---- Step 3: Build data-adaptive augmentation pipelines ----
        train_transform, eval_transform = build_augmentation_pipeline(
            data_props, mean, std
        )
        print("  [DataProcessor] Train augmentations: {}".format(train_transform))

        # ---- Step 4: Create PyTorch datasets ----
        train_ds = NASDataset(self.train_x, self.train_y, transform=train_transform)
        valid_ds = NASDataset(self.valid_x, self.valid_y, transform=eval_transform)
        test_ds  = NASDataset(self.test_x,  None,         transform=eval_transform)

        # ---- Step 5: Select batch size ----
        batch_size = estimate_batch_size(self.metadata['input_shape'])
        print("  [DataProcessor] Batch size selected: {}".format(batch_size))

        # Store for downstream use
        self.metadata['batch_size'] = batch_size

        # ---- Step 6: Build dataloaders ----
        # Use num_workers=0 for maximum compatibility across server environments
        # (avoids multiprocessing issues on some setups)
        num_workers = 0

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(train_ds) > batch_size),  # avoid empty loader
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # CRITICAL: test_loader must NOT shuffle and must NOT drop_last
        # (main.py asserts this — violation = instant crash)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        print("  [DataProcessor] Complete. Time remaining: ~{}".format(
            show_time(self.clock.check())))

        return train_loader, valid_loader, test_loader

    # ---- Private helpers ----

    def _compute_normalization_stats(self):
        """
        Compute per-channel mean and std from training data.

        Uses a subsample for speed on very large datasets.
        Returns (mean_per_channel, std_per_channel) as lists of floats.
        """
        x = self.train_x.astype(np.float32)

        # Handle 3D arrays (missing channel dim)
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]

        # Subsample if dataset is very large (>10k samples)
        if x.shape[0] > 10000:
            indices = np.random.choice(x.shape[0], 10000, replace=False)
            x = x[indices]

        # per-channel mean/std across (N, H, W)
        mean = np.mean(x, axis=(0, 2, 3))
        std  = np.std(x, axis=(0, 2, 3))

        # Avoid division by zero — clamp std to a small positive value
        std = np.maximum(std, 1e-6)

        return mean.tolist(), std.tolist()

    def _log_data_props(self, props):
        """Print a summary of detected data properties."""
        print("  [DataProcessor] Data properties:")
        print("    - Resolution: {}×{}, Channels: {}".format(
            props['height'], props['width'], props['channels']))
        print("    - Grayscale: {}, Small: {}, Square: {}".format(
            props['is_grayscale'], props['is_small'], props['is_square']))
        print("    - Spatial variance: {:.4f}".format(props['spatial_variance']))
        print("    - Num classes: {}".format(self.metadata['num_classes']))
        print("    - Train samples: {}".format(len(self.train_x)))
