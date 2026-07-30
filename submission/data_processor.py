"""Adaptive and memory-conscious data processing for unseen image datasets."""

import random

import numpy as np
import torch
import torchvision.transforms as transforms

from helpers import estimate_batch_size, inspect_data_properties, show_time


def _seed_worker(worker_id):
    """Make torchvision/Python augmentation deterministic in loader workers."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NASDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, transform=None):
        # Avoid a second full copy when the competition arrays are already
        # contiguous float32.
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        if self.x.dtype != torch.float32:
            self.x = self.x.float()
        if self.x.ndim == 3:
            self.x = self.x.unsqueeze(1)

        self.y = (
            torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long)
            if y is not None
            else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = self.x[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.y is None:
            return image
        return image, self.y[index]


def build_augmentation_pipeline(data_props, mean, std):
    """
    Choose conservative transforms from scale-aware data fingerprints.

    Unknown datasets make aggressive generic augmentation risky.  Structured
    and standardized inputs therefore never receive flips or rotations.
    """
    height = data_props["height"]
    width = data_props["width"]
    min_dim = min(height, width)
    normalize = transforms.Normalize(mean, std)

    sequence_confidence = max(
        data_props.get("sequence_width_confidence", 0.0),
        data_props.get("sequence_height_confidence", 0.0),
    )
    if sequence_confidence >= 0.55:
        print(
            "  [DataProcessor] Position-sensitive sequence/grid detected; "
            "disabling geometric augmentation"
        )
        pipeline = transforms.Compose([normalize])
        return pipeline, pipeline

    if data_props["is_small"]:
        pipeline = transforms.Compose([normalize])
        return pipeline, pipeline

    is_grayscale = data_props["is_grayscale"]
    is_structured = data_props.get("is_structured", is_grayscale)
    low_variance_color = data_props.get("low_variance_color", False)
    horizontal_flip_safe = data_props.get(
        "horizontal_flip_safe", not is_structured
    )
    vertical_flip_safe = data_props.get("vertical_flip_safe", False)
    augmentations = []

    if min_dim >= 12 and horizontal_flip_safe:
        augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
    if min_dim >= 12 and vertical_flip_safe and data_props["is_square"]:
        augmentations.append(transforms.RandomVerticalFlip(p=0.5))
    if not horizontal_flip_safe and not vertical_flip_safe:
        print("  [DataProcessor] Skipping flips (structured/synthetic data)")

    if min_dim >= 16:
        if is_grayscale and is_structured:
            # Mild translation helps glyphs. Rotations/reflections can change
            # their label and are deliberately avoided.  Ambiguous structured
            # inputs receive it stochastically so identity samples remain in
            # every epoch.
            augmentations.append(
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))],
                    p=0.50,
                )
            )
        else:
            pad = max(2, min_dim // 16)
            augmentations.append(
                transforms.RandomCrop((height, width), padding=pad)
            )

    augmentations.append(normalize)

    # RandomErasing is reserved for natural-looking large images.  After
    # normalization, an erased value of zero is the channel mean.
    if min_dim > 32 and not is_structured and not low_variance_color:
        augmentations.append(
            transforms.RandomErasing(
                p=0.20,
                scale=(0.02, 0.10),
                ratio=(0.5, 2.0),
                value=0,
            )
        )

    return transforms.Compose(augmentations), transforms.Compose([normalize])


class DataProcessor:
    def __init__(
        self,
        train_x,
        train_y,
        valid_x,
        valid_y,
        test_x,
        metadata,
        clock,
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock

    def process(self):
        print("  [DataProcessor] Analyzing data properties...")
        data_props = inspect_data_properties(self.train_x)
        horizontal_safe, vertical_safe = self._estimate_flip_safety(data_props)
        data_props["horizontal_flip_safe"] = horizontal_safe
        data_props["vertical_flip_safe"] = vertical_safe

        class_counts = np.bincount(
            np.asarray(self.train_y, dtype=np.int64),
            minlength=int(self.metadata["num_classes"]),
        )
        nonzero_counts = class_counts[class_counts > 0]
        data_props["class_imbalance_ratio"] = (
            float(nonzero_counts.max() / max(1.0, nonzero_counts.min()))
            if len(nonzero_counts)
            else 1.0
        )
        if len(nonzero_counts):
            inverse_sqrt = np.sqrt(
                float(nonzero_counts.mean()) / np.maximum(class_counts, 1)
            )
            inverse_sqrt = inverse_sqrt / max(1e-8, float(inverse_sqrt.mean()))
            data_props["class_weights"] = inverse_sqrt.astype(np.float32).tolist()
        else:
            data_props["class_weights"] = [1.0] * int(
                self.metadata["num_classes"]
            )
        data_props["train_samples"] = int(len(self.train_x))
        data_props["valid_samples"] = int(len(self.valid_x))
        self._log_data_props(data_props)
        self.metadata["data_props"] = data_props

        # Historical metadata can contain nominal dimensions (for example
        # 64x64 before a 60x60 preprocessing crop).  Models must use the actual
        # tensors.
        original_shape = self.metadata.get("input_shape", [len(self.train_x)])
        total_samples = int(original_shape[0])
        self.metadata["input_shape"] = [
            total_samples,
            data_props["channels"],
            data_props["height"],
            data_props["width"],
        ]
        print(
            "  [DataProcessor] Effective input_shape: {}".format(
                self.metadata["input_shape"]
            )
        )

        mean, std = self._compute_normalization_stats()
        print(
            "  [DataProcessor] Normalization — mean: {}, std: {}".format(
                [round(value, 4) for value in mean],
                [round(value, 4) for value in std],
            )
        )

        train_transform, eval_transform = build_augmentation_pipeline(
            data_props, mean, std
        )
        print("  [DataProcessor] Train augmentations: {}".format(train_transform))

        train_dataset = NASDataset(
            self.train_x, self.train_y, transform=train_transform
        )
        valid_dataset = NASDataset(
            self.valid_x, self.valid_y, transform=eval_transform
        )
        test_dataset = NASDataset(self.test_x, None, transform=eval_transform)

        batch_size = estimate_batch_size(self.metadata["input_shape"])
        self.metadata["batch_size"] = batch_size
        print("  [DataProcessor] Batch size selected: {}".format(batch_size))

        cuda = torch.cuda.is_available()
        num_workers = 2 if cuda else 0
        generator = torch.Generator()
        generator.manual_seed(1729)
        common = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": cuda,
            "worker_init_fn": _seed_worker if num_workers else None,
            "persistent_workers": bool(num_workers),
        }

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=len(train_dataset) > batch_size,
            generator=generator,
            **common
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            shuffle=False,
            drop_last=False,
            **common
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common
        )
        self.metadata["train_num_batches"] = len(train_loader)
        self.metadata["valid_num_batches"] = len(valid_loader)
        self.metadata["test_num_batches"] = len(test_loader)
        self.metadata["test_num_samples"] = len(test_dataset)

        print(
            "  [DataProcessor] Complete. Time remaining: ~{}".format(
                show_time(self.clock.check())
            )
        )
        return train_loader, valid_loader, test_loader

    def _compute_normalization_stats(self):
        raw = self.train_x
        if raw.shape[0] > 10000:
            # Deterministic subsampling avoids a float32 copy of the entire
            # training set.
            indices = np.linspace(0, raw.shape[0] - 1, 10000, dtype=np.int64)
            raw = raw[indices]
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim == 3:
            values = values[:, np.newaxis, :, :]

        mean = np.mean(values, axis=(0, 2, 3))
        std = np.maximum(np.std(values, axis=(0, 2, 3)), 1e-6)
        return mean.tolist(), std.tolist()

    def _estimate_flip_safety(self, data_props):
        """
        Cheap label-aware invariance probe using held-out nearest centroids.

        It avoids assuming that all RGB data is natural imagery: mirrored
        digits/text score substantially worse, while orientation-invariant
        aerial imagery tends to retain its class-centroid assignment.
        """
        if data_props["is_grayscale"] or min(
            data_props["height"], data_props["width"]
        ) < 12:
            return False, False

        sample_count = min(len(self.train_x), 768)
        if sample_count < max(32, 2 * int(self.metadata["num_classes"])):
            return (not data_props.get("is_structured", False)), False

        indices = np.linspace(
            0, len(self.train_x) - 1, sample_count, dtype=np.int64
        )
        images = np.asarray(self.train_x[indices], dtype=np.float32)
        labels = np.asarray(self.train_y, dtype=np.int64)[indices]
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]

        reference_mask = np.arange(sample_count) % 2 == 0
        evaluation_mask = ~reference_mask
        reference_images = images[reference_mask]
        reference_labels = labels[reference_mask]
        evaluation_images = images[evaluation_mask]
        evaluation_labels = labels[evaluation_mask]

        stride_h = max(1, data_props["height"] // 16)
        stride_w = max(1, data_props["width"] // 16)

        def features(batch):
            reduced = batch[:, :, ::stride_h, ::stride_w]
            return reduced.reshape(reduced.shape[0], -1).astype(np.float32)

        reference_features = features(reference_images)
        feature_mean = reference_features.mean(axis=0, keepdims=True)
        feature_std = np.maximum(
            reference_features.std(axis=0, keepdims=True), 1e-3
        )
        reference_features = (reference_features - feature_mean) / feature_std

        present_classes = np.unique(reference_labels)
        if len(present_classes) < 2:
            return False, False
        centroids = np.stack(
            [
                reference_features[reference_labels == class_id].mean(axis=0)
                for class_id in present_classes
            ],
            axis=0,
        )

        def centroid_accuracy(batch):
            values = (features(batch) - feature_mean) / feature_std
            distances = (
                np.sum(values * values, axis=1, keepdims=True)
                + np.sum(centroids * centroids, axis=1)[np.newaxis, :]
                - 2.0 * values.dot(centroids.T)
            )
            predictions = present_classes[np.argmin(distances, axis=1)]
            return float(np.mean(predictions == evaluation_labels))

        base_accuracy = centroid_accuracy(evaluation_images)
        random_accuracy = 1.0 / max(2, len(present_classes))
        if base_accuracy < max(0.10, 1.5 * random_accuracy):
            return (not data_props.get("is_structured", False)), False

        horizontal_accuracy = centroid_accuracy(
            np.flip(evaluation_images, axis=3)
        )
        vertical_accuracy = centroid_accuracy(np.flip(evaluation_images, axis=2))
        threshold = max(1.5 * random_accuracy, 0.90 * base_accuracy)
        horizontal_safe = horizontal_accuracy >= threshold
        vertical_safe = (
            data_props["is_square"]
            and data_props.get("low_variance_color", False)
            and vertical_accuracy >= threshold
        )
        return bool(horizontal_safe), bool(vertical_safe)

    def _log_data_props(self, props):
        print("  [DataProcessor] Data properties:")
        print(
            "    - Resolution: {}x{}, Channels: {}".format(
                props["height"], props["width"], props["channels"]
            )
        )
        print(
            "    - Grayscale: {}, Small: {}, Square: {}".format(
                props["is_grayscale"], props["is_small"], props["is_square"]
            )
        )
        print(
            "    - Structured: {}, standardized: {}, low-var RGB: {}".format(
                props.get("is_structured", False),
                props.get("is_standardized", False),
                props.get("low_variance_color", False),
            )
        )
        print(
            "    - Categorical grid: {} (binary={:.3f}, density={:.4f}, "
            "one-hot columns={:.3f}, rows={:.3f})".format(
                props.get("is_categorical_grid", False),
                props.get("binary_like_fraction", 0.0),
                props.get("active_density", 0.0),
                props.get("one_hot_column_ratio", 0.0),
                props.get("one_hot_row_ratio", 0.0),
            )
        )
        print(
            "    - Representation hypotheses: {}".format(
                {
                    name: round(float(confidence), 3)
                    for name, confidence in props.get(
                        "representation_hypotheses", {}
                    ).items()
                }
            )
        )
        print(
            "    - Flip safety: horizontal={}, vertical={}".format(
                props.get("horizontal_flip_safe", False),
                props.get("vertical_flip_safe", False),
            )
        )
        print(
            "    - Spatial variance: {:.4f}, normalized: {:.4f}".format(
                props["spatial_variance"],
                props.get("normalized_spatial_variance", 0.0),
            )
        )
        print(
            "    - Class imbalance ratio: {:.2f}".format(
                props["class_imbalance_ratio"]
            )
        )
        print("    - Num classes: {}".format(self.metadata["num_classes"]))
        print("    - Train samples: {}".format(len(self.train_x)))
