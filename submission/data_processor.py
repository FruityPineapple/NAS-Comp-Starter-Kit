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
        # Preserve the source dtype. Converting an entire uint8/float64
        # competition array to float32 can transiently double multi-gigabyte
        # datasets; conversion is therefore performed per example below.
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        if self.x.ndim == 3:
            self.x = self.x.unsqueeze(1)

        self.y = (
            torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long)
            if y is not None
            else None
        )
        self.transform = transform
        self.augmentation_policies = {}
        self.augmentation_policy = "default"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = self.x[index]
        if image.dtype != torch.float32:
            image = image.to(dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        if self.y is None:
            return image
        return image, self.y[index]

    def set_augmentation_policy(self, name):
        if name not in self.augmentation_policies:
            raise KeyError("Unknown augmentation policy: {}".format(name))
        self.augmentation_policy = str(name)
        self.transform = self.augmentation_policies[name]


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


def build_augmentation_portfolio(data_props, mean, std):
    """Return a small rule-safe policy portfolio with identity always present."""
    selected, evaluation = build_augmentation_pipeline(data_props, mean, std)
    policies = {"identity": evaluation}
    if repr(selected) != repr(evaluation):
        policies["conservative"] = selected

    # When flips passed the label-aware invariance probe, expose a lighter
    # alternative without crop/translation. The NAS controller can compare it
    # against identity and the conservative default on a common checkpoint.
    light = []
    if data_props.get("horizontal_flip_safe", False):
        light.append(transforms.RandomHorizontalFlip(p=0.5))
    if (
        data_props.get("vertical_flip_safe", False)
        and data_props.get("is_square", False)
    ):
        light.append(transforms.RandomVerticalFlip(p=0.5))
    if light:
        light.append(transforms.Normalize(mean, std))
        policies["safe_flips"] = transforms.Compose(light)

    selected_name = (
        "identity"
        if data_props.get("distribution_shift_detected", False)
        else ("conservative" if "conservative" in policies else "identity")
    )
    return policies, evaluation, selected_name


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
        data_props.update(self._estimate_distribution_shift())
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

        (
            augmentation_policies,
            eval_transform,
            selected_policy,
        ) = build_augmentation_portfolio(
            data_props, mean, std
        )
        train_transform = augmentation_policies[selected_policy]
        print(
            "  [DataProcessor] Augmentation candidates: {}; default={}".format(
                ",".join(augmentation_policies),
                selected_policy,
            )
        )
        print("  [DataProcessor] Train augmentations: {}".format(train_transform))

        train_dataset = NASDataset(
            self.train_x, self.train_y, transform=train_transform
        )
        train_dataset.augmentation_policies = dict(augmentation_policies)
        train_dataset.augmentation_policy = selected_policy
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
        self.metadata["augmentation_candidates"] = list(
            augmentation_policies
        )
        self.metadata["augmentation_policy"] = selected_policy

        print(
            "  [DataProcessor] Complete. Time remaining: ~{}".format(
                show_time(self.clock.check())
            )
        )
        return train_loader, valid_loader, test_loader

    def _compute_normalization_stats(
        self, max_samples=10000, max_chunk_megabytes=16
    ):
        """Compute channel moments with deterministic, byte-bounded chunks."""
        raw = self.train_x
        sample_count = min(int(raw.shape[0]), int(max_samples))
        indices = np.linspace(
            0, raw.shape[0] - 1, sample_count, dtype=np.int64
        )
        if raw.ndim == 3:
            channels, height, width = 1, raw.shape[1], raw.shape[2]
        else:
            channels, height, width = raw.shape[1:]
        bytes_per_sample = max(1, channels * height * width * 4)
        chunk_samples = max(
            1,
            int(max_chunk_megabytes * 1024 * 1024) // bytes_per_sample,
        )
        channel_sum = np.zeros(channels, dtype=np.float64)
        channel_square_sum = np.zeros(channels, dtype=np.float64)
        elements_per_channel = 0

        for start in range(0, sample_count, chunk_samples):
            chunk_indices = indices[start : start + chunk_samples]
            values = np.asarray(raw[chunk_indices], dtype=np.float32)
            if values.ndim == 3:
                values = values[:, np.newaxis, :, :]
            channel_sum += values.sum(axis=(0, 2, 3), dtype=np.float64)
            channel_square_sum += np.square(
                values, dtype=np.float32
            ).sum(axis=(0, 2, 3), dtype=np.float64)
            elements_per_channel += values.shape[0] * height * width

        denominator = max(1, elements_per_channel)
        mean = channel_sum / denominator
        variance = np.maximum(
            channel_square_sum / denominator - np.square(mean),
            1e-12,
        )
        std = np.maximum(np.sqrt(variance), 1e-6)
        self.metadata["normalization_samples"] = sample_count
        self.metadata["normalization_chunk_samples"] = chunk_samples
        return mean.tolist(), std.tolist()

    def _estimate_distribution_shift(self, max_samples=256):
        """Compare train/validation channel moments without fitting a model."""
        if len(self.valid_x) == 0 or len(self.train_x) == 0:
            return {
                "distribution_shift_score": 0.0,
                "distribution_shift_detected": False,
            }

        def moments(values):
            count = min(len(values), int(max_samples))
            indices = np.linspace(
                0, len(values) - 1, count, dtype=np.int64
            )
            sample = np.asarray(values[indices], dtype=np.float32)
            if sample.ndim == 3:
                sample = sample[:, np.newaxis, :, :]
            return (
                sample.mean(axis=(0, 2, 3), dtype=np.float64),
                sample.std(axis=(0, 2, 3), dtype=np.float64),
            )

        train_mean, train_std = moments(self.train_x)
        valid_mean, valid_std = moments(self.valid_x)
        scale = np.maximum(0.5 * (train_std + valid_std), 1e-4)
        mean_shift = float(np.mean(np.abs(train_mean - valid_mean) / scale))
        std_shift = float(
            np.mean(np.abs(train_std - valid_std) / np.maximum(scale, 1e-4))
        )
        score = mean_shift + 0.5 * std_shift
        return {
            "distribution_shift_score": score,
            "distribution_shift_detected": bool(score >= 0.75),
        }

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
        print(
            "    - Train/validation shift: {:.3f}{}".format(
                props.get("distribution_shift_score", 0.0),
                " (detected)"
                if props.get("distribution_shift_detected", False)
                else "",
            )
        )
        print("    - Num classes: {}".format(self.metadata["num_classes"]))
        print("    - Train samples: {}".format(len(self.train_x)))
