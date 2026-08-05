from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import normalize_image, remap_mask, resolve_data_path
from dataset import _augment, _crop_patch, _pad_to_patch


def _class_centered_crop(
    image: np.ndarray,
    mask: np.ndarray,
    raw_target: int,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose the densest of several crops containing a requested raw class."""
    image, mask = _pad_to_patch(image, mask, patch_size)
    target_y, target_x = np.where(mask == int(raw_target))
    if target_y.size == 0:
        return None

    height, width = mask.shape
    best: tuple[int, int] | None = None
    best_score = -1.0
    trials = min(12, max(4, target_y.size))
    jitter = patch_size // 5

    for _ in range(trials):
        point = int(rng.integers(0, target_y.size))
        centre_y = int(target_y[point])
        centre_x = int(target_x[point])
        top = centre_y - patch_size // 2 + int(rng.integers(-jitter, jitter + 1))
        left = centre_x - patch_size // 2 + int(rng.integers(-jitter, jitter + 1))
        top = int(np.clip(top, 0, height - patch_size))
        left = int(np.clip(left, 0, width - patch_size))
        candidate = mask[top : top + patch_size, left : left + patch_size]
        target_fraction = float(np.mean(candidate == int(raw_target)))
        annotated_fraction = float(np.mean((candidate != 0) & (candidate != 255)))
        score = target_fraction + 0.05 * annotated_fraction
        if score > best_score:
            best = (top, left)
            best_score = score

    if best is None:
        return None
    top, left = best
    return (
        image[top : top + patch_size, left : left + patch_size],
        mask[top : top + patch_size, left : left + patch_size],
    )


class ClassUniformPatchDataset(Dataset):
    """Class-uniform, target-centred patches for imbalanced tissue segmentation."""

    def __init__(
        self,
        rows: pd.DataFrame,
        data_root: str | Path,
        label_spec: dict[str, Any],
        patch_size: int,
        length: int,
        seed: int,
        min_annotated_fraction: float,
        roi_presence_path: str | Path,
        target_probability: float = 0.75,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_spec = label_spec
        self.patch_size = int(patch_size)
        self.length = int(length)
        self.seed = int(seed)
        self.min_annotated_fraction = float(min_annotated_fraction)
        self.target_probability = float(target_probability)
        self.epoch = 0

        with open(roi_presence_path, "r", encoding="utf-8") as handle:
            presence: dict[str, list[int]] = json.load(handle)

        self.class_to_rows: dict[int, list[int]] = {
            class_id: [] for class_id in range(label_spec["num_classes"])
        }
        for row_index, row in self.rows.iterrows():
            raw_values = presence.get(str(row["name"]), [])
            for raw_value in raw_values:
                class_id = label_spec["raw_to_train"].get(int(raw_value))
                if class_id is not None:
                    self.class_to_rows[int(class_id)].append(int(row_index))

        self.available_classes = np.asarray(
            [class_id for class_id, indices in self.class_to_rows.items() if indices],
            dtype=np.int64,
        )
        if self.available_classes.size != label_spec["num_classes"]:
            missing = sorted(set(range(label_spec["num_classes"])) - set(self.available_classes.tolist()))
            raise ValueError(f"Training split has no ROI for class IDs: {missing}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * max(1, self.length) + int(index))
        target_class: int | None = None
        if rng.random() < self.target_probability:
            target_class = int(rng.choice(self.available_classes))
            row_index = int(rng.choice(self.class_to_rows[target_class]))
        else:
            row_index = int(rng.integers(0, len(self.rows)))

        row = self.rows.iloc[row_index]
        image_path = resolve_data_path(self.data_root, row["image_path"])
        mask_path = resolve_data_path(self.data_root, row["annotation_path"])
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        with Image.open(mask_path) as mask_file:
            mask = np.asarray(mask_file)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask mismatch for {row['name']}: {image.shape[:2]} vs {mask.shape}")

        crop = None
        if target_class is not None:
            raw_target = int(self.label_spec["train_to_raw"][target_class])
            crop = _class_centered_crop(image, mask, raw_target, self.patch_size, rng)
        if crop is None:
            crop = _crop_patch(
                image,
                mask,
                self.patch_size,
                rng,
                self.min_annotated_fraction,
            )
        image, mask = crop
        image, mask = _augment(image, mask, rng)
        image = normalize_image(image)
        mask = remap_mask(mask, self.label_spec)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor, str(row["name"])
