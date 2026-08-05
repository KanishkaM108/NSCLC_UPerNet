from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import IGNORE_INDEX, normalize_image, remap_mask, resolve_data_path


GROUP_NAMES = [
    "Tumor",
    "Stroma + inflammation",
    "Macrophages",
    "Necrosis",
    "Rest",
]


def build_group_lookup(label_spec: dict[str, Any]) -> np.ndarray:
    """Map the official 16 train IDs to five prespecified tissue groups."""
    lookup = np.full(label_spec["num_classes"], 4, dtype=np.uint8)
    name_to_id = {name: index for index, name in enumerate(label_spec["class_names"])}
    required = [
        "Tumor epithelium",
        "Stroma",
        "Inflammation",
        "Macrophages",
        "Necrotic tissue",
    ]
    missing = [name for name in required if name not in name_to_id]
    if missing:
        raise ValueError(f"Official label map is missing: {missing}")
    lookup[name_to_id["Tumor epithelium"]] = 0
    lookup[name_to_id["Stroma"]] = 1
    lookup[name_to_id["Inflammation"]] = 1
    lookup[name_to_id["Macrophages"]] = 2
    lookup[name_to_id["Necrotic tissue"]] = 3
    return lookup


def to_grouped_mask(mask_16: np.ndarray, group_lookup: np.ndarray) -> np.ndarray:
    grouped = np.full(mask_16.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = mask_16 != IGNORE_INDEX
    grouped[valid] = group_lookup[mask_16[valid]]
    return grouped


def _pad_to_patch(
    image: np.ndarray, mask: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask = np.pad(
            mask,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=IGNORE_INDEX,
        )
    return image, mask


def _random_crop_origin(
    mask: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
    min_annotated_fraction: float,
) -> tuple[int, int]:
    height, width = mask.shape
    best = (0, 0)
    best_fraction = -1.0
    for _ in range(20):
        top = int(rng.integers(0, height - patch_size + 1))
        left = int(rng.integers(0, width - patch_size + 1))
        patch = mask[top : top + patch_size, left : left + patch_size]
        annotated_fraction = float(np.mean(patch != IGNORE_INDEX))
        if annotated_fraction > best_fraction:
            best = (top, left)
            best_fraction = annotated_fraction
        if annotated_fraction >= min_annotated_fraction:
            break
    return best


def _targeted_crop_origin(
    mask: np.ndarray,
    patch_size: int,
    target_group: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    coordinates = np.argwhere(mask == target_group)
    if len(coordinates) == 0:
        return None
    center_y, center_x = coordinates[int(rng.integers(0, len(coordinates)))]
    jitter = max(1, patch_size // 5)
    center_y += int(rng.integers(-jitter, jitter + 1))
    center_x += int(rng.integers(-jitter, jitter + 1))
    max_top = mask.shape[0] - patch_size
    max_left = mask.shape[1] - patch_size
    top = int(np.clip(center_y - patch_size // 2, 0, max_top))
    left = int(np.clip(center_x - patch_size // 2, 0, max_left))
    return top, left


def _augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    turns = int(rng.integers(0, 4))
    if turns:
        image = np.rot90(image, turns)
        mask = np.rot90(mask, turns)

    # Moderate stain/illumination variation. These ranges preserve morphology.
    image_float = image.astype(np.float32)
    brightness = float(rng.uniform(0.86, 1.14))
    contrast = float(rng.uniform(0.86, 1.14))
    saturation = float(rng.uniform(0.84, 1.16))
    channel_scale = rng.uniform(0.94, 1.06, size=(1, 1, 3)).astype(np.float32)
    image_float *= brightness * channel_scale
    channel_mean = image_float.mean(axis=(0, 1), keepdims=True)
    image_float = (image_float - channel_mean) * contrast + channel_mean
    gray = image_float.mean(axis=2, keepdims=True)
    image_float = gray + (image_float - gray) * saturation
    image = np.clip(image_float, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class GroupedPatchDataset(Dataset):
    """Five-class patches with class-targeted training crops.

    Row sampling remains uniform. The targeted crop is chosen only inside the
    selected training ROI, so no validation or test information is used.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        data_root: str | Path,
        label_spec: dict[str, Any],
        patch_size: int,
        length: int,
        seed: int,
        training: bool,
        min_annotated_fraction: float,
        targeted_crop_probability: float,
        target_group_probabilities: list[float],
        roi_presence_path: str | Path | None,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_spec = label_spec
        self.group_lookup = build_group_lookup(label_spec)
        self.patch_size = int(patch_size)
        self.length = int(length)
        self.seed = int(seed)
        self.training = bool(training)
        self.min_annotated_fraction = float(min_annotated_fraction)
        self.targeted_crop_probability = float(targeted_crop_probability)
        probabilities = np.asarray(target_group_probabilities, dtype=np.float64)
        if probabilities.shape != (len(GROUP_NAMES),) or np.any(probabilities < 0):
            raise ValueError("target_group_probabilities must contain five nonnegative values")
        self.target_group_probabilities = probabilities / probabilities.sum()
        self.group_to_rows = self._build_group_to_rows(roi_presence_path)
        self.epoch = 0

    def _build_group_to_rows(self, roi_presence_path):
        if not self.training or not roi_presence_path or not Path(roi_presence_path).exists():
            return {}
        with open(roi_presence_path, "r", encoding="utf-8") as handle:
            presence = json.load(handle)
        group_to_rows = {group_id: [] for group_id in range(len(GROUP_NAMES))}
        for row_index, row in enumerate(self.rows.itertuples(index=False)):
            groups = set()
            for raw_id in presence.get(str(row.name), []):
                train_id = self.label_spec["raw_to_train"].get(int(raw_id))
                if train_id is not None:
                    groups.add(int(self.group_lookup[train_id]))
            for group_id in groups:
                group_to_rows[group_id].append(row_index)
        return group_to_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(
            self.seed + self.epoch * max(1, self.length) + int(index)
        )
        target_group = None
        if self.training and rng.random() < self.targeted_crop_probability:
            target_group = int(
                rng.choice(len(GROUP_NAMES), p=self.target_group_probabilities)
            )
            candidate_rows = self.group_to_rows.get(target_group, [])
            if candidate_rows:
                row_index = int(candidate_rows[int(rng.integers(0, len(candidate_rows)))])
            else:
                row_index = int(rng.integers(0, len(self.rows)))
        else:
            row_index = (
                int(rng.integers(0, len(self.rows)))
                if self.training
                else int(index % len(self.rows))
            )
        row = self.rows.iloc[row_index]
        image_path = resolve_data_path(self.data_root, row["image_path"])
        mask_path = resolve_data_path(self.data_root, row["annotation_path"])

        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))
        with Image.open(mask_path) as mask_file:
            raw_mask = np.asarray(mask_file)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[..., 0]
        if image.shape[:2] != raw_mask.shape:
            raise ValueError(
                f"Image/mask size mismatch for {row['name']}: "
                f"{image.shape[:2]} vs {raw_mask.shape}"
            )

        mask_16 = remap_mask(raw_mask, self.label_spec)
        grouped_mask = to_grouped_mask(mask_16, self.group_lookup)
        image, grouped_mask = _pad_to_patch(image, grouped_mask, self.patch_size)

        origin = None
        if target_group is not None:
            origin = _targeted_crop_origin(
                grouped_mask, self.patch_size, target_group, rng
            )
        if origin is None:
            origin = _random_crop_origin(
                grouped_mask,
                self.patch_size,
                rng,
                self.min_annotated_fraction,
            )
        top, left = origin
        image = image[top : top + self.patch_size, left : left + self.patch_size]
        grouped_mask = grouped_mask[
            top : top + self.patch_size, left : left + self.patch_size
        ]

        if self.training:
            image, grouped_mask = _augment(image, grouped_mask, rng)
        image = normalize_image(image)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(grouped_mask)).long()
        return image_tensor, mask_tensor, str(row["name"])
