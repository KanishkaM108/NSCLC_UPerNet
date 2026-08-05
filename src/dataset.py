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


def _pad_to_patch(image: np.ndarray, mask: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image, mask


def _crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
    min_annotated_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_patch(image, mask, patch_size)
    height, width = mask.shape
    best = None
    best_fraction = -1.0
    for _ in range(16):
        top = int(rng.integers(0, height - patch_size + 1))
        left = int(rng.integers(0, width - patch_size + 1))
        mask_patch = mask[top : top + patch_size, left : left + patch_size]
        fraction = float(np.mean(mask_patch != 0))
        if fraction > best_fraction:
            best = (top, left)
            best_fraction = fraction
        if fraction >= min_annotated_fraction:
            break
    assert best is not None
    top, left = best
    return (
        image[top : top + patch_size, left : left + patch_size],
        mask[top : top + patch_size, left : left + patch_size],
    )


def _augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    rotations = int(rng.integers(0, 4))
    if rotations:
        image = np.rot90(image, rotations)
        mask = np.rot90(mask, rotations)

    image_float = image.astype(np.float32)
    brightness = float(rng.uniform(0.90, 1.10))
    contrast = float(rng.uniform(0.90, 1.10))
    saturation = float(rng.uniform(0.90, 1.10))
    image_float *= brightness
    channel_mean = image_float.mean(axis=(0, 1), keepdims=True)
    image_float = (image_float - channel_mean) * contrast + channel_mean
    gray = image_float.mean(axis=2, keepdims=True)
    image_float = gray + (image_float - gray) * saturation
    image = np.clip(image_float, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class IgnitePatchDataset(Dataset):
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
        class_weights_path: str | Path | None = None,
        roi_presence_path: str | Path | None = None,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_spec = label_spec
        self.patch_size = int(patch_size)
        self.length = int(length)
        self.seed = int(seed)
        self.training = bool(training)
        self.min_annotated_fraction = float(min_annotated_fraction)
        self.epoch = 0
        self.row_probabilities = self._build_row_probabilities(class_weights_path, roi_presence_path)

    def _build_row_probabilities(self, class_weights_path, roi_presence_path):
        if not self.training or not class_weights_path or not roi_presence_path:
            return None
        class_weights_file = Path(class_weights_path)
        presence_file = Path(roi_presence_path)
        if not class_weights_file.exists() or not presence_file.exists():
            return None
        with open(class_weights_file, "r", encoding="utf-8") as handle:
            class_weight_data = json.load(handle)
        with open(presence_file, "r", encoding="utf-8") as handle:
            presence = json.load(handle)
        weights = np.asarray(class_weight_data["weights"], dtype=np.float64)
        scores = []
        for row in self.rows.itertuples(index=False):
            raw_labels = presence.get(str(row.name), [])
            train_labels = [self.label_spec["raw_to_train"].get(int(value)) for value in raw_labels]
            train_labels = [value for value in train_labels if value is not None]
            scores.append(float(np.mean(weights[train_labels])) if train_labels else 1.0)
        scores = np.asarray(scores, dtype=np.float64)
        scores = np.clip(scores, 0.25, 5.0)
        return scores / scores.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * max(1, self.length) + int(index))
        if self.training:
            row_index = int(rng.choice(len(self.rows), p=self.row_probabilities))
        else:
            row_index = int(index % len(self.rows))
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
            raise ValueError(f"Image/mask size mismatch for {row['name']}: {image.shape[:2]} vs {mask.shape}")

        image, mask = _crop_patch(
            image,
            mask,
            self.patch_size,
            rng,
            self.min_annotated_fraction,
        )
        if self.training:
            image, mask = _augment(image, mask, rng)

        image = normalize_image(image)
        mask = remap_mask(mask, self.label_spec)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor, str(row["name"])

