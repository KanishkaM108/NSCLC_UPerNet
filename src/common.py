from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IGNORE_INDEX = 255
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return cfg


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def load_label_spec(label_map_path: str | Path) -> dict[str, Any]:
    with open(label_map_path, "r", encoding="utf-8") as handle:
        raw_map: dict[str, int] = json.load(handle)

    unannotated = [name for name, value in raw_map.items() if int(value) == 0]
    classes = sorted(
        ((name, int(value)) for name, value in raw_map.items() if int(value) != 0),
        key=lambda item: item[1],
    )
    if len(classes) != 16:
        raise ValueError(f"Expected 16 annotated classes, found {len(classes)}")

    class_names = [name for name, _ in classes]
    raw_values = [value for _, value in classes]
    raw_to_train = {raw_value: index for index, raw_value in enumerate(raw_values)}
    train_to_raw = {index: raw_value for raw_value, index in raw_to_train.items()}
    return {
        "raw_map": raw_map,
        "unannotated_names": unannotated,
        "class_names": class_names,
        "raw_values": raw_values,
        "raw_to_train": raw_to_train,
        "train_to_raw": train_to_raw,
        "num_classes": len(class_names),
    }


def remap_mask(raw_mask: np.ndarray, label_spec: dict[str, Any]) -> np.ndarray:
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., 0]
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for raw_value, train_value in label_spec["raw_to_train"].items():
        lut[int(raw_value)] = int(train_value)
    if raw_mask.min(initial=0) < 0 or raw_mask.max(initial=0) > 255:
        raise ValueError("Mask contains values outside the supported 0..255 range")
    return lut[raw_mask.astype(np.uint8, copy=False)]


def load_he_metadata(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {
        "patient_id",
        "name",
        "task",
        "image_path",
        "annotation_path",
        "split",
        "validation_fold",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    he = frame.loc[frame["task"].astype(str) == "he_tissue_segmentation"].copy()
    he["split"] = he["split"].astype(str).str.lower().str.strip()
    he["validation_fold"] = he["validation_fold"].fillna("").astype(str).str.strip()
    he = he.reset_index(drop=True)
    if he.empty:
        raise ValueError("No he_tissue_segmentation rows found in metadata")
    return he


def official_splits(he: pd.DataFrame, validation_fold: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official_train = he.loc[he["split"] == "train"].copy()
    test = he.loc[he["split"] == "test"].copy()
    val = official_train.loc[official_train["validation_fold"] == validation_fold].copy()
    train = official_train.loc[official_train["validation_fold"] != validation_fold].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Invalid split sizes: train={len(train)}, val={len(val)}, test={len(test)}. "
            f"Check validation_fold={validation_fold!r}."
        )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def resolve_data_path(data_root: str | Path, metadata_path: str) -> Path:
    data_root = Path(data_root)
    normalized = str(metadata_path).replace("\\", "/").strip()
    candidate_path = Path(normalized)
    parts = list(candidate_path.parts)
    lowered = [part.lower() for part in parts]
    # Official IGNITE metadata stores paths from the authors' machine, e.g.
    # /ignite_data_toolkit/data/images/he/.... Keep only the portable section.
    for anchor in ("images", "annotations"):
        if anchor in lowered:
            anchor_index = lowered.index(anchor)
            candidate_path = Path(*parts[anchor_index:])
            break
    else:
        if parts and parts[0].lower() == "data":
            candidate_path = Path(*parts[1:])
        elif candidate_path.is_absolute():
            return candidate_path
    return data_root / candidate_path


def context_variant(path: str | Path) -> Path:
    path = Path(path)
    candidate = path.with_name(f"{path.stem}_with_context{path.suffix}")
    return candidate if candidate.exists() else path


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def experiment_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["output_root"]) / cfg["experiment_name"]
    path.mkdir(parents=True, exist_ok=True)
    return path
