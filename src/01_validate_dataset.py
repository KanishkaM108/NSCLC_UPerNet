from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, load_he_metadata, load_label_spec, official_splits, resolve_data_path, save_json


def main() -> None:
    cfg = load_config()
    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    train, val, test = official_splits(he, cfg["validation_fold"])

    print("=" * 68)
    print("IGNITE H&E DATASET VALIDATION")
    print("=" * 68)
    print(f"Official H&E metadata rows : {len(he)}")
    print(f"Training rows              : {len(train)}")
    print(f"Validation rows ({cfg['validation_fold']})    : {len(val)}")
    print(f"Untouched test rows        : {len(test)}")
    print(f"Annotated classes          : {label_spec['num_classes']}")

    missing_files: list[str] = []
    mismatched_shapes: list[dict] = []
    invalid_values: dict[str, list[int]] = {}
    raw_counts: Counter[int] = Counter()
    roi_presence: dict[str, list[int]] = {}
    training_names = set(train["name"]).union(set(val["name"]))

    for row in tqdm(he.itertuples(index=False), total=len(he), desc="Checking pairs"):
        image_path = resolve_data_path(cfg["data_root"], row.image_path)
        mask_path = resolve_data_path(cfg["data_root"], row.annotation_path)
        if not image_path.exists():
            missing_files.append(str(image_path))
            continue
        if not mask_path.exists():
            missing_files.append(str(mask_path))
            continue
        with Image.open(image_path) as image_file:
            image_size = image_file.size
        with Image.open(mask_path) as mask_file:
            mask_size = mask_file.size
            if row.name in training_names:
                mask = np.asarray(mask_file)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                unique, counts = np.unique(mask, return_counts=True)
                roi_presence[str(row.name)] = [int(value) for value in unique if int(value) != 0]
                for value, count in zip(unique, counts):
                    raw_counts[int(value)] += int(count)
                invalid = [int(value) for value in unique if int(value) not in label_spec['raw_map'].values() and int(value) != 255]
                if invalid:
                    invalid_values[str(row.name)] = invalid
        if image_size != mask_size:
            mismatched_shapes.append({"name": row.name, "image": list(image_size), "mask": list(mask_size)})

    if missing_files or mismatched_shapes or invalid_values:
        print(f"Missing files        : {len(missing_files)}")
        print(f"Shape mismatches     : {len(mismatched_shapes)}")
        print(f"Invalid-label masks  : {len(invalid_values)}")
        save_json(
            {
                "missing_files": missing_files,
                "mismatched_shapes": mismatched_shapes,
                "invalid_values": invalid_values,
            },
            "artifacts/dataset_errors.json",
        )
        raise SystemExit("Dataset validation FAILED. See artifacts/dataset_errors.json")

    annotated_counts = np.asarray([raw_counts[value] for value in label_spec["raw_values"]], dtype=np.float64)
    frequencies = annotated_counts / max(1.0, annotated_counts.sum())
    positive = frequencies[frequencies > 0]
    median_frequency = float(np.median(positive)) if len(positive) else 1.0
    weights = np.divide(median_frequency, frequencies, out=np.ones_like(frequencies), where=frequencies > 0)
    weights = np.clip(weights, 0.25, 5.0)
    weights /= weights.mean()

    class_rows = []
    print("\nClass distribution across official train + validation masks:")
    for index, (name, raw_value) in enumerate(zip(label_spec["class_names"], label_spec["raw_values"])):
        class_rows.append(
            {
                "train_id": index,
                "raw_value": raw_value,
                "class_name": name,
                "pixels": int(annotated_counts[index]),
                "frequency": float(frequencies[index]),
                "loss_weight": float(weights[index]),
            }
        )
        print(f"  {index:2d} raw={raw_value:2d} {name:<26} {frequencies[index] * 100:8.4f}%  weight={weights[index]:.3f}")

    train_patients = set(train["patient_id"].astype(str)).union(set(val["patient_id"].astype(str)))
    test_patients = set(test["patient_id"].astype(str))
    patient_overlap = sorted(train_patients.intersection(test_patients))

    report = {
        "official_he_rows": len(he),
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "validation_fold": cfg["validation_fold"],
        "class_count": label_spec["num_classes"],
        "unannotated_raw_value": 0,
        "patient_overlap_train_test": patient_overlap,
        "classes": class_rows,
    }
    save_json(report, "artifacts/dataset_report.json")
    save_json(
        {"class_names": label_spec["class_names"], "weights": [float(value) for value in weights]},
        "artifacts/class_weights.json",
    )
    save_json(roi_presence, "artifacts/roi_class_presence.json")
    print("\nDataset validation PASSED")
    print("Saved artifacts/dataset_report.json")
    print("Saved artifacts/class_weights.json")


if __name__ == "__main__":
    main()

