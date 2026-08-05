from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    IGNORE_INDEX,
    context_variant,
    load_config,
    load_he_metadata,
    load_label_spec,
    official_splits,
    remap_mask,
    resolve_data_path,
    save_json,
)
from modeling import metrics_from_confusion, update_confusion  # noqa: E402


PAPER_OVERALL_F1 = 0.79
PAPER_GROUPED_F1 = 0.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the authors' official IGNITE H&E inference masks on the untouched test split."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--predictions",
        default="data/official/inference/he",
        help="Directory containing the 139 official *_with_context.png masks.",
    )
    parser.add_argument(
        "--output",
        default="outputs/official_nnunet_baseline/test_results",
    )
    return parser.parse_args()


def build_group_lookup(label_spec: dict) -> tuple[list[str], np.ndarray]:
    group_names = ["Tumor", "Stroma + inflammation", "Macrophages", "Necrosis", "Rest"]
    lookup = np.full(label_spec["num_classes"], 4, dtype=np.uint8)
    name_to_id = {name: index for index, name in enumerate(label_spec["class_names"])}
    lookup[name_to_id["Tumor epithelium"]] = 0
    lookup[name_to_id["Stroma"]] = 1
    lookup[name_to_id["Inflammation"]] = 1
    lookup[name_to_id["Macrophages"]] = 2
    lookup[name_to_id["Necrotic tissue"]] = 3
    return group_names, lookup


def prediction_path(prediction_root: Path, roi_name: str) -> Path:
    stem = Path(str(roi_name)).stem
    candidates = [
        prediction_root / f"{stem}_with_context.png",
        prediction_root / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing official prediction for {roi_name!r}. Expected {candidates[0]}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    _, _, test_rows = official_splits(he, cfg["validation_fold"])

    prediction_root = Path(args.predictions)
    if not prediction_root.is_dir():
        raise SystemExit(
            f"Official prediction directory not found: {prediction_root}\n"
            "Extract official_inference.zip into data\\official first."
        )

    prediction_files = list(prediction_root.glob("*.png"))
    if len(prediction_files) != len(test_rows):
        raise SystemExit(
            f"Expected {len(test_rows)} official H&E masks, found {len(prediction_files)} in {prediction_root}"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = label_spec["num_classes"]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    group_names, group_lookup = build_group_lookup(label_spec)
    grouped_confusion = np.zeros((len(group_names), len(group_names)), dtype=np.int64)
    allowed_raw = set(int(value) for value in label_spec["raw_values"])

    for row in tqdm(
        test_rows.itertuples(index=False),
        total=len(test_rows),
        desc="Official nnU-Net holdout",
    ):
        target_path = resolve_data_path(cfg["data_root"], row.annotation_path)
        if bool(cfg.get("use_test_context", True)):
            target_path = context_variant(target_path)
        pred_path = prediction_path(prediction_root, row.name)

        with Image.open(target_path) as handle:
            raw_target = np.asarray(handle)
        with Image.open(pred_path) as handle:
            raw_prediction = np.asarray(handle)

        if raw_target.ndim == 3:
            raw_target = raw_target[..., 0]
        if raw_prediction.ndim == 3:
            raw_prediction = raw_prediction[..., 0]
        if raw_target.shape != raw_prediction.shape:
            raise ValueError(
                f"Shape mismatch for {row.name}: target={raw_target.shape}, prediction={raw_prediction.shape}"
            )

        unexpected = set(int(value) for value in np.unique(raw_prediction)) - allowed_raw
        if unexpected:
            raise ValueError(f"Unexpected prediction labels in {pred_path}: {sorted(unexpected)}")

        target = remap_mask(raw_target, label_spec)
        prediction = remap_mask(raw_prediction, label_spec)
        update_confusion(confusion, prediction, target, num_classes)

        valid = target != IGNORE_INDEX
        grouped_target = np.full(target.shape, IGNORE_INDEX, dtype=np.uint8)
        grouped_target[valid] = group_lookup[target[valid]]
        grouped_prediction = group_lookup[prediction]
        update_confusion(
            grouped_confusion,
            grouped_prediction,
            grouped_target,
            len(group_names),
        )

    metrics = metrics_from_confusion(confusion, label_spec["class_names"])
    metrics["til_grouped"] = metrics_from_confusion(grouped_confusion, group_names)
    metrics["test_rois"] = len(test_rows)
    metrics["prediction_source"] = str(prediction_root)
    metrics["paper_reference"] = {
        "overall_16_class_f1": PAPER_OVERALL_F1,
        "grouped_f1": PAPER_GROUPED_F1,
    }

    save_json(metrics, output_dir / "metrics.json")
    np.save(output_dir / "confusion_matrix.npy", confusion)
    np.save(output_dir / "til_grouped_confusion_matrix.npy", grouped_confusion)
    with open(output_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["per_class"])

    overall_f1 = metrics["macro_f1_dice"]
    grouped_f1 = metrics["til_grouped"]["macro_f1_dice"]
    print("=" * 72)
    print("OFFICIAL IGNITE nnU-Net UNTOUCHED HOLDOUT RESULTS")
    print("=" * 72)
    print(f"Test ROIs       : {len(test_rows)}")
    print(f"Pixel accuracy  : {metrics['pixel_accuracy']:.4f}")
    print(f"Macro F1/Dice   : {overall_f1:.4f}")
    print(f"Mean IoU        : {metrics['mean_iou']:.4f}")
    print(f"Grouped F1      : {grouped_f1:.4f}")
    print(f"Paper rounded F1: {PAPER_OVERALL_F1:.2f} overall | {PAPER_GROUPED_F1:.2f} grouped")
    print(f"Results saved   : {output_dir}")


if __name__ == "__main__":
    main()
