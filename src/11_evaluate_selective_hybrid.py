from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
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
    seed_everything,
)
from modeling import metrics_from_confusion, update_confusion  # noqa: E402


# This is intentionally fixed before holdout evaluation. Do not tune it on the
# 139 test masks: doing that would leak test labels into method selection.
CONFIDENCE_THRESHOLD = 0.90
REQUIRED_FOLD_VOTES = 5


def load_overlap_module():
    module_path = Path(__file__).resolve().parent / "08_evaluate_ensemble_overlap.py"
    if not module_path.exists():
        raise SystemExit(
            f"Missing {module_path.name} in src. It is the five-fold ensemble script used earlier."
        )
    spec = importlib.util.spec_from_file_location("ensemble_overlap", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "sliding_starts",
        "pad_to_minimum",
        "blend_window",
        "build_group_lookup",
        "load_ensemble",
        "infer_batches_with_model",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise SystemExit(
            f"{module_path.name} is incompatible; missing functions: {', '.join(missing)}"
        )
    return module


def locate_official_prediction(root: Path, roi_name: str) -> Path:
    stem = Path(str(roi_name)).stem
    for candidate in (root / f"{stem}_with_context.png", root / f"{stem}.png"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Official prediction missing for {roi_name!r} in {root}")


def update_both_confusions(
    confusion: np.ndarray,
    grouped_confusion: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    group_lookup: np.ndarray,
    num_groups: int,
) -> None:
    update_confusion(confusion, prediction, target, num_classes)
    valid = target != IGNORE_INDEX
    grouped_target = np.full(target.shape, IGNORE_INDEX, dtype=np.uint8)
    grouped_target[valid] = group_lookup[target[valid]]
    grouped_prediction = group_lookup[prediction]
    update_confusion(
        grouped_confusion,
        grouped_prediction,
        grouped_target,
        num_groups,
    )


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

    overlap = load_overlap_module()
    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    _, _, test_rows = official_splits(he, "fold0")

    official_root = Path("data/official/inference/he")
    official_files = list(official_root.glob("*.png"))
    if len(official_files) != len(test_rows):
        raise SystemExit(
            f"Expected {len(test_rows)} official masks in {official_root}, found {len(official_files)}"
        )

    result_dir = (
        Path(cfg["output_root"])
        / "upernet_nnunet_selective_hybrid"
        / "test_results_overlap50"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    models, checkpoint_paths, checkpoint_epochs, checkpoint_scores, execution_mode = (
        overlap.load_ensemble(cfg, label_spec, device)
    )
    print(f"Hybrid execution mode: {execution_mode}")
    print(
        "Prespecified rule: use official nnU-Net by default; override only when "
        f"all {REQUIRED_FOLD_VOTES}/5 UPerNet folds agree and mean confidence >= "
        f"{CONFIDENCE_THRESHOLD:.2f}."
    )

    num_classes = label_spec["num_classes"]
    group_names, group_lookup = overlap.build_group_lookup(label_spec)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    grouped_confusion = np.zeros((len(group_names), len(group_names)), dtype=np.int64)
    official_confusion = np.zeros_like(confusion)
    official_grouped_confusion = np.zeros_like(grouped_confusion)
    upernet_confusion = np.zeros_like(confusion)
    upernet_grouped_confusion = np.zeros_like(grouped_confusion)

    total_valid_pixels = 0
    total_disagreements = 0
    total_overrides = 0
    state_path = result_dir / "progress_state.npz"
    start_row = 0
    if state_path.exists():
        state = np.load(state_path)
        start_row = int(state["next_row"])
        confusion = state["confusion"].astype(np.int64)
        grouped_confusion = state["grouped_confusion"].astype(np.int64)
        official_confusion = state["official_confusion"].astype(np.int64)
        official_grouped_confusion = state["official_grouped_confusion"].astype(np.int64)
        upernet_confusion = state["upernet_confusion"].astype(np.int64)
        upernet_grouped_confusion = state["upernet_grouped_confusion"].astype(np.int64)
        total_valid_pixels = int(state["total_valid_pixels"])
        total_disagreements = int(state["total_disagreements"])
        total_overrides = int(state["total_overrides"])
        print(f"Resuming saved hybrid evaluation at ROI {start_row + 1}/{len(test_rows)}")

    tile_size = int(cfg["patch_size"])
    stride = tile_size // 2
    eval_batch_size = int(cfg.get("evaluation_batch_size", 1))
    window = overlap.blend_window(tile_size)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        remaining = test_rows.iloc[start_row:]
        progress = tqdm(
            remaining.itertuples(index=False),
            total=len(test_rows),
            initial=start_row,
            desc="Selective UPerNet + nnU-Net hybrid",
        )
        for row_index, row in enumerate(progress, start=start_row):
            image_path = resolve_data_path(cfg["data_root"], row.image_path)
            target_path = resolve_data_path(cfg["data_root"], row.annotation_path)
            if bool(cfg.get("use_test_context", True)):
                image_path = context_variant(image_path)
                target_path = context_variant(target_path)
            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"))
            with Image.open(target_path) as handle:
                raw_target = np.asarray(handle)
            with Image.open(locate_official_prediction(official_root, row.name)) as handle:
                raw_official = np.asarray(handle)
            if raw_target.ndim == 3:
                raw_target = raw_target[..., 0]
            if raw_official.ndim == 3:
                raw_official = raw_official[..., 0]
            target = remap_mask(raw_target, label_spec)
            official_prediction = remap_mask(raw_official, label_spec)
            if target.shape != official_prediction.shape:
                raise ValueError(
                    f"Shape mismatch for {row.name}: target={target.shape}, official={official_prediction.shape}"
                )

            image_pad, target_pad, original_h, original_w = overlap.pad_to_minimum(
                image, target, tile_size
            )
            padded_h, padded_w = target_pad.shape
            tops = overlap.sliding_starts(padded_h, tile_size, stride)
            lefts = overlap.sliding_starts(padded_w, tile_size, stride)
            coordinates = [(top, left) for top in tops for left in lefts]
            weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
            for top, left in coordinates:
                weight_sum[top : top + tile_size, left : left + tile_size] += window

            ensemble_probability = np.zeros(
                (num_classes, padded_h, padded_w), dtype=np.float32
            )
            fold_votes = np.zeros((num_classes, padded_h, padded_w), dtype=np.uint8)
            for fold_index, model in enumerate(models):
                if execution_mode != "all_models_gpu_resident":
                    progress.set_postfix_str(f"fold {fold_index + 1}/5")
                    model.to(device)
                model_probability = np.zeros_like(ensemble_probability)
                overlap.infer_batches_with_model(
                    model,
                    image_pad,
                    coordinates,
                    tile_size,
                    eval_batch_size,
                    device,
                    window,
                    model_probability,
                )
                model_probability /= np.maximum(weight_sum[None, :, :], 1e-8)
                model_prediction = model_probability.argmax(axis=0)
                for class_id in range(num_classes):
                    fold_votes[class_id] += model_prediction == class_id
                ensemble_probability += model_probability / len(models)
                del model_probability, model_prediction
                if execution_mode != "all_models_gpu_resident":
                    model.to("cpu")
                    torch.cuda.empty_cache()

            upernet_prediction = ensemble_probability.argmax(axis=0).astype(np.uint8)
            upernet_confidence = np.take_along_axis(
                ensemble_probability,
                upernet_prediction[None, :, :],
                axis=0,
            )[0]
            agreeing_votes = np.take_along_axis(
                fold_votes,
                upernet_prediction[None, :, :],
                axis=0,
            )[0]

            upernet_prediction = upernet_prediction[:original_h, :original_w]
            upernet_confidence = upernet_confidence[:original_h, :original_w]
            agreeing_votes = agreeing_votes[:original_h, :original_w]
            target = target[:original_h, :original_w]
            official_prediction = official_prediction[:original_h, :original_w]

            disagreement = upernet_prediction != official_prediction
            override = (
                disagreement
                & (upernet_confidence >= CONFIDENCE_THRESHOLD)
                & (agreeing_votes >= REQUIRED_FOLD_VOTES)
            )
            hybrid_prediction = official_prediction.copy()
            hybrid_prediction[override] = upernet_prediction[override]

            update_both_confusions(
                confusion,
                grouped_confusion,
                hybrid_prediction,
                target,
                num_classes,
                group_lookup,
                len(group_names),
            )
            update_both_confusions(
                official_confusion,
                official_grouped_confusion,
                official_prediction,
                target,
                num_classes,
                group_lookup,
                len(group_names),
            )
            update_both_confusions(
                upernet_confusion,
                upernet_grouped_confusion,
                upernet_prediction,
                target,
                num_classes,
                group_lookup,
                len(group_names),
            )

            valid = target != IGNORE_INDEX
            total_valid_pixels += int(valid.sum())
            total_disagreements += int((disagreement & valid).sum())
            total_overrides += int((override & valid).sum())

            np.savez_compressed(
                state_path,
                next_row=np.asarray(row_index + 1),
                confusion=confusion,
                grouped_confusion=grouped_confusion,
                official_confusion=official_confusion,
                official_grouped_confusion=official_grouped_confusion,
                upernet_confusion=upernet_confusion,
                upernet_grouped_confusion=upernet_grouped_confusion,
                total_valid_pixels=np.asarray(total_valid_pixels),
                total_disagreements=np.asarray(total_disagreements),
                total_overrides=np.asarray(total_overrides),
            )
            del ensemble_probability, fold_votes, weight_sum

    metrics = metrics_from_confusion(confusion, label_spec["class_names"])
    metrics["til_grouped"] = metrics_from_confusion(grouped_confusion, group_names)
    official_metrics = metrics_from_confusion(official_confusion, label_spec["class_names"])
    official_metrics["til_grouped"] = metrics_from_confusion(
        official_grouped_confusion, group_names
    )
    upernet_metrics = metrics_from_confusion(upernet_confusion, label_spec["class_names"])
    upernet_metrics["til_grouped"] = metrics_from_confusion(
        upernet_grouped_confusion, group_names
    )

    metrics["method"] = "official nnU-Net with unanimous high-confidence five-fold UPerNet corrections"
    metrics["confidence_threshold"] = CONFIDENCE_THRESHOLD
    metrics["required_fold_votes"] = REQUIRED_FOLD_VOTES
    metrics["test_rois"] = len(test_rows)
    metrics["tile_size"] = tile_size
    metrics["stride"] = stride
    metrics["fold_checkpoints"] = [str(path) for path in checkpoint_paths]
    metrics["checkpoint_epochs"] = checkpoint_epochs
    metrics["checkpoint_validation_scores"] = checkpoint_scores
    metrics["execution_mode"] = execution_mode
    metrics["peak_allocated_vram_gib"] = float(torch.cuda.max_memory_allocated() / 2**30)
    metrics["valid_pixels"] = total_valid_pixels
    metrics["disagreement_pixels"] = total_disagreements
    metrics["override_pixels"] = total_overrides
    metrics["override_fraction"] = total_overrides / max(total_valid_pixels, 1)
    metrics["official_baseline_recomputed"] = official_metrics
    metrics["upernet_ensemble_recomputed"] = upernet_metrics
    metrics["change_vs_official"] = {
        "pixel_accuracy": metrics["pixel_accuracy"] - official_metrics["pixel_accuracy"],
        "macro_f1_dice": metrics["macro_f1_dice"] - official_metrics["macro_f1_dice"],
        "mean_iou": metrics["mean_iou"] - official_metrics["mean_iou"],
        "grouped_f1_dice": metrics["til_grouped"]["macro_f1_dice"]
        - official_metrics["til_grouped"]["macro_f1_dice"],
    }

    save_json(metrics, result_dir / "metrics.json")
    np.save(result_dir / "confusion_matrix.npy", confusion)
    np.save(result_dir / "til_grouped_confusion_matrix.npy", grouped_confusion)
    with open(result_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["per_class"])

    change = metrics["change_vs_official"]
    print("=" * 76)
    print("SELECTIVE UPerNet + OFFICIAL nnU-Net UNTOUCHED HOLDOUT RESULTS")
    print("=" * 76)
    print(f"Pixel accuracy : {metrics['pixel_accuracy']:.4f} ({change['pixel_accuracy']:+.4f})")
    print(f"Macro F1/Dice  : {metrics['macro_f1_dice']:.4f} ({change['macro_f1_dice']:+.4f})")
    print(f"Mean IoU       : {metrics['mean_iou']:.4f} ({change['mean_iou']:+.4f})")
    print(f"Grouped F1     : {metrics['til_grouped']['macro_f1_dice']:.4f} ({change['grouped_f1_dice']:+.4f})")
    print(f"Overrides      : {total_overrides:,} pixels ({metrics['override_fraction'] * 100:.4f}%)")
    print(f"Peak VRAM GiB  : {metrics['peak_allocated_vram_gib']:.2f}")
    print(f"Results saved  : {result_dir}")


if __name__ == "__main__":
    main()
