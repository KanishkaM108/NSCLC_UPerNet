from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    IGNORE_INDEX,
    context_variant,
    load_config,
    load_he_metadata,
    load_label_spec,
    normalize_image,
    official_splits,
    remap_mask,
    resolve_data_path,
    save_json,
    seed_everything,
)
from grouped_dataset import GROUP_NAMES, build_group_lookup, to_grouped_mask  # noqa: E402
from modeling import build_upernet, metrics_from_confusion, update_confusion  # noqa: E402


def grouped_label_spec() -> dict:
    return {"class_names": GROUP_NAMES, "num_classes": len(GROUP_NAMES)}


def sliding_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def pad_to_minimum(image: np.ndarray, target: np.ndarray, tile: int):
    height, width = target.shape
    pad_h = max(0, tile - height)
    pad_w = max(0, tile - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        target = np.pad(
            target,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=IGNORE_INDEX,
        )
    return image, target, height, width


def blend_window(tile: int) -> np.ndarray:
    axis = np.arange(tile, dtype=np.float32) - (tile - 1) / 2.0
    one_dimensional = np.exp(-0.5 * (axis / (tile * 0.22)) ** 2)
    window = np.outer(one_dimensional, one_dimensional)
    window /= window.max()
    return np.maximum(window, 0.01).astype(np.float32)


def transform_input(inputs: torch.Tensor, transform_id: int) -> torch.Tensor:
    turns = transform_id % 4
    flipped = transform_id >= 4
    transformed = torch.rot90(inputs, turns, dims=(-2, -1))
    if flipped:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed.contiguous()


def invert_output(probabilities: torch.Tensor, transform_id: int) -> torch.Tensor:
    turns = transform_id % 4
    flipped = transform_id >= 4
    if flipped:
        probabilities = torch.flip(probabilities, dims=(-1,))
    return torch.rot90(probabilities, -turns, dims=(-2, -1)).contiguous()


def infer_model(
    model,
    image_pad,
    coordinates,
    tile_size,
    batch_size,
    device,
    window,
    tta_transforms,
):
    probability_sum = np.zeros(
        (len(GROUP_NAMES), image_pad.shape[0], image_pad.shape[1]),
        dtype=np.float32,
    )
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        batch = [
            normalize_image(
                image_pad[top : top + tile_size, left : left + tile_size]
            ).transpose(2, 0, 1)
            for top, left in batch_coordinates
        ]
        inputs = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).float().to(device)
        tta_sum = None
        for transform_id in range(tta_transforms):
            transformed = transform_input(inputs, transform_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=transformed).logits
                logits = F.interpolate(
                    logits,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
                probabilities = torch.softmax(logits, dim=1)
            probabilities = invert_output(probabilities, transform_id).float()
            tta_sum = probabilities if tta_sum is None else tta_sum + probabilities
        probabilities_np = (tta_sum / float(tta_transforms)).cpu().numpy()
        for (top, left), probability in zip(batch_coordinates, probabilities_np):
            probability_sum[:, top : top + tile_size, left : left + tile_size] += (
                probability * window[None, :, :]
            )
    return probability_sum


def load_models(cfg, device):
    paths = [
        Path(cfg["output_root"]) / f"upernet_swin_tiny_grouped_fold{index}" / "best.pt"
        for index in range(5)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing grouped checkpoint(s):\n  " + "\n  ".join(missing))
    models = []
    scores = []
    for fold, path in enumerate(paths):
        print(f"Loading grouped fold {fold}: {path}")
        model = build_upernet(grouped_label_spec(), cfg["pretrained_checkpoint"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
        scores.append(max(1e-6, float(checkpoint.get("best_val_joint_score", 1.0))))
    weights = np.asarray(scores, dtype=np.float64)
    weights /= weights.sum()

    execution_mode = "all_models_gpu_resident"
    try:
        for model in models:
            model.to(device)
    except torch.cuda.OutOfMemoryError:
        for model in models:
            model.to("cpu")
        torch.cuda.empty_cache()
        execution_mode = "per_roi_cpu_offload"
    print(f"Validation-derived fold weights: {np.round(weights, 4).tolist()}")
    print(f"Execution mode: {execution_mode}")
    return models, paths, scores, weights, execution_mode


def save_figures(confusion: np.ndarray, metrics: dict, result_dir: Path) -> None:
    normalized = confusion.astype(np.float64)
    normalized /= np.maximum(1.0, normalized.sum(axis=1, keepdims=True))
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(GROUP_NAMES)), GROUP_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(GROUP_NAMES)), GROUP_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Grouped UPerNet ensemble: normalized holdout confusion")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrix.png", dpi=190)
    plt.close(fig)

    values = [row["f1_dice"] for row in metrics["per_class"]]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(GROUP_NAMES, values, color="tab:blue")
    axis.axvline(0.85, color="tab:red", linestyle="--", label="85% target")
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 / Dice")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "per_class_f1.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_grouped_fold0.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

    label_spec_16 = load_label_spec(cfg["label_map_json"])
    group_lookup = build_group_lookup(label_spec_16)
    he = load_he_metadata(cfg["metadata_csv"])
    _, _, test_rows = official_splits(he, "fold0")
    result_dir = (
        Path(cfg["output_root"])
        / "upernet_swin_tiny_grouped_5fold_ensemble"
        / "test_results_overlap50_d4"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    models, checkpoint_paths, checkpoint_scores, fold_weights, execution_mode = load_models(
        cfg, device
    )
    tile_size = int(cfg["patch_size"])
    stride = tile_size // 2
    batch_size = int(cfg.get("evaluation_batch_size", 1))
    tta_transforms = int(cfg.get("tta_transforms", 8))
    if tta_transforms not in (1, 4, 8):
        raise SystemExit("tta_transforms must be 1, 4, or 8")
    window = blend_window(tile_size)
    confusion = np.zeros((len(GROUP_NAMES), len(GROUP_NAMES)), dtype=np.int64)
    per_roi_rows = []

    state_path = result_dir / "progress_state.npz"
    start_row = 0
    if state_path.exists():
        state = np.load(state_path)
        start_row = int(state["next_row"])
        confusion = state["confusion"].astype(np.int64)
        print(f"Resuming at ROI {start_row + 1}/{len(test_rows)}")

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        progress = tqdm(
            test_rows.iloc[start_row:].itertuples(index=False),
            total=len(test_rows),
            initial=start_row,
            desc="Grouped 5-fold D4 holdout",
        )
        for row_index, row in enumerate(progress, start=start_row):
            image_path = resolve_data_path(cfg["data_root"], row.image_path)
            mask_path = resolve_data_path(cfg["data_root"], row.annotation_path)
            if bool(cfg.get("use_test_context", True)):
                image_path = context_variant(image_path)
                mask_path = context_variant(mask_path)
            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"))
            with Image.open(mask_path) as handle:
                raw_mask = np.asarray(handle)
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[..., 0]
            target = to_grouped_mask(remap_mask(raw_mask, label_spec_16), group_lookup)
            image_pad, target_pad, original_h, original_w = pad_to_minimum(
                image, target, tile_size
            )
            tops = sliding_starts(target_pad.shape[0], tile_size, stride)
            lefts = sliding_starts(target_pad.shape[1], tile_size, stride)
            coordinates = [(top, left) for top in tops for left in lefts]
            weight_sum = np.zeros(target_pad.shape, dtype=np.float32)
            for top, left in coordinates:
                weight_sum[top : top + tile_size, left : left + tile_size] += window

            ensemble = np.zeros(
                (len(GROUP_NAMES), *target_pad.shape), dtype=np.float32
            )
            for fold, (model, fold_weight) in enumerate(zip(models, fold_weights)):
                if execution_mode != "all_models_gpu_resident":
                    progress.set_postfix_str(f"fold {fold + 1}/5")
                    model.to(device)
                model_sum = infer_model(
                    model,
                    image_pad,
                    coordinates,
                    tile_size,
                    batch_size,
                    device,
                    window,
                    tta_transforms,
                )
                model_sum /= np.maximum(weight_sum[None, :, :], 1e-8)
                ensemble += float(fold_weight) * model_sum
                del model_sum
                if execution_mode != "all_models_gpu_resident":
                    model.to("cpu")
                    torch.cuda.empty_cache()

            prediction = ensemble.argmax(axis=0).astype(np.uint8)
            prediction = prediction[:original_h, :original_w]
            target = target[:original_h, :original_w]
            roi_confusion = np.zeros_like(confusion)
            update_confusion(
                roi_confusion, prediction, target, len(GROUP_NAMES)
            )
            confusion += roi_confusion
            roi_metrics = metrics_from_confusion(roi_confusion, GROUP_NAMES)
            per_roi_rows.append(
                {
                    "roi": str(row.name),
                    "pixel_accuracy": roi_metrics["pixel_accuracy"],
                    "macro_f1_dice": roi_metrics["macro_f1_dice"],
                    "mean_iou": roi_metrics["mean_iou"],
                }
            )
            np.savez_compressed(
                state_path,
                next_row=np.asarray(row_index + 1),
                confusion=confusion,
            )
            del ensemble, weight_sum

    metrics = metrics_from_confusion(confusion, GROUP_NAMES)
    support = np.asarray([row["support_pixels"] for row in metrics["per_class"]], dtype=np.float64)
    f1 = np.asarray([row["f1_dice"] for row in metrics["per_class"]], dtype=np.float64)
    metrics["weighted_f1_dice"] = float(np.sum(support * f1) / np.maximum(1.0, support.sum()))
    metrics["test_rois"] = len(test_rows)
    metrics["checkpoint_paths"] = [str(path) for path in checkpoint_paths]
    metrics["validation_joint_scores"] = checkpoint_scores
    metrics["ensemble_weights"] = fold_weights.tolist()
    metrics["tta_transforms"] = tta_transforms
    metrics["overlap_fraction"] = 0.5
    metrics["peak_vram_gib"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    metrics["target_threshold"] = 0.85
    metrics["passed_accuracy_85"] = metrics["pixel_accuracy"] >= 0.85
    metrics["passed_macro_f1_85"] = metrics["macro_f1_dice"] >= 0.85
    metrics["passed_both_85"] = bool(
        metrics["passed_accuracy_85"] and metrics["passed_macro_f1_85"]
    )

    np.save(result_dir / "confusion.npy", confusion)
    save_json(metrics, result_dir / "metrics.json")
    with open(result_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    with open(result_dir / "per_roi_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["roi", "pixel_accuracy", "macro_f1_dice", "mean_iou"])
        writer.writeheader()
        writer.writerows(per_roi_rows)
    save_figures(confusion, metrics, result_dir)
    if state_path.exists():
        state_path.unlink()

    print("=" * 72)
    print("DIRECT FIVE-CLASS GROUPED UPerNet ENSEMBLE — UNTOUCHED HOLDOUT")
    print("=" * 72)
    print(f"Test ROIs       : {len(test_rows)}")
    print(f"Pixel accuracy  : {metrics['pixel_accuracy']:.4f}")
    print(f"Macro F1/Dice   : {metrics['macro_f1_dice']:.4f}")
    print(f"Weighted F1     : {metrics['weighted_f1_dice']:.4f}")
    print(f"Mean IoU        : {metrics['mean_iou']:.4f}")
    print(f"85% BOTH target : {'PASSED' if metrics['passed_both_85'] else 'NOT YET PASSED'}")
    print(f"Results saved   : {result_dir}")


if __name__ == "__main__":
    main()
