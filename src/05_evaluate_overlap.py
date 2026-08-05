from __future__ import annotations

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

from common import (
    IGNORE_INDEX,
    context_variant,
    experiment_dir,
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
from modeling import build_upernet, metrics_from_confusion, update_confusion


def sliding_starts(length: int, tile: int, stride: int) -> list[int]:
    """Return starts that cover the whole axis and always include the far edge."""
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
    """Positive Gaussian window: centre predictions dominate overlapping edges."""
    axis = np.arange(tile, dtype=np.float32) - (tile - 1) / 2.0
    sigma = tile * 0.22
    one_dimensional = np.exp(-0.5 * (axis / sigma) ** 2)
    window = np.outer(one_dimensional, one_dimensional)
    window /= window.max()
    return np.maximum(window, 0.01).astype(np.float32)


def color_palette(num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    return np.asarray(
        [(np.asarray(cmap(i % 20)[:3]) * 255).astype(np.uint8) for i in range(num_classes)]
    )


def save_qualitative(image, target, prediction, class_names, path):
    palette = color_palette(len(class_names))
    target_rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    pred_rgb = palette[prediction]
    valid = target != IGNORE_INDEX
    target_rgb[valid] = palette[target[valid]]
    overlay = (0.55 * image + 0.45 * pred_rgb).astype(np.uint8)
    max_side = 1400
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1.0:
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = np.asarray(Image.fromarray(image).resize(new_size, Image.Resampling.LANCZOS))
        target_rgb = np.asarray(
            Image.fromarray(target_rgb).resize(new_size, Image.Resampling.NEAREST)
        )
        pred_rgb = np.asarray(Image.fromarray(pred_rgb).resize(new_size, Image.Resampling.NEAREST))
        overlay = np.asarray(Image.fromarray(overlay).resize(new_size, Image.Resampling.LANCZOS))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [image, target_rgb, pred_rgb, overlay]
    titles = ["H&E", "Ground truth", "Overlap-blended prediction", "Overlay"]
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_metric_figures(confusion, metrics, class_names, output_dir):
    normalized = confusion.astype(np.float64)
    normalized /= np.maximum(1.0, normalized.sum(axis=1, keepdims=True))
    fig, axis = plt.subplots(figsize=(12, 10))
    matrix_image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Row-normalized overlap-blended test confusion matrix")
    fig.colorbar(matrix_image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=190)
    plt.close(fig)

    values = [row["f1_dice"] for row in metrics["per_class"]]
    fig, axis = plt.subplots(figsize=(10, 6))
    order = np.argsort(values)
    axis.barh(np.asarray(class_names)[order], np.asarray(values)[order], color="tab:green")
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 / Dice")
    axis.set_title("Per-class overlap-blended holdout performance")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_f1.png", dpi=190)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    _, _, test_rows = official_splits(he, cfg["validation_fold"])
    out_dir = experiment_dir(cfg)
    checkpoint_path = out_dir / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}. Train first.")

    result_dir = out_dir / "test_results_overlap50"
    qualitative_dir = result_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = build_upernet(label_spec, cfg["pretrained_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    num_classes = label_spec["num_classes"]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    grouped_names = ["Tumor", "Stroma + inflammation", "Macrophages", "Necrosis", "Rest"]
    group_lookup = np.full(num_classes, 4, dtype=np.uint8)
    class_name_to_id = {name: index for index, name in enumerate(label_spec["class_names"])}
    group_lookup[class_name_to_id["Tumor epithelium"]] = 0
    group_lookup[class_name_to_id["Stroma"]] = 1
    group_lookup[class_name_to_id["Inflammation"]] = 1
    group_lookup[class_name_to_id["Macrophages"]] = 2
    group_lookup[class_name_to_id["Necrotic tissue"]] = 3
    grouped_confusion = np.zeros((len(grouped_names), len(grouped_names)), dtype=np.int64)

    tile_size = int(cfg["patch_size"])
    stride = tile_size // 2
    eval_batch_size = int(cfg.get("evaluation_batch_size", 1))
    visual_limit = int(cfg.get("qualitative_examples", 6))
    window = blend_window(tile_size)

    with torch.inference_mode():
        progress = tqdm(test_rows.itertuples(index=False), total=len(test_rows), desc="Overlap holdout")
        for row_index, row in enumerate(progress):
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
            target = remap_mask(raw_mask, label_spec)
            image_pad, target_pad, original_h, original_w = pad_to_minimum(
                image, target, tile_size
            )
            padded_h, padded_w = target_pad.shape

            probability_sum = np.zeros(
                (num_classes, padded_h, padded_w), dtype=np.float32
            )
            weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
            tops = sliding_starts(padded_h, tile_size, stride)
            lefts = sliding_starts(padded_w, tile_size, stride)
            coordinates = [(top, left) for top in tops for left in lefts]

            for start in range(0, len(coordinates), eval_batch_size):
                batch_coordinates = coordinates[start : start + eval_batch_size]
                batch = []
                for top, left in batch_coordinates:
                    tile = image_pad[top : top + tile_size, left : left + tile_size]
                    batch.append(normalize_image(tile).transpose(2, 0, 1))
                inputs = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).float().to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(pixel_values=inputs).logits
                    logits = F.interpolate(
                        logits,
                        size=(tile_size, tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probabilities = torch.softmax(logits, dim=1)
                probabilities = probabilities.float().cpu().numpy()
                for (top, left), probability in zip(batch_coordinates, probabilities):
                    probability_sum[
                        :, top : top + tile_size, left : left + tile_size
                    ] += probability * window[None, :, :]
                    weight_sum[top : top + tile_size, left : left + tile_size] += window

            probability_sum /= np.maximum(weight_sum[None, :, :], 1e-8)
            prediction = probability_sum.argmax(axis=0).astype(np.uint8)
            prediction = prediction[:original_h, :original_w]
            target = target[:original_h, :original_w]

            update_confusion(confusion, prediction, target, num_classes)
            valid = target != IGNORE_INDEX
            grouped_target = np.full(target.shape, IGNORE_INDEX, dtype=np.uint8)
            grouped_target[valid] = group_lookup[target[valid]]
            grouped_prediction = group_lookup[prediction]
            update_confusion(
                grouped_confusion,
                grouped_prediction,
                grouped_target,
                len(grouped_names),
            )

            if row_index < visual_limit:
                save_qualitative(
                    image[:original_h, :original_w],
                    target,
                    prediction,
                    label_spec["class_names"],
                    qualitative_dir / f"{row.name}.png",
                )

            del probability_sum, weight_sum

    metrics = metrics_from_confusion(confusion, label_spec["class_names"])
    metrics["til_grouped"] = metrics_from_confusion(grouped_confusion, grouped_names)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["test_rois"] = len(test_rows)
    metrics["tile_size"] = tile_size
    metrics["stride"] = stride
    metrics["overlap_fraction"] = 0.5
    metrics["blend_window"] = "gaussian"

    old_metrics_path = out_dir / "test_results" / "metrics.json"
    if old_metrics_path.exists():
        with open(old_metrics_path, "r", encoding="utf-8") as handle:
            old_metrics = json.load(handle)
        metrics["change_vs_nonoverlap"] = {
            "pixel_accuracy": metrics["pixel_accuracy"] - old_metrics["pixel_accuracy"],
            "macro_f1_dice": metrics["macro_f1_dice"] - old_metrics["macro_f1_dice"],
            "mean_iou": metrics["mean_iou"] - old_metrics["mean_iou"],
            "grouped_f1_dice": metrics["til_grouped"]["macro_f1_dice"]
            - old_metrics["til_grouped"]["macro_f1_dice"],
        }

    save_json(metrics, result_dir / "metrics.json")
    np.save(result_dir / "confusion_matrix.npy", confusion)
    np.save(result_dir / "til_grouped_confusion_matrix.npy", grouped_confusion)
    with open(result_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    save_metric_figures(confusion, metrics, label_spec["class_names"], result_dir)

    print("=" * 68)
    print("OVERLAP-BLENDED UNTOUCHED HOLDOUT TEST RESULTS")
    print("=" * 68)
    print(f"Pixel accuracy : {metrics['pixel_accuracy']:.4f}")
    print(f"Macro F1/Dice  : {metrics['macro_f1_dice']:.4f}")
    print(f"Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"Grouped F1     : {metrics['til_grouped']['macro_f1_dice']:.4f}")
    if "change_vs_nonoverlap" in metrics:
        change = metrics["change_vs_nonoverlap"]
        print(f"Macro F1 change: {change['macro_f1_dice']:+.4f}")
        print(f"Grouped change : {change['grouped_f1_dice']:+.4f}")
    print(f"Results saved  : {result_dir}")


if __name__ == "__main__":
    main()
