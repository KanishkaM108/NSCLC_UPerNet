from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def pad_to_tiles(image: np.ndarray, mask: np.ndarray, tile: int):
    height, width = mask.shape
    pad_h = (tile - height % tile) % tile
    pad_w = (tile - width % tile) % tile
    image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_pad = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=IGNORE_INDEX)
    return image_pad, mask_pad, height, width


def color_palette(num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    return np.asarray([(np.asarray(cmap(i % 20)[:3]) * 255).astype(np.uint8) for i in range(num_classes)])


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
        target_rgb = np.asarray(Image.fromarray(target_rgb).resize(new_size, Image.Resampling.NEAREST))
        pred_rgb = np.asarray(Image.fromarray(pred_rgb).resize(new_size, Image.Resampling.NEAREST))
        overlay = np.asarray(Image.fromarray(overlay).resize(new_size, Image.Resampling.LANCZOS))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, panel, title in zip(axes, [image, target_rgb, pred_rgb, overlay], ["H&E", "Ground truth", "Prediction", "Overlay"]):
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
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Row-normalized test confusion matrix")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=190)
    plt.close(fig)

    values = [row["f1_dice"] for row in metrics["per_class"]]
    fig, axis = plt.subplots(figsize=(10, 6))
    order = np.argsort(values)
    axis.barh(np.asarray(class_names)[order], np.asarray(values)[order], color="tab:blue")
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 / Dice")
    axis.set_title("Per-class holdout test performance")
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
    result_dir = out_dir / "test_results"
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
    eval_batch_size = int(cfg.get("evaluation_batch_size", 1))
    visual_limit = int(cfg.get("qualitative_examples", 6))

    with torch.inference_mode():
        for row_index, row in enumerate(tqdm(test_rows.itertuples(index=False), total=len(test_rows), desc="Holdout test")):
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
            image_pad, target_pad, original_h, original_w = pad_to_tiles(image, target, tile_size)
            padded_h, padded_w = target_pad.shape
            full_prediction = np.zeros((padded_h, padded_w), dtype=np.uint8) if row_index < visual_limit else None

            coordinates = [(top, left) for top in range(0, padded_h, tile_size) for left in range(0, padded_w, tile_size)]
            for start in range(0, len(coordinates), eval_batch_size):
                batch_coordinates = coordinates[start : start + eval_batch_size]
                batch = []
                for top, left in batch_coordinates:
                    tile = image_pad[top : top + tile_size, left : left + tile_size]
                    tile = normalize_image(tile).transpose(2, 0, 1)
                    batch.append(tile)
                inputs = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).float().to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(pixel_values=inputs).logits
                    logits = torch.nn.functional.interpolate(
                        logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False
                    )
                predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                for (top, left), prediction in zip(batch_coordinates, predictions):
                    target_tile = target_pad[top : top + tile_size, left : left + tile_size]
                    update_confusion(confusion, prediction, target_tile, num_classes)
                    valid = target_tile != IGNORE_INDEX
                    grouped_target = np.full(target_tile.shape, IGNORE_INDEX, dtype=np.uint8)
                    grouped_target[valid] = group_lookup[target_tile[valid]]
                    grouped_prediction = group_lookup[prediction]
                    update_confusion(
                        grouped_confusion,
                        grouped_prediction,
                        grouped_target,
                        len(grouped_names),
                    )
                    if full_prediction is not None:
                        full_prediction[top : top + tile_size, left : left + tile_size] = prediction

            if full_prediction is not None:
                save_qualitative(
                    image,
                    target[:original_h, :original_w],
                    full_prediction[:original_h, :original_w],
                    label_spec["class_names"],
                    qualitative_dir / f"{row.name}.png",
                )

    metrics = metrics_from_confusion(confusion, label_spec["class_names"])
    metrics["til_grouped"] = metrics_from_confusion(grouped_confusion, grouped_names)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["test_rois"] = len(test_rows)
    metrics["tile_size"] = tile_size
    save_json(metrics, result_dir / "metrics.json")
    np.save(result_dir / "confusion_matrix.npy", confusion)
    np.save(result_dir / "til_grouped_confusion_matrix.npy", grouped_confusion)
    with open(result_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    save_metric_figures(confusion, metrics, label_spec["class_names"], result_dir)

    print("=" * 68)
    print("UNTOUCHED HOLDOUT TEST RESULTS")
    print("=" * 68)
    print(f"Pixel accuracy : {metrics['pixel_accuracy']:.4f}")
    print(f"Macro F1/Dice  : {metrics['macro_f1_dice']:.4f}")
    print(f"Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"Grouped F1     : {metrics['til_grouped']['macro_f1_dice']:.4f}")
    print(f"Results saved  : {result_dir}")


if __name__ == "__main__":
    main()
