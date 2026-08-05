from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    IGNORE_INDEX,
    load_config,
    load_he_metadata,
    load_label_spec,
    official_splits,
    save_json,
    seed_everything,
)
from grouped_dataset import GROUP_NAMES, GroupedPatchDataset  # noqa: E402
from modeling import (  # noqa: E402
    build_upernet,
    cosine_warmup_lambda,
    freeze_batchnorm_statistics,
    metrics_from_confusion,
    multiclass_dice_loss,
    resize_logits,
    update_confusion,
)


def grouped_label_spec() -> dict:
    return {"class_names": GROUP_NAMES, "num_classes": len(GROUP_NAMES)}


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Class-sensitive Tversky: restrain stroma FP and rare-class FN."""
    logits = resize_logits(logits, target)
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float

    alpha = logits.new_tensor([0.35, 0.62, 0.30, 0.25, 0.45])
    beta = logits.new_tensor([0.65, 0.38, 0.70, 0.75, 0.55])
    dimensions = (0, 2, 3)
    tp = torch.sum(probabilities * one_hot, dimensions)
    fp = torch.sum(probabilities * (1.0 - one_hot) * valid_float, dimensions)
    fn = torch.sum((1.0 - probabilities) * one_hot, dimensions)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    present = torch.sum(one_hot, dimensions) > 0
    if not present.any():
        return logits.sum() * 0.0
    return torch.pow(1.0 - score[present], 0.75).mean()


def hard_pixel_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    logits = resize_logits(logits, target)
    loss_map = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    valid_losses = loss_map[target != IGNORE_INDEX]
    if valid_losses.numel() == 0:
        return logits.sum() * 0.0
    count = max(1, int(round(valid_losses.numel() * fraction)))
    return torch.topk(valid_losses, k=count, sorted=False).values.mean()


def boundary_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable boundary supervision without any test-derived setting."""
    logits = resize_logits(logits, target)
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float

    def gradient_map(values: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(values, kernel_size=3, stride=1, padding=1)
        minimum = -F.max_pool2d(-values, kernel_size=3, stride=1, padding=1)
        return (maximum - minimum) * valid_float

    predicted_boundary = gradient_map(probabilities)
    true_boundary = gradient_map(one_hot)
    dimensions = (0, 2, 3)
    numerator = 2.0 * torch.sum(predicted_boundary * true_boundary, dimensions) + 1.0
    denominator = (
        torch.sum(predicted_boundary, dimensions)
        + torch.sum(true_boundary, dimensions)
        + 1.0
    )
    present = torch.sum(true_boundary, dimensions) > 0
    if not present.any():
        return logits.sum() * 0.0
    return (1.0 - numerator[present] / denominator[present]).mean()


def refinement_loss(logits, target, class_weights, cfg):
    logits = resize_logits(logits, target)
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
        label_smoothing=float(cfg.get("label_smoothing", 0.01)),
    )
    dice = multiclass_dice_loss(logits, target, len(GROUP_NAMES))
    tversky = focal_tversky_loss(logits, target)
    hard = hard_pixel_ce(
        logits,
        target,
        class_weights,
        float(cfg.get("hard_pixel_fraction", 0.20)),
    )
    boundary = boundary_dice_loss(logits, target)
    total = (
        float(cfg.get("ce_weight", 0.35)) * ce
        + float(cfg.get("dice_weight", 0.90)) * dice
        + float(cfg.get("tversky_weight", 0.45)) * tversky
        + float(cfg.get("hard_ce_weight", 0.15)) * hard
        + float(cfg.get("boundary_weight", 0.15)) * boundary
    )
    parts = [ce, dice, tversky, hard, boundary]
    return total, [float(value.detach().cpu()) for value in parts]


def evaluate_patches(model, loader, device, class_weights, cfg):
    model.eval()
    confusion = np.zeros((len(GROUP_NAMES), len(GROUP_NAMES)), dtype=np.int64)
    losses = []
    with torch.inference_mode():
        for images, masks, _ in tqdm(loader, desc="Refinement validation", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, _ = refinement_loss(logits, masks, class_weights, cfg)
            predictions = resize_logits(logits, masks).argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            for prediction, target in zip(predictions, targets):
                update_confusion(confusion, prediction, target, len(GROUP_NAMES))
            losses.append(float(loss.cpu()))
    metrics = metrics_from_confusion(confusion, GROUP_NAMES)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    f1 = {row["class_name"]: float(row["f1_dice"]) for row in metrics["per_class"]}
    critical_floor = min(f1["Macrophages"], f1["Necrosis"])
    base_joint = min(metrics["pixel_accuracy"], metrics["macro_f1_dice"])
    metrics["critical_class_floor"] = float(critical_floor)
    metrics["selection_score"] = float(base_joint + 0.08 * critical_floor)
    return metrics


def checkpoint_payload(model, optimizer, scheduler, epoch, best_score, metrics, cfg):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_joint_score": float(min(metrics["pixel_accuracy"], metrics["macro_f1_dice"])),
        "best_val_selection_score": float(best_score),
        "best_val_pixel_accuracy": float(metrics["pixel_accuracy"]),
        "best_val_macro_f1": float(metrics["macro_f1_dice"]),
        "best_validation_metrics": metrics,
        "config": cfg,
        "class_names": GROUP_NAMES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

    source_path = Path(cfg["source_grouped_checkpoint"])
    if not source_path.exists():
        raise SystemExit(f"Missing grouped checkpoint: {source_path}")
    label_spec_16 = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    train_rows, val_rows, _ = official_splits(he, cfg["validation_fold"])
    dataset_arguments = {
        "data_root": cfg["data_root"],
        "label_spec": label_spec_16,
        "patch_size": cfg["patch_size"],
        "min_annotated_fraction": cfg["min_annotated_fraction"],
        "targeted_crop_probability": cfg["targeted_crop_probability"],
        "target_group_probabilities": cfg["target_group_probabilities"],
        "roi_presence_path": "artifacts/roi_class_presence.json",
    }
    train_dataset = GroupedPatchDataset(
        train_rows,
        length=int(cfg["train_patches_per_epoch"]),
        seed=int(cfg["seed"]),
        training=True,
        **dataset_arguments,
    )
    val_dataset = GroupedPatchDataset(
        val_rows,
        length=len(val_rows) * int(cfg["validation_patches_per_roi"]),
        seed=int(cfg["seed"]) + 200_000,
        training=False,
        **dataset_arguments,
    )
    loader_arguments = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": False,
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, **loader_arguments)
    val_loader = DataLoader(val_dataset, **loader_arguments)

    device = torch.device("cuda")
    model = build_upernet(grouped_label_spec(), cfg["pretrained_checkpoint"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"], strict=True)
    model.to(device)

    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (heads if name.startswith(("decode_head", "auxiliary_head")) else backbone).append(parameter)
    optimizer = AdamW(
        [
            {"params": backbone, "lr": float(cfg["backbone_learning_rate"])},
            {"params": heads, "lr": float(cfg["head_learning_rate"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    accumulation = int(cfg["gradient_accumulation"])
    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / accumulation)))
    scheduler = LambdaLR(
        optimizer,
        partial(
            cosine_warmup_lambda,
            warmup_steps=steps_per_epoch * int(cfg["warmup_epochs"]),
            total_steps=steps_per_epoch * int(cfg["epochs"]),
        ),
    )
    scaler = torch.amp.GradScaler("cuda")
    class_weights = torch.tensor(cfg["group_class_weights"], dtype=torch.float32, device=device)

    out_dir = Path(cfg["output_root"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "resolved_config.json")
    best_score = -1.0
    best_metrics = None
    history = []
    patience = 0
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Refining from: {source_path}")
    print("Selection uses validation only: min(accuracy, macro F1) + 0.08 x rare-class floor.")
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        started = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        freeze_batchnorm_statistics(model)
        optimizer.zero_grad(set_to_none=True)
        running = np.zeros(6, dtype=np.float64)
        progress = tqdm(train_loader, desc=f"Refine {cfg['validation_fold']} {epoch}/{cfg['epochs']}")
        for step, (images, masks, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, parts = refinement_loss(logits, masks, class_weights, cfg)
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += [float(loss.detach().cpu()), *parts]
            progress.set_postfix(loss=f"{running[0] / step:.4f}")

        metrics = evaluate_patches(model, val_loader, device, class_weights, cfg)
        row = {
            "epoch": epoch,
            "train_loss": running[0] / len(train_loader),
            "train_ce": running[1] / len(train_loader),
            "train_dice": running[2] / len(train_loader),
            "train_tversky": running[3] / len(train_loader),
            "train_hard_ce": running[4] / len(train_loader),
            "train_boundary": running[5] / len(train_loader),
            "val_accuracy": metrics["pixel_accuracy"],
            "val_macro_f1": metrics["macro_f1_dice"],
            "val_miou": metrics["mean_iou"],
            "critical_floor": metrics["critical_class_floor"],
            "selection_score": metrics["selection_score"],
            "minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        score = float(metrics["selection_score"])
        if score > best_score:
            best_score = score
            best_metrics = metrics
            patience = 0
            torch.save(
                checkpoint_payload(model, optimizer, scheduler, epoch, best_score, metrics, cfg),
                out_dir / "best.pt",
            )
            save_json(metrics, out_dir / "best_validation_metrics.json")
            status = "NEW BEST"
        else:
            patience += 1
            status = f"patience {patience}/{cfg['early_stopping_patience']}"
        torch.save(
            checkpoint_payload(model, optimizer, scheduler, epoch, best_score, best_metrics, cfg),
            out_dir / "last.pt",
        )
        print(
            f"Epoch {epoch}: accuracy={metrics['pixel_accuracy']:.4f}, "
            f"macro F1={metrics['macro_f1_dice']:.4f}, "
            f"rare floor={metrics['critical_class_floor']:.4f}, "
            f"score={score:.4f} [{status}]"
        )
        if patience >= int(cfg["early_stopping_patience"]):
            print("Early stopping activated")
            break

    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")
    print(f"Refinement complete. Best validation selection score: {best_score:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
