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
    experiment_dir,
    load_config,
    load_he_metadata,
    load_label_spec,
    official_splits,
    save_json,
    seed_everything,
)
from grouped_dataset import GROUP_NAMES, GroupedPatchDataset, build_group_lookup  # noqa: E402
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


def initialize_from_16_class_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    label_spec_16: dict,
) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise SystemExit(f"Missing source 16-class checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    destination = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)

    # Convert the learned 16-way classifier into a meaningful five-way
    # initialization by averaging source rows belonging to each tissue group.
    group_lookup = build_group_lookup(label_spec_16)
    converted = {}
    for key in (
        "decode_head.classifier.weight",
        "decode_head.classifier.bias",
        "auxiliary_head.classifier.weight",
        "auxiliary_head.classifier.bias",
    ):
        if key not in source or key not in destination:
            continue
        old_tensor = source[key]
        new_tensor = destination[key].clone()
        if old_tensor.shape[0] != len(group_lookup) or new_tensor.shape[0] != len(GROUP_NAMES):
            continue
        for group_id in range(len(GROUP_NAMES)):
            member_ids = np.flatnonzero(group_lookup == group_id).tolist()
            new_tensor[group_id] = old_tensor[member_ids].mean(dim=0)
        converted[key] = new_tensor
    model.load_state_dict(converted, strict=False)
    print(
        f"Warm-started {len(compatible)} matching tensors and "
        f"{len(converted)} grouped classifier tensors from {path}"
    )


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    gamma: float = 0.75,
) -> torch.Tensor:
    logits = resize_logits(logits, target)
    valid = target != IGNORE_INDEX
    safe_target = target.clone()
    safe_target[~valid] = 0
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float
    dimensions = (0, 2, 3)
    true_positive = torch.sum(probabilities * one_hot, dimensions)
    false_positive = torch.sum(probabilities * (1.0 - one_hot) * valid_float, dimensions)
    false_negative = torch.sum((1.0 - probabilities) * one_hot, dimensions)
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    present = torch.sum(one_hot, dimensions) > 0
    if not present.any():
        return logits.sum() * 0.0
    return torch.pow(1.0 - score[present], gamma).mean()


def grouped_loss(logits, target, class_weights, cfg):
    logits = resize_logits(logits, target)
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
        label_smoothing=float(cfg.get("label_smoothing", 0.02)),
    )
    dice = multiclass_dice_loss(logits, target, len(GROUP_NAMES))
    tversky = focal_tversky_loss(logits, target)
    total = (
        float(cfg.get("ce_weight", 0.55)) * ce
        + float(cfg.get("dice_weight", 1.00)) * dice
        + float(cfg.get("tversky_weight", 0.45)) * tversky
    )
    return total, float(ce.detach().cpu()), float(dice.detach().cpu()), float(tversky.detach().cpu())


def evaluate_patches(model, loader, device, class_weights, cfg):
    model.eval()
    confusion = np.zeros((len(GROUP_NAMES), len(GROUP_NAMES)), dtype=np.int64)
    losses = []
    with torch.inference_mode():
        for images, masks, _ in tqdm(loader, desc="Grouped validation", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, _, _, _ = grouped_loss(logits, masks, class_weights, cfg)
            predictions = resize_logits(logits, masks).argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            for prediction, target in zip(predictions, targets):
                update_confusion(confusion, prediction, target, len(GROUP_NAMES))
            losses.append(float(loss.cpu()))
    metrics = metrics_from_confusion(confusion, GROUP_NAMES)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["joint_min_accuracy_macro_f1"] = min(
        metrics["pixel_accuracy"], metrics["macro_f1_dice"]
    )
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_joint, metrics, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_joint_score": best_joint,
            "best_val_pixel_accuracy": float(metrics["pixel_accuracy"]),
            "best_val_macro_f1": float(metrics["macro_f1_dice"]),
            "config": cfg,
            "class_names": GROUP_NAMES,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

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
        seed=int(cfg["seed"]) + 100_000,
        training=False,
        **dataset_arguments,
    )
    loader_args = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": False,
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, **loader_args)
    val_loader = DataLoader(val_dataset, **loader_args)

    device = torch.device("cuda")
    model = build_upernet(grouped_label_spec(), cfg["pretrained_checkpoint"])
    initialize_from_16_class_checkpoint(
        model, cfg["source_16class_checkpoint"], label_spec_16
    )
    model.to(device)

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("decode_head") or name.startswith("auxiliary_head"):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": float(cfg["backbone_learning_rate"])},
            {"params": head_parameters, "lr": float(cfg["head_learning_rate"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    accumulation = int(cfg["gradient_accumulation"])
    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / accumulation)))
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = steps_per_epoch * int(cfg["warmup_epochs"])
    scheduler = LambdaLR(
        optimizer,
        partial(
            cosine_warmup_lambda,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    scaler = torch.amp.GradScaler("cuda")
    class_weights = torch.tensor(
        cfg["group_class_weights"], dtype=torch.float32, device=device
    )

    out_dir = experiment_dir(cfg)
    save_json(cfg, out_dir / "resolved_config.json")
    start_epoch = 1
    best_joint = -1.0
    best_metrics = {"pixel_accuracy": 0.0, "macro_f1_dice": 0.0}
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_joint = float(checkpoint["best_val_joint_score"])
        print(f"Resumed from epoch {start_epoch - 1}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Grouped endpoint: {GROUP_NAMES}")
    print(f"Train/validation ROIs: {len(train_rows)}/{len(val_rows)}")
    print(
        "Checkpoint selection score = min(validation pixel accuracy, "
        "validation macro F1)."
    )
    patience_count = 0
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        freeze_batchnorm_statistics(model)
        optimizer.zero_grad(set_to_none=True)
        running = np.zeros(4, dtype=np.float64)
        progress = tqdm(train_loader, desc=f"Grouped fold {cfg['validation_fold']} {epoch}/{cfg['epochs']}")
        for step, (images, masks, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, ce, dice, tversky = grouped_loss(
                    logits, masks, class_weights, cfg
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += [float(loss.detach().cpu()), ce, dice, tversky]
            progress.set_postfix(loss=f"{running[0] / step:.4f}")

        metrics = evaluate_patches(model, val_loader, device, class_weights, cfg)
        row = {
            "epoch": epoch,
            "train_loss": running[0] / len(train_loader),
            "train_ce": running[1] / len(train_loader),
            "train_dice_loss": running[2] / len(train_loader),
            "train_tversky_loss": running[3] / len(train_loader),
            "val_loss": metrics["loss"],
            "val_pixel_accuracy": metrics["pixel_accuracy"],
            "val_macro_f1": metrics["macro_f1_dice"],
            "val_mean_iou": metrics["mean_iou"],
            "val_joint_min": metrics["joint_min_accuracy_macro_f1"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "epoch_minutes": (time.time() - epoch_start) / 60.0,
        }
        history.append(row)
        with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        joint = float(metrics["joint_min_accuracy_macro_f1"])
        if joint > best_joint:
            best_joint = joint
            best_metrics = metrics
            patience_count = 0
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_joint,
                metrics,
                cfg,
            )
            save_json(metrics, out_dir / "best_validation_metrics.json")
            status = "NEW BEST"
        else:
            patience_count += 1
            status = f"patience {patience_count}/{cfg['early_stopping_patience']}"
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_joint,
            best_metrics,
            cfg,
        )
        print(
            f"Epoch {epoch}: loss={row['train_loss']:.4f}, "
            f"val accuracy={metrics['pixel_accuracy']:.4f}, "
            f"val macro F1={metrics['macro_f1_dice']:.4f}, "
            f"joint={joint:.4f} [{status}]"
        )
        if patience_count >= int(cfg["early_stopping_patience"]):
            print("Early stopping activated")
            break

    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")
    print(f"Training complete. Best validation joint score: {best_joint:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

