from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import experiment_dir, load_config, load_he_metadata, load_label_spec, official_splits, save_json, seed_everything
from dataset import IgnitePatchDataset
from modeling import (
    build_upernet,
    cosine_warmup_lambda,
    hybrid_loss,
    metrics_from_confusion,
    resize_logits,
    update_confusion,
)


def evaluate_patches(model, loader, device, class_weights, cfg, class_names):
    model.eval()
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    losses = []
    with torch.inference_mode():
        for images, masks, _ in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, _, _ = hybrid_loss(logits, masks, class_weights, cfg["ce_weight"], cfg["dice_weight"])
            logits = resize_logits(logits, masks)
            predictions = logits.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            for prediction, target in zip(predictions, targets):
                update_confusion(confusion, prediction, target, len(class_names))
            losses.append(float(loss.cpu()))
    metrics = metrics_from_confusion(confusion, class_names)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Hybrid loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], color="tab:green")
    axes[1].set_title("Validation macro F1/Dice")
    axes[2].plot(epochs, [row["val_mean_iou"] for row in history], color="tab:orange")
    axes[2].set_title("Validation mean IoU")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1, cfg, label_spec):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_macro_f1": best_f1,
            "config": cfg,
            "class_names": label_spec["class_names"],
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    train_rows, val_rows, _ = official_splits(he, cfg["validation_fold"])
    class_weights_file = Path("artifacts/class_weights.json")
    if not class_weights_file.exists():
        raise SystemExit("Run python src\\01_validate_dataset.py first")

    train_dataset = IgnitePatchDataset(
        train_rows,
        cfg["data_root"],
        label_spec,
        cfg["patch_size"],
        cfg["train_patches_per_epoch"],
        cfg["seed"],
        True,
        cfg["min_annotated_fraction"],
        class_weights_file if cfg["rare_class_sampling"] else None,
        "artifacts/roi_class_presence.json" if cfg["rare_class_sampling"] else None,
    )
    val_dataset = IgnitePatchDataset(
        val_rows,
        cfg["data_root"],
        label_spec,
        cfg["patch_size"],
        len(val_rows) * int(cfg["validation_patches_per_roi"]),
        cfg["seed"] + 100_000,
        False,
        cfg["min_annotated_fraction"],
    )
    common_loader_args = {
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, **common_loader_args)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False, **common_loader_args)

    device = torch.device("cuda")
    model = build_upernet(label_spec, cfg["pretrained_checkpoint"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    accumulation = int(cfg["gradient_accumulation"])
    optimizer_steps_per_epoch = max(1, int(np.ceil(len(train_loader) / accumulation)))
    total_steps = optimizer_steps_per_epoch * int(cfg["epochs"])
    warmup_steps = optimizer_steps_per_epoch * int(cfg["warmup_epochs"])
    scheduler = LambdaLR(optimizer, partial(cosine_warmup_lambda, warmup_steps=warmup_steps, total_steps=total_steps))
    scaler = torch.amp.GradScaler("cuda")
    with open(class_weights_file, "r", encoding="utf-8") as handle:
        class_weights = torch.tensor(json.load(handle)["weights"], dtype=torch.float32, device=device)

    out_dir = experiment_dir(cfg)
    save_json(cfg, out_dir / "resolved_config.json")
    start_epoch = 1
    best_f1 = -1.0
    history: list[dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_val_macro_f1"])
        history_path = out_dir / "training_history.csv"
        if history_path.exists():
            import pandas as pd

            history = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed from epoch {start_epoch - 1}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train/validation ROIs: {len(train_rows)}/{len(val_rows)}")
    print(f"Patch size: {cfg['patch_size']} | Batch: {cfg['batch_size']} | Accumulation: {accumulation}")
    patience_count = 0
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for step, (images, masks, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, ce, dice = hybrid_loss(logits, masks, class_weights, cfg["ce_weight"], cfg["dice_weight"])
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += float(loss.detach().cpu())
            running_ce += ce
            running_dice += dice
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        val_metrics = evaluate_patches(model, val_loader, device, class_weights, cfg, label_spec["class_names"])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "train_ce": running_ce / len(train_loader),
            "train_dice_loss": running_dice / len(train_loader),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1_dice"],
            "val_mean_iou": val_metrics["mean_iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_minutes": (time.time() - epoch_start) / 60.0,
        }
        history.append(row)
        with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, out_dir / "training_curves.png")
        current_f1 = float(val_metrics["macro_f1_dice"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_count = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, best_f1, cfg, label_spec)
            save_json(val_metrics, out_dir / "best_validation_metrics.json")
            status = "NEW BEST"
        else:
            patience_count += 1
            status = f"patience {patience_count}/{cfg['early_stopping_patience']}"
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, best_f1, cfg, label_spec)
        print(
            f"Epoch {epoch}: train loss={row['train_loss']:.4f}, val F1={current_f1:.4f}, "
            f"val mIoU={row['val_mean_iou']:.4f} [{status}]"
        )
        if patience_count >= int(cfg["early_stopping_patience"]):
            print("Early stopping activated")
            break

    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")
    print(f"Training complete. Best validation macro F1/Dice: {best_f1:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

