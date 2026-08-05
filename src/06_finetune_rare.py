from __future__ import annotations

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
    freeze_batchnorm_statistics,
    hybrid_loss,
    metrics_from_confusion,
    resize_logits,
    update_confusion,
)
from rare_dataset import ClassUniformPatchDataset


EXPERIMENT_NAME = "upernet_swin_tiny_class_uniform_fold0"
BASE_EXPERIMENT = "upernet_swin_tiny_hybrid_fold0"


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
                loss, _, _ = hybrid_loss(
                    logits,
                    masks,
                    class_weights,
                    cfg["ce_weight"],
                    cfg["dice_weight"],
                )
            predictions = resize_logits(logits, masks).argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            for prediction, target in zip(predictions, targets):
                update_confusion(confusion, prediction, target, len(class_names))
            losses.append(float(loss.cpu()))
    metrics = metrics_from_confusion(confusion, class_names)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


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
        axis.set_xlabel("Fine-tuning epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    cfg.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "fine_tune_source_experiment": BASE_EXPERIMENT,
            "fine_tune_epochs": 12,
            "fine_tune_backbone_lr": 1.0e-5,
            "fine_tune_head_lr": 3.0e-5,
            "train_patches_per_epoch": 600,
            "target_class_crop_probability": 0.75,
            "early_stopping_patience": 4,
        }
    )
    seed_everything(int(cfg["seed"]) + 606)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")

    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    train_rows, val_rows, _ = official_splits(he, cfg["validation_fold"])
    class_weights_file = Path("artifacts/class_weights.json")
    presence_file = Path("artifacts/roi_class_presence.json")
    if not class_weights_file.exists() or not presence_file.exists():
        raise SystemExit("Missing validation artifacts. Run src\\01_validate_dataset.py first.")

    base_checkpoint = Path(cfg["output_root"]) / BASE_EXPERIMENT / "best.pt"
    if not base_checkpoint.exists():
        raise SystemExit(f"Missing baseline checkpoint: {base_checkpoint}")

    train_dataset = ClassUniformPatchDataset(
        train_rows,
        cfg["data_root"],
        label_spec,
        cfg["patch_size"],
        cfg["train_patches_per_epoch"],
        int(cfg["seed"]) + 606,
        cfg["min_annotated_fraction"],
        presence_file,
        cfg["target_class_crop_probability"],
    )
    val_dataset = IgnitePatchDataset(
        val_rows,
        cfg["data_root"],
        label_spec,
        cfg["patch_size"],
        len(val_rows) * int(cfg["validation_patches_per_roi"]),
        int(cfg["seed"]) + 100_000,
        False,
        cfg["min_annotated_fraction"],
    )
    loader_args = {
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        **loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        **loader_args,
    )

    device = torch.device("cuda")
    model = build_upernet(label_spec, cfg["pretrained_checkpoint"])
    source = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"])
    model.to(device)

    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if "decode_head" in name or "auxiliary_head" in name:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": cfg["fine_tune_backbone_lr"]},
            {"params": head_parameters, "lr": cfg["fine_tune_head_lr"]},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    accumulation = int(cfg["gradient_accumulation"])
    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / accumulation)))
    total_steps = steps_per_epoch * int(cfg["fine_tune_epochs"])
    warmup_steps = steps_per_epoch
    scheduler = LambdaLR(
        optimizer,
        partial(
            cosine_warmup_lambda,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    scaler = torch.amp.GradScaler("cuda")
    with open(class_weights_file, "r", encoding="utf-8") as handle:
        class_weights = torch.tensor(
            json.load(handle)["weights"],
            dtype=torch.float32,
            device=device,
        )

    out_dir = experiment_dir(cfg)
    cfg["fine_tune_source_checkpoint"] = str(base_checkpoint)
    save_json(cfg, out_dir / "resolved_config.json")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loaded baseline checkpoint: {base_checkpoint}")
    print("Method: 75% class-uniform target-centred crops; 25% ordinary crops")
    print(f"Train/validation ROIs: {len(train_rows)}/{len(val_rows)}")
    print("Checking baseline on the unchanged validation patches...")
    initial_metrics = evaluate_patches(
        model,
        val_loader,
        device,
        class_weights,
        cfg,
        label_spec["class_names"],
    )
    save_json(initial_metrics, out_dir / "initial_validation_metrics.json")
    best_f1 = float(initial_metrics["macro_f1_dice"])
    save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, 0, best_f1, cfg, label_spec)
    print(f"Initial validation macro F1/Dice: {best_f1:.4f}")

    history: list[dict] = []
    patience_count = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, int(cfg["fine_tune_epochs"]) + 1):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        freeze_batchnorm_statistics(model)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0
        progress = tqdm(train_loader, desc=f"Rare fine-tune {epoch}/{cfg['fine_tune_epochs']}")
        for step, (images, masks, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=images).logits
                loss, ce, dice = hybrid_loss(
                    logits,
                    masks,
                    class_weights,
                    cfg["ce_weight"],
                    cfg["dice_weight"],
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
            running_loss += float(loss.detach().cpu())
            running_ce += ce
            running_dice += dice
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        val_metrics = evaluate_patches(
            model,
            val_loader,
            device,
            class_weights,
            cfg,
            label_spec["class_names"],
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "train_ce": running_ce / len(train_loader),
            "train_dice_loss": running_dice / len(train_loader),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1_dice"],
            "val_mean_iou": val_metrics["mean_iou"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
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
            f"Fine-tune epoch {epoch}: loss={row['train_loss']:.4f}, "
            f"val F1={current_f1:.4f}, val mIoU={row['val_mean_iou']:.4f} [{status}]"
        )
        if patience_count >= int(cfg["early_stopping_patience"]):
            print("Early stopping activated")
            break

    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")
    print(f"Rare-class fine-tuning complete. Best validation macro F1/Dice: {best_f1:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
