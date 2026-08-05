from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common import IGNORE_INDEX


def build_upernet(label_spec: dict[str, Any], checkpoint_name: str):
    from transformers import UperNetForSemanticSegmentation

    id2label = {index: name for index, name in enumerate(label_spec["class_names"])}
    label2id = {name: index for index, name in id2label.items()}
    model = UperNetForSemanticSegmentation.from_pretrained(
        checkpoint_name,
        num_labels=label_spec["num_classes"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.loss_ignore_index = IGNORE_INDEX
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, ValueError):
        pass
    return model


def resize_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = resize_logits(logits, target)
    valid = target != IGNORE_INDEX
    safe_target = target.clone()
    safe_target[~valid] = 0
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probabilities = probabilities * valid_float
    one_hot = one_hot * valid_float
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dims)
    denominator = torch.sum(probabilities + one_hot, dims)
    present = torch.sum(one_hot, dims) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    if present.any():
        return 1.0 - dice[present].mean()
    return logits.sum() * 0.0


def hybrid_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    ce_weight: float,
    dice_weight: float,
) -> tuple[torch.Tensor, float, float]:
    logits = resize_logits(logits, target)
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE_INDEX)
    dice = multiclass_dice_loss(logits, target, logits.shape[1])
    total = float(ce_weight) * ce + float(dice_weight) * dice
    return total, float(ce.detach().cpu()), float(dice.detach().cpu())


def update_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray, num_classes: int) -> None:
    valid = target != IGNORE_INDEX
    target_valid = target[valid].astype(np.int64, copy=False)
    pred_valid = prediction[valid].astype(np.int64, copy=False)
    encoded = target_valid * num_classes + pred_valid
    confusion += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    confusion = confusion.astype(np.float64)
    tp = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    present = support > 0
    total = confusion.sum()
    per_class = []
    for index, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": index,
                "class_name": name,
                "support_pixels": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_dice": float(f1[index]),
                "iou": float(iou[index]),
            }
        )
    return {
        "pixel_accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1_dice": float(f1[present].mean()) if present.any() else 0.0,
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "frequency_weighted_iou": float((support[present] * iou[present]).sum() / support[present].sum()) if present.any() else 0.0,
        "per_class": per_class,
    }


def cosine_warmup_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))



def freeze_batchnorm_statistics(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
