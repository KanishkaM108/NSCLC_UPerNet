from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, load_he_metadata, load_label_spec, official_splits, seed_everything
from dataset import IgnitePatchDataset
from modeling import build_upernet, hybrid_loss


def main() -> None:
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected")
    class_weights_path = Path("artifacts/class_weights.json")
    if not class_weights_path.exists():
        raise SystemExit("Run: python src\\01_validate_dataset.py")

    label_spec = load_label_spec(cfg["label_map_json"])
    he = load_he_metadata(cfg["metadata_csv"])
    train_rows, _, _ = official_splits(he, cfg["validation_fold"])
    dataset = IgnitePatchDataset(
        train_rows,
        cfg["data_root"],
        label_spec,
        cfg["patch_size"],
        length=1,
        seed=cfg["seed"],
        training=True,
        min_annotated_fraction=cfg["min_annotated_fraction"],
        class_weights_path=class_weights_path if cfg["rare_class_sampling"] else None,
        roi_presence_path="artifacts/roi_class_presence.json" if cfg["rare_class_sampling"] else None,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
    images, masks, names = next(iter(loader))
    device = torch.device("cuda")
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Input       : {tuple(images.shape)}")
    print(f"Mask        : {tuple(masks.shape)}")
    print(f"ROI         : {names[0]}")
    print("Loading pretrained UPerNet-Swin-Tiny (first run may download weights)...")
    model = build_upernet(label_spec, cfg["pretrained_checkpoint"]).to(device)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    with open(class_weights_path, "r", encoding="utf-8") as handle:
        class_weights = torch.tensor(json.load(handle)["weights"], dtype=torch.float32, device=device)
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=images)
            loss, ce, dice = hybrid_loss(
                outputs.logits,
                masks,
                class_weights,
                cfg["ce_weight"],
                cfg["dice_weight"],
            )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise SystemExit(
            "GPU out of memory at the configured patch size. Change patch_size in config.json from 512 to 384, then rerun."
        ) from error
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Logits      : {tuple(outputs.logits.shape)}")
    print(f"Total loss  : {float(loss.detach().cpu()):.5f}")
    print(f"CE / Dice   : {ce:.5f} / {dice:.5f}")
    print(f"Peak VRAM   : {peak_gb:.2f} GiB")
    print("SMOKE TEST PASSED (forward, backward, and optimizer step)")


if __name__ == "__main__":
    main()

