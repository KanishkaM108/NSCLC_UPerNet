from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path


GROUPS = ["Tumor", "Stroma + inflammation", "Macrophages", "Necrosis", "Rest"]


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing validation file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def class_f1(metrics: dict) -> dict[str, float]:
    return {row["class_name"]: float(row["f1_dice"]) for row in metrics["per_class"]}


def main() -> None:
    output_root = Path("outputs")
    report_dir = Path("artifacts") / "grouped_refinement_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        original_path = output_root / f"upernet_swin_tiny_grouped_fold{fold}" / "best_validation_metrics.json"
        refined_path = output_root / f"upernet_swin_tiny_grouped_refined_fold{fold}" / "best_validation_metrics.json"
        original = load(original_path)
        refined = load(refined_path)
        old_f1, new_f1 = class_f1(original), class_f1(refined)
        row = {
            "fold": fold,
            "original_accuracy": float(original["pixel_accuracy"]),
            "refined_accuracy": float(refined["pixel_accuracy"]),
            "original_macro_f1": float(original["macro_f1_dice"]),
            "refined_macro_f1": float(refined["macro_f1_dice"]),
        }
        for group in GROUPS:
            key = group.lower().replace(" + ", "_").replace(" ", "_")
            row[f"original_{key}_f1"] = old_f1[group]
            row[f"refined_{key}_f1"] = new_f1[group]
        rows.append(row)
        shutil.copy2(refined_path, report_dir / f"fold{fold}_refined_metrics.json")
        history = refined_path.parent / "training_history.csv"
        if history.exists():
            shutil.copy2(history, report_dir / f"fold{fold}_refined_history.csv")

    means = {}
    numeric_keys = [key for key in rows[0] if key != "fold"]
    for key in numeric_keys:
        means[key] = sum(row[key] for row in rows) / len(rows)
    macro_gain = means["refined_macro_f1"] - means["original_macro_f1"]
    accuracy_change = means["refined_accuracy"] - means["original_accuracy"]
    rare_old = min(means["original_macrophages_f1"], means["original_necrosis_f1"])
    rare_new = min(means["refined_macrophages_f1"], means["refined_necrosis_f1"])
    eligible = macro_gain >= 0.005 and accuracy_change >= -0.003 and rare_new > rare_old
    summary = {
        "folds": rows,
        "mean": means,
        "macro_f1_gain": macro_gain,
        "accuracy_change": accuracy_change,
        "original_rare_class_floor": rare_old,
        "refined_rare_class_floor": rare_new,
        "eligible_for_one_final_holdout_evaluation": eligible,
        "gate": "macro F1 gain >= 0.005, accuracy drop <= 0.003, rare-class floor improves",
    }
    with open(report_dir / "validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(report_dir / "fold_comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    zip_path = downloads / "REFINEMENT_VALIDATION_DIAGNOSTICS.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(report_dir.iterdir()):
            archive.write(path, arcname=path.name)
    print("=" * 72)
    print("VALIDATION-ONLY REFINEMENT DECISION")
    print("=" * 72)
    print(f"Mean accuracy change : {accuracy_change:+.4f}")
    print(f"Mean macro F1 gain   : {macro_gain:+.4f}")
    print(f"Rare-class floor     : {rare_old:.4f} -> {rare_new:.4f}")
    print(f"Decision             : {'ELIGIBLE FOR ONE FINAL TEST' if eligible else 'KEEP ORIGINAL; DO NOT RETEST'}")
    print(f"Upload this file     : {zip_path}")


if __name__ == "__main__":
    main()
