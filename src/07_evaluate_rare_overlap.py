from __future__ import annotations

import importlib.util
from pathlib import Path


EXPERIMENT_NAME = "upernet_swin_tiny_class_uniform_fold0"


def main() -> None:
    script_path = Path(__file__).resolve().parent / "05_evaluate_overlap.py"
    if not script_path.exists():
        raise SystemExit(f"Missing required overlap evaluator: {script_path}")
    spec = importlib.util.spec_from_file_location("overlap_evaluator", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {script_path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_load_config = evaluator.load_config

    def load_rare_config(path="config.json"):
        cfg = original_load_config(path)
        cfg["experiment_name"] = EXPERIMENT_NAME
        return cfg

    evaluator.load_config = load_rare_config
    evaluator.main()


if __name__ == "__main__":
    main()
