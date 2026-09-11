#!/usr/bin/env python3
"""Generate deterministic within-dataset rPPG-Toolbox configs for Table 1/2 baselines."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "third_party" / "rPPG-Toolbox"
TEMPLATES = {
    "TSCAN": "UBFC-rPPG_UBFC-rPPG_MMPD_TSCAN_BASIC.yaml",
    "PhysNet": "UBFC-rPPG_UBFC-rPPG_UBFC-PHYS_PHYSNET_BASIC.yaml",
    "DeepPhys": "UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml",
    "PhysFormer": "UBFC-rPPG_UBFC-rPPG_MMPD_PHYSFORMER_BASIC.yaml",
    "EfficientPhys": "UBFC-rPPG_UBFC-rPPG_PURE_EFFICIENTPHYS.yaml",
}
SPLITS = {"TRAIN": (0.0, 0.6), "VALID": (0.6, 0.8), "TEST": (0.8, 1.0)}
MMPD_INFO = {
    "LIGHT": [1, 2, 3, 4],
    "MOTION": [1, 2, 3, 4],
    "EXERCISE": [1, 2],
    "SKIN_COLOR": [3, 4, 5, 6],
    "GENDER": [1, 2],
    "GLASSER": [1, 2],
    "HAIR_COVER": [1, 2],
    "MAKEUP": [1, 2],
}


def build_config(model: str, dataset_key: str) -> dict:
    template_path = TOOLBOX / "configs" / "train_configs" / TEMPLATES[model]
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    source_data = copy.deepcopy(config["TRAIN"]["DATA"])
    dataset_name = "UBFC-rPPG" if dataset_key == "ubfc" else "MMPD"
    raw_path = (
        ROOT / "data" / "raw" / "UBFC-rPPG" / "DATASET_2"
        if dataset_key == "ubfc"
        else ROOT / "data" / "raw" / "MMPD-mini"
    )
    cache_root = ROOT / "data" / "cache" / "rppg-toolbox" / dataset_key / model.lower()
    experiment_name = f"{dataset_name}_{model}_within_622"

    for section, (begin, end) in SPLITS.items():
        data = copy.deepcopy(source_data)
        data.pop("FILE_LIST_PATH", None)
        data.update(
            {
                "FS": 30,
                "DATASET": dataset_name,
                "DO_PREPROCESS": True,
                "DATA_PATH": str(raw_path),
                "CACHED_PATH": str(cache_root),
                "EXP_DATA_NAME": experiment_name,
                "BEGIN": begin,
                "END": end,
            }
        )
        if dataset_key == "mmpd-mini":
            data["INFO"] = copy.deepcopy(MMPD_INFO)
        else:
            data.pop("INFO", None)
        config.setdefault(section, {})["DATA"] = data

    config["BASE"] = [""]
    config["TOOLBOX_MODE"] = "train_and_test"
    config["TRAIN"]["BATCH_SIZE"] = 4
    config["TRAIN"]["MODEL_FILE_NAME"] = f"{dataset_key}_{dataset_key}_{dataset_key}_{model}"
    config["TEST"]["USE_LAST_EPOCH"] = False
    config["TEST"]["METRICS"] = ["MAE", "RMSE", "MAPE", "Pearson"]
    config["DEVICE"] = "cpu"
    config["NUM_OF_GPU_TRAIN"] = 1
    config["NUM_WORKERS"] = 0
    config["LOG"] = {"PATH": str(ROOT / "outputs" / "toolbox" / dataset_key / model)}
    config["INFERENCE"]["BATCH_SIZE"] = 4
    config["INFERENCE"]["EVALUATION_METHOD"] = "FFT"
    config["INFERENCE"]["EVALUATION_WINDOW"] = {
        "USE_SMALLER_WINDOW": False,
        "WINDOW_SIZE": 10,
    }
    config["INFERENCE"]["MODEL_PATH"] = ""
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "toolbox_baselines")
    args = parser.parse_args()
    manifest: list[dict[str, str]] = []
    for dataset_key in ("ubfc", "mmpd-mini"):
        output_dir = args.output / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        for model in TEMPLATES:
            path = output_dir / f"{model.lower()}.yaml"
            path.write_text(
                yaml.safe_dump(build_config(model, dataset_key), sort_keys=False),
                encoding="utf-8",
            )
            manifest.append({"dataset": dataset_key, "model": model, "config": str(path)})
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
