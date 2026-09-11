#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml

from lightrppg.engine import evaluate, train


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    current = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled one-factor reproduction trials.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(plan["base_config"]).read_text(encoding="utf-8"))
    output_root = Path(plan["output_root"])
    trials = plan["trials"]
    if args.dry_run:
        print(json.dumps({"plan": str(args.plan), "trials": trials}, ensure_ascii=False, indent=2))
        return

    results: list[dict[str, Any]] = []
    trial_configs: dict[str, dict[str, Any]] = {}
    trial_checkpoints: dict[str, Path] = {}
    for trial in trials:
        name = str(trial["name"])
        config = copy.deepcopy(base)
        for key, value in trial.get("overrides", {}).items():
            set_dotted(config, key, value)
        trial_output = output_root / "trials" / name
        config["experiment"]["name"] = f"{config['experiment']['name']}_{name}"
        config["experiment"]["output_dir"] = str(trial_output)
        trial_output.mkdir(parents=True, exist_ok=True)
        (trial_output / "trial_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        resume = trial_output / "last.pt"
        checkpoint = train(config, resume if resume.exists() else None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result = {
            "trial": name,
            "selection_eligible": bool(trial.get("selection_eligible", True)),
            "reason": trial.get("reason", ""),
            "best_epoch": int(state["epoch"]),
            "best_valid_loss": float(state["valid_loss"]),
            "checkpoint": str(checkpoint),
        }
        results.append(result)
        trial_configs[name] = config
        trial_checkpoints[name] = checkpoint
        (output_root / "iteration_state.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    eligible = [row for row in results if row["selection_eligible"]]
    if not eligible:
        raise RuntimeError("Iteration plan has no selection-eligible trials")
    selected = min(eligible, key=lambda row: row["best_valid_loss"])
    selected_name = str(selected["trial"])
    selected_config = trial_configs[selected_name]
    selected_checkpoint = trial_checkpoints[selected_name]
    metrics = evaluate(selected_config, selected_checkpoint, split="test")
    targets = json.loads(Path("configs/paper_targets.json").read_text(encoding="utf-8"))
    target = targets["accuracy"][plan["paper_dataset"]][plan["paper_model"]]
    delta = {metric: float(metrics[metric]) - float(value) for metric, value in target.items()}

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, output_root / "best.pt")
    shutil.copy2(Path(selected_config["experiment"]["output_dir"]) / "metrics.json", output_root / "metrics.json")
    shutil.copy2(
        Path(selected_config["experiment"]["output_dir"]) / "per_video_metrics.csv",
        output_root / "per_video_metrics.csv",
    )
    (output_root / "selected_config.json").write_text(
        json.dumps(selected_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "selection_rule": "minimum validation Negative Pearson loss; test metrics evaluated once for the selected eligible trial",
        "dataset_variant": plan.get("dataset_variant", plan["paper_dataset"]),
        "selected_trial": selected_name,
        "paper_target": target,
        "local_test_metrics": metrics,
        "delta_local_minus_paper": delta,
        "trials": results,
    }
    (output_root / "iteration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "trial_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
