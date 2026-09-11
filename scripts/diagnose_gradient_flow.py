#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lightrppg.engine import choose_device, make_dataset
from lightrppg.losses import NegativePearsonLoss
from lightrppg.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(int(config["experiment"]["seed"]))
    device = choose_device(str(config["train"].get("device", "auto")))
    dataset = make_dataset(config["data"], args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.train()

    prediction = model(batch["video"].to(device))
    target = batch["label"].to(device)
    loss = NegativePearsonLoss()(prediction, target)
    loss.backward()

    squared_norms: dict[str, float] = defaultdict(float)
    nonzero_parameters: dict[str, int] = defaultdict(int)
    total_parameters: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        total_parameters[group] += 1
        if parameter.grad is not None:
            norm = float(parameter.grad.detach().norm())
            squared_norms[group] += norm * norm
            if norm > 0:
                nonzero_parameters[group] += 1

    pred = prediction.detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    correlations = []
    for row_pred, row_truth in zip(pred, truth, strict=True):
        correlations.append(float(np.corrcoef(row_pred, row_truth)[0, 1]))
    gammas = {
        name: float(parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if name.endswith(".gamma") and parameter.numel() == 1
    }
    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "batch_source_ids": list(batch["source_id"]),
        "loss": float(loss.detach()),
        "prediction_std_per_sample": pred.std(axis=1).tolist(),
        "target_std_per_sample": truth.std(axis=1).tolist(),
        "pearson_per_sample": correlations,
        "gradient_norm_by_top_level_module": {
            group: squared_norms[group] ** 0.5 for group in sorted(total_parameters)
        },
        "nonzero_gradient_parameter_tensors": {
            group: {
                "nonzero": nonzero_parameters[group],
                "total": total_parameters[group],
            }
            for group in sorted(total_parameters)
        },
        "cca_gamma": gammas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
