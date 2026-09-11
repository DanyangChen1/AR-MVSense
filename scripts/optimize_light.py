#!/usr/bin/env python3
"""Controlled Light-rPPGViT deployment trials for settings omitted by the paper."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import disable_observer
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lightrppg.config import load_config
from lightrppg.engine import choose_device, make_dataset, seed_everything, validation_loss
from lightrppg.losses import NegativePearsonLoss
from lightrppg.metrics import evaluate_collections
from lightrppg.models import LightRPPGViT, build_model, parameter_count
from lightrppg.optimization import (
    apply_output_channel_pruning,
    fake_quantizer_count,
    make_pruning_permanent,
    prepare_eager_qat,
    summarize_pruned_weights,
)


def load_float_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = build_model(config["model"])
    if not isinstance(model, LightRPPGViT):
        raise TypeError("optimize_light.py requires a Light-rPPGViT configuration")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    return model.to(device)


def loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    effective = int(config["train"]["batch_size"])
    micro = int(config["train"].get("micro_batch_size", effective))
    workers = int(config["data"].get("num_workers", 0))
    train = DataLoader(make_dataset(config["data"], "train"), batch_size=micro, shuffle=True,
                       num_workers=workers, generator=torch.Generator().manual_seed(int(config["experiment"]["seed"])))
    valid = DataLoader(make_dataset(config["data"], "valid"), batch_size=micro, shuffle=False,
                       num_workers=workers)
    return train, valid


def fit_trial(
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    lr: float,
    desc: str,
    disable_observer_after: int | None = None,
) -> list[dict[str, float]]:
    train_loader, valid_loader = loaders(config)
    criterion = NegativePearsonLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    effective = int(config["train"]["batch_size"])
    micro = int(config["train"].get("micro_batch_size", effective))
    accumulation = effective // micro
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        if disable_observer_after is not None and epoch > disable_observer_after:
            model.apply(disable_observer)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        pending = 0
        progress = tqdm(train_loader, desc=f"{desc} epoch {epoch}", leave=False)
        for index, batch in enumerate(progress):
            video = batch["video"].to(device)
            target = batch["label"].to(device)
            loss = criterion(model(video), target)
            (loss / accumulation).backward()
            pending += 1
            if pending == accumulation or index + 1 == len(train_loader):
                if pending != accumulation:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(accumulation / pending)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 5.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            total += float(loss.detach()) * len(video)
            count += len(video)
        train_loss = total / max(count, 1)
        valid_loss = validation_loss(model, valid_loader, criterion, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"trial": desc, **history[-1]}))
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, config: dict[str, Any], split: str, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dataset = make_dataset(config["data"], split)
    loader = DataLoader(dataset, batch_size=int(config["train"].get("micro_batch_size", 1)), shuffle=False,
                        num_workers=int(config["data"].get("num_workers", 0)))
    model.eval()
    collections: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for batch in tqdm(loader, desc=f"evaluate {split}", leave=False):
        prediction = model(batch["video"].to(device)).cpu().numpy()
        target = batch["label"].numpy()
        for index, source_id in enumerate(batch["source_id"]):
            collections[str(source_id)].append((int(batch["start"][index]), prediction[index], target[index]))
    settings = config.get("evaluation", {})
    return evaluate_collections(
        collections,
        fps=float(config["data"]["fps"]),
        low_hz=float(settings.get("low_hz", 0.75)),
        high_hz=float(settings.get("high_hz", 2.5)),
        diff_signal=bool(config["data"].get("label_diff", True)),
    )


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, report: dict[str, Any]) -> None:
    qat = report.get("qat", {}).get("evaluated") is True
    pruning = report.get("structured_pruning", {}).get("evaluated") is True
    reparam = report.get("reparameterization", {}).get("verified") is True
    report["status"] = "measured" if qat and pruning and reparam else "partial"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("reparameterization", "qat", "pruning", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/optimization/light_rppgvit.json"))
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--qat-lr", type=float, default=1e-3)
    parser.add_argument("--qat-backend", default="x86")
    parser.add_argument("--pruning-amount", type=float, default=0.2)
    parser.add_argument("--pruning-epochs", type=int, default=5)
    parser.add_argument("--pruning-lr", type=float, default=1e-3)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    device = choose_device(str(config["train"].get("device", "auto")))
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {
        "paper_settings_disclosed": False,
        "interpretation": "Controlled inferred settings; not claimed as the authors' undisclosed recipe.",
        "source_checkpoint": str(args.checkpoint),
    }
    modes = {args.mode} if args.mode != "all" else {"reparameterization", "qat", "pruning"}

    if "reparameterization" in modes:
        model = load_float_model(config, args.checkpoint, device).eval()
        deployed = model.deployed_copy().to(device)
        sample = make_dataset(config["data"], "valid")[0]["video"][None].to(device)
        with torch.no_grad():
            error = float((model(sample) - deployed(sample)).abs().max())
        report["reparameterization"] = {
            "verified": error <= 1e-4,
            "max_abs_error": error,
            "training_parameters": parameter_count(model),
            "deploy_parameters": parameter_count(deployed),
        }
        write_report(args.output, report)

    if "qat" in modes:
        model = prepare_eager_qat(load_float_model(config, args.checkpoint, device), args.qat_backend).to(device)
        history = fit_trial(model, config, device, args.qat_epochs, args.qat_lr, "qat",
                            disable_observer_after=max(1, args.qat_epochs - 2))
        metrics, rows = evaluate_model(model, config, "test", device)
        checkpoint = args.output.parent / "light_rppgvit_qat_fakequant.pt"
        torch.save({"model": model.state_dict(), "config": config, "settings": vars(args)}, checkpoint)
        save_rows(args.output.parent / "light_rppgvit_qat_per_video.csv", rows)
        report["qat"] = {
            "evaluated": True,
            "inferred": True,
            "backend": args.qat_backend,
            "activation_bits": 8,
            "activation_granularity": "per-tensor",
            "weight_bits": 8,
            "weight_granularity": "per-output-channel",
            "fake_quantizers": fake_quantizer_count(model),
            "epochs": args.qat_epochs,
            "learning_rate": args.qat_lr,
            "metrics": metrics,
            "history": history,
            "checkpoint": str(checkpoint),
            "limitation": "Fake-quantized QAT evaluation; integer kernel conversion is backend-dependent.",
        }
        write_report(args.output, report)

    if "pruning" in modes:
        model = load_float_model(config, args.checkpoint, device)
        initial_parameters = parameter_count(model)
        summary = apply_output_channel_pruning(model, args.pruning_amount)
        history = fit_trial(model, config, device, args.pruning_epochs, args.pruning_lr, "structured_pruning")
        metrics, rows = evaluate_model(model, config, "test", device)
        total, nonzero = summarize_pruned_weights(model)
        make_pruning_permanent(model)
        checkpoint = args.output.parent / "light_rppgvit_structured_pruned.pt"
        torch.save({"model": model.state_dict(), "config": config, "settings": vars(args)}, checkpoint)
        save_rows(args.output.parent / "light_rppgvit_structured_pruned_per_video.csv", rows)
        report["structured_pruning"] = {
            "evaluated": True,
            "inferred": True,
            "criterion": "L1 output-channel",
            "requested_amount_per_module": args.pruning_amount,
            "modules_pruned": summary.modules,
            "weight_elements": total,
            "nonzero_weight_elements": nonzero,
            "measured_weight_sparsity": 1.0 - nonzero / total,
            "dense_parameter_slots": initial_parameters,
            "epochs": args.pruning_epochs,
            "learning_rate": args.pruning_lr,
            "metrics": metrics,
            "history": history,
            "checkpoint": str(checkpoint),
            "limitation": "Shape-preserving channel masks; no claim of latency reduction without physical compaction/backend support.",
        }
        write_report(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
