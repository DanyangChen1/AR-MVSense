from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import CachedRPPGDataset
from .losses import NegativePearsonLoss
from .metrics import evaluate_collections
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(config: dict[str, Any], split: str) -> CachedRPPGDataset:
    return CachedRPPGDataset(
        config["cache_path"],
        split,
        clip_length=int(config["clip_length"]),
        clip_stride=int(config.get("clip_stride", config["clip_length"])),
        image_size=int(config["image_size"]),
        label_diff=bool(config.get("label_diff", True)),
        frame_diff_mode=str(config.get("frame_diff_mode", "simple")),
        diff_alignment=str(config.get("diff_alignment", "extra_frame")),
        split_mode=str(config.get("split_mode", "numeric")),
        split_seed=int(config.get("split_seed", 42)),
    )


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        prediction = model(batch["video"].to(device))
        loss = criterion(prediction, batch["label"].to(device))
        total += float(loss) * len(prediction)
        count += len(prediction)
    return total / max(count, 1)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "valid_loss", "learning_rate"]
        )
        writer.writeheader()
        writer.writerows(history)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: dict[str, Any], resume_path: str | Path | None = None) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    output = Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    device = choose_device(str(config["train"].get("device", "auto")))
    train_dataset = make_dataset(config["data"], "train")
    valid_dataset = make_dataset(config["data"], "valid")
    if train_dataset.subjects != valid_dataset.subjects:
        raise RuntimeError("Train and validation datasets resolved different subject splits")
    (output / "split_subjects.json").write_text(
        json.dumps(train_dataset.subjects, indent=2), encoding="utf-8"
    )
    generator = torch.Generator().manual_seed(seed)
    effective_batch_size = int(config["train"]["batch_size"])
    micro_batch_size = int(config["train"].get("micro_batch_size", effective_batch_size))
    if micro_batch_size < 1 or effective_batch_size % micro_batch_size:
        raise ValueError("train.micro_batch_size must be a positive divisor of train.batch_size")
    accumulation_steps = effective_batch_size // micro_batch_size
    grad_clip_config = config["train"].get("grad_clip")
    grad_clip = None if grad_clip_config is None else float(grad_clip_config)
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("train.grad_clip must be positive or null")
    print(
        json.dumps(
            {
                "device": str(device),
                "effective_batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "gradient_clip_norm": grad_clip,
            }
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    model = build_model(config["model"]).to(device)
    criterion = NegativePearsonLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    scheduler_config = config["train"].get("scheduler")
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if scheduler_config:
        scheduler_name = str(scheduler_config["name"]).lower()
        if scheduler_name == "cosine":
            decay_epochs = int(scheduler_config.get("decay_epochs", config["train"]["epochs"]))
            if decay_epochs < 1:
                raise ValueError("train.scheduler.decay_epochs must be positive")
            initial_lr = float(config["train"]["lr"])
            minimum_factor = float(scheduler_config.get("min_lr", 0.0)) / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda completed: minimum_factor
                + (1.0 - minimum_factor)
                * (1.0 + math.cos(math.pi * min(completed / decay_epochs, 1.0)))
                / 2.0,
            )
        else:
            raise ValueError(f"Unknown learning-rate scheduler: {scheduler_name}")
    best = float("inf")
    best_path = output / "best.pt"
    history: list[dict[str, float]] = []
    start_epoch = 1
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        best = float(checkpoint.get("best_valid_loss", checkpoint.get("valid_loss", best)))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        if "data_generator_state" in checkpoint:
            generator.set_state(checkpoint["data_generator_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        print(json.dumps({"resumed_from": str(resume_path), "start_epoch": start_epoch}))

    total_epochs = int(config["train"]["epochs"])
    if start_epoch > total_epochs:
        if not best_path.exists():
            raise FileNotFoundError(f"Training is complete but best checkpoint is missing: {best_path}")
        return best_path

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        accumulation_count = 0
        for batch_index, batch in enumerate(progress):
            video = batch["video"].to(device)
            target = batch["label"].to(device)
            prediction = model(video)
            loss = criterion(prediction, target)
            (loss / accumulation_steps).backward()
            accumulation_count += 1
            should_step = accumulation_count == accumulation_steps or batch_index + 1 == len(train_loader)
            if should_step:
                if accumulation_count != accumulation_steps:
                    correction = accumulation_steps / accumulation_count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
            total += float(loss.detach()) * len(video)
            count += len(video)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        train_loss = total / max(count, 1)
        valid_loss = validation_loss(model, valid_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        _write_history(output / "history.csv", history)
        print(json.dumps(row))
        if valid_loss < best:
            best = valid_loss
            _atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "valid_loss": best,
                },
                best_path,
            )
        if scheduler is not None:
            scheduler.step()
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": config,
                "epoch": epoch,
                "best_valid_loss": best,
                "history": history,
                "data_generator_state": generator.get_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
            },
            output / "last.pt",
        )
    return best_path


@torch.no_grad()
def evaluate(
    config: dict[str, Any], checkpoint_path: str | Path, split: str = "test"
) -> dict[str, float]:
    device = choose_device(str(config["train"].get("device", "auto")))
    dataset = make_dataset(config["data"], split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    collections: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for batch in tqdm(loader, desc="evaluation"):
        prediction = model(batch["video"].to(device)).cpu().numpy()
        target = batch["label"].numpy()
        for idx, source_id in enumerate(batch["source_id"]):
            collections[str(source_id)].append((int(batch["start"][idx]), prediction[idx], target[idx]))
    settings = config.get("evaluation", {})
    metrics, rows = evaluate_collections(
        collections,
        fps=float(config["data"]["fps"]),
        low_hz=float(settings.get("low_hz", 0.75)),
        high_hz=float(settings.get("high_hz", 2.5)),
        diff_signal=bool(config["data"].get("label_diff", True)),
    )
    output = Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "test" else f"_{split}"
    (output / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / f"per_video_metrics{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics
