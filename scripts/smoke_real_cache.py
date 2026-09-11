#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lightrppg.datasets import CachedRPPGDataset
from lightrppg.losses import NegativePearsonLoss
from lightrppg.models import LightRPPGViT, RPPGViT


def main() -> None:
    parser = argparse.ArgumentParser(description="One real-cache forward/backward smoke step.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--model", choices=("rppgvit", "light"), default="light")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    dataset = CachedRPPGDataset(
        args.cache,
        split=args.split,
        clip_length=args.frames,
        clip_stride=args.frames,
        image_size=args.size,
    )
    sample = dataset[0]
    video = sample["video"].unsqueeze(0)
    label = sample["label"].unsqueeze(0)
    if args.model == "light":
        model = LightRPPGViT(dim=32, rep_depth=1, tasa_depth=1, dropout=0.0)
    else:
        model = RPPGViT(
            dim=32,
            heads=4,
            cca_depth=1,
            swta_depth=1,
            window_size=8,
            spatial_tokens=4,
            dropout=0.0,
        )
    prediction = model(video)
    loss = NegativePearsonLoss()(prediction, label)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    result = {
        "cache": str(args.cache),
        "available_subjects": dataset.subjects,
        "records_in_split": len(dataset),
        "source_id": sample["source_id"],
        "video_shape": list(video.shape),
        "label_shape": list(label.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "gradients_finite": gradients_finite,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if prediction.shape != label.shape or not gradients_finite:
        raise RuntimeError("Real-cache smoke step failed")


if __name__ == "__main__":
    main()
