#!/usr/bin/env python3
"""Recalculate paper metrics from an rPPG-Toolbox saved prediction pickle."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "rPPG-Toolbox"))
from evaluation.post_process import calculate_metric_per_video  # noqa: E402


def reform(chunks: dict) -> np.ndarray:
    ordered = [value for _, value in sorted(chunks.items(), key=lambda item: item[0])]
    return torch.cat(ordered, dim=0).detach().cpu().numpy().reshape(-1)


def calculate(payload: dict) -> dict[str, object]:
    predictions = payload["predictions"]
    labels = payload["labels"]
    diff_flag = payload.get("label_type") == "DiffNormalized"
    fs = int(payload.get("fs", 30))
    predicted_hr: list[float] = []
    reference_hr: list[float] = []
    for subject in sorted(predictions):
        prediction = reform(predictions[subject])
        label = reform(labels[subject])
        if len(prediction) < 9:
            continue
        gt_hr, pred_hr, _, _ = calculate_metric_per_video(
            prediction, label, fs=fs, diff_flag=diff_flag, hr_method="FFT"
        )
        reference_hr.append(float(gt_hr))
        predicted_hr.append(float(pred_hr))
    gt = np.asarray(reference_hr)
    pred = np.asarray(predicted_hr)
    if gt.size < 2:
        raise ValueError("Need at least two test subjects/windows to compute all paper metrics")
    error = pred - gt
    metrics: dict[str, object] = {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE": float(np.mean(np.abs(error / gt)) * 100),
        "Pearson": float(np.corrcoef(pred, gt)[0, 1]),
        "n_windows": int(gt.size),
        "evaluation_method": "rPPG-Toolbox whole-video FFT",
        "label_type": payload.get("label_type"),
        "fs": fs,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("rb") as handle:
        payload = pickle.load(handle)
    metrics = calculate(payload)
    metrics["source_prediction_pickle"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
