from __future__ import annotations

import numpy as np
import torch

from scripts.collect_toolbox_metrics import calculate


def test_collect_toolbox_metrics_from_subject_chunks() -> None:
    fs = 30
    frames = 600
    time = np.arange(frames) / fs
    predictions = {}
    labels = {}
    for subject, bpm in (("subject1", 60), ("subject2", 72), ("subject3", 84)):
        waveform = np.sin(2 * np.pi * (bpm / 60) * time).astype(np.float32)
        chunks = {
            1: torch.from_numpy(waveform[300:]),
            0: torch.from_numpy(waveform[:300]),
        }
        predictions[subject] = chunks
        labels[subject] = {key: value.clone() for key, value in chunks.items()}
    metrics = calculate(
        {
            "predictions": predictions,
            "labels": labels,
            "label_type": "Raw",
            "fs": fs,
        }
    )
    assert metrics["n_windows"] == 3
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0
    assert np.isclose(metrics["Pearson"], 1.0)
