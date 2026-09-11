from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy import signal


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(first) < 2 or len(first) != len(second):
        return float("nan")
    if first.std() < 1e-12 or second.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def reconstruct_and_filter(
    values: np.ndarray,
    fps: float,
    low_hz: float = 0.75,
    high_hz: float = 2.5,
    diff_signal: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if diff_signal:
        values = np.cumsum(values)
    values = signal.detrend(values, type="linear")
    if len(values) >= 9:
        b, a = signal.butter(1, [low_hz / (fps / 2), high_hz / (fps / 2)], btype="bandpass")
        values = signal.filtfilt(b, a, values)
    return values


def fft_hr(values: np.ndarray, fps: float, low_hz: float = 0.75, high_hz: float = 2.5) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    nfft = 1 if not len(values) else 2 ** int(np.ceil(np.log2(len(values))))
    frequency, power = signal.periodogram(values, fs=fps, nfft=nfft, detrend=False)
    mask = (frequency >= low_hz) & (frequency <= high_hz)
    if not np.any(mask):
        return float("nan")
    return float(frequency[mask][np.argmax(power[mask])] * 60.0)


def summarize_hr(predicted_hr: np.ndarray, target_hr: np.ndarray) -> dict[str, float]:
    predicted_hr = np.asarray(predicted_hr, dtype=np.float64)
    target_hr = np.asarray(target_hr, dtype=np.float64)
    error = predicted_hr - target_hr
    pearson = safe_pearson(predicted_hr, target_hr)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error) / np.maximum(np.abs(target_hr), 1e-8)) * 100.0),
        "Pearson": pearson,
        "n_videos": float(len(error)),
    }


def evaluate_collections(
    collections: dict[str, list[tuple[int, np.ndarray, np.ndarray]]],
    fps: float,
    low_hz: float = 0.75,
    high_hz: float = 2.5,
    diff_signal: bool = True,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    rows: list[dict[str, float | str]] = []
    for source_id, pieces in sorted(collections.items()):
        ordered = sorted(pieces, key=lambda item: item[0])
        prediction = np.concatenate([item[1] for item in ordered])
        target = np.concatenate([item[2] for item in ordered])
        prediction = reconstruct_and_filter(prediction, fps, low_hz, high_hz, diff_signal)
        target = reconstruct_and_filter(target, fps, low_hz, high_hz, diff_signal)
        waveform_r = safe_pearson(prediction, target)
        rows.append(
            {
                "source_id": source_id,
                "pred_hr": fft_hr(prediction, fps, low_hz, high_hz),
                "gt_hr": fft_hr(target, fps, low_hz, high_hz),
                "waveform_pearson": waveform_r,
            }
        )
    metrics = summarize_hr(
        np.asarray([row["pred_hr"] for row in rows], dtype=float),
        np.asarray([row["gt_hr"] for row in rows], dtype=float),
    )
    metrics["waveform_pearson"] = float(np.nanmean([row["waveform_pearson"] for row in rows]))
    return metrics, rows
