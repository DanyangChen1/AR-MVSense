from __future__ import annotations

import math

import numpy as np
from scipy import signal


def mean_rgb(frames: np.ndarray) -> np.ndarray:
    return np.asarray(frames, dtype=np.float64).mean(axis=(1, 2))


def green(frames: np.ndarray, fps: float = 30.0) -> np.ndarray:
    del fps
    return mean_rgb(frames)[:, 1]


def pos(frames: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Plane-orthogonal-to-skin implementation matching rPPG-Toolbox."""
    rgb = mean_rgb(frames)
    length = len(rgb)
    output = np.zeros(length, dtype=np.float64)
    window = math.ceil(1.6 * fps)
    projection = np.asarray([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for end in range(window, length + 1):
        start = end - window
        normalized = rgb[start:end] / np.maximum(rgb[start:end].mean(axis=0), 1e-8)
        components = projection @ normalized.T
        pulse = components[0] + components[0].std() / max(components[1].std(), 1e-8) * components[1]
        output[start:end] += pulse - pulse.mean()
    return output


def chrom(frames: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Windowed CHROM baseline."""
    rgb = mean_rgb(frames)
    normalized = rgb / np.maximum(rgb.mean(axis=0), 1e-8) - 1.0
    x = 3.0 * normalized[:, 0] - 2.0 * normalized[:, 1]
    y = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    b, a = signal.butter(3, [0.75 / (fps / 2), 2.5 / (fps / 2)], btype="bandpass")
    xf = signal.filtfilt(b, a, x)
    yf = signal.filtfilt(b, a, y)
    return xf - xf.std() / max(yf.std(), 1e-8) * yf


METHODS = {"GREEN": green, "POS": pos, "CHROM": chrom}

