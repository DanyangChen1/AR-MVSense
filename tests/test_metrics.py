import numpy as np
import pytest

from lightrppg.metrics import fft_hr, reconstruct_and_filter, summarize_hr


def test_fft_recovers_known_heart_rate() -> None:
    fps = 30
    seconds = 30
    frequency = 1.25
    time = np.arange(fps * seconds) / fps
    waveform = np.sin(2 * np.pi * frequency * time)
    assert abs(fft_hr(waveform, fps) - 75.0) < 1.0


def test_metrics_are_consistent() -> None:
    result = summarize_hr(np.array([60.0, 80.0]), np.array([62.0, 78.0]))
    assert result["MAE"] == 2.0
    assert result["RMSE"] == 2.0
    assert result["Pearson"] == pytest.approx(1.0)
