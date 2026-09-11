import numpy as np

from lightrppg.preprocessing import normalize_video_axes


def test_normalize_video_axes_handles_scipy_layout() -> None:
    video = np.zeros((180, 80, 60, 3), dtype=np.uint8)
    assert normalize_video_axes(video).shape == (180, 80, 60, 3)


def test_normalize_video_axes_handles_hdf5_reversed_layout() -> None:
    video = np.zeros((3, 60, 80, 180), dtype=np.uint8)
    assert normalize_video_axes(video).shape == (180, 60, 80, 3)
