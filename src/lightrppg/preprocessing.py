from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def numeric_key(path: Path) -> tuple[int, ...]:
    values = re.findall(r"\d+", path.name)
    return tuple(int(value) for value in values) or (0,)


def resample_signal(signal: np.ndarray, length: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(signal) == length:
        return signal
    old = np.linspace(0.0, 1.0, len(signal), endpoint=True)
    new = np.linspace(0.0, 1.0, length, endpoint=True)
    return np.interp(new, old, signal).astype(np.float32)


def detect_fixed_face(frames: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    for index in range(min(30, len(frames))):
        gray = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces):
            x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
            return (int(x), int(y), int(width), int(height)), f"haar_frame_{index}"
    height, width = frames[0].shape[:2]
    side = int(min(height, width) * 0.8)
    return ((width - side) // 2, (height - side) // 2, side, side), "center_fallback"


def crop_resize(frames: np.ndarray, bbox: tuple[int, int, int, int], size: int) -> np.ndarray:
    x, y, width, height = bbox
    output = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        crop = frame[y : y + height, x : x + width]
        output[index] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return output


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    success, frame = capture.read()
    while success:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = capture.read()
    capture.release()
    if not frames:
        raise ValueError(f"No readable frames in {path}")
    return np.stack(frames), fps


def read_video_cropped(path: Path, size: int) -> tuple[np.ndarray, float, tuple[int, int, int, int], str]:
    """Stream a large video and retain only resized fixed-ROI frames in memory."""
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    probe: list[np.ndarray] = []
    for _ in range(30):
        success, frame = capture.read()
        if not success:
            break
        probe.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not probe:
        raise ValueError(f"No readable frames in {path}")
    bbox, detector_status = detect_fixed_face(np.stack(probe))
    x, y, width, height = bbox
    capture = cv2.VideoCapture(str(path))
    cropped: list[np.ndarray] = []
    success, frame = capture.read()
    while success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = rgb[y : y + height, x : x + width]
        cropped.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
        success, frame = capture.read()
    capture.release()
    return np.stack(cropped).astype(np.uint8), fps, bbox, detector_status


def read_ubfc_wave(path: Path) -> np.ndarray:
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    return np.asarray([float(value) for value in first_line.split()], dtype=np.float32)


def _decode_mat_string(value: object) -> str:
    array = np.asarray(value).squeeze()
    if array.dtype.kind in {"U", "S"}:
        return str(array.tolist())
    if array.dtype.kind in {"u", "i"} and array.ndim == 1:
        return "".join(chr(int(item)) for item in array if int(item) > 0)
    return str(array.tolist())


def normalize_video_axes(frames: np.ndarray) -> np.ndarray:
    """Normalize common MATLAB/HDF5 layouts to (time, height, width, RGB)."""
    frames = np.squeeze(np.asarray(frames))
    if frames.ndim != 4:
        raise ValueError(f"Expected a four-dimensional video array, got {frames.shape}")
    color_candidates = [axis for axis, length in enumerate(frames.shape) if length in (3, 4)]
    if not color_candidates:
        raise ValueError(f"Could not locate an RGB channel axis in {frames.shape}")
    color_axis = frames.ndim - 1 if frames.shape[-1] in (3, 4) else color_candidates[0]
    frames = np.moveaxis(frames, color_axis, -1)[..., :3]
    # MMPD clips are much longer than either spatial dimension.  MATLAB v7.3
    # files opened by h5py commonly reverse the apparent axis order.
    time_axis = int(np.argmax(frames.shape[:-1]))
    frames = np.moveaxis(frames, time_axis, 0)
    return frames


def read_mmpd_mat(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    try:
        content = loadmat(path, squeeze_me=True)
        frames = np.asarray(content["video"])
        signal = np.asarray(content["GT_ppg"]).reshape(-1)
        info = {
            key: _decode_mat_string(content[key])
            for key in ("light", "motion", "exercise", "skin_color", "gender", "glasser", "hair_cover", "makeup")
            if key in content
        }
    except (NotImplementedError, ValueError, OSError):
        with h5py.File(path, "r") as handle:
            frames = np.asarray(handle["video"])
            signal = np.asarray(handle["GT_ppg"]).reshape(-1)
            info = {}
    frames = normalize_video_axes(frames)
    if frames.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(frames)) <= 1.5 else 1.0
        frames = np.clip(np.rint(frames * scale), 0, 255).astype(np.uint8)
    return frames, signal.astype(np.float32), info


def _save_cache(
    output: Path,
    frames: np.ndarray,
    signal: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "frames.npy", frames, allow_pickle=False)
    np.save(output / "bvp.npy", signal.astype(np.float32), allow_pickle=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def preprocess_ubfc(
    input_root: Path,
    output_root: Path,
    size: int,
    force: bool = False,
    skip_incomplete: bool = False,
) -> None:
    subjects = sorted(input_root.glob("subject*"), key=numeric_key)
    if not subjects:
        raise FileNotFoundError(f"No subject folders under {input_root}")
    for subject in tqdm(subjects, desc="UBFC preprocessing"):
        missing = [name for name in ("vid.avi", "ground_truth.txt") if not (subject / name).exists()]
        if missing:
            if skip_incomplete:
                continue
            raise FileNotFoundError(f"Missing {', '.join(missing)} in {subject}")
        target = output_root / subject.name
        if (target / "metadata.json").exists() and not force:
            continue
        face_frames, fps, bbox, detector_status = read_video_cropped(subject / "vid.avi", size)
        bvp = resample_signal(read_ubfc_wave(subject / "ground_truth.txt"), len(face_frames))
        _save_cache(
            target,
            face_frames,
            bvp,
            {
                "dataset": "UBFC-rPPG",
                "source_id": subject.name,
                "subject_id": subject.name,
                "fps_original": fps,
                "fps": 30,
                "frame_count": len(face_frames),
                "cache_size": size,
                "bbox_xywh": bbox,
                "roi_status": detector_status,
            },
        )


def preprocess_mmpd(input_root: Path, output_root: Path, size: int, force: bool = False) -> None:
    subjects = sorted(input_root.glob("subject*"), key=numeric_key)
    if not subjects:
        raise FileNotFoundError(f"No subject folders under {input_root}")
    files = [path for subject in subjects for path in sorted(subject.glob("*.mat"), key=numeric_key)]
    for path in tqdm(files, desc="MMPD preprocessing"):
        subject_id = path.parent.name
        source_id = f"{subject_id}_{path.stem}"
        target = output_root / source_id
        if (target / "metadata.json").exists() and not force:
            continue
        frames, signal, info = read_mmpd_mat(path)
        bbox, detector_status = detect_fixed_face(frames)
        face_frames = crop_resize(frames, bbox, size)
        bvp = resample_signal(signal, len(face_frames))
        _save_cache(
            target,
            face_frames,
            bvp,
            {
                "dataset": "MMPD",
                "source_id": source_id,
                "subject_id": subject_id,
                "fps_original": 30,
                "fps": 30,
                "frame_count": len(face_frames),
                "cache_size": size,
                "bbox_xywh": bbox,
                "roi_status": detector_status,
                "conditions": info,
            },
        )
