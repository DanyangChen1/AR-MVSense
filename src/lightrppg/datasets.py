from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _subject_number(subject_id: str) -> int:
    match = re.search(r"\d+", subject_id)
    if not match:
        raise ValueError(f"Subject ID has no number: {subject_id}")
    return int(match.group())


def subject_split(
    subject_ids: list[str], mode: str = "numeric", seed: int = 42
) -> dict[str, list[str]]:
    unique = sorted(set(subject_ids), key=_subject_number)
    if mode == "seeded_random":
        random.Random(seed).shuffle(unique)
    elif mode != "numeric":
        raise ValueError(f"Unknown split mode: {mode}")
    train_end = int(0.6 * len(unique))
    valid_end = int(0.8 * len(unique))
    return {"train": unique[:train_end], "valid": unique[train_end:valid_end], "test": unique[valid_end:]}


@dataclass(frozen=True)
class ClipRecord:
    directory: Path
    source_id: str
    subject_id: str
    start: int


def temporal_difference_normalize(
    frames: np.ndarray, mode: str = "simple", eps: float = 1e-6
) -> np.ndarray:
    frames = frames.astype(np.float32) / 255.0
    difference = frames[1:] - frames[:-1]
    if mode == "relative":
        difference = difference / (frames[1:] + frames[:-1] + eps)
    elif mode != "simple":
        raise ValueError(f"Unknown frame difference mode: {mode}")
    return difference / max(float(difference.std()), eps)


def signal_difference_normalize(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    difference = np.diff(signal.astype(np.float32))
    return difference / max(float(difference.std()), eps)


class CachedRPPGDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        cache_path: str | Path,
        split: str,
        clip_length: int = 128,
        clip_stride: int = 128,
        image_size: int = 128,
        label_diff: bool = True,
        frame_diff_mode: str = "simple",
        diff_alignment: str = "extra_frame",
        split_mode: str = "numeric",
        split_seed: int = 42,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.cache_path = Path(cache_path)
        self.clip_length = clip_length
        self.image_size = image_size
        self.label_diff = label_diff
        self.frame_diff_mode = frame_diff_mode
        if diff_alignment not in {"extra_frame", "zero_tail"}:
            raise ValueError(f"Unknown difference alignment: {diff_alignment}")
        self.diff_alignment = diff_alignment
        directories = sorted(path.parent for path in self.cache_path.glob("*/metadata.json"))
        if not directories:
            raise FileNotFoundError(f"No preprocessed samples in {self.cache_path}")
        metadata = [json.loads((directory / "metadata.json").read_text(encoding="utf-8")) for directory in directories]
        split_map = subject_split(
            [str(item["subject_id"]) for item in metadata], mode=split_mode, seed=split_seed
        )
        selected = set(split_map[split])
        self.subjects = split_map
        self.records: list[ClipRecord] = []
        for directory, item in zip(directories, metadata):
            if item["subject_id"] not in selected:
                continue
            frame_count = int(item["frame_count"])
            required = clip_length + 1 if label_diff and diff_alignment == "extra_frame" else clip_length
            for start in range(0, frame_count - required + 1, clip_stride):
                self.records.append(
                    ClipRecord(directory, str(item["source_id"]), str(item["subject_id"]), start)
                )
        if not self.records:
            raise ValueError(f"No clips in split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        required = (
            self.clip_length + 1
            if self.label_diff and self.diff_alignment == "extra_frame"
            else self.clip_length
        )
        frames_file = np.load(record.directory / "frames.npy", mmap_mode="r")
        frames = np.asarray(frames_file[record.start : record.start + required])
        if frames.shape[1] != self.image_size or frames.shape[2] != self.image_size:
            frames = np.stack(
                [cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR) for frame in frames]
            )
        signal_file = np.load(record.directory / "bvp.npy", mmap_mode="r")
        signal = np.asarray(signal_file[record.start : record.start + required])
        if self.label_diff:
            video = temporal_difference_normalize(frames, self.frame_diff_mode)
            label = signal_difference_normalize(signal)
            if self.diff_alignment == "zero_tail":
                video = np.concatenate((video, np.zeros_like(video[:1])), axis=0)
                label = np.concatenate((label, np.zeros_like(label[:1])), axis=0)
        else:
            video = (frames.astype(np.float32) / 255.0 - 0.5) / 0.25
            label = (signal.astype(np.float32) - signal.mean()) / max(float(signal.std()), 1e-6)
        video = torch.from_numpy(np.ascontiguousarray(video.transpose(0, 3, 1, 2))).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(label)).float()
        return {
            "video": video,
            "label": label_tensor,
            "source_id": record.source_id,
            "subject_id": record.subject_id,
            "start": record.start,
        }
