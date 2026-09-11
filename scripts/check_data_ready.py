#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def check_ubfc(root: Path, deep: bool) -> dict[str, object]:
    directories = sorted(path for path in root.glob("subject*") if path.is_dir())
    missing_video = [path.name for path in directories if not (path / "vid.avi").exists()]
    missing_label = [path.name for path in directories if not (path / "ground_truth.txt").exists()]
    invalid: list[str] = []
    subject_checks: list[dict[str, object]] = []
    if deep:
        for directory in directories:
            label = directory / "ground_truth.txt"
            video = directory / "vid.avi"
            row: dict[str, object] = {"subject": directory.name}
            label_length: int | None = None
            if label.exists():
                lines = label.read_text(encoding="utf-8", errors="ignore").splitlines()
                lengths = [len(np.fromstring(line, sep=" ")) for line in lines[:3]]
                row["label_line_lengths"] = lengths
                label_length = lengths[0] if len(lengths) == 3 and len(set(lengths)) == 1 else None
                if len(lines) < 3 or min(lengths, default=0) < 129 or len(set(lengths)) != 1:
                    invalid.append(f"{directory.name}:label")
            if video.exists():
                capture = cv2.VideoCapture(str(video))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                opened = capture.isOpened()
                first_ok, _ = capture.read() if opened else (False, None)
                if opened and frame_count > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                    last_ok, _ = capture.read()
                else:
                    last_ok = False
                capture.release()
                row.update(
                    {
                        "video_bytes": video.stat().st_size,
                        "video_frames": frame_count,
                        "fps": fps,
                        "width": width,
                        "height": height,
                        "decode_first": first_ok,
                        "decode_last": last_ok,
                        "labels_match_video_frames": label_length == frame_count,
                    }
                )
                if (
                    not opened
                    or frame_count < 129
                    or fps <= 0
                    or width <= 0
                    or height <= 0
                    or not first_ok
                    or not last_ok
                    or label_length != frame_count
                ):
                    invalid.append(f"{directory.name}:video")
            subject_checks.append(row)
    ready = len(directories) == 42 and not missing_video and not missing_label and not invalid
    return {
        "dataset": "UBFC-rPPG",
        "root": str(root),
        "ready": ready,
        "subject_directories": len(directories),
        "expected_subject_directories": 42,
        "videos": sum((path / "vid.avi").exists() for path in directories),
        "labels": sum((path / "ground_truth.txt").exists() for path in directories),
        "missing_video_subjects": missing_video,
        "missing_label_subjects": missing_label,
        "invalid_files": invalid,
        "deep_checked": deep,
        "subject_checks": subject_checks,
        "display_restrictions": {
            "subject27": "Do not show face images in PPT, websites, reports, or publications. Training/evaluation use is permitted by the supplied agreement table."
        },
    }


def check_mmpd(root: Path, deep: bool) -> dict[str, object]:
    directories = sorted(path for path in root.glob("subject*") if path.is_dir())
    files = [path for directory in directories for path in directory.glob("*.mat")]
    invalid = [str(path.relative_to(root)) for path in files if deep and path.stat().st_size < 1024]
    ready = len(directories) == 33 and len(files) == 660 and not invalid
    return {
        "dataset": "MMPD",
        "root": str(root),
        "ready": ready,
        "subject_directories": len(directories),
        "expected_subject_directories": 33,
        "mat_files": len(files),
        "expected_mat_files": 660,
        "invalid_files": invalid,
        "deep_checked": deep,
        "variant": "mini_or_full_must_be_recorded_separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ubfc", "mmpd"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_ubfc(args.root, args.deep) if args.dataset == "ubfc" else check_mmpd(args.root, args.deep)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
