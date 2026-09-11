#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lightrppg.baselines import METHODS
from lightrppg.datasets import subject_split
from lightrppg.metrics import evaluate_collections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), default="POS")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write summary JSON and a sibling *_per_video.csv audit table.",
    )
    args = parser.parse_args()
    directories = sorted(path.parent for path in args.cache.glob("*/metadata.json"))
    metadata = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in directories]
    test_subjects = set(subject_split([str(item["subject_id"]) for item in metadata])["test"])
    collections = {}
    for directory, item in zip(directories, metadata):
        if item["subject_id"] not in test_subjects:
            continue
        frames = np.load(directory / "frames.npy", mmap_mode="r")
        target = np.load(directory / "bvp.npy", mmap_mode="r")
        prediction = METHODS[args.method](frames, args.fps)
        collections[str(item["source_id"])] = [(0, prediction, np.asarray(target))]
    metrics, rows = evaluate_collections(collections, args.fps, diff_signal=False)
    payload = {
        "method": args.method,
        "cache": str(args.cache),
        "split": "numeric subject-level 6:2:2 test partition",
        "test_subjects": sorted(test_subjects, key=lambda item: int(''.join(filter(str.isdigit, item)))),
        **metrics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        per_video = args.output.with_name(f"{args.output.stem}_per_video.csv")
        with per_video.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
