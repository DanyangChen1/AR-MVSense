#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lightrppg.metrics import fft_hr, reconstruct_and_filter, summarize_hr


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UBFC three-line ground-truth files.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(args.root.glob("subject*/ground_truth.txt"))
    if not files:
        raise FileNotFoundError(f"No UBFC labels under {args.root}")

    inferred_rates: list[float] = []
    fft_estimates: list[float] = []
    monitor_means: list[float] = []
    invalid_hr_samples = 0
    lengths: list[int] = []
    rows: list[dict[str, object]] = []
    for path in files:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 3:
            raise ValueError(f"Expected BVP, HR and timestamp lines in {path}")
        bvp, monitor_hr, timestamps = (np.fromstring(line, sep=" ") for line in lines[:3])
        if not (len(bvp) == len(monitor_hr) == len(timestamps)):
            raise ValueError(f"Ground-truth line lengths differ in {path}")
        valid_hr = monitor_hr[(monitor_hr >= 40.0) & (monitor_hr <= 240.0)]
        invalid_hr_samples += int(len(monitor_hr) - len(valid_hr))
        inferred_fps = float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))
        estimate = fft_hr(
            reconstruct_and_filter(bvp, inferred_fps, diff_signal=False), inferred_fps
        )
        reference = float(valid_hr.mean())
        inferred_rates.append(inferred_fps)
        fft_estimates.append(estimate)
        monitor_means.append(reference)
        lengths.append(len(bvp))
        rows.append(
            {
                "subject": path.parent.name,
                "samples": len(bvp),
                "inferred_sampling_hz": inferred_fps,
                "bvp_fft_hr_bpm": estimate,
                "valid_monitor_hr_mean_bpm": reference,
            }
        )

    report = {
        "scope_note": "Label-only audit; this is not a model accuracy result.",
        "subjects_available": len(files),
        "expected_subjects": 42,
        "samples_per_subject": {
            "min": min(lengths),
            "median": float(np.median(lengths)),
            "max": max(lengths),
        },
        "timestamp_inferred_sampling_hz": {
            "min": min(inferred_rates),
            "median": float(np.median(inferred_rates)),
            "max": max(inferred_rates),
        },
        "invalid_monitor_hr_samples_outside_40_240_bpm": invalid_hr_samples,
        "bvp_fft_vs_valid_monitor_hr_mean": summarize_hr(
            np.asarray(fft_estimates), np.asarray(monitor_means)
        ),
        "subjects": rows,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
