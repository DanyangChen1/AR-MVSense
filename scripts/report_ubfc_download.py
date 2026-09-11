#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Report exact UBFC download progress.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/UBFC-rPPG/DATASET_2"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/kagglehub"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/ubfc_kaggle_inventory.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inventory_payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    inventory = {
        item["name"]: int(item["totalBytes"])
        for item in inventory_payload["datasetFiles"]
    }
    cache_files = (
        args.cache
        / "datasets/malekdinarito/ubfc-rppg-dataset/versions/1"
    )
    rows: list[dict[str, object]] = []
    complete_files = 0
    downloaded_bytes = 0
    for name, expected in sorted(inventory.items()):
        raw_path = args.raw / name
        cache_path = cache_files / name
        raw_bytes = raw_path.stat().st_size if raw_path.exists() else 0
        cache_bytes = cache_path.stat().st_size if cache_path.exists() else 0
        complete = raw_bytes == expected
        current = expected if complete else min(max(raw_bytes, cache_bytes), expected)
        downloaded_bytes += current
        complete_files += int(complete)
        rows.append(
            {
                "path": name,
                "expected_bytes": expected,
                "raw_bytes": raw_bytes,
                "cache_bytes": cache_bytes,
                "downloaded_bytes": current,
                "complete": complete,
            }
        )

    total_bytes = sum(inventory.values())
    video_rows = [row for row in rows if str(row["path"]).endswith("/vid.avi")]
    label_rows = [row for row in rows if str(row["path"]).endswith("/ground_truth.txt")]
    report = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inventory_files": len(rows),
        "complete_files": complete_files,
        "complete_videos": sum(bool(row["complete"]) for row in video_rows),
        "expected_videos": len(video_rows),
        "complete_labels": sum(bool(row["complete"]) for row in label_rows),
        "expected_labels": len(label_rows),
        "downloaded_bytes": downloaded_bytes,
        "expected_bytes": total_bytes,
        "downloaded_percent": round(100.0 * downloaded_bytes / total_bytes, 3),
        "active_or_partial": [
            row for row in rows if not row["complete"] and row["downloaded_bytes"]
        ],
        "missing": [row["path"] for row in rows if not row["downloaded_bytes"]],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
