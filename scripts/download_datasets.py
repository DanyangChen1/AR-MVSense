#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gdown
import requests


UBFC_DATASET2_ID = "1q4vWuF2GJvKP5xyeX8dxaJ2fmq97-4ai"
UBFC_AGREEMENT_ID = "1oT_t9EHB_SzqwoQt-7VWvyLpFgCoLZT1"
UBFC_KAGGLE_HANDLE = "malekdinarito/ubfc-rppg-dataset/versions/1"
UBFC_KAGGLE_FILES_API = (
    "https://www.kaggle.com/api/v1/datasets/list/"
    "malekdinarito/ubfc-rppg-dataset?pageSize=200"
)
UBFC_SUBJECTS = (
    1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23,
    24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49,
)


def _ubfc_target(dataset_output: Path, remote_path: str) -> Path:
    relative = Path(remote_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe Drive path: {remote_path}")
    return dataset_output / relative


def _write_manifest(
    dataset_output: Path,
    failures: list[dict[str, str]],
    *,
    source: str,
) -> None:
    manifest_path = Path("data/download_manifest.json")
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    videos = list(dataset_output.glob("subject*/vid.avi"))
    labels = list(dataset_output.glob("subject*/ground_truth.txt"))
    complete = len(videos) == 42 and len(labels) == 42
    manifest["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    entry = manifest.setdefault("UBFC-rPPG", {})
    entry.update(
        {
            "status": "complete" if complete else "download_incomplete",
            "requested_subset": "DATASET_2 (42 subjects)",
            "download_source": source,
            "source_provenance": {
                "official": "https://sites.google.com/view/ybenezeth/ubfcrppg",
                "kaggle_mirror": "https://www.kaggle.com/datasets/malekdinarito/ubfc-rppg-dataset",
                "kaggle_version": 1,
                "kaggle_declared_license": "CC0: Public Domain",
                "verification": (
                    "Exact official 42-subject directory set; subject1/ground_truth.txt is byte-identical "
                    "to the official Drive copy (SHA-256 "
                    "836852f57f9cc9e04ecab134ebe506f831ad3e7d95ee73771ab8ac32ce845b67)."
                ),
                "image_restriction": (
                    "UBFC_Agreement.xlsx permits sharing subject27 data but forbids showing that "
                    "subject's image in presentations, websites, reports, or publications."
                ),
            },
            "labels_downloaded": len(labels),
            "labels_expected": 42,
            "videos_downloaded": len(videos),
            "videos_expected": 42,
            "latest_attempt": (
                f"Per-file {source} resume completed successfully."
                if complete
                else (
                    f"Per-file {source} resume left {len(failures)} failed request(s); see last_failures."
                    if failures
                    else f"{source} inventory checked; missing large videos were not attempted in this limited probe."
                )
            ),
            "last_failures": failures,
            "next_action": (
                "Run preprocessing and training."
                if complete
                else "Rerun the same command; completed files are reused and interrupted files resume."
            ),
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _download_ubfc_official(dataset_output: Path, video_limit: int | None) -> list[dict[str, str]]:
    inventory = gdown.download_folder(
        id=UBFC_DATASET2_ID,
        output=str(dataset_output),
        quiet=True,
        skip_download=True,
    )
    selected = [
        item
        for item in inventory
        if Path(item.path).name in {"ground_truth.txt", "vid.avi"}
        and Path(item.path).parent.name.startswith("subject")
    ]
    selected.sort(key=lambda item: (Path(item.path).name == "vid.avi", item.path))
    attempted_videos = 0
    failures: list[dict[str, str]] = []
    for item in selected:
        target = _ubfc_target(dataset_output, item.path)
        if target.exists() and target.stat().st_size > 0:
            continue
        is_video = target.name == "vid.avi"
        if is_video and video_limit is not None and attempted_videos >= video_limit:
            continue
        attempted_videos += int(is_video)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = gdown.download(id=item.id, output=str(target), quiet=False, resume=True)
            if not result or not target.exists() or target.stat().st_size == 0:
                raise RuntimeError("gdown returned without a non-empty file")
        except Exception as error:  # keep independent files resumable after one quota failure
            if target.exists() and target.stat().st_size == 0:
                target.unlink()
            failures.append(
                {
                    "path": item.path,
                    "drive_id": item.id,
                    "error": str(error).splitlines()[0],
                }
            )
    return failures


def _download_ubfc_kaggle(
    dataset_output: Path,
    video_limit: int | None,
    workers: int,
    retries: int,
) -> list[dict[str, str]]:
    cache_root = Path(os.environ.get("KAGGLEHUB_CACHE", ".cache/kagglehub")).expanduser().resolve()
    os.environ["KAGGLEHUB_CACHE"] = str(cache_root)
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Kaggle mirror download requires `pip install kagglehub`.") from error

    response = requests.get(UBFC_KAGGLE_FILES_API, timeout=30)
    response.raise_for_status()
    inventory = {
        item["name"]: int(item["totalBytes"])
        for item in response.json().get("datasetFiles", [])
    }
    expected_paths = {
        f"subject{subject}/{filename}"
        for subject in UBFC_SUBJECTS
        for filename in ("ground_truth.txt", "vid.avi")
    }
    if set(inventory) != expected_paths:
        missing = sorted(expected_paths - set(inventory))
        extra = sorted(set(inventory) - expected_paths)
        raise RuntimeError(f"Unexpected Kaggle UBFC inventory; missing={missing}, extra={extra}")

    candidates: list[tuple[str, Path, int]] = []
    selected_videos = 0
    for subject in UBFC_SUBJECTS:
        for filename in ("ground_truth.txt", "vid.avi"):
            target = dataset_output / f"subject{subject}" / filename
            remote_path = f"subject{subject}/{filename}"
            expected_bytes = inventory[remote_path]
            if target.exists() and target.stat().st_size == expected_bytes:
                continue
            is_video = filename == "vid.avi"
            if is_video and video_limit is not None and selected_videos >= video_limit:
                continue
            selected_videos += int(is_video)
            candidates.append((remote_path, target, expected_bytes))

    def fetch(candidate: tuple[str, Path, int]) -> tuple[str, str | None]:
        remote_path, target, expected_bytes = candidate
        cached = (
            cache_root
            / "datasets/malekdinarito/ubfc-rppg-dataset/versions/1"
            / remote_path
        )
        cached.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size != expected_bytes and not cached.exists():
            target.replace(cached)
        last_error = "download did not start"
        for attempt in range(1, retries + 1):
            try:
                result = Path(
                    kagglehub.dataset_download(
                        UBFC_KAGGLE_HANDLE,
                        path=remote_path,
                    )
                )
                if not result.exists() or result.stat().st_size != expected_bytes:
                    actual = result.stat().st_size if result.exists() else 0
                    raise RuntimeError(
                        f"download size mismatch: expected {expected_bytes} bytes, got {actual}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                result.replace(target)
                return remote_path, None
            except Exception as error:  # partial cache is reused by the next attempt
                last_error = f"attempt {attempt}/{retries}: {str(error).splitlines()[0]}"
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))
        return remote_path, last_error

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, candidate) for candidate in candidates]
        for future in as_completed(futures):
            remote_path, error = future.result()
            if error:
                failures.append({"path": remote_path, "source": "kaggle", "error": error})
    return failures


def download_ubfc(
    output: Path,
    video_limit: int | None = None,
    source: str = "official",
    workers: int = 1,
    retries: int = 3,
) -> None:
    dataset_output = output / "DATASET_2"
    dataset_output.mkdir(parents=True, exist_ok=True)
    if source == "official":
        failures = _download_ubfc_official(dataset_output, video_limit)
    elif source == "kaggle":
        failures = _download_ubfc_kaggle(dataset_output, video_limit, workers, retries)
    else:
        raise ValueError(f"Unknown UBFC source: {source}")

    access = Path("data/access")
    access.mkdir(parents=True, exist_ok=True)
    agreement = access / "UBFC_Agreement.xlsx"
    if not agreement.exists():
        try:
            gdown.download(id=UBFC_AGREEMENT_ID, output=str(agreement), quiet=False)
        except Exception as error:
            failures.append(
                {"path": "UBFC_Agreement.xlsx", "drive_id": UBFC_AGREEMENT_ID, "error": str(error).splitlines()[0]}
            )
    _write_manifest(dataset_output, failures, source=source)
    videos = list(dataset_output.glob("subject*/vid.avi"))
    labels = list(dataset_output.glob("subject*/ground_truth.txt"))
    if len(videos) != 42 or len(labels) != 42:
        raise RuntimeError(
            f"Incomplete UBFC download: {len(videos)}/42 videos, {len(labels)}/42 labels. "
            "Rerun the same command later; completed files are reused."
        )


def prepare_mmpd(output: Path) -> None:
    source = Path("third_party/MMPD_rPPG_dataset")
    access = Path("data/access")
    access.mkdir(parents=True, exist_ok=True)
    for filename in ("MMPD_Release_Agreement.pdf", "Data Usage Protocol.pdf"):
        if (source / filename).exists():
            shutil.copy2(source / filename, access / filename)
    output.mkdir(parents=True, exist_ok=True)
    print("MMPD requires an approved release agreement; no anonymous data URL is available.")
    print("Complete data/access/MMPD_Release_Agreement.pdf and use the email template in data/access/.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ubfc", "mmpd"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--video-limit",
        type=int,
        help="For quota probes, attempt at most this many missing UBFC videos; omitted means all.",
    )
    parser.add_argument(
        "--source",
        choices=("official", "kaggle"),
        default="official",
        help="UBFC source. Use the verified Kaggle mirror when the official Drive quota is exhausted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent Kaggle downloads. Use 2-4 on slow links; each file remains independently resumable.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Per-file Kaggle attempts; each retry resumes from the verified cache length.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.retries < 1 or args.retries > 10:
        parser.error("--retries must be between 1 and 10")
    if args.dataset == "ubfc":
        download_ubfc(args.output, args.video_limit, args.source, args.workers, args.retries)
    else:
        prepare_mmpd(args.output)


if __name__ == "__main__":
    main()
