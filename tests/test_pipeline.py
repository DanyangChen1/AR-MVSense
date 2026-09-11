import json
from pathlib import Path

import numpy as np

from lightrppg.engine import evaluate, train


def _write_synthetic_cache(root: Path) -> None:
    rng = np.random.default_rng(7)
    fps = 30
    length = 17
    time = np.arange(length) / fps
    for subject in range(1, 11):
        directory = root / f"subject{subject}"
        directory.mkdir(parents=True)
        pulse = np.sin(2 * np.pi * (1.0 + 0.02 * subject) * time).astype(np.float32)
        base = rng.integers(70, 170, size=(16, 16, 3), dtype=np.uint8)
        frames = np.stack(
            [np.clip(base.astype(float) + value * np.array([2.0, 6.0, 1.0]), 0, 255) for value in pulse]
        ).astype(np.uint8)
        np.save(directory / "frames.npy", frames, allow_pickle=False)
        np.save(directory / "bvp.npy", pulse, allow_pickle=False)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset": "synthetic",
                    "source_id": f"subject{subject}",
                    "subject_id": f"subject{subject}",
                    "fps": fps,
                    "frame_count": length,
                    "cache_size": 16,
                }
            ),
            encoding="utf-8",
        )


def test_training_and_evaluation_pipeline(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    _write_synthetic_cache(cache)
    config = {
        "experiment": {"name": "synthetic", "output_dir": str(output), "seed": 42},
        "data": {
            "dataset": "synthetic",
            "cache_path": str(cache),
            "fps": 30,
            "image_size": 16,
            "clip_length": 16,
            "clip_stride": 16,
            "label_diff": True,
            "frame_diff_mode": "simple",
            "diff_alignment": "extra_frame",
            "split_mode": "numeric",
            "split_seed": 42,
            "num_workers": 0,
        },
        "model": {
            "name": "light_rppgvit",
            "dim": 16,
            "rep_depth": 1,
            "tasa_depth": 1,
            "pconv_ratio": 0.25,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "micro_batch_size": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
            "scheduler": {"name": "cosine", "decay_epochs": 1, "min_lr": 0.0001},
            "device": "cpu",
            "grad_clip": 5.0,
        },
        "evaluation": {"low_hz": 0.75, "high_hz": 2.5},
    }
    checkpoint = train(config)
    config["train"]["epochs"] = 2
    checkpoint = train(config, output / "last.pt")
    metrics = evaluate(config, checkpoint)
    assert checkpoint.exists()
    assert (output / "history.csv").exists()
    assert len((output / "history.csv").read_text(encoding="utf-8").splitlines()) == 3
    assert (output / "metrics.json").exists()
    assert metrics["n_videos"] == 2
