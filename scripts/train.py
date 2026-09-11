#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lightrppg.config import load_config
from lightrppg.engine import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a checkpoint path, or from OUTPUT_DIR/last.pt when omitted.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    resume = args.resume
    if resume == "auto":
        candidate = Path(config["experiment"]["output_dir"]) / "last.pt"
        resume = candidate if candidate.exists() else None
        if resume is None:
            print(json.dumps({"resume": "requested_auto_but_no_checkpoint_found"}))
    checkpoint = train(config, resume)
    print(f"best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
