#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lightrppg.preprocessing import preprocess_mmpd, preprocess_ubfc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("ubfc", "mmpd"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Preprocess only complete samples; useful while a resumable dataset download is still running.",
    )
    args = parser.parse_args()
    if args.dataset == "ubfc":
        preprocess_ubfc(args.input, args.output, args.size, args.force, args.skip_incomplete)
    else:
        preprocess_mmpd(args.input, args.output, args.size, args.force)


if __name__ == "__main__":
    main()
