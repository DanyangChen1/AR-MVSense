#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lightrppg.config import load_config
from lightrppg.engine import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    args = parser.parse_args()
    print(json.dumps(evaluate(load_config(args.config), args.checkpoint, args.split), indent=2))


if __name__ == "__main__":
    main()
