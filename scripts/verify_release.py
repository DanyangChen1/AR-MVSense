#!/usr/bin/env python3
"""Verify release weight hashes and load every bundled model checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from lightrppg.config import load_config
    from lightrppg.models import build_model

    manifest_path = root / 'weights' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checks = []
    for entry in manifest['weights']:
        path = root / entry['path']
        actual = sha256(path)
        if actual != entry['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {path}')
        if entry['framework'] == 'lightrppg':
            config = load_config(root / entry['config'])
            model = build_model(config['model'])
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
        elif entry['framework'] == 'rppg-toolbox':
            toolbox = root / 'third_party' / 'rPPG-Toolbox'
            sys.path.insert(0, str(toolbox))
            from neural_methods.model.TS_CAN import TSCAN
            model = torch.nn.DataParallel(TSCAN(frame_depth=10, img_size=72))
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        else:
            raise RuntimeError(f"Unknown framework: {entry['framework']}")
        checks.append({'name': entry['name'], 'sha256': actual, 'load': 'ok'})
    print(json.dumps({'release_verified': True, 'weights': checks}, indent=2))


if __name__ == '__main__':
    main()
