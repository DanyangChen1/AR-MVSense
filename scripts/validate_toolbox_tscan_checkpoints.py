#!/usr/bin/env python3
"""Deterministically compute validation MSE for TSCAN model checkpoints."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    toolbox = root / 'third_party' / 'rPPG-Toolbox'
    sys.path.insert(0, str(toolbox))

    from config import get_config
    from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
    from neural_methods.model.TS_CAN import TSCAN

    config = get_config(SimpleNamespace(config_file=str(Path(args.config).resolve())))
    valid_data = UBFCrPPGLoader(
        name='valid',
        data_path=config.VALID.DATA.DATA_PATH,
        config_data=config.VALID.DATA,
        device=config.DEVICE,
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        num_workers=config.NUM_WORKERS,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(config.DEVICE)
    model = TSCAN(
        frame_depth=config.MODEL.TSCAN.FRAME_DEPTH,
        img_size=config.VALID.DATA.PREPROCESS.RESIZE.H,
    ).to(device)
    model = torch.nn.DataParallel(model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
    criterion = torch.nn.MSELoss()
    base_len = config.NUM_OF_GPU_TRAIN * config.MODEL.TSCAN.FRAME_DEPTH
    results = {}

    for checkpoint_arg in args.checkpoints:
        checkpoint = Path(checkpoint_arg).resolve()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=checkpoint.stem, ncols=100):
                data, labels = batch[0].to(device), batch[1].to(device)
                n, depth, channels, height, width = data.shape
                data = data.view(n * depth, channels, height, width)
                labels = labels.view(-1, 1)
                usable = (n * depth) // base_len * base_len
                losses.append(criterion(model(data[:usable]), labels[:usable]).item())
        mse = float(np.mean(np.asarray(losses)))
        results[checkpoint.name] = mse
        print(json.dumps({'checkpoint': str(checkpoint), 'validation_mse': mse}))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
