#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "rPPG-Toolbox"))

import torch

from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN


def count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def record(name: str, model: torch.nn.Module, sample: torch.Tensor, *args: object) -> dict[str, object]:
    model.eval()
    with torch.inference_mode():
        output = model(sample, *args) if args else model(sample)
    primary = output[0] if isinstance(output, tuple) else output
    result = {"parameters": count(model), "input_shape": list(sample.shape), "output_shape": list(primary.shape)}
    del model, sample, output, primary
    gc.collect()
    return result


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(4)
    results = {
        "TSCAN": record("TSCAN", TSCAN(frame_depth=10, img_size=72), torch.randn(20, 6, 72, 72)),
        "DeepPhys": record("DeepPhys", DeepPhys(img_size=72), torch.randn(20, 6, 72, 72)),
        "EfficientPhys": record(
            "EfficientPhys", EfficientPhys(frame_depth=10, img_size=72), torch.randn(21, 3, 72, 72)
        ),
        "PhysNet": record(
            "PhysNet", PhysNet_padding_Encoder_Decoder_MAX(frames=32), torch.randn(1, 3, 32, 72, 72)
        ),
        "PhysFormer": record(
            "PhysFormer",
            ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(32, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
                num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7,
            ),
            torch.randn(1, 3, 32, 128, 128),
            2.0,
        ),
    }
    output = ROOT / "outputs" / "baselines" / "toolbox_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
