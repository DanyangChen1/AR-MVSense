#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "rPPG-Toolbox"))

import torch
from fvcore.nn import FlopCountAnalysis

from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN


def primary(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def profile(
    model: torch.nn.Module,
    sample: torch.Tensor,
    call: Callable[[], Any],
    represented_frames: int,
    warmup: int = 1,
    iterations: int = 3,
) -> dict[str, object]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            primary(call())
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = primary(call())
            times.append((time.perf_counter() - start) * 1000)
    if isinstance(model, ViT_ST_ST_Compact3_TDC_gra_sharp):
        analysis = FlopCountAnalysis(model, (sample, 2.0))
    else:
        analysis = FlopCountAnalysis(model, sample)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    known = int(analysis.total())
    scale = 128.0 / represented_frames
    normalized = int(round(known * scale))
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(sample.shape),
        "represented_output_frames": represented_frames,
        "clip_latency_ms_median": statistics.median(times),
        "per_output_frame_ms_median": statistics.median(times) / represented_frames,
        "fvcore_known_fma_ops_actual_input": known,
        "fvcore_known_fma_ops_normalized_128_frames": normalized,
        "paper_comparable_known_fma_ops_normalized_128_frames": normalized,
        "mul_add_as_two_ops_128_frame_lower_bound": 2 * normalized,
        "unsupported_operators": dict(analysis.unsupported_ops()),
        "normalization_note": (
            "exact 128-frame input" if represented_frames == 128
            else "TSM requires multiples of 10; operation count scaled linearly from 120 outputs"
        ),
    }
    del output
    return result


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(4)
    results: dict[str, object] = {}

    model = TSCAN(frame_depth=10, img_size=72)
    sample = torch.randn(120, 6, 72, 72)
    results["TSCAN"] = profile(model, sample, lambda: model(sample), 120)
    del model, sample
    gc.collect()

    model = DeepPhys(img_size=72)
    sample = torch.randn(128, 6, 72, 72)
    results["DeepPhys"] = profile(model, sample, lambda: model(sample), 128)
    del model, sample
    gc.collect()

    model = EfficientPhys(frame_depth=10, img_size=72)
    sample = torch.randn(121, 3, 72, 72)
    results["EfficientPhys"] = profile(model, sample, lambda: model(sample), 120)
    del model, sample
    gc.collect()

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
    sample = torch.randn(1, 3, 128, 72, 72)
    results["PhysNet"] = profile(model, sample, lambda: model(sample), 128)
    del model, sample
    gc.collect()

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(128, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
        num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7,
    )
    sample = torch.randn(1, 3, 128, 128, 128)
    results["PhysFormer"] = profile(model, sample, lambda: model(sample, 2.0), 128)

    report = {
        "metadata": {
            "device": "Apple M2 CPU",
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "warmup": 1,
            "iterations": 3,
            "latency_note": "Local CPU clip latency is not comparable to Snapdragon XR2 per-frame latency",
            "operation_note": "Paper TSCAN/DeepPhys values match the one-FMA count; unsupported operators still make some models lower bounds",
        },
        "models": results,
    }
    output = ROOT / "outputs" / "benchmarks" / "toolbox_baselines_apple_m2_cpu.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
