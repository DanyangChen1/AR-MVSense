#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lightrppg.engine import choose_device
from lightrppg.models import LightRPPGViT, RPPGViT, parameter_count


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_known_operations(model: torch.nn.Module, sample: torch.Tensor) -> dict[str, object]:
    """Count supported graph operations; unsupported operators keep this a lower bound."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise RuntimeError("Install fvcore to use --count-ops") from error
    analysis = FlopCountAnalysis(model, sample)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    known = int(analysis.total())
    return {
        "fvcore_known_fma_ops": known,
        "paper_comparable_known_fma_ops": known,
        "mul_add_as_two_ops_lower_bound": 2 * known,
        "unsupported_operators": dict(analysis.unsupported_ops()),
    }


def profile(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
    count_ops: bool,
) -> dict[str, object]:
    device = sample.device
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize(device)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    result: dict[str, object] = {
        "device": str(device),
        "parameters": parameter_count(model),
        "clip_latency_ms_median": statistics.median(times),
        "clip_latency_ms_mean": statistics.mean(times),
        "per_frame_ms_median": statistics.median(times) / sample.shape[1],
    }
    if count_ops:
        result.update(count_known_operations(model, sample))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("rppgvit", "light", "all"), default="all")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--count-ops", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = choose_device(args.device)
    sample = torch.randn(1, args.frames, 3, args.size, args.size, device=device)
    models: list[tuple[str, torch.nn.Module]] = []
    if args.model in {"rppgvit", "all"}:
        models.append(("rPPGViT", RPPGViT()))
    if args.model in {"light", "all"}:
        training_model = LightRPPGViT().eval()
        models.append(("Light-rPPGViT-training", training_model))
        models.append(("Light-rPPGViT-deploy", training_model.deployed_copy()))
    results = {}
    for name, model in models:
        results[name] = profile(
            model.to(device), sample, args.warmup, args.iterations, args.count_ops
        )
    report = {
        "metadata": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "input_shape": [1, args.frames, 3, args.size, args.size],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "operation_count_note": (
                "Paper baselines TSCAN/DeepPhys match FVCore when one fused multiply-add counts as one; "
                "paper_comparable_known_fma_ops follows that convention but excludes listed unsupported operators"
            ),
        },
        "models": results,
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
