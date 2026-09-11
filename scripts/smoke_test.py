#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lightrppg.losses import NegativePearsonLoss
from lightrppg.models import LightRPPGViT, RPPGViT, parameter_count


def synthetic_batch(batch: int = 2, frames: int = 32, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    phase = torch.linspace(0, 4 * torch.pi, frames)
    target = torch.sin(phase)[None].repeat(batch, 1)
    video = torch.randn(batch, frames, 3, size, size) * 0.05
    video[:, :, 1] += target[:, :, None, None] * 0.15
    return video, target


def one_step(model: torch.nn.Module, video: torch.Tensor, target: torch.Tensor) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction = model(video)
    loss = NegativePearsonLoss()(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    video, target = synthetic_batch()
    standard = RPPGViT(dim=32, heads=4, cca_depth=1, swta_depth=2, window_size=8, spatial_tokens=4, dropout=0.0)
    light = LightRPPGViT(dim=32, rep_depth=1, tasa_depth=2, dropout=0.0)
    standard_loss = one_step(standard, video, target)
    light.eval()
    with torch.inference_mode():
        before = light(video)
    deployed = light.deployed_copy()
    with torch.inference_mode():
        after = deployed(video)
    max_deploy_error = float((before - after).abs().max())
    light.train()
    light_loss = one_step(light, video, target)
    result = {
        "standard_output": list(standard(video).shape),
        "light_output": list(light(video).shape),
        "standard_loss": standard_loss,
        "light_loss": light_loss,
        "repconv_max_abs_error": max_deploy_error,
        "default_rppgvit_parameters": parameter_count(RPPGViT()),
        "default_light_parameters_training": parameter_count(LightRPPGViT()),
        "default_light_parameters_deploy": parameter_count(LightRPPGViT().eval().deployed_copy()),
    }
    if max_deploy_error > 1e-4:
        raise RuntimeError(f"Reparameterization mismatch: {max_deploy_error}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

