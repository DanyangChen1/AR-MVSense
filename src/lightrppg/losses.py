from __future__ import annotations

import torch
from torch import nn


class NegativePearsonLoss(nn.Module):
    """Mean batch-wise 1 - Pearson correlation."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
        pred = prediction - prediction.mean(dim=-1, keepdim=True)
        truth = target - target.mean(dim=-1, keepdim=True)
        numerator = (pred * truth).sum(dim=-1)
        denominator = pred.square().sum(dim=-1).sqrt() * truth.square().sum(dim=-1).sqrt()
        correlation = numerator / denominator.clamp_min(self.eps)
        return (1.0 - correlation).mean()

