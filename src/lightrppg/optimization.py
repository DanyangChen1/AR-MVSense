from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
from torch.nn.utils import prune


@dataclass(frozen=True)
class PruningSummary:
    modules: int
    total_weight_elements: int
    nonzero_weight_elements: int

    @property
    def sparsity(self) -> float:
        if self.total_weight_elements == 0:
            return 0.0
        return 1.0 - self.nonzero_weight_elements / self.total_weight_elements


def prepare_eager_qat(model: nn.Module, backend: str = "x86") -> nn.Module:
    """Insert explicit 8-bit fake-quantization nodes for a controlled QAT trial.

    The paper does not disclose its quantizer. This function intentionally uses
    PyTorch's named default configuration so the inferred setting is executable
    and recorded, rather than silently implying that it is the authors' setup.
    """

    model.train()
    model.qconfig = get_default_qat_qconfig(backend)
    return prepare_qat(model, inplace=False)


def fake_quantizer_count(model: nn.Module) -> int:
    return sum("FakeQuantize" in type(module).__name__ for module in model.modules())


def apply_output_channel_pruning(model: nn.Module, amount: float) -> PruningSummary:
    """Apply L1 output-channel masks without changing tensor shapes.

    Shape-preserving masks make the trial valid for residual and normalization
    paths. They measure structured sparsity, not backend-specific wall-clock
    acceleration; physical channel compaction must be reported separately.
    """

    if not 0.0 < amount < 1.0:
        raise ValueError("Structured pruning amount must be between 0 and 1")
    modules = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if module.weight.ndim < 2 or module.weight.shape[0] <= 1:
                continue
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            modules += 1
    summary = summarize_pruned_weights(model)
    return PruningSummary(modules, summary[0], summary[1])


def summarize_pruned_weights(model: nn.Module) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            weight = module.weight.detach()
            total += weight.numel()
            nonzero += int(torch.count_nonzero(weight))
    return total, nonzero


def make_pruning_permanent(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
