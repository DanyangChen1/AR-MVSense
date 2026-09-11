import torch

from lightrppg.models import LightRPPGViT
from lightrppg.optimization import (
    apply_output_channel_pruning,
    fake_quantizer_count,
    make_pruning_permanent,
    prepare_eager_qat,
)


def tiny_model() -> LightRPPGViT:
    return LightRPPGViT(dim=16, rep_depth=1, tasa_depth=1, pconv_ratio=0.25)


def test_eager_qat_inserts_fake_quant_and_runs() -> None:
    model = prepare_eager_qat(tiny_model())
    output = model(torch.randn(1, 8, 3, 32, 32))
    assert output.shape == (1, 8)
    assert fake_quantizer_count(model) > 0


def test_structured_pruning_masks_whole_output_channels() -> None:
    model = tiny_model().eval()
    summary = apply_output_channel_pruning(model, 0.25)
    assert summary.modules > 0
    assert 0.0 < summary.sparsity < 1.0
    assert model(torch.randn(1, 8, 3, 32, 32)).shape == (1, 8)
    make_pruning_permanent(model)
    assert not any(hasattr(module, "weight_orig") for module in model.modules())
